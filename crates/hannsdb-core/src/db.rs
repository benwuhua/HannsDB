use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap, HashMap, HashSet};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, RwLock};

use hannsdb_index::descriptor::{ScalarIndexDescriptor, VectorIndexDescriptor};
use hannsdb_index::scalar::{InvertedScalarIndex, RangeOp, ScalarValue};
use hannsdb_index::sparse::{SparseIndexBackend, SparseVectorData};

use crate::catalog::{CollectionMetadata, IndexCatalog, ManifestMetadata};
use crate::document::{
    field_value_to_scalar, validate_documents, validate_schema_primary_vector_descriptor,
    validate_schema_secondary_vector_descriptors, validate_vector_index_descriptor,
    CollectionSchema, Document, DocumentUpdate, FieldValue, ScalarFieldSchema, SparseVector,
    VectorFieldSchema,
};
use crate::pk::{PrimaryKeyMode, PrimaryKeyRegistry};
use crate::query::{
    compare_hits, distance_by_metric, parse_filter, project_hits_output_fields,
    resolve_vector_descriptor_for_field, search_by_metric, search_sparse_bruteforce,
    sort_hits_by_field, ComparisonOp, FilterExpr, QueryContext, QueryExecutor, QueryPlan,
    QueryPlanner, SearchHit,
};
#[cfg(feature = "hanns-backend")]
use crate::segment::index_runtime::ann_search_with_bitset;
use crate::segment::index_runtime::{
    ann_blob_path, ann_ids_path, ann_search, build_optimized_ann_state, invalidate_ann_blobs,
    CachedSearchState,
};
#[cfg(all(test, feature = "hanns-backend"))]
use crate::segment::{append_record_ids, append_records, SegmentPaths};
use crate::segment::{
    SegmentManager, SegmentMetadata, SegmentSet, SegmentWriter, TombstoneMask, VersionSet,
};
use crate::storage::compaction::compact_immutable_segments;
use crate::storage::paths::{
    collection_paths_for_dir, collection_paths_for_root, manifest_path, wal_path, CollectionPaths,
};
use crate::storage::persist;
use crate::storage::primary_keys::{
    assign_internal_ids_for_public_keys, display_key_for_internal_id, load_primary_key_registry,
    resolve_public_keys_to_internal_ids, upsert_public_keys_with_internal_ids,
};
use crate::storage::recovery::{collection_name_for_wal_record, WalReplayPlan};
use crate::storage::segment_io::{
    load_external_ids_for_segment_or_empty, load_payloads_or_empty,
    load_payloads_with_fields_or_empty, load_primary_dense_rows_for_segment_or_empty,
    load_shadowed_live_records, load_shadowed_live_vector_records, load_sparse_vectors_or_empty,
    load_vectors_or_empty, materialize_active_segment_arrow_snapshots,
    materialize_forward_store_snapshot, persisted_ann_exists,
};
use crate::storage::tombstone;
use crate::storage::wal::{
    append_wal_record, load_wal_records, load_wal_records_or_empty, truncate_wal, WalRecord,
};
use crate::wal::{AddColumnBackfill, AlterColumnMigration};

const DEFAULT_EF_SEARCH: usize = 32;
const DEFAULT_NPROBE: usize = 32;

fn validate_add_column_backfill(
    field: &ScalarFieldSchema,
    backfill: Option<&AddColumnBackfill>,
) -> io::Result<()> {
    let Some(backfill) = backfill else {
        return Ok(());
    };

    if field.array {
        return Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "add_column constant backfill does not support array fields yet",
        ));
    }

    match backfill {
        AddColumnBackfill::Constant { value: None } => {
            if !field.nullable {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "null constant backfill requires a nullable field",
                ));
            }
        }
        AddColumnBackfill::Constant { value: Some(value) } => match (&field.data_type, value) {
            (crate::document::FieldType::String, FieldValue::String(_))
            | (crate::document::FieldType::Int64, FieldValue::Int64(_))
            | (crate::document::FieldType::Int32, FieldValue::Int32(_))
            | (crate::document::FieldType::UInt32, FieldValue::UInt32(_))
            | (crate::document::FieldType::UInt64, FieldValue::UInt64(_))
            | (crate::document::FieldType::Float, FieldValue::Float(_))
            | (crate::document::FieldType::Float64, FieldValue::Float64(_))
            | (crate::document::FieldType::Bool, FieldValue::Bool(_)) => {}
            (crate::document::FieldType::VectorFp32, _)
            | (crate::document::FieldType::VectorFp16, _)
            | (crate::document::FieldType::VectorSparse, _)
            | (_, FieldValue::Array(_)) => {
                return Err(io::Error::new(
                    io::ErrorKind::Unsupported,
                    "add_column constant backfill only supports scalar field types",
                ));
            }
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "constant backfill value does not match the destination field type",
                ));
            }
        },
    }

    Ok(())
}

fn widen_field_value(
    value: &FieldValue,
    migration: AlterColumnMigration,
) -> io::Result<FieldValue> {
    match migration {
        AlterColumnMigration::Int32ToInt64 | AlterColumnMigration::RenameAndInt32ToInt64 => {
            match value {
                FieldValue::Int32(v) => Ok(FieldValue::Int64(*v as i64)),
                FieldValue::Int64(v) => Ok(FieldValue::Int64(*v)),
                _ => Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "existing field value does not match int32 -> int64 widening",
                )),
            }
        }
        AlterColumnMigration::UInt32ToUInt64 | AlterColumnMigration::RenameAndUInt32ToUInt64 => {
            match value {
                FieldValue::UInt32(v) => Ok(FieldValue::UInt64(*v as u64)),
                FieldValue::UInt64(v) => Ok(FieldValue::UInt64(*v)),
                _ => Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "existing field value does not match uint32 -> uint64 widening",
                )),
            }
        }
        AlterColumnMigration::FloatToFloat64 | AlterColumnMigration::RenameAndFloatToFloat64 => {
            match value {
                FieldValue::Float(v) => Ok(FieldValue::Float64(*v as f64)),
                FieldValue::Float64(v) => Ok(FieldValue::Float64(*v)),
                _ => Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "existing field value does not match float -> float64 widening",
                )),
            }
        }
    }
}

pub struct HannsDb {
    root: PathBuf,
    read_only: bool,
    collection_handles: RwLock<HashMap<String, Arc<CollectionHandle>>>,
}

pub use crate::db_types::{CollectionInfo, CollectionSegmentInfo};
pub use crate::query::DocumentHit;

pub struct CollectionHandle {
    segment_manager: SegmentManager,
    version_set: RwLock<VersionSet>,
    search_cache: Mutex<HashMap<String, CachedSearchState>>,
    scalar_cache: Mutex<HashMap<String, InvertedScalarIndex>>,
    sparse_index_cache: Mutex<HashMap<String, Box<dyn SparseIndexBackend>>>,
    document_cache: Mutex<Option<Arc<HashMap<i64, Document>>>>,
}

impl HannsDb {
    pub fn open(root: &Path) -> io::Result<Self> {
        Self::open_internal(root, false)
    }

    pub fn open_read_only(root: &Path) -> io::Result<Self> {
        Self::open_internal(root, true)
    }

    fn open_internal(root: &Path, read_only: bool) -> io::Result<Self> {
        fs::create_dir_all(root)?;
        fs::create_dir_all(root.join("collections"))?;

        let manifest_path = manifest_path(root);
        if !manifest_path.exists() {
            ManifestMetadata::new("hannsdb-local", Vec::new()).save_to_path(&manifest_path)?;
        } else {
            let _ = ManifestMetadata::load_from_path(&manifest_path)?;
        }

        let mut db = Self {
            root: root.to_path_buf(),
            read_only,
            collection_handles: RwLock::new(HashMap::new()),
        };
        db.replay_wal_if_needed()?;
        Ok(db)
    }

    fn require_write(&self) -> io::Result<()> {
        if self.read_only {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "database is opened in read-only mode",
            ));
        }
        Ok(())
    }

    pub fn open_collection_handle(&self, name: &str) -> io::Result<Arc<CollectionHandle>> {
        if let Some(handle) = self
            .collection_handles
            .read()
            .expect("collection handles rwlock poisoned")
            .get(name)
            .cloned()
        {
            return Ok(handle);
        }

        let mut handles = self
            .collection_handles
            .write()
            .expect("collection handles rwlock poisoned");
        if let Some(handle) = handles.get(name) {
            return Ok(Arc::clone(handle));
        }

        let paths = self.collection_paths(name);
        let _ = CollectionMetadata::load_from_path(&paths.collection_meta)?;
        let segment_manager = SegmentManager::new(paths.dir.clone());
        let version_set = segment_manager.version_set()?;
        let handle = Arc::new(CollectionHandle::new(segment_manager, version_set));
        handles.insert(name.to_string(), Arc::clone(&handle));
        Ok(handle)
    }

    pub fn create_collection(
        &mut self,
        name: &str,
        dimension: usize,
        metric: &str,
    ) -> io::Result<()> {
        self.create_collection_with_schema(
            name,
            &CollectionSchema::new("vector", dimension, metric, Vec::new()),
        )
    }

    pub fn create_collection_with_schema(
        &mut self,
        name: &str,
        schema: &CollectionSchema,
    ) -> io::Result<()> {
        self.require_write()?;
        self.create_collection_with_schema_internal(name, schema, true)
    }

    fn create_collection_with_schema_internal(
        &mut self,
        name: &str,
        schema: &CollectionSchema,
        log_wal: bool,
    ) -> io::Result<()> {
        let _primary_vector = schema.primary_vector().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "collection primary vector '{}' is not defined in schema vectors",
                    schema.primary_vector_name()
                ),
            )
        })?;
        validate_schema_primary_vector_descriptor(schema)?;
        validate_schema_secondary_vector_descriptors(schema)?;
        let dimension = schema.dimension();
        if dimension == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "collection dimension must be > 0",
            ));
        }

        let paths = self.collection_paths(name);
        if paths.collection_meta.exists() {
            return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                format!("collection already exists: {name}"),
            ));
        }

        if log_wal {
            append_wal_record(
                &wal_path(&self.root),
                &WalRecord::CreateCollection {
                    collection: name.to_string(),
                    schema: schema.clone(),
                },
            )?;
        }

        fs::create_dir_all(&paths.dir)?;

        let collection = CollectionMetadata::new_with_schema(name, schema.clone());
        collection.save_to_path(&paths.collection_meta)?;
        PrimaryKeyRegistry::new(PrimaryKeyMode::Numeric, 1).save_to_path(&paths.primary_keys)?;

        let segment = SegmentMetadata::new("seg-0001", dimension, 0, 0);
        segment.save_to_path(&paths.segment_meta)?;

        let tombstone = TombstoneMask::new(0);
        tombstone.save_to_path(&paths.tombstones)?;

        let mut manifest = ManifestMetadata::load_from_path(&manifest_path(&self.root))?;
        if !manifest.collections.iter().any(|entry| entry == name) {
            manifest.collections.push(name.to_string());
            manifest.save_to_path(&manifest_path(&self.root))?;
        }

        self.invalidate_search_cache(name);
        Ok(())
    }

    pub fn add_column(&mut self, collection: &str, field: ScalarFieldSchema) -> io::Result<()> {
        self.require_write()?;
        self.add_column_internal(collection, field, None, true)
    }

    pub fn add_column_with_backfill(
        &mut self,
        collection: &str,
        field: ScalarFieldSchema,
        backfill: Option<AddColumnBackfill>,
    ) -> io::Result<()> {
        self.require_write()?;
        self.add_column_internal(collection, field, backfill, true)
    }

    fn add_column_internal(
        &mut self,
        collection: &str,
        field: ScalarFieldSchema,
        backfill: Option<AddColumnBackfill>,
        log_wal: bool,
    ) -> io::Result<()> {
        let paths = self.collection_paths(collection);
        let mut meta = CollectionMetadata::load_from_path(&paths.collection_meta)?;

        crate::catalog::schema_mutation::add_field_to_schema(&mut meta.fields, field.clone())?;
        validate_add_column_backfill(&field, backfill.as_ref())?;

        if log_wal {
            append_wal_record(
                &wal_path(&self.root),
                &WalRecord::AddColumn {
                    collection: collection.to_string(),
                    field: field.clone(),
                    backfill: backfill.clone(),
                },
            )?;
        }

        meta.save_to_path(&paths.collection_meta)?;
        self.invalidate_search_cache(collection);
        if let Some(backfill) = backfill.as_ref() {
            self.apply_add_column_backfill(collection, &field, backfill)?;
        }
        Ok(())
    }

    fn apply_add_column_backfill(
        &mut self,
        collection: &str,
        field: &ScalarFieldSchema,
        backfill: &AddColumnBackfill,
    ) -> io::Result<()> {
        let state = self
            .open_collection_handle(collection)?
            .cached_document_state()?;
        if state.is_empty() {
            return Ok(());
        }

        let mut ids: Vec<i64> = state.keys().copied().collect();
        ids.sort_unstable();

        let updates = match backfill {
            AddColumnBackfill::Constant { value: Some(value) } => ids
                .into_iter()
                .map(|id| DocumentUpdate {
                    id,
                    fields: [(field.name.clone(), Some::<FieldValue>(value.clone()))]
                        .into_iter()
                        .collect(),
                    vectors: BTreeMap::new(),
                    sparse_vectors: BTreeMap::new(),
                })
                .collect::<Vec<_>>(),
            AddColumnBackfill::Constant { value: None } => return Ok(()),
        };

        self.update_documents_internal(collection, &updates, false)?;
        Ok(())
    }

    pub fn drop_column(&mut self, collection: &str, field_name: &str) -> io::Result<()> {
        self.require_write()?;
        self.drop_column_internal(collection, field_name, true)
    }

    fn drop_column_internal(
        &mut self,
        collection: &str,
        field_name: &str,
        log_wal: bool,
    ) -> io::Result<()> {
        let paths = self.collection_paths(collection);
        let mut meta = CollectionMetadata::load_from_path(&paths.collection_meta)?;

        if log_wal {
            append_wal_record(
                &wal_path(&self.root),
                &WalRecord::DropColumn {
                    collection: collection.to_string(),
                    field_name: field_name.to_string(),
                },
            )?;
        }

        crate::catalog::schema_mutation::remove_field_from_schema(&mut meta.fields, field_name)?;
        meta.save_to_path(&paths.collection_meta)?;
        self.invalidate_search_cache(collection);
        Ok(())
    }

    pub fn alter_column(
        &mut self,
        collection: &str,
        old_name: &str,
        new_name: &str,
    ) -> io::Result<()> {
        self.require_write()?;
        self.alter_column_internal(collection, old_name, new_name, None, None, true)
    }

    pub fn alter_column_with_field_schema(
        &mut self,
        collection: &str,
        old_name: &str,
        new_name: &str,
        field: ScalarFieldSchema,
        migration: AlterColumnMigration,
    ) -> io::Result<()> {
        self.require_write()?;
        self.alter_column_internal(
            collection,
            old_name,
            new_name,
            Some(field),
            Some(migration),
            true,
        )
    }

    fn alter_column_internal(
        &mut self,
        collection: &str,
        old_name: &str,
        new_name: &str,
        field: Option<ScalarFieldSchema>,
        migration: Option<AlterColumnMigration>,
        log_wal: bool,
    ) -> io::Result<()> {
        let paths = self.collection_paths(collection);
        let mut meta = CollectionMetadata::load_from_path(&paths.collection_meta)?;

        if let (Some(field), Some(migration)) = (&field, migration) {
            self.validate_alter_column_migration(collection, &meta, old_name, field, migration)?;
        }

        if log_wal {
            append_wal_record(
                &wal_path(&self.root),
                &WalRecord::AlterColumn {
                    collection: collection.to_string(),
                    old_name: old_name.to_string(),
                    new_name: new_name.to_string(),
                    field: field.clone(),
                    migration,
                },
            )?;
        }

        if let (Some(field), Some(migration)) = (&field, migration) {
            self.apply_alter_column_migration(collection, &mut meta, old_name, field, migration)?;
        } else {
            crate::catalog::schema_mutation::rename_field_in_schema(
                &mut meta.fields,
                old_name,
                new_name,
            )?;
        }
        meta.save_to_path(&paths.collection_meta)?;
        self.invalidate_search_cache(collection);
        Ok(())
    }

    fn validate_alter_column_migration(
        &self,
        collection: &str,
        meta: &CollectionMetadata,
        old_name: &str,
        target_field: &ScalarFieldSchema,
        migration: AlterColumnMigration,
    ) -> io::Result<()> {
        let current_field = meta
            .fields
            .iter()
            .find(|field| field.name == old_name)
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::NotFound,
                    format!("field not found: {old_name}"),
                )
            })?;

        let is_rename = target_field.name != old_name;
        if target_field.name.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "field_schema migration requires a non-empty target field name",
            ));
        }
        if is_rename
            && meta
                .fields
                .iter()
                .any(|field| field.name == target_field.name && field.name != old_name)
        {
            return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                format!("field already exists: {}", target_field.name),
            ));
        }
        if target_field.nullable != current_field.nullable {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "nullable changes are not supported in this lane",
            ));
        }
        if target_field.array != current_field.array {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "array changes are not supported in this lane",
            ));
        }

        let indexed = self
            .list_scalar_indexes(collection)?
            .into_iter()
            .any(|descriptor| descriptor.field_name == old_name);
        if indexed {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "field_schema migration is not supported for fields with scalar index descriptors",
            ));
        }

        let valid_pair = match migration {
            AlterColumnMigration::Int32ToInt64 => {
                !is_rename
                    && current_field.data_type == crate::document::FieldType::Int32
                    && target_field.data_type == crate::document::FieldType::Int64
            }
            AlterColumnMigration::UInt32ToUInt64 => {
                !is_rename
                    && current_field.data_type == crate::document::FieldType::UInt32
                    && target_field.data_type == crate::document::FieldType::UInt64
            }
            AlterColumnMigration::FloatToFloat64 => {
                !is_rename
                    && current_field.data_type == crate::document::FieldType::Float
                    && target_field.data_type == crate::document::FieldType::Float64
            }
            AlterColumnMigration::RenameAndInt32ToInt64 => {
                is_rename
                    && current_field.data_type == crate::document::FieldType::Int32
                    && target_field.data_type == crate::document::FieldType::Int64
            }
            AlterColumnMigration::RenameAndUInt32ToUInt64 => {
                is_rename
                    && current_field.data_type == crate::document::FieldType::UInt32
                    && target_field.data_type == crate::document::FieldType::UInt64
            }
            AlterColumnMigration::RenameAndFloatToFloat64 => {
                is_rename
                    && current_field.data_type == crate::document::FieldType::Float
                    && target_field.data_type == crate::document::FieldType::Float64
            }
        };

        if !valid_pair {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "field_schema migration supports only the widening subset in this lane",
            ));
        }

        Ok(())
    }

    fn apply_alter_column_migration(
        &mut self,
        collection: &str,
        meta: &mut CollectionMetadata,
        old_name: &str,
        target_field: &ScalarFieldSchema,
        migration: AlterColumnMigration,
    ) -> io::Result<()> {
        let state = self
            .open_collection_handle(collection)?
            .cached_document_state()?;
        let mut ids: Vec<i64> = state.keys().copied().collect();
        ids.sort_unstable();
        let mut updates = Vec::new();

        for id in ids {
            let Some(document) = state.get(&id) else {
                continue;
            };
            let source_name = if document.fields.contains_key(old_name) {
                old_name
            } else if document.fields.contains_key(target_field.name.as_str()) {
                target_field.name.as_str()
            } else {
                continue;
            };
            let value = document
                .fields
                .get(source_name)
                .expect("checked source field presence");
            let widened = widen_field_value(value, migration)?;
            let already_final = source_name == target_field.name.as_str() && &widened == value;
            if already_final {
                continue;
            }
            let mut fields = BTreeMap::new();
            if source_name == old_name && target_field.name != old_name {
                fields.insert(old_name.to_string(), None);
            }
            fields.insert(target_field.name.clone(), Some::<FieldValue>(widened));
            updates.push(DocumentUpdate {
                id,
                fields,
                vectors: BTreeMap::new(),
                sparse_vectors: BTreeMap::new(),
            });
        }

        let field = meta
            .fields
            .iter_mut()
            .find(|field| field.name == old_name)
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::NotFound,
                    format!("field not found: {old_name}"),
                )
            })?;
        field.name = target_field.name.clone();
        field.data_type = target_field.data_type.clone();
        field.nullable = target_field.nullable;
        field.array = target_field.array;

        if !updates.is_empty() {
            let paths = self.collection_paths(collection);
            meta.save_to_path(&paths.collection_meta)?;
            self.update_documents_internal(collection, &updates, false)?;
        }

        Ok(())
    }

    pub fn add_vector_field(
        &mut self,
        collection: &str,
        field: VectorFieldSchema,
    ) -> io::Result<()> {
        self.require_write()?;
        self.add_vector_field_internal(collection, field, true)
    }

    fn add_vector_field_internal(
        &mut self,
        collection: &str,
        field: VectorFieldSchema,
        log_wal: bool,
    ) -> io::Result<()> {
        let paths = self.collection_paths(collection);
        let mut meta = CollectionMetadata::load_from_path(&paths.collection_meta)?;

        // Also check scalar fields for name collisions.
        if meta.fields.iter().any(|f| f.name == field.name) {
            return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                format!("a scalar field with name '{}' already exists", field.name),
            ));
        }

        crate::catalog::schema_mutation::add_vector_field_to_schema(
            &mut meta.vectors,
            field.clone(),
        )?;

        if log_wal {
            append_wal_record(
                &wal_path(&self.root),
                &WalRecord::AddVectorField {
                    collection: collection.to_string(),
                    field,
                },
            )?;
        }

        meta.save_to_path(&paths.collection_meta)?;
        self.invalidate_search_cache(collection);
        Ok(())
    }

    pub fn drop_vector_field(&mut self, collection: &str, field_name: &str) -> io::Result<()> {
        self.require_write()?;
        self.drop_vector_field_internal(collection, field_name, true)
    }

    fn drop_vector_field_internal(
        &mut self,
        collection: &str,
        field_name: &str,
        log_wal: bool,
    ) -> io::Result<()> {
        let paths = self.collection_paths(collection);
        let mut meta = CollectionMetadata::load_from_path(&paths.collection_meta)?;

        crate::catalog::schema_mutation::remove_vector_field_from_schema(
            &mut meta.vectors,
            field_name,
            &meta.primary_vector,
        )?;

        if log_wal {
            append_wal_record(
                &wal_path(&self.root),
                &WalRecord::DropVectorField {
                    collection: collection.to_string(),
                    field_name: field_name.to_string(),
                },
            )?;
        }

        // Drop any associated vector index descriptor.
        if paths.index_catalog.exists() {
            if let Ok(mut catalog) = IndexCatalog::load_from_path(&paths.index_catalog) {
                catalog.drop_vector_index(field_name);
                catalog.save_to_path(&paths.index_catalog)?;
            }
        }

        // Remove persisted ANN blobs for this field.
        let ann_blob = ann_blob_path(&paths.dir, field_name);
        if ann_blob.exists() {
            let _ = fs::remove_file(&ann_blob);
        }
        let ann_ids = ann_ids_path(&paths.dir, field_name);
        if ann_ids.exists() {
            let _ = fs::remove_file(&ann_ids);
        }

        meta.save_to_path(&paths.collection_meta)?;
        self.invalidate_search_cache(collection);
        Ok(())
    }

    pub fn drop_collection(&mut self, name: &str) -> io::Result<()> {
        self.require_write()?;
        self.drop_collection_internal(name, true)
    }

    fn drop_collection_internal(&mut self, name: &str, log_wal: bool) -> io::Result<()> {
        let paths = self.collection_paths(name);
        if !paths.dir.exists() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!("collection not found: {name}"),
            ));
        }

        if log_wal {
            append_wal_record(
                &wal_path(&self.root),
                &WalRecord::DropCollection {
                    collection: name.to_string(),
                },
            )?;
        }

        fs::remove_dir_all(&paths.dir)?;

        let mut manifest = ManifestMetadata::load_from_path(&manifest_path(&self.root))?;
        manifest.collections.retain(|entry| entry != name);
        manifest.save_to_path(&manifest_path(&self.root))?;
        self.collection_handles
            .write()
            .expect("collection handles rwlock poisoned")
            .remove(name);
        self.invalidate_search_cache(name);
        Ok(())
    }

    pub fn list_collections(&self) -> io::Result<Vec<String>> {
        let manifest = ManifestMetadata::load_from_path(&manifest_path(&self.root))?;
        Ok(manifest.collections)
    }

    pub fn create_vector_index(
        &self,
        collection: &str,
        descriptor: VectorIndexDescriptor,
    ) -> io::Result<()> {
        let paths = self.collection_paths(collection);
        let metadata = CollectionMetadata::load_from_path(&paths.collection_meta)?;
        let vector = if let Some(vector) = metadata
            .vectors
            .iter()
            .find(|vector| vector.name == descriptor.field_name)
        {
            vector
        } else {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "vector field '{}' is not defined in collection '{}'",
                    descriptor.field_name, collection
                ),
            ));
        };
        validate_vector_index_descriptor(vector.dimension, &descriptor)?;

        let mut catalog = IndexCatalog::load_from_path(&paths.index_catalog)?;
        catalog.upsert_vector_index(descriptor);
        catalog.save_to_path(&paths.index_catalog)?;
        invalidate_ann_blobs(&paths.dir)?;
        self.invalidate_search_cache(collection);
        Ok(())
    }

    pub fn drop_vector_index(&self, collection: &str, field_name: &str) -> io::Result<()> {
        let paths = self.collection_paths(collection);
        let _ = CollectionMetadata::load_from_path(&paths.collection_meta)?;
        let mut catalog = IndexCatalog::load_from_path(&paths.index_catalog)?;
        if !catalog.drop_vector_index(field_name) {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!(
                    "vector index descriptor not found for field '{}' in '{}'",
                    field_name, collection
                ),
            ));
        }
        catalog.save_to_path(&paths.index_catalog)?;
        invalidate_ann_blobs(&paths.dir)?;
        self.invalidate_search_cache(collection);
        Ok(())
    }

    pub fn list_vector_indexes(&self, collection: &str) -> io::Result<Vec<VectorIndexDescriptor>> {
        let paths = self.collection_paths(collection);
        let _ = CollectionMetadata::load_from_path(&paths.collection_meta)?;
        Ok(IndexCatalog::load_from_path(&paths.index_catalog)?.vector_indexes)
    }

    pub fn create_scalar_index(
        &self,
        collection: &str,
        descriptor: ScalarIndexDescriptor,
    ) -> io::Result<()> {
        let paths = self.collection_paths(collection);
        let metadata = CollectionMetadata::load_from_path(&paths.collection_meta)?;
        if !metadata
            .fields
            .iter()
            .any(|field| field.name == descriptor.field_name)
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "scalar field '{}' is not defined in collection '{}'",
                    descriptor.field_name, collection
                ),
            ));
        }

        let mut catalog = IndexCatalog::load_from_path(&paths.index_catalog)?;
        catalog.upsert_scalar_index(descriptor);
        catalog.save_to_path(&paths.index_catalog)?;
        Ok(())
    }

    pub fn drop_scalar_index(&self, collection: &str, field_name: &str) -> io::Result<()> {
        let paths = self.collection_paths(collection);
        let _ = CollectionMetadata::load_from_path(&paths.collection_meta)?;
        let mut catalog = IndexCatalog::load_from_path(&paths.index_catalog)?;
        if !catalog.drop_scalar_index(field_name) {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!(
                    "scalar index descriptor not found for field '{}' in '{}'",
                    field_name, collection
                ),
            ));
        }
        catalog.save_to_path(&paths.index_catalog)?;
        Ok(())
    }

    pub fn list_scalar_indexes(&self, collection: &str) -> io::Result<Vec<ScalarIndexDescriptor>> {
        let paths = self.collection_paths(collection);
        let _ = CollectionMetadata::load_from_path(&paths.collection_meta)?;
        Ok(IndexCatalog::load_from_path(&paths.index_catalog)?.scalar_indexes)
    }

    pub fn flush_collection(&self, name: &str) -> io::Result<()> {
        self.open_collection_handle(name)?.flush()?;
        // After successful flush, all WAL entries are reflected in the segment
        // files on disk. Truncate the WAL to reclaim space and avoid redundant
        // replay on the next open.
        let wal = wal_path(&self.root);
        if wal.exists() {
            truncate_wal(&wal)?;
        }
        Ok(())
    }

    pub fn optimize_collection(&self, name: &str) -> io::Result<()> {
        self.open_collection_handle(name)?.optimize()?;
        // After successful optimize, all data is persisted in segment files and
        // ANN index blobs. Truncate the WAL since replay is no longer needed.
        let wal = wal_path(&self.root);
        if wal.exists() {
            truncate_wal(&wal)?;
        }
        Ok(())
    }

    pub fn compact_collection(&mut self, name: &str) -> io::Result<()> {
        self.require_write()?;
        self.compact_collection_internal(name, true)
    }

    fn compact_collection_internal(&mut self, name: &str, log_wal: bool) -> io::Result<()> {
        let paths = self.collection_paths(name);
        let collection_meta = CollectionMetadata::load_from_path(&paths.collection_meta)?;

        let result = compact_immutable_segments(&paths, &collection_meta)?;

        if result.compacted_segment_id.is_empty() {
            return Ok(());
        }

        if log_wal {
            append_wal_record(
                &wal_path(&self.root),
                &WalRecord::CompactCollection {
                    collection_name: name.to_string(),
                    compacted_segment_id: result.compacted_segment_id,
                },
            )?;
        }

        self.invalidate_search_cache(name);
        Ok(())
    }

    pub fn get_collection_info(&self, name: &str) -> io::Result<CollectionInfo> {
        self.open_collection_handle(name)?.collection_info()
    }

    pub fn list_collection_segments(&self, name: &str) -> io::Result<Vec<CollectionSegmentInfo>> {
        self.open_collection_handle(name)?.list_segments()
    }

    pub fn insert(
        &mut self,
        collection: &str,
        external_ids: &[i64],
        vectors: &[f32],
    ) -> io::Result<usize> {
        self.require_write()?;
        self.insert_internal(collection, external_ids, vectors, true)
    }

    fn insert_internal(
        &mut self,
        collection: &str,
        external_ids: &[i64],
        vectors: &[f32],
        log_wal: bool,
    ) -> io::Result<usize> {
        let paths = self.collection_paths(collection);
        let collection_meta = CollectionMetadata::load_from_path(&paths.collection_meta)?;

        if collection_meta.dimension == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "collection dimension must be > 0",
            ));
        }

        if vectors.len() % collection_meta.dimension != 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "vector count must be aligned to dimension",
            ));
        }
        let expected = vectors.len() / collection_meta.dimension;
        if external_ids.len() != expected {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "external id count must match vector count",
            ));
        }

        let mut new_ids = HashSet::with_capacity(external_ids.len());
        for external_id in external_ids {
            if !new_ids.insert(*external_id) {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("duplicate external id in batch: {external_id}"),
                ));
            }
        }

        let existing_live = self.fetch_documents(collection, external_ids)?;
        if let Some(existing) = existing_live.first() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("external id already exists: {}", existing.id),
            ));
        }

        if log_wal {
            append_wal_record(
                &wal_path(&self.root),
                &WalRecord::Insert {
                    collection: collection.to_string(),
                    ids: external_ids.to_vec(),
                    vectors: vectors.to_vec(),
                },
            )?;
        }

        let batch_len = external_ids.len();
        let empty_payloads = vec![BTreeMap::new(); batch_len];
        let empty_vectors = vec![BTreeMap::new(); batch_len];
        let empty_sparse = vec![BTreeMap::new(); batch_len];
        let writer = self.segment_writer(&paths);
        let inserted = writer
            .append_records(
                &collection_meta,
                external_ids,
                vectors,
                &empty_payloads,
                &empty_vectors,
                &empty_sparse,
                collection_meta.primary_is_fp16(),
            )?
            .inserted;
        invalidate_ann_blobs(&paths.dir)?;
        if self.should_auto_compact(collection)? {
            self.compact_collection_internal(collection, true)?;
        }
        self.invalidate_search_cache(collection);
        Ok(inserted)
    }

    pub fn insert_documents(
        &mut self,
        collection: &str,
        documents: &[Document],
    ) -> io::Result<usize> {
        self.require_write()?;
        self.insert_documents_internal(collection, documents, true)
    }

    pub fn insert_documents_with_primary_keys(
        &mut self,
        collection: &str,
        keyed_documents: &[(String, Document)],
    ) -> io::Result<usize> {
        self.require_write()?;
        if keyed_documents.is_empty() {
            return Ok(0);
        }

        let paths = self.collection_paths(collection);
        let mut collection_meta = CollectionMetadata::load_from_path(&paths.collection_meta)?;
        let public_keys = keyed_documents
            .iter()
            .map(|(public_key, _)| public_key.clone())
            .collect::<Vec<_>>();
        let internal_ids = keyed_documents
            .iter()
            .map(|(_, document)| document.id)
            .collect::<Vec<_>>();

        upsert_public_keys_with_internal_ids(
            &paths,
            &mut collection_meta,
            &public_keys,
            &internal_ids,
        )?;

        let documents = keyed_documents
            .iter()
            .map(|(_, document)| document.clone())
            .collect::<Vec<_>>();
        self.insert_documents_internal(collection, &documents, true)
    }

    fn insert_documents_internal(
        &mut self,
        collection: &str,
        documents: &[Document],
        log_wal: bool,
    ) -> io::Result<usize> {
        let paths = self.collection_paths(collection);
        let collection_meta = CollectionMetadata::load_from_path(&paths.collection_meta)?;

        validate_documents(documents, &collection_meta)?;

        let requested_ids = documents
            .iter()
            .map(|document| document.id)
            .collect::<Vec<_>>();
        let existing_live = self.fetch_documents(collection, &requested_ids)?;
        if let Some(existing) = existing_live.first() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("external id already exists: {}", existing.id),
            ));
        }

        if log_wal {
            append_wal_record(
                &wal_path(&self.root),
                &WalRecord::InsertDocuments {
                    collection: collection.to_string(),
                    documents: documents.to_vec(),
                },
            )?;
        }

        let writer = self.segment_writer(&paths);
        let inserted = writer
            .append_documents(&collection_meta, documents)?
            .inserted;
        invalidate_ann_blobs(&paths.dir)?;
        if self.should_auto_compact(collection)? {
            self.compact_collection_internal(collection, true)?;
        }
        self.invalidate_search_cache(collection);
        Ok(inserted)
    }

    pub fn upsert_documents(
        &mut self,
        collection: &str,
        documents: &[Document],
    ) -> io::Result<usize> {
        self.require_write()?;
        self.upsert_documents_internal(collection, documents, true)
    }

    pub fn upsert_documents_with_primary_keys(
        &mut self,
        collection: &str,
        keyed_documents: &[(String, Document)],
    ) -> io::Result<usize> {
        self.require_write()?;
        if keyed_documents.is_empty() {
            return Ok(0);
        }

        let paths = self.collection_paths(collection);
        let mut collection_meta = CollectionMetadata::load_from_path(&paths.collection_meta)?;
        let public_keys = keyed_documents
            .iter()
            .map(|(public_key, _)| public_key.clone())
            .collect::<Vec<_>>();
        let internal_ids = keyed_documents
            .iter()
            .map(|(_, document)| document.id)
            .collect::<Vec<_>>();

        upsert_public_keys_with_internal_ids(
            &paths,
            &mut collection_meta,
            &public_keys,
            &internal_ids,
        )?;

        let documents = keyed_documents
            .iter()
            .map(|(_, document)| document.clone())
            .collect::<Vec<_>>();
        self.upsert_documents_internal(collection, &documents, true)
    }

    fn upsert_documents_internal(
        &mut self,
        collection: &str,
        documents: &[Document],
        log_wal: bool,
    ) -> io::Result<usize> {
        let paths = self.collection_paths(collection);
        let collection_meta = CollectionMetadata::load_from_path(&paths.collection_meta)?;

        validate_documents(documents, &collection_meta)?;

        if log_wal {
            append_wal_record(
                &wal_path(&self.root),
                &WalRecord::UpsertDocuments {
                    collection: collection.to_string(),
                    documents: documents.to_vec(),
                },
            )?;
        }

        let document_ids = documents
            .iter()
            .map(|document| document.id)
            .collect::<Vec<_>>();
        self.mark_live_ids_deleted_across_segments(&paths, &document_ids)?;

        let writer = self.segment_writer(&paths);
        let inserted = writer
            .append_documents(&collection_meta, documents)?
            .inserted;
        invalidate_ann_blobs(&paths.dir)?;
        if self.should_auto_compact(collection)? {
            self.compact_collection_internal(collection, true)?;
        }
        self.invalidate_search_cache(collection);
        Ok(inserted)
    }

    /// Partially update existing documents.
    ///
    /// For each `DocumentUpdate`, fields/vectors set to `Some(..)` replace the
    /// current value; `None` means "keep current". The old row is tombstoned
    /// and a merged row is appended.
    pub fn update_documents(
        &mut self,
        collection: &str,
        updates: &[DocumentUpdate],
    ) -> io::Result<usize> {
        self.require_write()?;
        self.update_documents_internal(collection, updates, true)
    }

    fn update_documents_internal(
        &mut self,
        collection: &str,
        updates: &[DocumentUpdate],
        log_wal: bool,
    ) -> io::Result<usize> {
        if updates.is_empty() {
            return Ok(0);
        }

        // Fetch existing documents.
        let ids: Vec<i64> = updates.iter().map(|u| u.id).collect();
        let existing = self.fetch_documents(collection, &ids)?;
        if existing.is_empty() {
            return Ok(0);
        }

        let existing_map: HashMap<i64, Document> =
            existing.into_iter().map(|doc| (doc.id, doc)).collect();

        // Merge updates into full documents.
        let merged: Vec<Document> = updates
            .iter()
            .filter_map(|update| {
                let existing = existing_map.get(&update.id)?;
                let mut fields = existing.fields.clone();
                for (key, value) in &update.fields {
                    match value {
                        Some(v) => {
                            fields.insert(key.clone(), v.clone());
                        }
                        None => {
                            fields.remove(key);
                        }
                    }
                }
                let mut vectors = existing.vectors.clone();
                for (key, value) in &update.vectors {
                    match value {
                        Some(v) => {
                            vectors.insert(key.clone(), v.clone());
                        }
                        None => {
                            vectors.remove(key);
                        }
                    }
                }
                let mut sparse_vectors = existing.sparse_vectors.clone();
                for (key, value) in &update.sparse_vectors {
                    match value {
                        Some(v) => {
                            sparse_vectors.insert(key.clone(), v.clone());
                        }
                        None => {
                            sparse_vectors.remove(key);
                        }
                    }
                }
                Some(Document {
                    id: update.id,
                    fields,
                    vectors,
                    sparse_vectors,
                })
            })
            .collect();

        if merged.is_empty() {
            return Ok(0);
        }

        if log_wal {
            append_wal_record(
                &wal_path(&self.root),
                &WalRecord::UpdateDocuments {
                    collection: collection.to_string(),
                    updates: updates.to_vec(),
                },
            )?;
        }

        // Delegate to upsert (which does tombstone+append), but skip its own
        // WAL write since we already logged the UpdateDocuments record.
        self.upsert_documents_internal(collection, &merged, false)
    }

    pub fn fetch_documents(
        &self,
        collection: &str,
        external_ids: &[i64],
    ) -> io::Result<Vec<Document>> {
        self.open_collection_handle(collection)?
            .fetch_documents(external_ids)
    }

    pub fn fetch_documents_by_primary_keys(
        &self,
        collection: &str,
        public_keys: &[String],
    ) -> io::Result<Vec<Document>> {
        let ids = self.resolve_query_ids_by_primary_keys(collection, public_keys)?;
        self.fetch_documents(collection, &ids)
    }

    pub fn delete(&mut self, collection: &str, external_ids: &[i64]) -> io::Result<usize> {
        self.require_write()?;
        self.delete_internal(collection, external_ids, true)
    }

    pub fn delete_by_primary_keys(
        &mut self,
        collection: &str,
        public_keys: &[String],
    ) -> io::Result<usize> {
        self.require_write()?;
        let ids = self.resolve_query_ids_by_primary_keys(collection, public_keys)?;
        self.delete_internal(collection, &ids, true)
    }

    pub fn assign_internal_ids_for_primary_keys(
        &mut self,
        collection: &str,
        public_keys: &[String],
    ) -> io::Result<Vec<i64>> {
        self.require_write()?;
        let paths = self.collection_paths(collection);
        let mut collection_meta = CollectionMetadata::load_from_path(&paths.collection_meta)?;
        assign_internal_ids_for_public_keys(&paths, &mut collection_meta, public_keys)
    }

    pub fn resolve_query_ids_by_primary_keys(
        &self,
        collection: &str,
        public_keys: &[String],
    ) -> io::Result<Vec<i64>> {
        let paths = self.collection_paths(collection);
        let collection_meta = CollectionMetadata::load_from_path(&paths.collection_meta)?;
        resolve_public_keys_to_internal_ids(&paths, &collection_meta, public_keys)
    }

    pub fn display_primary_keys_for_document_ids(
        &self,
        collection: &str,
        internal_ids: &[i64],
    ) -> io::Result<Vec<String>> {
        let paths = self.collection_paths(collection);
        let collection_meta = CollectionMetadata::load_from_path(&paths.collection_meta)?;
        let registry = load_primary_key_registry(&paths, &collection_meta)?;
        Ok(internal_ids
            .iter()
            .map(|internal_id| display_key_for_internal_id(&registry, *internal_id))
            .collect())
    }

    pub fn delete_by_filter(&mut self, collection: &str, filter: &str) -> io::Result<usize> {
        self.require_write()?;
        let filter_expr = parse_filter(filter)?;
        let matching_ids = self.collect_latest_live_filtered_ids(collection, &filter_expr)?;
        if matching_ids.is_empty() {
            return Ok(0);
        }

        let matching_ids = matching_ids.into_iter().collect::<Vec<_>>();
        self.delete_internal(collection, &matching_ids, true)
    }

    fn delete_internal(
        &mut self,
        collection: &str,
        external_ids: &[i64],
        log_wal: bool,
    ) -> io::Result<usize> {
        let paths = self.collection_paths(collection);

        if log_wal {
            append_wal_record(
                &wal_path(&self.root),
                &WalRecord::Delete {
                    collection: collection.to_string(),
                    ids: external_ids.to_vec(),
                },
            )?;
        }

        let result = tombstone::mark_ids_deleted(&paths, external_ids)?;

        invalidate_ann_blobs(&paths.dir)?;
        if self.should_auto_compact(collection)? {
            self.compact_collection_internal(collection, true)?;
        }
        self.invalidate_search_cache(collection);
        Ok(result.deleted_count)
    }

    fn collect_latest_live_filtered_ids(
        &self,
        collection: &str,
        filter_expr: &FilterExpr,
    ) -> io::Result<BTreeSet<i64>> {
        let paths = self.collection_paths(collection);
        let collection_meta = CollectionMetadata::load_from_path(&paths.collection_meta)?;
        let segment_paths = SegmentManager::new(paths.dir.clone()).segment_paths()?;

        let mut seen_ids = HashSet::new();
        let mut matching_ids = BTreeSet::new();

        for segment in segment_paths {
            let segment_meta = SegmentMetadata::load_from_path(&segment.metadata)?;
            let dense_rows = load_primary_dense_rows_for_segment_or_empty(
                &segment,
                &segment_meta,
                &collection_meta.primary_vector,
                collection_meta.dimension,
                collection_meta.primary_is_fp16(),
            )?;
            let records = dense_rows.primary_vectors;
            let stored_ids = dense_rows.external_ids;
            let payloads = load_payloads_or_empty(&segment, &segment_meta, stored_ids.len())?;
            let tombstone = TombstoneMask::load_from_path(&segment.tombstones)?;

            let row_limit = segment_meta
                .record_count
                .min(stored_ids.len())
                .min(records.len() / collection_meta.dimension);
            if row_limit == 0 {
                continue;
            }

            for row_idx in (0..row_limit).rev() {
                let external_id = stored_ids[row_idx];
                if !seen_ids.insert(external_id) {
                    continue;
                }
                if tombstone.is_deleted(row_idx) {
                    continue;
                }
                if filter_expr.matches(&payloads[row_idx]) {
                    matching_ids.insert(external_id);
                }
            }
        }

        Ok(matching_ids)
    }

    pub fn search(
        &self,
        collection: &str,
        query: &[f32],
        top_k: usize,
    ) -> io::Result<Vec<SearchHit>> {
        self.search_with_ef(collection, query, top_k, DEFAULT_EF_SEARCH)
    }

    pub fn search_with_ef(
        &self,
        collection: &str,
        query: &[f32],
        top_k: usize,
        _ef_search: usize,
    ) -> io::Result<Vec<SearchHit>> {
        self.open_collection_handle(collection)?
            .search_with_ef(query, top_k, _ef_search)
    }

    pub fn query_documents(
        &self,
        collection: &str,
        query: &[f32],
        top_k: usize,
        filter: Option<&str>,
    ) -> io::Result<Vec<DocumentHit>> {
        self.open_collection_handle(collection)?
            .query_documents(query, top_k, filter)
    }

    pub fn query_with_context(
        &self,
        collection: &str,
        context: &QueryContext,
    ) -> io::Result<Vec<DocumentHit>> {
        self.open_collection_handle(collection)?
            .query_with_context(context)
    }

    /// Search a sparse vector field by brute force.
    ///
    /// Returns top-k hits ranked by sparse inner product (descending similarity).
    pub fn search_sparse(
        &self,
        collection: &str,
        field_name: &str,
        query: &SparseVector,
        top_k: usize,
    ) -> io::Result<Vec<SearchHit>> {
        self.open_collection_handle(collection)?
            .search_sparse(field_name, query, top_k)
    }

    fn collection_paths(&self, name: &str) -> CollectionPaths {
        collection_paths_for_root(&self.root, name)
    }

    fn segment_writer(&self, paths: &CollectionPaths) -> SegmentWriter {
        SegmentWriter::new(paths.dir.clone(), SegmentManager::new(paths.dir.clone()))
    }

    fn mark_live_ids_deleted_across_segments(
        &self,
        paths: &CollectionPaths,
        external_ids: &[i64],
    ) -> io::Result<()> {
        tombstone::mark_live_ids_deleted_across_segments(paths, external_ids)?;
        Ok(())
    }

    fn should_auto_compact(&self, collection: &str) -> io::Result<bool> {
        let paths = self.collection_paths(collection);
        if !paths.segment_set.exists() {
            return Ok(false);
        }
        let version_set = VersionSet::load_from_path(&paths.segment_set)?;
        Ok(SegmentSet::should_compact(
            version_set.immutable_segment_ids().len(),
        ))
    }

    fn invalidate_search_cache(&self, collection: &str) {
        let handle = {
            let handles = self
                .collection_handles
                .read()
                .expect("collection handles rwlock poisoned");
            handles.get(collection).cloned()
        };
        if let Some(handle) = handle {
            let _ = handle.refresh_version_set();
            handle.invalidate_search_cache();
        }
    }

    fn replay_wal_if_needed(&mut self) -> io::Result<()> {
        let records = load_wal_records_or_empty(&wal_path(&self.root))?;
        if records.is_empty() {
            return Ok(());
        }

        let plan = WalReplayPlan::build(&records);
        if !plan.has_owned_collections()
            || !plan.requires_replay(&manifest_path(&self.root), |collection| {
                self.collection_paths(collection)
            })?
        {
            return Ok(());
        }

        let manifest_path = manifest_path(&self.root);
        let mut manifest = ManifestMetadata::load_from_path(&manifest_path)?;
        for collection in plan.collections_to_reset() {
            let paths = self.collection_paths(collection);
            if paths.dir.exists() {
                fs::remove_dir_all(&paths.dir)?;
            }
            manifest.collections.retain(|entry| entry != collection);
            self.invalidate_search_cache(collection);
        }
        manifest.save_to_path(&manifest_path)?;
        fs::create_dir_all(self.root.join("collections"))?;

        for record in &records {
            if plan.owns(collection_name_for_wal_record(record)) {
                self.apply_wal_record(record)?;
            }
        }
        Ok(())
    }

    fn apply_wal_record(&mut self, record: &WalRecord) -> io::Result<()> {
        match record {
            WalRecord::CreateCollection { collection, schema } => {
                self.create_collection_with_schema_internal(collection, schema, false)
            }
            WalRecord::DropCollection { collection } => {
                let paths = self.collection_paths(collection);
                if !paths.dir.exists() {
                    return Ok(());
                }
                self.drop_collection_internal(collection, false)
            }
            WalRecord::Insert {
                collection,
                ids,
                vectors,
            } => self
                .insert_internal(collection, ids, vectors, false)
                .map(|_| ()),
            WalRecord::InsertDocuments {
                collection,
                documents,
            } => self
                .insert_documents_internal(collection, documents, false)
                .map(|_| ()),
            WalRecord::UpsertDocuments {
                collection,
                documents,
            } => self
                .upsert_documents_internal(collection, documents, false)
                .map(|_| ()),
            WalRecord::Delete { collection, ids } => {
                self.delete_internal(collection, ids, false).map(|_| ())
            }
            WalRecord::CompactCollection {
                collection_name, ..
            } => self.compact_collection_internal(collection_name, false),
            WalRecord::UpdateDocuments {
                collection,
                updates,
            } => self
                .update_documents_internal(collection, updates, false)
                .map(|_| ()),
            WalRecord::AddColumn {
                collection,
                field,
                backfill,
            } => self.add_column_internal(collection, field.clone(), backfill.clone(), false),
            WalRecord::DropColumn {
                collection,
                field_name,
            } => self.drop_column_internal(collection, field_name, false),
            WalRecord::AlterColumn {
                collection,
                old_name,
                new_name,
                field,
                migration,
            } => self.alter_column_internal(
                collection,
                old_name,
                new_name,
                field.clone(),
                *migration,
                false,
            ),
            WalRecord::AddVectorField { collection, field } => {
                self.add_vector_field_internal(collection, field.clone(), false)
            }
            WalRecord::DropVectorField {
                collection,
                field_name,
            } => self.drop_vector_field_internal(collection, field_name, false),
        }
    }
}

impl CollectionHandle {
    fn new(segment_manager: SegmentManager, version_set: VersionSet) -> Self {
        Self {
            segment_manager,
            version_set: RwLock::new(version_set),
            search_cache: Mutex::new(HashMap::new()),
            scalar_cache: Mutex::new(HashMap::new()),
            sparse_index_cache: Mutex::new(HashMap::new()),
            document_cache: Mutex::new(None),
        }
    }

    pub fn search(&self, query: &[f32], top_k: usize) -> io::Result<Vec<SearchHit>> {
        self.search_with_ef(query, top_k, DEFAULT_EF_SEARCH)
    }

    pub fn search_with_ef(
        &self,
        query: &[f32],
        top_k: usize,
        _ef_search: usize,
    ) -> io::Result<Vec<SearchHit>> {
        let field_name = self.collection_primary_vector_name()?;
        self.search_field_with_ef_internal(&field_name, query, top_k, _ef_search)
    }

    /// Brute-force search over a sparse vector field.
    ///
    /// Returns top-k hits ranked by sparse inner product (descending similarity).
    pub fn search_sparse(
        &self,
        field_name: &str,
        query: &SparseVector,
        top_k: usize,
    ) -> io::Result<Vec<SearchHit>> {
        let paths = self.collection_paths();
        let collection_meta = CollectionMetadata::load_from_path(&paths.collection_meta)?;

        // Verify the field is a sparse vector field in the schema.
        collection_meta
            .vectors
            .iter()
            .find(|v| {
                v.name == field_name && v.data_type == crate::document::FieldType::VectorSparse
            })
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!(
                        "sparse vector field '{}' is not defined in collection '{}'",
                        field_name, collection_meta.name
                    ),
                )
            })?;

        // Try cached sparse index first.
        {
            let sparse_cache = self
                .sparse_index_cache
                .lock()
                .expect("sparse index cache mutex poisoned");
            if let Some(index) = sparse_cache.get(field_name) {
                let query_data = SparseVectorData::new(query.indices.clone(), query.values.clone());
                let hits = index.search(&query_data, top_k, None).map_err(|e| {
                    io::Error::new(io::ErrorKind::Other, format!("sparse index search: {e:?}"))
                })?;
                return Ok(hits
                    .into_iter()
                    .map(|hit| SearchHit {
                        id: hit.id,
                        distance: -hit.score, // score → distance (negate for consistency)
                    })
                    .collect());
            }
        }

        // Fallback: brute-force scan.
        let segment_paths = self.segment_manager.segment_paths()?;

        // Collect all live sparse vectors for the target field across segments.
        let mut all_sparse: Vec<SparseVector> = Vec::new();
        let mut all_ids: Vec<i64> = Vec::new();
        let mut shadowed: HashSet<i64> = HashSet::new();

        for segment in &segment_paths {
            let segment_meta = SegmentMetadata::load_from_path(&segment.metadata)?;
            let stored_ids = load_external_ids_for_segment_or_empty(segment, &segment_meta)?;
            let sparse_rows =
                load_sparse_vectors_or_empty(&segment.sparse_vectors, stored_ids.len())?;
            let tombstone = TombstoneMask::load_from_path(&segment.tombstones)?;

            for row_idx in (0..stored_ids.len()).rev() {
                let ext_id = stored_ids[row_idx];
                if !shadowed.insert(ext_id) {
                    continue;
                }
                if tombstone.is_deleted(row_idx) {
                    continue;
                }
                if let Some(sv) = sparse_rows[row_idx].get(field_name) {
                    all_sparse.push(sv.clone());
                    all_ids.push(ext_id);
                }
            }
        }

        let tombstone = TombstoneMask::new(all_ids.len());
        search_sparse_bruteforce(&all_sparse, &all_ids, &tombstone, query, top_k)
    }

    pub fn query_with_context(&self, context: &QueryContext) -> io::Result<Vec<DocumentHit>> {
        let paths = self.collection_paths();
        let collection_meta = CollectionMetadata::load_from_path(&paths.collection_meta)?;
        let index_catalog = IndexCatalog::load_from_path(&paths.index_catalog)?;
        let query_by_id_documents = match context.query_by_id.as_ref() {
            Some(query_by_id) => self.fetch_documents(query_by_id)?,
            None => Vec::new(),
        };
        let plan = QueryPlanner::build(
            &collection_meta,
            &index_catalog,
            context,
            &query_by_id_documents,
        )?;
        match plan {
            QueryPlan::LegacySingleVector(plan) => {
                let ef_search = plan.ef_search.unwrap_or(DEFAULT_EF_SEARCH);
                let nprobe = plan.nprobe.unwrap_or(DEFAULT_NPROBE);

                // Resolve the effective search parameter: when nprobe is explicitly
                // provided, it overrides ef_search for IVF indexes (the IVF backend
                // maps the ef_search argument to nprobe internally).
                let state = self.cached_search_state_for_field(&plan.field_name)?;
                let is_ivf = matches!(
                    state.descriptor.kind,
                    hannsdb_index::descriptor::VectorIndexKind::Ivf
                        | hannsdb_index::descriptor::VectorIndexKind::IvfUsq
                );
                let effective_search_param = if is_ivf && plan.nprobe.is_some() {
                    nprobe
                } else {
                    ef_search
                };

                #[cfg(feature = "hanns-backend")]
                if let Some(filter) = &plan.filter {
                    // Pre-filter path: construct BitsetView from filter evaluation
                    // and pass it to the ANN backend for efficient pre-filtered search.
                    if let Some(optimized) = state.optimized_ann.as_ref() {
                        // Fetch payloads for all indexed vectors to evaluate filter
                        let documents = self.fetch_documents(&optimized.ann_external_ids)?;
                        let bitset =
                            hannsdb_index::bitset::filter_to_bitset(documents.len(), |i| {
                                filter.matches(&documents[i].fields)
                            });
                        let ann_hits = match &plan.vector {
                            crate::query::QueryVector::Dense(v) => ann_search_with_bitset(
                                optimized.backend.as_ref(),
                                &optimized.ann_external_ids,
                                &optimized.metric,
                                v,
                                plan.top_k,
                                effective_search_param,
                                &bitset,
                            )?,
                            crate::query::QueryVector::Sparse(_) => {
                                return Err(io::Error::new(
                                    io::ErrorKind::InvalidInput,
                                    "sparse queries cannot use ANN pre-filter path",
                                ));
                            }
                        };
                        // Hits already have correct external IDs; fetch for output fields
                        let fetched = self
                            .fetch_documents(&ann_hits.iter().map(|h| h.id).collect::<Vec<_>>())?;
                        let mut doc_hits: Vec<DocumentHit> = ann_hits
                            .into_iter()
                            .zip(fetched)
                            .map(|(hit, document)| DocumentHit {
                                id: hit.id,
                                distance: hit.distance,
                                fields: document.fields,
                                vectors: BTreeMap::new(),
                                sparse_vectors: BTreeMap::new(),
                                group_key: None,
                            })
                            .collect();
                        project_hits_output_fields(&mut doc_hits, plan.output_fields.as_deref());
                        return Ok(doc_hits);
                    }
                }

                // Fallback: overfetch + post-filter (non-hanns-backend or no ANN)
                const FILTER_OVERFETCH_FACTOR: usize = 4;
                let search_k = if plan.filter.is_some() {
                    plan.top_k.saturating_mul(FILTER_OVERFETCH_FACTOR)
                } else {
                    plan.top_k
                };
                let query_vec = match &plan.vector {
                    crate::query::QueryVector::Dense(v) => v.clone(),
                    crate::query::QueryVector::Sparse(_) => {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidInput,
                            "sparse queries cannot use the legacy single-vector fast path",
                        ));
                    }
                };
                let mut hits = self.query_documents_for_field_with_ef_internal(
                    &collection_meta,
                    &plan.field_name,
                    &query_vec,
                    search_k,
                    effective_search_param,
                    context.include_vector,
                )?;

                if let Some(filter) = &plan.filter {
                    hits.retain(|hit| filter.matches(&hit.fields));
                    hits.truncate(plan.top_k);
                }

                if let Some(order_by) = &plan.order_by {
                    sort_hits_by_field(&mut hits, &order_by.field_name, order_by.descending);
                }

                project_hits_output_fields(&mut hits, plan.output_fields.as_deref());
                Ok(hits)
            }
            QueryPlan::BruteForce(plan) => {
                let mut hits =
                    QueryExecutor::execute(&self.segment_manager, &collection_meta, &plan)?;
                if context.include_vector {
                    self.materialize_document_hit_vectors(&collection_meta, &mut hits)?;
                }
                project_hits_output_fields(&mut hits, plan.output_fields.as_deref());
                Ok(hits)
            }
        }
    }

    pub fn optimize(&self) -> io::Result<()> {
        let paths = self.collection_paths();
        let collection_name = self.segment_manager.collection_name().to_string();
        let collection_meta = CollectionMetadata::load_from_path(&paths.collection_meta)?;
        let index_catalog = IndexCatalog::load_from_path(&paths.index_catalog)?;

        let mut cache = self
            .search_cache
            .lock()
            .expect("search cache mutex poisoned");

        for vector_schema in &collection_meta.vectors {
            let field_name = &vector_schema.name;
            let Some(descriptor) =
                resolve_vector_descriptor_for_field(&collection_meta, &index_catalog, field_name)?
            else {
                continue;
            };

            let mut state = self.load_search_state_for_field(field_name)?;
            let mut blob_bytes = None;
            state.optimized_ann = Some(build_optimized_ann_state(&state, Some(&mut blob_bytes))?);

            #[cfg(feature = "hanns-backend")]
            if let Some(ref bytes) = blob_bytes {
                if let Some(ref ann) = state.optimized_ann {
                    match persist::persist_ann_blob(
                        &paths.dir,
                        field_name,
                        bytes,
                        &ann.ann_external_ids,
                    ) {
                        Ok(()) => {
                            log::info!(
                                "Persisted ANN index ({} bytes) for field '{}' in collection '{}'",
                                bytes.len(),
                                field_name,
                                collection_name
                            );
                        }
                        Err(e) => {
                            log::warn!("Failed to persist ANN index for '{}': {e}", field_name);
                        }
                    }
                }
            }

            cache.insert(state.field_name.clone(), state);

            // Suppress unused-variable warnings when hanns-backend is off
            let _ = (&descriptor, &blob_bytes);
        }

        // Build scalar indexes for fields that have descriptors in the catalog.
        if !index_catalog.scalar_indexes.is_empty() {
            let segment_paths = self.segment_manager.segment_paths()?;
            let scalar_indexes = persist::build_scalar_indexes_from_segments(
                &segment_paths,
                &index_catalog.scalar_indexes,
            )?;

            let mut scalar_cache = self
                .scalar_cache
                .lock()
                .expect("scalar cache mutex poisoned");
            for (field_name, index) in scalar_indexes {
                log::info!(
                    "Built scalar index for field '{}' in collection '{}' ({} indexed IDs)",
                    field_name,
                    collection_name,
                    index.all_indexed_ids().len(),
                );
                scalar_cache.insert(field_name, index);
            }
        }

        // Build sparse indexes for sparse vector fields.
        let segment_paths = self.segment_manager.segment_paths()?;
        let sparse_indexes =
            persist::build_sparse_indexes_from_segments(&paths, &segment_paths, &collection_meta)?;

        if !sparse_indexes.is_empty() {
            let mut sparse_cache = self
                .sparse_index_cache
                .lock()
                .expect("sparse index cache mutex poisoned");
            for (field_name, index) in sparse_indexes {
                log::info!(
                    "Built sparse index for field '{}' in collection '{}' ({} vectors)",
                    field_name,
                    collection_name,
                    index.len(),
                );
                sparse_cache.insert(field_name, index);
            }
        }

        Ok(())
    }

    pub fn version_set(&self) -> io::Result<VersionSet> {
        Ok(self
            .version_set
            .read()
            .expect("version_set rwlock poisoned")
            .clone())
    }

    fn collection_info(&self) -> io::Result<CollectionInfo> {
        let paths = self.collection_paths();
        let collection_meta = CollectionMetadata::load_from_path(&paths.collection_meta)?;

        let mut record_count = 0usize;
        let mut deleted_count = 0usize;
        for segment in self.segment_manager.segment_paths()? {
            let metadata = SegmentMetadata::load_from_path(&segment.metadata)?;
            record_count += metadata.record_count;
            deleted_count += metadata.deleted_count;
        }
        let live_count = record_count.saturating_sub(deleted_count);

        // Build index_completeness: check the search cache for each vector field.
        // If the cache has an entry with optimized_ann populated, that field is fully indexed.
        let mut index_completeness = BTreeMap::new();
        let cache = self
            .search_cache
            .lock()
            .expect("search cache mutex poisoned");
        for vector_schema in &collection_meta.vectors {
            let has_persisted_ann = persisted_ann_exists(
                &paths.dir,
                vector_schema.name.as_str(),
                vector_schema.name == collection_meta.primary_vector,
            );
            let completeness = if live_count == 0 {
                // No data: vacuously fully indexed.
                1.0
            } else if let Some(state) = cache.get(&vector_schema.name) {
                if state.optimized_ann.is_some() {
                    1.0
                } else if has_persisted_ann {
                    1.0
                } else {
                    0.0
                }
            } else if has_persisted_ann {
                1.0
            } else {
                0.0
            };
            index_completeness.insert(vector_schema.name.clone(), completeness);
        }

        Ok(CollectionInfo {
            name: collection_meta.name,
            dimension: collection_meta.dimension,
            metric: collection_meta.metric,
            record_count,
            deleted_count,
            live_count,
            index_completeness,
        })
    }

    fn flush(&self) -> io::Result<()> {
        let paths = self.collection_paths();
        let collection_meta = CollectionMetadata::load_from_path(&paths.collection_meta)?;
        let active_segment = self.segment_manager.active_segment_path()?;
        let _ = SegmentMetadata::load_from_path(&active_segment.metadata)?;
        let _ = TombstoneMask::load_from_path(&active_segment.tombstones)?;
        let root = paths
            .dir
            .parent()
            .and_then(Path::parent)
            .map(Path::to_path_buf)
            .unwrap_or_else(|| paths.dir.clone());
        let _ = load_wal_records(&wal_path(&root))?;
        materialize_active_segment_arrow_snapshots(&active_segment, &collection_meta)?;
        materialize_forward_store_snapshot(&active_segment, &collection_meta)?;
        Ok(())
    }

    fn list_segments(&self) -> io::Result<Vec<CollectionSegmentInfo>> {
        let paths = self.collection_paths();
        let _ = CollectionMetadata::load_from_path(&paths.collection_meta)?;
        self.segment_manager
            .segment_paths()?
            .into_iter()
            .map(|segment| {
                let metadata = SegmentMetadata::load_from_path(&segment.metadata)?;
                Ok(CollectionSegmentInfo {
                    id: segment.segment_id.clone(),
                    live_count: metadata.record_count.saturating_sub(metadata.deleted_count),
                    dead_count: metadata.deleted_count,
                    ann_ready: segment.ann_dir().exists(),
                })
            })
            .collect()
    }

    fn query_documents(
        &self,
        query: &[f32],
        top_k: usize,
        filter: Option<&str>,
    ) -> io::Result<Vec<DocumentHit>> {
        self.query_documents_with_ef(query, top_k, filter, DEFAULT_EF_SEARCH)
    }

    fn query_documents_with_ef(
        &self,
        query: &[f32],
        top_k: usize,
        filter: Option<&str>,
        ef_search: usize,
    ) -> io::Result<Vec<DocumentHit>> {
        self.query_documents_with_ef_internal(None, query, top_k, filter, ef_search, false)
    }

    fn query_documents_with_ef_internal(
        &self,
        collection_meta: Option<&CollectionMetadata>,
        query: &[f32],
        top_k: usize,
        filter: Option<&str>,
        ef_search: usize,
        include_vector: bool,
    ) -> io::Result<Vec<DocumentHit>> {
        match filter.map(str::trim) {
            None | Some("") => {
                let field_name = match collection_meta {
                    Some(collection_meta) => collection_meta.primary_vector.clone(),
                    None => self.collection_primary_vector_name()?,
                };
                let hits =
                    self.search_field_with_ef_internal(&field_name, query, top_k, ef_search)?;
                let documents =
                    self.fetch_documents(&hits.iter().map(|hit| hit.id).collect::<Vec<_>>())?;
                if include_vector && documents.len() != hits.len() {
                    return Err(io::Error::new(
                        io::ErrorKind::NotFound,
                        format!(
                            "failed to resolve {} typed query hit(s) for vector materialization, got {}",
                            hits.len(),
                            documents.len()
                        ),
                    ));
                }
                Ok(hits
                    .into_iter()
                    .zip(documents)
                    .map(|(hit, document)| {
                        let vectors =
                            if include_vector {
                                document.vectors_with_primary(
                                collection_meta
                                    .expect("collection metadata must exist when include_vector")
                                    .primary_vector
                                    .as_str(),
                            ).clone()
                            } else {
                                BTreeMap::new()
                            };
                        DocumentHit {
                            id: hit.id,
                            distance: hit.distance,
                            fields: document.fields,
                            vectors,
                            sparse_vectors: BTreeMap::new(),
                            group_key: None,
                        }
                    })
                    .collect())
            }
            Some(filter) => {
                let mut hits = self.query_documents_with_filter(query, top_k, filter)?;
                if include_vector {
                    let collection_meta = collection_meta
                        .expect("collection metadata must exist when include_vector");
                    self.materialize_document_hit_vectors(collection_meta, &mut hits)?;
                }
                Ok(hits)
            }
        }
    }

    fn query_documents_for_field_with_ef_internal(
        &self,
        collection_meta: &CollectionMetadata,
        field_name: &str,
        query: &[f32],
        top_k: usize,
        ef_search: usize,
        include_vector: bool,
    ) -> io::Result<Vec<DocumentHit>> {
        let hits = self.search_field_with_ef_internal(field_name, query, top_k, ef_search)?;
        let documents = self.fetch_documents(&hits.iter().map(|hit| hit.id).collect::<Vec<_>>())?;
        if include_vector && documents.len() != hits.len() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!(
                    "failed to resolve {} typed query hit(s) for vector materialization, got {}",
                    hits.len(),
                    documents.len()
                ),
            ));
        }
        Ok(hits
            .into_iter()
            .zip(documents)
            .map(|(hit, document)| {
                let vectors = if include_vector {
                    document
                        .vectors_with_primary(collection_meta.primary_vector.as_str())
                        .clone()
                } else {
                    BTreeMap::new()
                };
                DocumentHit {
                    id: hit.id,
                    distance: hit.distance,
                    fields: document.fields,
                    vectors,
                    sparse_vectors: BTreeMap::new(),
                    group_key: None,
                }
            })
            .collect())
    }

    fn search_field_with_ef_internal(
        &self,
        field_name: &str,
        query: &[f32],
        top_k: usize,
        ef_search: usize,
    ) -> io::Result<Vec<SearchHit>> {
        let state = self.cached_search_state_for_field(field_name)?;
        let ef_search = ef_search.max(1);
        if let Some(optimized_ann) = state.optimized_ann.as_ref() {
            let optimized_snapshot = (
                Arc::clone(&optimized_ann.backend),
                Arc::clone(&optimized_ann.ann_external_ids),
                optimized_ann.metric.clone(),
            );
            return ann_search(
                optimized_snapshot.0.as_ref(),
                optimized_snapshot.1.as_slice(),
                &optimized_snapshot.2,
                query,
                top_k,
                ef_search,
            );
        }

        search_by_metric(
            &state.records,
            &state.external_ids,
            state.dimension,
            state.tombstone.as_ref(),
            query,
            top_k,
            &state.metric,
        )
    }

    fn fetch_documents(&self, external_ids: &[i64]) -> io::Result<Vec<Document>> {
        let state = self.cached_document_state()?;
        let docs = &*state;
        Ok(external_ids
            .iter()
            .filter_map(|id| docs.get(id).cloned())
            .collect())
    }

    fn materialize_document_hit_vectors(
        &self,
        collection_meta: &CollectionMetadata,
        hits: &mut [DocumentHit],
    ) -> io::Result<()> {
        if hits.is_empty() {
            return Ok(());
        }

        let documents = self.fetch_documents(&hits.iter().map(|hit| hit.id).collect::<Vec<_>>())?;
        if documents.len() != hits.len() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                "failed to resolve one or more typed query hits for vector materialization",
            ));
        }

        for (hit, document) in hits.iter_mut().zip(documents) {
            hit.vectors = document
                .vectors_with_primary(collection_meta.primary_vector.as_str())
                .clone();
            hit.sparse_vectors = document.sparse_vectors.clone();
        }
        Ok(())
    }

    fn query_documents_with_filter(
        &self,
        query: &[f32],
        top_k: usize,
        filter: &str,
    ) -> io::Result<Vec<DocumentHit>> {
        #[derive(Debug, Clone)]
        struct RankedDocumentHit {
            hit: DocumentHit,
        }

        impl PartialEq for RankedDocumentHit {
            fn eq(&self, other: &Self) -> bool {
                self.hit.id == other.hit.id
                    && self.hit.distance.to_bits() == other.hit.distance.to_bits()
            }
        }

        impl Eq for RankedDocumentHit {}

        impl PartialOrd for RankedDocumentHit {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                Some(self.cmp(other))
            }
        }

        impl Ord for RankedDocumentHit {
            fn cmp(&self, other: &Self) -> Ordering {
                compare_hits(&self.hit, &other.hit)
            }
        }

        if top_k == 0 {
            return Ok(Vec::new());
        }
        let filter_expr = parse_filter(filter)?;
        let paths = self.collection_paths();
        let collection_meta = CollectionMetadata::load_from_path(&paths.collection_meta)?;
        let segment_paths = self.segment_manager.segment_paths()?;

        // Compute the minimal set of payload fields needed for filter evaluation.
        let filter_fields: Vec<String> = filter_expr.referenced_fields().into_iter().collect();
        let projection = if filter_fields.is_empty() {
            None
        } else {
            Some(filter_fields.as_slice())
        };

        // Try to extract candidate IDs from scalar indexes.
        // For now we handle the simplest case: a single Eq or InList clause
        // on a field that has a scalar index.
        let indexed_candidates = self.try_scalar_index_candidates(&filter_expr);

        let mut heap: BinaryHeap<RankedDocumentHit> = BinaryHeap::new();
        for segment in segment_paths {
            let segment_meta = SegmentMetadata::load_from_path(&segment.metadata)?;
            let dense_rows = load_primary_dense_rows_for_segment_or_empty(
                &segment,
                &segment_meta,
                &collection_meta.primary_vector,
                collection_meta.dimension,
                collection_meta.primary_is_fp16(),
            )?;
            let records = dense_rows.primary_vectors;
            let stored_ids = dense_rows.external_ids;
            let payloads = if projection.is_some() {
                load_payloads_with_fields_or_empty(
                    &segment,
                    &segment_meta,
                    stored_ids.len(),
                    projection,
                )?
            } else {
                load_payloads_or_empty(&segment, &segment_meta, stored_ids.len())?
            };
            let tombstone = TombstoneMask::load_from_path(&segment.tombstones)?;

            if stored_ids.len().saturating_mul(collection_meta.dimension) != records.len() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "records and ids are not aligned",
                ));
            }

            for (row_idx, vector) in records.chunks_exact(collection_meta.dimension).enumerate() {
                if tombstone.is_deleted(row_idx) {
                    continue;
                }
                let ext_id = stored_ids[row_idx];

                // If the scalar index gave us a candidate set, skip rows
                // whose external ID is not in the set.
                if let Some(ref candidates) = indexed_candidates {
                    if !candidates.contains(&ext_id) {
                        continue;
                    }
                }

                let fields = &payloads[row_idx];
                if !filter_expr.matches(fields) {
                    continue;
                }
                let candidate = RankedDocumentHit {
                    hit: DocumentHit {
                        id: ext_id,
                        distance: distance_by_metric(query, vector, &collection_meta.metric)?,
                        fields: fields.clone(),
                        vectors: BTreeMap::new(),
                        sparse_vectors: BTreeMap::new(),
                        group_key: None,
                    },
                };

                if heap.len() < top_k {
                    heap.push(candidate);
                    continue;
                }

                if let Some(worst) = heap.peek() {
                    if candidate.cmp(worst) == Ordering::Less {
                        let _ = heap.pop();
                        heap.push(candidate);
                    }
                }
            }
        }

        let mut hits = heap.into_iter().map(|entry| entry.hit).collect::<Vec<_>>();
        hits.sort_by(compare_hits);
        Ok(hits)
    }

    /// Attempt to use scalar indexes to narrow down candidate IDs.
    ///
    /// Returns `Some(set)` if an index-accelerated candidate set was produced,
    /// or `None` if no scalar index could be applied (fall back to brute-force).
    fn try_scalar_index_candidates(&self, filter_expr: &FilterExpr) -> Option<BTreeSet<i64>> {
        let scalar_cache = self.scalar_cache.lock().ok()?;
        if scalar_cache.is_empty() {
            return None;
        }

        match filter_expr {
            FilterExpr::Clause { field, op, value } => {
                let index = scalar_cache.get(field)?;
                let scalar_value = field_value_to_scalar(value);
                match op {
                    ComparisonOp::Eq => Some(index.lookup_eq(&scalar_value)),
                    ComparisonOp::Ne => Some(index.lookup_range(RangeOp::Ne, &scalar_value)),
                    ComparisonOp::Gt => Some(index.lookup_range(RangeOp::Gt, &scalar_value)),
                    ComparisonOp::Gte => Some(index.lookup_range(RangeOp::Gte, &scalar_value)),
                    ComparisonOp::Lt => Some(index.lookup_range(RangeOp::Lt, &scalar_value)),
                    ComparisonOp::Lte => Some(index.lookup_range(RangeOp::Lte, &scalar_value)),
                }
            }
            FilterExpr::InList {
                field,
                negated,
                values,
            } => {
                let index = scalar_cache.get(field)?;
                let scalar_values: Vec<ScalarValue> =
                    values.iter().map(field_value_to_scalar).collect();
                Some(index.lookup_in(&scalar_values, *negated))
            }
            // AND: intersect candidates from each sub-expression.
            FilterExpr::And(exprs) => {
                let mut result: Option<BTreeSet<i64>> = None;
                for expr in exprs {
                    let sub = self.try_scalar_index_candidates(expr);
                    match (result.take(), sub) {
                        (None, Some(s)) => result = Some(s),
                        (Some(r), Some(s)) => {
                            let intersection: BTreeSet<i64> = r.intersection(&s).cloned().collect();
                            result = Some(intersection);
                        }
                        (prior, None) => result = prior,
                    }
                }
                result
            }
            // OR: union candidates from each sub-expression.
            FilterExpr::Or(exprs) => {
                let mut result: Option<BTreeSet<i64>> = None;
                for expr in exprs {
                    let sub = self.try_scalar_index_candidates(expr);
                    match (result.take(), sub) {
                        (None, Some(s)) => result = Some(s),
                        (Some(r), Some(s)) => {
                            let union: BTreeSet<i64> = r.union(&s).cloned().collect();
                            result = Some(union);
                        }
                        (prior, None) => result = prior,
                    }
                }
                result
            }
            FilterExpr::HasPrefix {
                field,
                pattern,
                negated,
            } => {
                let index = scalar_cache.get(field)?;
                let ids = index.lookup_prefix(pattern)?;
                if *negated {
                    None
                } else {
                    Some(ids)
                }
            }
            FilterExpr::HasSuffix {
                field,
                pattern,
                negated,
            } => {
                let index = scalar_cache.get(field)?;
                let ids = index.lookup_suffix(pattern)?;
                if *negated {
                    None
                } else {
                    Some(ids)
                }
            }
            // NOT and NullCheck are harder to accelerate; fall back.
            _ => None,
        }
    }

    fn collection_paths(&self) -> CollectionPaths {
        collection_paths_for_dir(self.segment_manager.collection_dir())
    }

    fn collection_primary_vector_name(&self) -> io::Result<String> {
        let paths = self.collection_paths();
        let collection_meta = CollectionMetadata::load_from_path(&paths.collection_meta)?;
        Ok(collection_meta.primary_vector)
    }

    fn load_search_state_for_field(&self, field_name: &str) -> io::Result<CachedSearchState> {
        let paths = self.collection_paths();
        let collection_meta = CollectionMetadata::load_from_path(&paths.collection_meta)?;
        let index_catalog = IndexCatalog::load_from_path(&paths.index_catalog)?;
        let descriptor = resolve_vector_descriptor_for_field(&collection_meta, &index_catalog, field_name)?
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::NotFound,
                    format!(
                        "vector field '{}' does not have a resolvable index descriptor in collection '{}'",
                        field_name, collection_meta.name
                    ),
                )
            })?;
        let vector_schema = collection_meta
            .vectors
            .iter()
            .find(|vector| vector.name == field_name)
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!(
                        "vector field '{}' is not defined in collection '{}'",
                        field_name, collection_meta.name
                    ),
                )
            })?;

        let (records, external_ids) = if field_name == collection_meta.primary_vector {
            load_shadowed_live_records(
                &self.segment_manager,
                vector_schema.dimension,
                &collection_meta.primary_vector,
                collection_meta.primary_is_fp16(),
            )?
        } else {
            load_shadowed_live_vector_records(
                &self.segment_manager,
                vector_schema.dimension,
                field_name,
                &collection_meta.primary_vector,
            )?
        };

        Ok(CachedSearchState {
            records: Arc::new(records),
            external_ids: Arc::new(external_ids),
            tombstone: Arc::new(TombstoneMask::new(0)),
            dimension: vector_schema.dimension,
            metric: descriptor
                .metric
                .clone()
                .unwrap_or_else(|| collection_meta.metric.clone())
                .to_ascii_lowercase(),
            field_name: field_name.to_string(),
            descriptor,
            optimized_ann: None,
        })
    }

    fn cached_search_state_for_field(&self, field_name: &str) -> io::Result<CachedSearchState> {
        if let Some(state) = self
            .search_cache
            .lock()
            .expect("search cache mutex poisoned")
            .get(field_name)
            .cloned()
        {
            return Ok(state);
        }

        let state = self
            .try_load_persisted_ann_state(field_name)?
            .unwrap_or(self.load_search_state_for_field(field_name)?);

        let mut cache = self
            .search_cache
            .lock()
            .expect("search cache mutex poisoned");
        if let Some(state) = cache.get(field_name) {
            return Ok(state.clone());
        }
        cache.insert(field_name.to_string(), state.clone());
        Ok(state)
    }

    fn cached_document_state(&self) -> io::Result<Arc<HashMap<i64, Document>>> {
        {
            let cache = self
                .document_cache
                .lock()
                .expect("document cache mutex poisoned");
            if let Some(ref state) = *cache {
                return Ok(state.clone());
            }
        }

        let state = self.build_document_cache()?;

        let mut cache = self
            .document_cache
            .lock()
            .expect("document cache mutex poisoned");
        if let Some(ref existing) = *cache {
            return Ok(existing.clone());
        }
        *cache = Some(state.clone());
        Ok(state)
    }

    fn build_document_cache(&self) -> io::Result<Arc<HashMap<i64, Document>>> {
        let paths = self.collection_paths();
        let collection_meta = CollectionMetadata::load_from_path(&paths.collection_meta)?;
        let segment_paths = self.segment_manager.segment_paths()?;

        let mut documents: HashMap<i64, Document> = HashMap::new();
        let mut shadowed_ids: HashSet<i64> = HashSet::new();

        for segment in &segment_paths {
            let segment_meta = SegmentMetadata::load_from_path(&segment.metadata)?;
            let dense_rows = load_primary_dense_rows_for_segment_or_empty(
                &segment,
                &segment_meta,
                &collection_meta.primary_vector,
                collection_meta.dimension,
                collection_meta.primary_is_fp16(),
            )?;
            let stored_ids = dense_rows.external_ids;
            let records = dense_rows.primary_vectors;
            let payloads = load_payloads_or_empty(&segment, &segment_meta, stored_ids.len())?;
            let vectors = load_vectors_or_empty(
                &segment,
                &segment_meta,
                &collection_meta.primary_vector,
                stored_ids.len(),
            )?;
            let sparse = load_sparse_vectors_or_empty(&segment.sparse_vectors, stored_ids.len())?;
            let tombstone = TombstoneMask::load_from_path(&segment.tombstones)?;

            if stored_ids.len().saturating_mul(collection_meta.dimension) != records.len() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "records and ids are not aligned",
                ));
            }

            for row_idx in (0..stored_ids.len()).rev() {
                let ext_id = stored_ids[row_idx];
                if !shadowed_ids.insert(ext_id) {
                    continue;
                }
                if tombstone.is_deleted(row_idx) {
                    continue;
                }
                let start = row_idx * collection_meta.dimension;
                let end = start + collection_meta.dimension;
                let mut doc_vectors = vectors[row_idx].clone();
                doc_vectors.insert(
                    collection_meta.primary_vector.clone(),
                    records[start..end].to_vec(),
                );
                documents.insert(
                    ext_id,
                    Document {
                        id: ext_id,
                        fields: payloads[row_idx].clone(),
                        vectors: doc_vectors,
                        sparse_vectors: sparse.get(row_idx).cloned().unwrap_or_default(),
                    },
                );
            }
        }

        Ok(Arc::new(documents))
    }

    fn try_load_persisted_ann_state(
        &self,
        field_name: &str,
    ) -> io::Result<Option<CachedSearchState>> {
        #[cfg(not(feature = "hanns-backend"))]
        {
            let _ = field_name;
            return Ok(None);
        }

        #[cfg(feature = "hanns-backend")]
        {
            let paths = self.collection_paths();
            let collection_meta = CollectionMetadata::load_from_path(&paths.collection_meta)?;
            let index_catalog = IndexCatalog::load_from_path(&paths.index_catalog)?;

            // Prepare fallback external IDs from segment data.
            let fallback_ids = {
                let state = self.load_search_state_for_field(field_name)?;
                Arc::clone(&state.external_ids)
            };

            persist::load_persisted_ann_from_disk(
                &paths,
                &collection_meta,
                &index_catalog,
                field_name,
                fallback_ids,
            )
        }
    }

    fn invalidate_search_cache(&self) {
        let mut cache = self
            .search_cache
            .lock()
            .expect("search cache mutex poisoned");
        cache.clear();
        let mut doc_cache = self
            .document_cache
            .lock()
            .expect("document cache mutex poisoned");
        *doc_cache = None;
        let mut sparse_cache = self
            .sparse_index_cache
            .lock()
            .expect("sparse index cache mutex poisoned");
        sparse_cache.clear();
    }

    fn refresh_version_set(&self) -> io::Result<()> {
        let version_set = self.segment_manager.version_set()?;
        let mut snapshot = self
            .version_set
            .write()
            .expect("version_set rwlock poisoned");
        *snapshot = version_set;
        Ok(())
    }
}

#[cfg(feature = "hanns-backend")]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::document::{
        CollectionSchema, Document, FieldType, FieldValue, ScalarFieldSchema, VectorFieldSchema,
        VectorIndexSchema,
    };
    use crate::segment::load_payloads_arrow;
    use std::path::Path;
    use tempfile::tempdir;

    #[test]
    fn secondary_indexed_field_search_returns_correct_results_without_optimize() {
        let tempdir = tempdir().expect("tempdir");
        let db = seeded_db_with_secondary_hnsw_indexed_collection(tempdir.path());
        let handle = db
            .open_collection_handle("docs")
            .expect("open collection handle");

        let hits = handle
            .search_field_with_ef_internal("title", &[0.0_f32, 0.0], 3, 64)
            .expect("search secondary vector field");

        assert_eq!(
            hits.iter().map(|hit| hit.id).collect::<Vec<_>>(),
            vec![1, 3, 2]
        );

        let cache = handle
            .search_cache
            .lock()
            .expect("search cache mutex poisoned");
        let state = cache.get("title").expect("cached secondary search state");
        assert!(
            state.optimized_ann.is_none(),
            "secondary field should not have ANN runtime until optimize() is called or persisted blob is loaded"
        );
    }

    #[test]
    fn primary_plain_search_keeps_cached_state_unoptimized_without_optimize() {
        let tempdir = tempdir().expect("tempdir");
        let db = seeded_db_with_secondary_hnsw_indexed_collection(tempdir.path());
        let handle = db
            .open_collection_handle("docs")
            .expect("open collection handle");

        let hits = db
            .search("docs", &[0.0_f32, 0.0], 1)
            .expect("plain primary search");

        assert_eq!(hits.iter().map(|hit| hit.id).collect::<Vec<_>>(), vec![2]);

        let cache = handle
            .search_cache
            .lock()
            .expect("search cache mutex poisoned");
        let state = cache.get("dense").expect("cached primary search state");
        assert!(
            state.optimized_ann.is_none(),
            "plain primary search should stay unoptimized until optimize() or persisted HNSW load"
        );
    }

    #[test]
    fn secondary_ivf_indexed_field_search_returns_correct_results_without_optimize() {
        let tempdir = tempdir().expect("tempdir");
        let db = seeded_db_with_secondary_ivf_indexed_collection(tempdir.path());
        let handle = db
            .open_collection_handle("docs")
            .expect("open collection handle");

        let hits = handle
            .search_field_with_ef_internal("title", &[0.0_f32, 0.0], 3, 64)
            .expect("search secondary ivf vector field");

        assert_eq!(
            hits.iter().map(|hit| hit.id).collect::<Vec<_>>(),
            vec![1, 3, 2]
        );

        let cache = handle
            .search_cache
            .lock()
            .expect("search cache mutex poisoned");
        let state = cache.get("title").expect("cached secondary search state");
        assert!(
            state.optimized_ann.is_none(),
            "secondary IVF field should not have ANN runtime until optimize() is called or persisted blob is loaded"
        );
    }

    #[test]
    fn build_optimized_ann_state_skips_serialization_when_bytes_not_requested() {
        let tempdir = tempdir().expect("tempdir");
        let db = seeded_db_with_secondary_hnsw_indexed_collection(tempdir.path());
        let handle = db
            .open_collection_handle("docs")
            .expect("open collection handle");
        let state = handle
            .load_search_state_for_field("title")
            .expect("load secondary state");

        let optimized = build_optimized_ann_state(&state, None)
            .expect("build optimized ann state without requesting serialized bytes");

        let hits = ann_search(
            optimized.backend.as_ref(),
            optimized.ann_external_ids.as_slice(),
            &optimized.metric,
            &[0.0_f32, 0.0],
            3,
            64,
        )
        .expect("ann search");
        assert_eq!(
            hits.iter().map(|hit| hit.id).collect::<Vec<_>>(),
            vec![1, 3, 2]
        );
    }

    fn seeded_db_with_secondary_hnsw_indexed_collection(root: &Path) -> HannsDb {
        seeded_db_with_secondary_indexed_collection(
            root,
            VectorIndexSchema::hnsw(Some("l2"), 16, 128),
        )
    }

    fn seeded_db_with_secondary_ivf_indexed_collection(root: &Path) -> HannsDb {
        seeded_db_with_secondary_indexed_collection(root, VectorIndexSchema::ivf(Some("l2"), 8))
    }

    fn seeded_db_with_secondary_indexed_collection(
        root: &Path,
        secondary_index: VectorIndexSchema,
    ) -> HannsDb {
        let mut db = HannsDb::open(root).expect("open db");
        let mut schema = CollectionSchema::new("dense", 2, "l2", Vec::new());
        schema
            .vectors
            .push(VectorFieldSchema::new("title", 2).with_index_param(secondary_index));
        db.create_collection_with_schema("docs", &schema)
            .expect("create collection");
        db.insert_documents(
            "docs",
            &[
                Document::with_named_vectors(
                    1,
                    [],
                    "dense",
                    vec![5.0_f32, 5.0],
                    [("title".to_string(), vec![0.0_f32, 0.0])],
                ),
                Document::with_named_vectors(
                    2,
                    [],
                    "dense",
                    vec![0.0_f32, 0.0],
                    [("title".to_string(), vec![2.0_f32, 0.0])],
                ),
                Document::with_named_vectors(
                    3,
                    [],
                    "dense",
                    vec![1.0_f32, 1.0],
                    [("title".to_string(), vec![1.0_f32, 0.0])],
                ),
            ],
        )
        .expect("insert documents");
        db
    }

    #[test]
    fn pre_append_rollover_reloads_active_segment_sidecars() {
        let tempdir = tempdir().expect("tempdir");
        let mut db = HannsDb::open(tempdir.path()).expect("open db");
        let schema = CollectionSchema::new(
            "vector",
            2,
            "l2",
            vec![ScalarFieldSchema::new("kind", FieldType::String)],
        );
        db.create_collection_with_schema("docs", &schema)
            .expect("create collection");
        db.insert_documents(
            "docs",
            &[
                Document::new(
                    1,
                    [("kind".into(), FieldValue::String("old".into()))],
                    vec![1.0, 0.0],
                ),
                Document::new(
                    2,
                    [("kind".into(), FieldValue::String("old".into()))],
                    vec![0.8, 0.0],
                ),
                Document::new(
                    3,
                    [("kind".into(), FieldValue::String("old".into()))],
                    vec![0.6, 0.0],
                ),
            ],
        )
        .expect("insert initial docs");
        assert_eq!(
            db.delete_by_filter("docs", "kind == \"old\"")
                .expect("delete old docs"),
            3
        );

        db.insert_documents(
            "docs",
            &[Document::new(
                4,
                [("kind".into(), FieldValue::String("new".into()))],
                vec![0.0, 0.0],
            )],
        )
        .expect("insert replacement doc");

        let hits = db
            .query_documents("docs", &[0.0, 0.0], 10, Some("kind == \"new\""))
            .expect("query replacement doc");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].id, 4);

        let paths = collection_paths_for_root(tempdir.path(), "docs");
        let active_segment = SegmentManager::new(paths.dir.clone())
            .active_segment_path()
            .expect("active segment path");
        let payloads = load_payloads_arrow(&active_segment.payloads_arrow)
            .expect("load active segment payloads");
        assert_eq!(
            payloads.len(),
            1,
            "new active segment should only persist its own row"
        );
    }

    #[test]
    fn auto_compact_triggers_after_threshold() {
        use crate::segment::segment_set::COMPACTION_THRESHOLD;
        let tempdir = tempdir().expect("tempdir");
        let mut db = HannsDb::open(tempdir.path()).expect("open db");
        db.create_collection("docs", 2, "l2").expect("create");

        let dim = 2usize;

        // Manually create COMPACTION_THRESHOLD immutable segments + an active one.
        let paths = collection_paths_for_root(tempdir.path(), "docs");
        let seg_manager = SegmentManager::new(paths.dir.clone());

        // Create immutable segments
        let mut immutable_ids = Vec::new();
        for i in 0..COMPACTION_THRESHOLD {
            let seg_id = format!("seg-{:06}", i + 1);
            seg_manager
                .create_segment_dir(&seg_id, dim)
                .expect("create seg");
            let seg_paths =
                SegmentPaths::from_segment_dir(paths.segments_dir.join(&seg_id), seg_id.clone());
            let ids = vec![i as i64 * 10 + 1];
            let vectors = vec![1.0_f32, 0.0];
            append_records(&seg_paths.records, dim, &vectors).expect("append");
            append_record_ids(&seg_paths.external_ids, &ids).expect("append ids");
            let mut meta = SegmentMetadata::load_from_path(&seg_paths.metadata).expect("meta");
            meta.record_count = 1;
            meta.save_to_path(&seg_paths.metadata).expect("save meta");
            TombstoneMask::new(1)
                .save_to_path(&seg_paths.tombstones)
                .expect("save tombstones");
            immutable_ids.push(seg_id);
        }

        // Create active segment
        let active_id = format!("seg-{:06}", COMPACTION_THRESHOLD + 1);
        seg_manager
            .create_segment_dir(&active_id, dim)
            .expect("create active seg");
        let active_paths =
            SegmentPaths::from_segment_dir(paths.segments_dir.join(&active_id), active_id.clone());
        let active_ids = vec![999];
        let active_vectors = vec![1.0_f32, 0.0];
        append_records(&active_paths.records, dim, &active_vectors).expect("append");
        append_record_ids(&active_paths.external_ids, &active_ids).expect("append ids");
        let mut meta = SegmentMetadata::load_from_path(&active_paths.metadata).expect("meta");
        meta.record_count = 1;
        meta.save_to_path(&active_paths.metadata)
            .expect("save meta");
        TombstoneMask::new(1)
            .save_to_path(&active_paths.tombstones)
            .expect("save tombstones");

        let vs = VersionSet::new(active_id, immutable_ids);
        vs.save_to_path(&seg_manager.version_set_path())
            .expect("save vs");

        // Call compact_collection directly — same code path as auto-trigger.
        db.compact_collection("docs").expect("compact");

        let segments = db.list_collection_segments("docs").expect("segments");
        assert!(
            segments.len() <= 2,
            "expected compaction to reduce segments to <= 2 (active + compacted), got {}",
            segments.len()
        );

        // Verify data is intact.
        let info = db.get_collection_info("docs").expect("info");
        assert_eq!(
            info.live_count,
            COMPACTION_THRESHOLD + 1,
            "all live documents should survive compaction"
        );

        let hits = db.search("docs", &[1.0, 0.0], 5).expect("search");
        assert_eq!(hits.len(), COMPACTION_THRESHOLD + 1);
    }

    #[test]
    fn compact_output_materializes_forward_store_artifacts() {
        let tempdir = tempdir().expect("tempdir");
        let mut db = HannsDb::open(tempdir.path()).expect("open db");
        db.create_collection("docs", 2, "l2").expect("create");

        let dim = 2usize;
        let paths = collection_paths_for_root(tempdir.path(), "docs");
        let seg_manager = SegmentManager::new(paths.dir.clone());

        // Create 3 immutable segments with data.
        let mut immutable_ids = Vec::new();
        for i in 0..3 {
            let seg_id = format!("seg-{:06}", i + 1);
            seg_manager
                .create_segment_dir(&seg_id, dim)
                .expect("create seg");
            let seg_paths =
                SegmentPaths::from_segment_dir(paths.segments_dir.join(&seg_id), seg_id.clone());
            let ids = vec![i as i64 * 10 + 1];
            let vectors = vec![1.0_f32, 0.0];
            append_records(&seg_paths.records, dim, &vectors).expect("append");
            append_record_ids(&seg_paths.external_ids, &ids).expect("append ids");
            let mut meta = SegmentMetadata::load_from_path(&seg_paths.metadata).expect("meta");
            meta.record_count = 1;
            meta.save_to_path(&seg_paths.metadata).expect("save meta");
            TombstoneMask::new(1)
                .save_to_path(&seg_paths.tombstones)
                .expect("save tombstones");
            immutable_ids.push(seg_id);
        }

        // Create active segment.
        let active_id = "seg-0004".to_string();
        seg_manager
            .create_segment_dir(&active_id, dim)
            .expect("create active seg");
        let vs = VersionSet::new(active_id, immutable_ids);
        vs.save_to_path(&seg_manager.version_set_path())
            .expect("save vs");

        db.compact_collection("docs").expect("compact");

        // Verify compacted segment keeps compatibility arrow files and writes
        // authoritative forward_store artifacts.
        let version_set = VersionSet::load_from_path(&paths.segment_set).expect("vs");
        let immutables = version_set.immutable_segment_ids();
        assert_eq!(immutables.len(), 1, "should have 1 compacted segment");

        let compacted_dir = paths.segments_dir.join(&immutables[0]);
        assert!(
            compacted_dir.join("payloads.arrow").exists(),
            "compacted segment should have payloads.arrow"
        );
        assert!(
            compacted_dir.join("vectors.arrow").exists(),
            "compacted segment should have vectors.arrow"
        );
        assert!(
            compacted_dir.join("forward_store.json").exists(),
            "compacted segment should persist forward_store descriptor"
        );
        assert!(
            compacted_dir.join("forward_store.arrow").exists(),
            "compacted segment should persist forward_store Arrow IPC artifact"
        );
        assert!(
            compacted_dir.join("forward_store.parquet").exists(),
            "compacted segment should persist forward_store Parquet artifact"
        );
    }

    #[test]
    fn atomic_save_no_tmp_residual() {
        let tempdir = tempdir().expect("tempdir");
        let path = tempdir.path().join("test.json");
        let vs = VersionSet::single("seg-0001");
        vs.save_to_path(&path).expect("save");
        assert!(path.exists(), "file should exist");
        assert!(
            !path.with_extension("tmp").exists(),
            "tmp file should not remain after atomic save"
        );
        let loaded = VersionSet::load_from_path(&path).expect("load");
        assert_eq!(loaded, vs);
    }

    #[test]
    fn order_by_scalar_field_ascending() {
        use crate::query::{OrderBy, QueryContext, QueryVector, VectorQuery};
        let tempdir = tempdir().expect("tempdir");
        let mut db = HannsDb::open(tempdir.path()).expect("open db");
        db.create_collection("docs", 2, "l2").expect("create");

        db.insert_documents(
            "docs",
            &[
                Document::new(1, [("score".into(), FieldValue::Int64(10))], vec![1.0, 0.0]),
                Document::new(2, [("score".into(), FieldValue::Int64(30))], vec![0.5, 0.0]),
                Document::new(3, [("score".into(), FieldValue::Int64(20))], vec![0.8, 0.0]),
            ],
        )
        .expect("insert");

        let ctx = QueryContext {
            top_k: 10,
            queries: vec![VectorQuery {
                field_name: "vector".to_string(),
                vector: QueryVector::Dense(vec![0.0, 0.0]),
                param: None,
            }],
            order_by: Some(OrderBy {
                field_name: "score".to_string(),
                descending: false,
            }),
            ..Default::default()
        };

        let hits = db.query_with_context("docs", &ctx).expect("query");
        assert_eq!(hits.len(), 3);
        assert_eq!(hits[0].id, 1);
        assert_eq!(hits[1].id, 3);
        assert_eq!(hits[2].id, 2);
    }

    #[test]
    fn order_by_scalar_field_descending() {
        use crate::query::{OrderBy, QueryContext, QueryVector, VectorQuery};
        let tempdir = tempdir().expect("tempdir");
        let mut db = HannsDb::open(tempdir.path()).expect("open db");
        db.create_collection("docs", 2, "l2").expect("create");

        db.insert_documents(
            "docs",
            &[
                Document::new(
                    1,
                    [("name".into(), FieldValue::String("alice".into()))],
                    vec![1.0, 0.0],
                ),
                Document::new(
                    2,
                    [("name".into(), FieldValue::String("charlie".into()))],
                    vec![0.5, 0.0],
                ),
                Document::new(
                    3,
                    [("name".into(), FieldValue::String("bob".into()))],
                    vec![0.8, 0.0],
                ),
            ],
        )
        .expect("insert");

        let ctx = QueryContext {
            top_k: 10,
            queries: vec![VectorQuery {
                field_name: "vector".to_string(),
                vector: QueryVector::Dense(vec![0.0, 0.0]),
                param: None,
            }],
            order_by: Some(OrderBy {
                field_name: "name".to_string(),
                descending: true,
            }),
            ..Default::default()
        };

        let hits = db.query_with_context("docs", &ctx).expect("query");
        assert_eq!(hits.len(), 3);
        assert_eq!(hits[0].id, 2);
        assert_eq!(hits[1].id, 3);
        assert_eq!(hits[2].id, 1);
    }

    #[test]
    fn order_by_rejects_vector_field() {
        use crate::query::{OrderBy, QueryContext, QueryVector, VectorQuery};
        let tempdir = tempdir().expect("tempdir");
        let mut db = HannsDb::open(tempdir.path()).expect("open db");
        db.create_collection("docs", 2, "l2").expect("create");

        let ctx = QueryContext {
            top_k: 10,
            queries: vec![VectorQuery {
                field_name: "vector".to_string(),
                vector: QueryVector::Dense(vec![0.0, 0.0]),
                param: None,
            }],
            order_by: Some(OrderBy {
                field_name: "vector".to_string(),
                descending: false,
            }),
            ..Default::default()
        };

        let result = db.query_with_context("docs", &ctx);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
    }

    #[test]
    fn order_by_rejects_empty_field_name() {
        use crate::query::{OrderBy, QueryContext, QueryVector, VectorQuery};
        let tempdir = tempdir().expect("tempdir");
        let mut db = HannsDb::open(tempdir.path()).expect("open db");
        db.create_collection("docs", 2, "l2").expect("create");

        let ctx = QueryContext {
            top_k: 10,
            queries: vec![VectorQuery {
                field_name: "vector".to_string(),
                vector: QueryVector::Dense(vec![0.0, 0.0]),
                param: None,
            }],
            order_by: Some(OrderBy {
                field_name: "".to_string(),
                descending: false,
            }),
            ..Default::default()
        };

        let result = db.query_with_context("docs", &ctx);
        assert!(result.is_err());
    }

    #[test]
    fn sparse_vector_crud_and_search() {
        use crate::document::{FieldType, SparseVector, VectorFieldSchema};
        let tempdir = tempdir().expect("tempdir");
        let mut db = HannsDb::open(tempdir.path()).expect("open db");

        // Create collection with a primary dense vector and a sparse vector field.
        let mut schema = CollectionSchema::new("dense", 2, "l2", Vec::new());
        schema.vectors.push(VectorFieldSchema {
            name: "sparse_title".to_string(),
            data_type: FieldType::VectorSparse,
            dimension: 0, // sparse vectors have no fixed dimension
            index_param: None,
            bm25_params: None,
        });
        db.create_collection_with_schema("docs", &schema)
            .expect("create collection");

        let docs = &[
            Document::with_sparse_vectors(
                1,
                [("label".into(), FieldValue::String("a".into()))],
                "dense",
                vec![1.0_f32, 0.0],
                [(
                    "sparse_title".to_string(),
                    SparseVector::new(vec![0, 3, 7], vec![1.0, 2.0, 0.5]),
                )],
            ),
            Document::with_sparse_vectors(
                2,
                [("label".into(), FieldValue::String("b".into()))],
                "dense",
                vec![0.0_f32, 1.0],
                [(
                    "sparse_title".to_string(),
                    SparseVector::new(vec![1, 3, 5], vec![0.5, 3.0, 1.0]),
                )],
            ),
            Document::with_sparse_vectors(
                3,
                [("label".into(), FieldValue::String("c".into()))],
                "dense",
                vec![0.5_f32, 0.5],
                [(
                    "sparse_title".to_string(),
                    SparseVector::new(vec![0, 5], vec![2.0, 1.0]),
                )],
            ),
        ];
        db.insert_documents("docs", docs).expect("insert documents");

        // Fetch and verify sparse vectors round-trip.
        let fetched = db.fetch_documents("docs", &[1, 2, 3]).expect("fetch");
        assert_eq!(fetched.len(), 3);
        let doc1 = fetched.iter().find(|d| d.id == 1).expect("doc 1");
        let sv1 = doc1
            .sparse_vectors
            .get("sparse_title")
            .expect("sparse field");
        assert_eq!(sv1.indices, vec![0, 3, 7]);
        assert_eq!(sv1.values, vec![1.0, 2.0, 0.5]);

        // Search sparse with a query that has high overlap with doc 2.
        // Query: indices [1, 3, 5], values [1.0, 1.0, 1.0]
        // IP(doc1): index 3 → 1.0*2.0 = 2.0
        // IP(doc2): index 1→0.5*1.0, 3→3.0*1.0, 5→1.0*1.0 = 4.5
        // IP(doc3): index 5→1.0*1.0 = 1.0
        // Distance = -IP, so smallest distance wins: doc2 (−4.5) < doc1 (−2.0) < doc3 (−1.0)
        let query = SparseVector::new(vec![1, 3, 5], vec![1.0, 1.0, 1.0]);
        let hits = db
            .search_sparse("docs", "sparse_title", &query, 3)
            .expect("sparse search");
        assert_eq!(hits.len(), 3);
        assert_eq!(hits[0].id, 2);
        assert!(hits[0].distance < hits[1].distance);
        assert_eq!(hits[1].id, 1);
        assert_eq!(hits[2].id, 3);
    }

    #[test]
    fn sparse_inner_product_correctness() {
        use crate::document::SparseVector;
        use crate::query::sparse_inner_product;

        let a = SparseVector::new(vec![0, 3, 7], vec![1.0, 2.0, 0.5]);
        let b = SparseVector::new(vec![1, 3, 5], vec![0.5, 3.0, 1.0]);
        // Only index 3 matches: 2.0 * 3.0 = 6.0
        assert!((sparse_inner_product(&a, &b) - 6.0).abs() < 1e-6);

        // Self product: 1*1 + 2*2 + 0.5*0.5 = 5.25
        assert!((sparse_inner_product(&a, &a) - 5.25).abs() < 1e-6);

        // No overlap
        let c = SparseVector::new(vec![10, 20], vec![1.0, 1.0]);
        assert!((sparse_inner_product(&a, &c)).abs() < 1e-6);
    }
}
