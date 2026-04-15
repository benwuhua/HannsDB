use std::collections::HashSet;
use std::io;

use crate::catalog::CollectionMetadata;
use crate::pk::{PrimaryKeyMode, PrimaryKeyRegistry};
use crate::storage::paths::CollectionPaths;
use crate::storage::segment_io::load_all_collection_ids;

pub(crate) fn load_primary_key_registry(
    paths: &CollectionPaths,
    collection_meta: &CollectionMetadata,
) -> io::Result<PrimaryKeyRegistry> {
    if paths.primary_keys.exists() {
        let registry = PrimaryKeyRegistry::load_from_path(&paths.primary_keys)?;
        if registry.mode != collection_meta.primary_key_mode {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "primary key registry mode does not match collection metadata",
            ));
        }
        return Ok(registry);
    }

    let next_internal_id = collection_max_internal_id(paths)?.saturating_add(1).max(1);
    let mut registry =
        PrimaryKeyRegistry::new(collection_meta.primary_key_mode.clone(), next_internal_id);
    if registry.mode == PrimaryKeyMode::String {
        populate_numeric_keys_into_registry(paths, &mut registry)?;
    }
    Ok(registry)
}

pub(crate) fn save_primary_key_registry(
    paths: &CollectionPaths,
    collection_meta: &CollectionMetadata,
    registry: &PrimaryKeyRegistry,
) -> io::Result<()> {
    if registry.mode != collection_meta.primary_key_mode {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "primary key registry mode does not match collection metadata",
        ));
    }
    registry.save_to_path(&paths.primary_keys)
}

pub(crate) fn ensure_string_primary_key_mode(
    paths: &CollectionPaths,
    collection_meta: &mut CollectionMetadata,
) -> io::Result<PrimaryKeyRegistry> {
    let mut registry = load_primary_key_registry(paths, collection_meta)?;
    if collection_meta.primary_key_mode == PrimaryKeyMode::String {
        return Ok(registry);
    }

    collection_meta.primary_key_mode = PrimaryKeyMode::String;
    collection_meta.save_to_path(&paths.collection_meta)?;

    registry.mode = PrimaryKeyMode::String;
    populate_numeric_keys_into_registry(paths, &mut registry)?;
    save_primary_key_registry(paths, collection_meta, &registry)?;
    Ok(registry)
}

pub(crate) fn assign_internal_ids_for_public_keys(
    paths: &CollectionPaths,
    collection_meta: &mut CollectionMetadata,
    public_keys: &[String],
) -> io::Result<Vec<i64>> {
    if public_keys.is_empty() {
        return Ok(Vec::new());
    }

    if collection_meta.primary_key_mode == PrimaryKeyMode::Numeric
        && public_keys.iter().all(|key| key.parse::<i64>().is_ok())
    {
        return public_keys
            .iter()
            .map(|key| parse_numeric_public_key(key))
            .collect();
    }

    let mut registry = ensure_string_primary_key_mode(paths, collection_meta)?;
    let mut ids = Vec::with_capacity(public_keys.len());
    for public_key in public_keys {
        if let Some(existing) = registry.key_to_id.get(public_key).copied() {
            ids.push(existing);
            continue;
        }

        let internal_id = registry.next_internal_id;
        registry.next_internal_id = registry.next_internal_id.saturating_add(1).max(1);
        registry.key_to_id.insert(public_key.clone(), internal_id);
        registry.id_to_key.insert(internal_id, public_key.clone());
        ids.push(internal_id);
    }

    save_primary_key_registry(paths, collection_meta, &registry)?;
    Ok(ids)
}

pub(crate) fn resolve_public_keys_to_internal_ids(
    paths: &CollectionPaths,
    collection_meta: &CollectionMetadata,
    public_keys: &[String],
) -> io::Result<Vec<i64>> {
    match collection_meta.primary_key_mode {
        PrimaryKeyMode::Numeric => public_keys
            .iter()
            .map(|key| parse_numeric_public_key(key))
            .collect(),
        PrimaryKeyMode::String => {
            let registry = load_primary_key_registry(paths, collection_meta)?;
            Ok(public_keys
                .iter()
                .filter_map(|public_key| registry.key_to_id.get(public_key).copied())
                .collect())
        }
    }
}

pub(crate) fn upsert_public_keys_with_internal_ids(
    paths: &CollectionPaths,
    collection_meta: &mut CollectionMetadata,
    public_keys: &[String],
    internal_ids: &[i64],
) -> io::Result<()> {
    if public_keys.len() != internal_ids.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "public key count must match internal id count",
        ));
    }

    if collection_meta.primary_key_mode == PrimaryKeyMode::Numeric
        && public_keys.iter().all(|key| key.parse::<i64>().is_ok())
    {
        for (public_key, internal_id) in public_keys.iter().zip(internal_ids) {
            let parsed = parse_numeric_public_key(public_key)?;
            if parsed != *internal_id {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!(
                        "numeric public key '{public_key}' must match document id {internal_id}"
                    ),
                ));
            }
        }
        return Ok(());
    }

    let mut registry = ensure_string_primary_key_mode(paths, collection_meta)?;
    for (public_key, internal_id) in public_keys.iter().zip(internal_ids) {
        if let Some(existing_id) = registry.key_to_id.get(public_key).copied() {
            if existing_id == *internal_id {
                // Idempotent: same key → same ID, no-op.
                continue;
            }
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "public key already mapped to different id: {public_key} -> {existing_id}, got {internal_id}"
                ),
            ));
        }
        if let Some(existing_key) = registry.id_to_key.get(internal_id) {
            if existing_key == public_key {
                // Idempotent: same ID → same key, no-op.
                continue;
            }
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "internal id already registered to different key: {internal_id} -> {existing_key}, got {public_key}"
                ),
            ));
        }
        registry.key_to_id.insert(public_key.clone(), *internal_id);
        registry.id_to_key.insert(*internal_id, public_key.clone());
        registry.next_internal_id = registry.next_internal_id.max(internal_id.saturating_add(1));
    }

    save_primary_key_registry(paths, collection_meta, &registry)
}

pub(crate) fn display_key_for_internal_id(
    registry: &PrimaryKeyRegistry,
    internal_id: i64,
) -> String {
    match registry.mode {
        PrimaryKeyMode::Numeric => internal_id.to_string(),
        PrimaryKeyMode::String => registry
            .id_to_key
            .get(&internal_id)
            .cloned()
            .unwrap_or_else(|| internal_id.to_string()),
    }
}

pub(crate) fn parse_numeric_public_key(key: &str) -> io::Result<i64> {
    key.parse::<i64>().map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("public key must parse to i64 in numeric mode: {key}"),
        )
    })
}

fn populate_numeric_keys_into_registry(
    paths: &CollectionPaths,
    registry: &mut PrimaryKeyRegistry,
) -> io::Result<()> {
    for internal_id in load_all_collection_ids(paths)? {
        let public_key = internal_id.to_string();
        registry.key_to_id.insert(public_key.clone(), internal_id);
        registry.id_to_key.insert(internal_id, public_key);
        registry.next_internal_id = registry.next_internal_id.max(internal_id.saturating_add(1));
    }
    Ok(())
}

fn collection_max_internal_id(paths: &CollectionPaths) -> io::Result<i64> {
    let mut max_internal_id = 0i64;
    for internal_id in load_all_collection_ids(paths)? {
        max_internal_id = max_internal_id.max(internal_id);
    }
    Ok(max_internal_id)
}
