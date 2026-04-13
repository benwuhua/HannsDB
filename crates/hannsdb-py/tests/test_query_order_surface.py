import hannsdb
import pytest


def test_query_order_by_surface_is_exported():
    order_by = hannsdb.QueryOrderBy(field_name="group")

    assert order_by.field_name == "group"
    assert order_by.descending is False


def test_query_context_accepts_order_by_surface():
    context = hannsdb.QueryContext(
        top_k=3,
        order_by=hannsdb.QueryOrderBy(field_name="group", descending=True),
    )

    assert context.order_by.field_name == "group"
    assert context.order_by.descending is True


def test_real_collection_query_orders_by_scalar_field(tmp_path):
    schema = hannsdb.CollectionSchema(
        name="docs",
        primary_vector="dense",
        fields=[
            hannsdb.FieldSchema(name="group", data_type="int64"),
            hannsdb.FieldSchema(name="rank", data_type="int64"),
        ],
        vectors=[
            hannsdb.VectorSchema(
                name="dense",
                data_type="vector_fp32",
                dimension=2,
            )
        ],
    )
    collection = hannsdb.create_and_open(str(tmp_path), schema)
    collection.insert(
        [
            hannsdb.Doc(id="user-a", vector=[0.0, 0.0], fields={"group": 1, "rank": 2}),
            hannsdb.Doc(id="user-b", vector=[0.1, 0.0], fields={"group": 1, "rank": 1}),
            hannsdb.Doc(id="user-c", vector=[0.2, 0.0], fields={"group": 2, "rank": 3}),
        ]
    )

    result = collection.query(
        query_context=hannsdb.QueryContext(
            top_k=3,
            query_by_id=["user-b"],
            output_fields=["rank"],
            order_by=hannsdb.QueryOrderBy(field_name="rank"),
        )
    )

    assert [doc.id for doc in result] == ["user-b", "user-a", "user-c"]
    assert [doc.field("rank") for doc in result] == [1, 2, 3]

    collection.destroy()


def test_real_collection_query_orders_by_scalar_field_via_legacy_kwargs(tmp_path):
    schema = hannsdb.CollectionSchema(
        name="docs",
        primary_vector="dense",
        fields=[hannsdb.FieldSchema(name="rank", data_type="int64")],
        vectors=[
            hannsdb.VectorSchema(
                name="dense",
                data_type="vector_fp32",
                dimension=2,
            )
        ],
    )
    collection = hannsdb.create_and_open(str(tmp_path), schema)
    collection.insert(
        [
            hannsdb.Doc(id="user-a", vector=[0.0, 0.0], fields={"rank": 2}),
            hannsdb.Doc(id="user-b", vector=[0.1, 0.0], fields={"rank": 1}),
            hannsdb.Doc(id="user-c", vector=[0.2, 0.0], fields={"rank": 3}),
        ]
    )

    result = collection.query(
        query_by_id=["user-b"],
        output_fields=["rank"],
        order_by=hannsdb.QueryOrderBy(field_name="rank"),
    )

    assert [doc.id for doc in result] == ["user-b", "user-a", "user-c"]
    assert [doc.field("rank") for doc in result] == [1, 2, 3]

    collection.destroy()


def test_real_collection_query_rejects_vector_order_by_field(tmp_path):
    schema = hannsdb.CollectionSchema(
        name="docs",
        primary_vector="dense",
        fields=[hannsdb.FieldSchema(name="group", data_type="int64")],
        vectors=[
            hannsdb.VectorSchema(
                name="dense",
                data_type="vector_fp32",
                dimension=2,
            )
        ],
    )
    collection = hannsdb.create_and_open(str(tmp_path), schema)
    collection.insert(
        [
            hannsdb.Doc(id="user-a", vector=[0.0, 0.0], fields={"group": 1}),
            hannsdb.Doc(id="user-b", vector=[0.1, 0.0], fields={"group": 1}),
        ]
    )

    with pytest.raises(Exception, match="order_by"):
        collection.query(
            query_context=hannsdb.QueryContext(
                top_k=2,
                query_by_id=["user-b"],
                order_by=hannsdb.QueryOrderBy(field_name="dense"),
            )
        )

    collection.destroy()
