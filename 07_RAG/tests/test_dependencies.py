from app.dependencies import (
    get_document_service,
    get_embedding_service,
    get_ingestion_service,
    get_vector_store,
)


def test_embedding_service_is_cached():

    service_1 = get_embedding_service()
    service_2 = get_embedding_service()

    assert service_1 is service_2


def test_vector_store_is_cached():

    store_1 = get_vector_store()
    store_2 = get_vector_store()

    assert store_1 is store_2


def test_ingestion_service_is_cached():

    service_1 = get_ingestion_service()
    service_2 = get_ingestion_service()

    assert service_1 is service_2


def test_document_service_is_cached():

    service_1 = get_document_service()
    service_2 = get_document_service()

    assert service_1 is service_2