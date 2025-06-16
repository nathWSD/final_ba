from src.ingestion_pipeline.helper_functions import format_qdrant_results_to_string, TimeMeasurer

from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding
from fastembed import SparseTextEmbedding
from typing import List, Any, Optional, Dict, Tuple
import traceback
from dotenv import load_dotenv
import os
load_dotenv(dotenv_path=os.path.join(os.getcwd(), '.env'), override=True)

    
# --- Configuration ---
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_TIMEOUT =  int(os.getenv("QDRANT_TIMEOUT"))
BATCH_SIZE = 4 


COLLECTION_BASE_NAME = os.getenv("COLLECTION_BASE_NAME")
DENSE_MODEL_KEY = os.getenv("DENSE_MODEL_KEY")
SPARSE_MODEL_KEY = os.getenv("SPARSE_MODEL_KEY")

# Collection Names
DENSE_COLLECTION_NAME = f"{COLLECTION_BASE_NAME}_dense"
SPARSE_COLLECTION_NAME = f"{COLLECTION_BASE_NAME}_sparse"
HYBRID_COLLECTION_NAME = f"{COLLECTION_BASE_NAME}_hybrid"

dense_embedding_model = TextEmbedding(model_name=DENSE_MODEL_KEY)
sparse_embedding_model = SparseTextEmbedding(model_name=SPARSE_MODEL_KEY)
sample_embedding = list(dense_embedding_model.embed(["sample text"]))
DENSE_VECTOR_SIZE = len(sample_embedding[0])


# --- Qdrant Client ---
client = QdrantClient(QDRANT_URL, timeout=QDRANT_TIMEOUT)


# 1. Dense Only Config
dense_vectors_config = {
    DENSE_MODEL_KEY: models.VectorParams(
        size=DENSE_VECTOR_SIZE,
        distance=models.Distance.COSINE,
    )
}

# 2. Sparse Only Config
sparse_vectors_config = {
    SPARSE_MODEL_KEY: models.SparseVectorParams() 
}

# 3. Hybrid Config (Combines Dense and Sparse)
hybrid_vectors_config = dense_vectors_config 
hybrid_sparse_vectors_config = sparse_vectors_config 


def retrieve_chunks(
    client: QdrantClient,
    query: str,
    collection_name: str,
    n: int,
    dense_model: Any,
    sparse_model: Any,
    dense_model_key: str,
    sparse_model_key: str
) -> List[models.ScoredPoint]:
    """
    Retrieves the top 'n' relevant chunks using client.query_points, applying
    the correct arguments (query=raw_vector, using=vector_name) for simple
    searches and the Query object for hybrid fusion.

    Args:
        client: An initialized QdrantClient instance.
        query: The user's search query string.
        collection_name: The name of the Qdrant collection to search in.
        n: The maximum number of chunks to retrieve.
        dense_model: The instantiated dense embedding model.
        sparse_model: The instantiated sparse embedding model.
        dense_model_key: The key used for dense vectors in Qdrant.
        sparse_model_key: The key used for sparse vectors in Qdrant.

    Returns:
        A list of Qdrant ScoredPoint objects, containing the results.
        Returns an empty list if the collection type is unknown or an error occurs.
    """
    print(f"\n--- Retrieving top {n} chunks for query: '{query}' ---")
    print(f"Searching in collection: {collection_name}")

    search_result: List[models.ScoredPoint] = []
    query_response: Optional[models.QueryResponse] = None

    try:
        if collection_name == DENSE_COLLECTION_NAME:
            print("Executing search using client.query_points (dense)...")
            dense_query_vector_raw = list(dense_model.embed([query]))[0]
            dense_query_vector_list: List[float] = dense_query_vector_raw.tolist()
            query_response = client.query_points(
                collection_name=collection_name,
                query=dense_query_vector_list, 
                using=dense_model_key,       
                limit=n,
                with_payload=True,
                with_vectors=False
            )

        elif collection_name == SPARSE_COLLECTION_NAME:

            print("Executing search using client.query_points (sparse)...")
            sparse_query_vector_obj = list(sparse_model.embed([query]))[0]
            sparse_query_vector_dict = sparse_query_vector_obj.as_object()
            sparse_query_vector_data = models.SparseVector(
                 indices=sparse_query_vector_dict['indices'],
                 values=sparse_query_vector_dict['values']
             )
            query_response = client.query_points(
                collection_name=collection_name,
                query=sparse_query_vector_data, 
                using=sparse_model_key,        
                limit=n,
                with_payload=True,
                with_vectors=False
            )

        elif collection_name == HYBRID_COLLECTION_NAME:
            print("Executing search using client.query_points (hybrid)...")
            dense_query_vector_raw = list(dense_model.embed([query]))[0]
            dense_query_vector_list = dense_query_vector_raw.tolist()
            
            sparse_query_vector_obj = list(sparse_model.embed([query]))[0]
            sparse_query_vector_dict = sparse_query_vector_obj.as_object()

            sparse_query_vector_data_for_prefetch = models.SparseVector(
                 indices=sparse_query_vector_dict['indices'],
                 values=sparse_query_vector_dict['values']
             )
            prefetch_list = [
                models.Prefetch(
                    query=dense_query_vector_list,  
                    using=dense_model_key,
                    limit=n * 5 
                ),
                models.Prefetch(
                    query=sparse_query_vector_data_for_prefetch, 
                    using=sparse_model_key,
                    limit=n * 5
                )
            ]

            # Define the fusion query type
            fusion_query = models.FusionQuery(fusion=models.Fusion.RRF)
            query_response = client.query_points(
                collection_name=collection_name,
                prefetch=prefetch_list,       
                query=fusion_query,          
                limit=n,                    
                with_payload=True,
                with_vectors=False
            )

        else:
            print(f"Error: Unknown collection name '{collection_name}'. Cannot determine search type.")
            return []
        if query_response:
            search_result = query_response.points
        else:
             search_result = [] 
        print(f"Found {len(search_result)} results.")
        return search_result
    except Exception as e:
        print(f"An error occurred during search in collection '{collection_name}': {e}")
        traceback.print_exc()
        return []
    
def all_vector_retrieve(search_query: str, num_results: int) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Retrieves dense, sparse, and hybrid search results using timed blocks,
    formats them, and returns packaged dictionaries containing contexts and
    timings for each type.

    Args:
        search_query: The query string.
        num_results: Number of results to retrieve for each type.

    Returns:
        A tuple containing three dictionaries:
        (dense_package, sparse_package, hybrid_package)
        Each package dictionary has keys like:
            '<type>_context': formatted context string
            '<type>_retrieval_time': float (seconds for retrieve_chunks)
    """
    print(f"\nRunning vector retrieval for query: '{search_query}'")
    measurer = TimeMeasurer()

    # --- Dense Processing ---
    dense_raw_results = None
    dense_context = ""
    with measurer.measure('dense_retrieval_time'):
        dense_raw_results = retrieve_chunks(
            client=client, query=search_query, collection_name=DENSE_COLLECTION_NAME,
            n=num_results, dense_model=dense_embedding_model, sparse_model=sparse_embedding_model,
            dense_model_key=DENSE_MODEL_KEY, sparse_model_key=SPARSE_MODEL_KEY
        )
    if dense_raw_results:
            dense_context = format_qdrant_results_to_string(dense_raw_results)

    # --- Sparse Processing ---
    sparse_raw_results = None
    sparse_context = ""
    with measurer.measure('sparse_retrieval_time'):
        sparse_raw_results = retrieve_chunks(
            client=client, query=search_query, collection_name=SPARSE_COLLECTION_NAME,
            n=num_results, dense_model=dense_embedding_model, sparse_model=sparse_embedding_model,
            dense_model_key=DENSE_MODEL_KEY, sparse_model_key=SPARSE_MODEL_KEY
        )
    if sparse_raw_results:
            sparse_context = format_qdrant_results_to_string(sparse_raw_results)

    # --- Hybrid Processing ---
    hybrid_raw_results = None
    hybrid_context = ""
    with measurer.measure('hybrid_retrieval_time'):
        hybrid_raw_results = retrieve_chunks(
            client=client, query=search_query, collection_name=HYBRID_COLLECTION_NAME,
            n=num_results, dense_model=dense_embedding_model, sparse_model=sparse_embedding_model,
            dense_model_key=DENSE_MODEL_KEY, sparse_model_key=SPARSE_MODEL_KEY
        )
    if hybrid_raw_results:
            hybrid_context = format_qdrant_results_to_string(hybrid_raw_results)

    dense_package = {
        "context": dense_context,
        "retrieval_time_taken": measurer.get_timing('dense_retrieval_time'),
    }
    sparse_package = {
        "context": sparse_context,
        "retrieval_time_taken": measurer.get_timing('sparse_retrieval_time'),
    }
    hybrid_package = {
        "context": hybrid_context,
        "retrieval_time_taken": measurer.get_timing('hybrid_retrieval_time'),
    }
    return dense_package, sparse_package, hybrid_package
    
    
