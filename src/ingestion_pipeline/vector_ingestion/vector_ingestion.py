import os
import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from qdrant_client import QdrantClient, models
import re
from datasets import Dataset
import tqdm
from fastembed import TextEmbedding
from fastembed import SparseTextEmbedding
from dotenv import load_dotenv
from typing import Any
import logging
from src.ingestion_pipeline import helper_functions 


load_dotenv(dotenv_path=os.path.join(os.getcwd(), '.env'), override=True)

# --- Configuration ---
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_TIMEOUT = int(os.getenv("QDRANT_TIMEOUT"))
BATCH_SIZE = 4 
QDRANT_PORT= int(os.getenv("QDRANT_PORT"))

COLLECTION_BASE_NAME = os.getenv("COLLECTION_BASE_NAME")

# Collection Names
DENSE_COLLECTION_NAME = f"{COLLECTION_BASE_NAME}_dense"
SPARSE_COLLECTION_NAME = f"{COLLECTION_BASE_NAME}_sparse"
HYBRID_COLLECTION_NAME = f"{COLLECTION_BASE_NAME}_hybrid"

# Embedding Model Names (Keys for Qdrant)
DENSE_MODEL_KEY = os.getenv("DENSE_MODEL_KEY")
SPARSE_MODEL_KEY = os.getenv("SPARSE_MODEL_KEY")

def parse_response_to_json(response_text):
        # Look for JSON pattern
    json_pattern = r'```json\s*([\s\S]*?)\s*```'
    json_match = re.search(json_pattern, response_text)
        
    if json_match:
            # Extract JSON from code block
       json_str = json_match.group(1).strip()
       title = json.loads(json_str)
       title = title['title']
       return title
   
def generate_title(text, model, temperature):
    prompt_template = """
    You are an expert at information extraction you are provided with a chunk of text 
    Your Job is to look at the text understand its general meaning an generate a title for that text
    Return ONLY a JSON object with the following structure:
    ```json
    {{
      "title":""
    }}
    ```

    Text to analyze:
    {input_text}
    """
     # Set up the chain
    prompt = ChatPromptTemplate.from_template(prompt_template)
    llm = helper_functions.gemini_llm(model, temperature)
    chain = prompt | llm | StrOutputParser()
            
            # Invoke the chain
    response = chain.invoke({"input_text": text})
            
            
    # Parse the response
    extraction_result = parse_response_to_json(response)
    return extraction_result
            
            
def create_dataset(texts, model, temperature):
    data = {
        "_id": [],
        "title": [],
        "text": []
    }
    
     # Process each chunk
    for i, chunk_text in enumerate(texts):
        print(f"Processing chunk {i+1}/{len(texts)}")
        chunk_id = i+1
        
        title = generate_title(chunk_text, model, temperature)
        
        data["_id"].append(chunk_id)
        data["title"].append(title)
        data["text"].append(chunk_text)
    
    return Dataset.from_dict(data)

   
def create_collections():    
    # --- Instantiate Embedding Models ---
    try:
        dense_embedding_model = TextEmbedding(model_name=DENSE_MODEL_KEY)
        print("Dense model loaded.")
        sparse_embedding_model = SparseTextEmbedding(model_name=SPARSE_MODEL_KEY) 
        print("Sparse model loaded.")
    except Exception as e:
        print(f"Error loading embedding models: {e}")
        exit() 

    # --- Get Dense Embedding Size (Needed for config) ---
    print("Calculating dense embedding size...")
    try:
        sample_embedding = list(dense_embedding_model.embed(["sample text"]))
        DENSE_VECTOR_SIZE = len(sample_embedding[0])
        print(f"Detected Dense Vector Size: {DENSE_VECTOR_SIZE}")
    except Exception as e:
        print(f"Error determining dense vector size: {e}")
        exit()

    # --- Qdrant Client ---
    client = QdrantClient(QDRANT_URL, timeout=QDRANT_TIMEOUT)
    
    try:
        collections_response = client.get_collections()
        existing_collections = collections_response.collections

        if not existing_collections:
            print("No existing collections found to delete.")
        else:
            print(f"Found {len(existing_collections)} collections. Attempting deletion...")
            for collection_desc in existing_collections:
                collection_name = collection_desc.name
                print(f"Deleting collection: '{collection_name}'...")
                try:
                    client.delete_collection(collection_name=collection_name, timeout=QDRANT_TIMEOUT)
                    print(f"Collection '{collection_name}' deleted successfully.")
                except Exception as delete_error:
                    print(f"Error deleting collection '{collection_name}': {delete_error}")
            print("Finished attempting to delete existing collections.")

    except Exception as list_error:
        print(f"Could not retrieve collection list to perform deletions: {list_error}")

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

    print("\n--- Creating/Recreating Collections ---")
    # Dense Collection
    try:
        print(f"Creating Dense collection: {DENSE_COLLECTION_NAME}")
        client.create_collection(
            collection_name=DENSE_COLLECTION_NAME,
            vectors_config=dense_vectors_config,
        )
        print(f"Collection '{DENSE_COLLECTION_NAME}' created.")
    except Exception as e:
        print(f"Error creating dense collection: {e}")

    # Sparse Collection
    try:
        print(f"Creating Sparse collection: {SPARSE_COLLECTION_NAME}")
        client.create_collection(
            collection_name=SPARSE_COLLECTION_NAME,
            vectors_config=None, # No dense vectors needed
            sparse_vectors_config=sparse_vectors_config
        )
        print(f"Collection '{SPARSE_COLLECTION_NAME}' created.")
    except Exception as e:
        print(f"Error creating sparse collection: {e}")

    # Hybrid Collection
    try:
        print(f"Creating Hybrid collection: {HYBRID_COLLECTION_NAME}")
        client.create_collection(
            collection_name=HYBRID_COLLECTION_NAME,
            vectors_config=hybrid_vectors_config,
            sparse_vectors_config=hybrid_sparse_vectors_config
        )
        print(f"Collection '{HYBRID_COLLECTION_NAME}' created.")
    except Exception as e:
        print(f"Error creating hybrid collection: {e}")

    return dense_embedding_model, sparse_embedding_model, client
    
# --- Function for Data Ingestion ---
def ingest_data(dataset: Any, dense_embedding_model, sparse_embedding_model, client):
    print("\n--- Starting Data Ingestion ---")
    total_items = None
    try:
      if hasattr(dataset, '__len__'): total_items = len(dataset)
      elif hasattr(dataset, 'num_rows'): total_items = dataset.num_rows
    except TypeError: print("Could not determine dataset length.")

    progress_bar = tqdm.tqdm(
        dataset.iter(batch_size=BATCH_SIZE),
        total= (total_items // BATCH_SIZE) if total_items else None,
        desc="Ingesting Batches"
    )

    for batch in progress_bar:
        # Get text data for the batch
        texts_to_embed = batch["text"]
        if not isinstance(texts_to_embed, list): 
             texts_to_embed = list(texts_to_embed)

        # Calculate both types of embeddings
        try:
            dense_embeddings_batch = list(dense_embedding_model.embed(texts_to_embed))
            sparse_embeddings_batch = list(sparse_embedding_model.embed(texts_to_embed))
        except Exception as e:
            print(f"\nError during embedding calculation for a batch: {e}")
            continue 

        # Prepare points for each collection
        dense_points = []
        sparse_points = []
        hybrid_points = []

        for i, _ in enumerate(batch["_id"]):
            doc_id = int(batch["_id"][i])
            payload_content = {
                "_id": batch["_id"][i], 
                "title": batch["title"][i],
                "text": texts_to_embed[i], 
            }

            # Point for Dense Collection
            dense_points.append(models.PointStruct(
                id=doc_id,
                vector={DENSE_MODEL_KEY: dense_embeddings_batch[i].tolist()},
                payload=payload_content
            ))

            # Point for Sparse Collection
            sparse_points.append(models.PointStruct(
                id=doc_id,
                # Sparse vectors go in the main 'vector' dict, keyed by the name from config
                vector={SPARSE_MODEL_KEY: sparse_embeddings_batch[i].as_object()},
                payload=payload_content
            ))

            # Point for Hybrid Collection
            hybrid_points.append(models.PointStruct(
                id=doc_id,
                vector={
                    DENSE_MODEL_KEY: dense_embeddings_batch[i].tolist(),
                    SPARSE_MODEL_KEY: sparse_embeddings_batch[i].as_object()
                },
                payload=payload_content
            ))

        try:
            if dense_points:
                client.upload_points(DENSE_COLLECTION_NAME, points=dense_points, wait=False)
            if sparse_points:
                client.upload_points(SPARSE_COLLECTION_NAME, points=sparse_points, wait=False)
            if hybrid_points:
                client.upload_points(HYBRID_COLLECTION_NAME, points=hybrid_points, wait=False)
        except Exception as e:
            print(f"\nError uploading batch to Qdrant: {e}")

    print("--- Ingestion Finished ---")    
    
    
def run_vector_ingestion(dataset_path, model="gemini-2.0-flash-lite", temperature = 0.7):
    
    texts = helper_functions.create_texts_data(dataset_path)

    dataset = create_dataset(texts, model, temperature)   

    dense_embedding_model, sparse_embedding_model, client = create_collections()
    ingest_data(dataset= dataset, dense_embedding_model= dense_embedding_model, sparse_embedding_model=sparse_embedding_model , client = client) 
    logging.info("***** ingest done successfully *****")
   

   
    