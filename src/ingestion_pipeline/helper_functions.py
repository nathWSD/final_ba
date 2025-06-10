from langchain_text_splitters import TokenTextSplitter
from sentence_transformers import SentenceTransformer
import pandas as pd
import time
import numpy as np
from datetime import datetime
import networkx as nx
import uuid
import igraph as ig
import leidenalg as la
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import logging
from contextlib import contextmanager
from typing import Dict
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from neo4j import GraphDatabase, exceptions as neo4j_exceptions
from rapidfuzz import fuzz
import re
import string
from typing import List, Dict, Any, Tuple, Set
import matplotlib.cm as cm 
from sklearn.cluster import AgglomerativeClustering

load_dotenv(dotenv_path=os.path.join(os.getcwd(), '.env'), override=True)

CHUNK_SIZE= int(os.getenv('CHUNK_SIZE'))
OVERLAP= int(os.getenv('OVERLAP'))
NEO4J_URI=os.getenv('NEO4J_URI')  
NEO4J_USERNAME=os.getenv('NEO4J_USERNAME')
NEO4J_PASSWORD=os.getenv('NEO4J_PASSWORD')
NEO4J_DATABASE=os.getenv('NEO4J_DB') 
DEFAULT_EDGE_WIDTH = 1.5
DEFAULT_ARROW_SIZE = 15
DEFAULT_EDGE_FONT_SIZE = 7
DEFAULT_EDGE_FONT_COLOR = 'darkred'
DEFAULT_EDGE_COLOR = 'gray'
DEFAULT_NODE_ALPHA = 0.9
DEFAULT_EDGE_ALPHA = 0.6
DEFAULT_NODE_FONT_SIZE = 8
DEFAULT_NODE_TYPE = 'Unknown'
DEFAULT_LAYOUT_TYPE = 'spring'
SEED=42
SIMILARITY_THRESHOLD = 85 

GENERIC_TYPES_FOR_CANONICAL = {'MISC', 'REBEL_ENTITY', None}
        
def create_texts_data(dataset_path):

    with open(dataset_path, 'r', encoding='utf-8') as file: 
        content = file.read()

    text_splitter = TokenTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=OVERLAP)

    texts = text_splitter.split_text(content)
    
    return texts

def gemini_llm(model_name, temperature):
    llm = ChatGoogleGenerativeAI(
            model=model_name, 
            temperature=temperature,
            google_api_key=os.getenv('GOOGLE_API_KEY')
        )
    return llm


# --- create_neo4j_driver (Connects to Neo4j running in Compose) ---
def create_neo4j_driver(max_retries=5, delay=5): 
    
    driver = None
    retries = 0
    logging.info(f"Attempting to connect to Neo4j at {NEO4J_URI}...")
    while retries < max_retries:
        try:
            driver_attempt = GraphDatabase.driver(
                NEO4J_URI,
                auth=(NEO4J_USERNAME, NEO4J_PASSWORD),
                # Increase connection timeout as container might be starting
                connection_timeout=30.0
            )
            driver_attempt.verify_connectivity()

            with driver_attempt.session(database=NEO4J_DATABASE) as session:
                 result = session.run("RETURN 1 as test")
                 if result.single()["test"] == 1:
                     logging.info(f"Successfully connected to Neo4j database '{NEO4J_DATABASE}' via {NEO4J_URI} after {retries} retries.")
                     driver = driver_attempt
                     break # Success

            driver_attempt.close()
            raise ConnectionError("Neo4j connection verified but test query failed.")

        except neo4j_exceptions.AuthError as e:
            logging.error(f"Authentication error connecting to Neo4j ({NEO4J_URI}): {e}")
            logging.error("Check NEO4J_USERNAME and NEO4J_PASSWORD in .env match the container setup.")
            if 'driver_attempt' in locals() and driver_attempt: driver_attempt.close()
            return None

        # Catch common connection errors when container might not be ready
        except (neo4j_exceptions.ServiceUnavailable, ConnectionRefusedError, OSError, neo4j_exceptions.DriverError) as e:
            last_exception = e
            retries += 1
            logging.warning(f"Neo4j not available yet at {NEO4J_URI}. Retrying ({retries}/{max_retries})... Error: {e}")
            if 'driver_attempt' in locals() and driver_attempt: driver_attempt.close()
            time.sleep(delay) # Wait before retrying

        except Exception as e:
            logging.error(f"An unexpected error occurred during Neo4j connection: {e}")
            last_exception = e
            if 'driver_attempt' in locals() and driver_attempt: driver_attempt.close()
            return None # Stop on other unexpected errors

    if driver is None:
        logging.error(f"Failed to connect to Neo4j ({NEO4J_URI}) after {max_retries} retries.")
        if 'last_exception' in locals():
             logging.error(f"Last error: {last_exception}")
    return driver
    
        
def format_qdrant_results_to_string(results) -> str:
    """
    Formats a list of Qdrant ScoredPoint results into a single string document.

    Each document's title and text are combined, separated by a newline.
    Each document block (title + text) is separated by two newlines.

    Args:
        results: A list of ScoredPoint objects, each expected to have a
                 payload dictionary containing 'title' and 'text' keys.

    Returns:
        A single string concatenating the formatted title and text
        from all results. Returns an empty string if the input list is empty.
    """
    if not results:
        return ""

    document_parts = []
    for point in results:
        # Ensure the point has a payload dictionary
        if isinstance(point.payload, dict):
            # Safely get title and text, providing defaults if keys are missing
            title = point.payload.get('title', 'No Title Provided')
            text = point.payload.get('text', 'No Text Provided')

            # Format: Title on one line, Text on the next
            formatted_part = f"title: {title}\n text: {text}"
            document_parts.append(formatted_part)
        else:
            print(f"Warning: ScoredPoint with id {point.id} has invalid or missing payload: {point.payload}")


    # Join each document part (title + text) with two newlines
    final_string = "\n\n".join(document_parts)
    return final_string


def add_embeddings_to_graph(driver, graph_prefix="DL"):
    """
    Add embedding fields and create vector indexes for the specified graph,
    processing nodes in batches fetched iteratively from the database.
    """

    # Apply prefix to labels
    entity_label = f"{graph_prefix}__Entity__"
    community_label = f"{graph_prefix}__Community__"
    chunk_label = f"{graph_prefix}__Chunk__"

    # Create index names with prefix
    entity_index_name = f"{graph_prefix}_entity_embeddings"
    community_index_name = f"{graph_prefix}_community_embeddings"
    chunk_index_name = f"{graph_prefix}_chunk_embeddings"

    # Load the model
    print(f"Loading {os.getenv('DENSE_MODEL_KEY')} model...")
    try:
        model = SentenceTransformer(os.getenv('DENSE_MODEL_KEY'))
        EMBEDDING_DIMENSION = model.get_sentence_embedding_dimension()
        print(f"Model loaded, embedding dimension: {EMBEDDING_DIMENSION}")
    except Exception as e:
        print(f"FATAL: Could not load embedding model: {e}")
        return False 


    batch_size = 32
    print(f"Using batch size: {batch_size}")

    # --- Process Chunks ---
    print(f"\nProcessing embeddings for {graph_prefix} text chunks...")
    processed_chunk_count = 0
    while True: # Loop until no more unembedded chunks are found
        chunk_query = f"""
        MATCH (c:{chunk_label})
        WHERE c.embedding IS NULL
        RETURN c.id AS id, c.text AS text
        LIMIT {batch_size}
        """
        try:
            result = driver.execute_query(chunk_query, database_=NEO4J_DATABASE)
            chunks_batch = [(record["id"], record.get("text", "")) for record in result.records]
        except Exception as e:
            print(f"ERROR fetching chunk batch: {e}. Stopping chunk processing.")
            break # Stop processing chunks on DB error

        if not chunks_batch:
            print("No more unembedded chunks found.")
            break # Exit the while loop for chunks

        print(f"  Processing batch of {len(chunks_batch)} chunks...")
        texts = [text if text else "Empty chunk" for _, text in chunks_batch]

        try:
            embeddings = model.encode(texts)
        except Exception as e:
            print(f"ERROR during chunk embedding generation: {e}. Skipping this batch.")
            time.sleep(2) # Avoid tight loop on persistent errors
            continue # Skip to the next iteration

        # Update chunks in Neo4j
        updates_failed = 0
        for j, (chunk_id, _) in enumerate(chunks_batch):
            embedding = embeddings[j].tolist()
            update_query = f"MATCH (c:{chunk_label} {{id: $id}}) SET c.embedding = $embedding"
            try:
                driver.execute_query(update_query, id=chunk_id, embedding=embedding, database_=NEO4J_DATABASE)
            except Exception as e:
                 print(f"  ERROR updating embedding for chunk {chunk_id}: {e}")
                 updates_failed += 1

        processed_chunk_count += len(chunks_batch) - updates_failed
        print(f"  Finished batch. Total chunks processed so far: {processed_chunk_count}")
        if updates_failed > 0:
            print(f"  ({updates_failed} updates failed in this batch)")
        # The loop automatically fetches the next batch if more exist

    # --- Process Entities ---
    print(f"\nProcessing embeddings for {graph_prefix} entities...")
    processed_entity_count = 0
    while True: # Loop until no more unembedded entities are found
        entity_query = f"""
        MATCH (e:{entity_label})
        WHERE e.embedding IS NULL
        RETURN e.id AS id, e.title AS title, e.description AS description
        LIMIT {batch_size}
        """
        try:
            result = driver.execute_query(entity_query, database_=NEO4J_DATABASE)
            entities_batch = [(record["id"], record.get("title", ""), record.get("description", "")) for record in result.records]
        except Exception as e:
            print(f"ERROR fetching entity batch: {e}. Stopping entity processing.")
            break

        if not entities_batch:
            print("No more unembedded entities found.")
            break

        print(f"  Processing batch of {len(entities_batch)} entities...")
        texts = [f"{title} {desc}".strip() for _, title, desc in entities_batch]
        texts = [text if text else "Unknown entity" for text in texts]

        try:
            embeddings = model.encode(texts)
        except Exception as e:
            print(f"ERROR during entity embedding generation: {e}. Skipping this batch.")
            time.sleep(2)
            continue

        updates_failed = 0
        for j, (entity_id, _, _) in enumerate(entities_batch):
            embedding = embeddings[j].tolist()
            update_query = f"MATCH (e:{entity_label} {{id: $id}}) SET e.embedding = $embedding"
            try:
                driver.execute_query(update_query, id=entity_id, embedding=embedding, database_=NEO4J_DATABASE)
            except Exception as e:
                print(f"  ERROR updating embedding for entity {entity_id}: {e}")
                updates_failed += 1

        processed_entity_count += len(entities_batch) - updates_failed
        print(f"  Finished batch. Total entities processed so far: {processed_entity_count}")
        if updates_failed > 0:
            print(f"  ({updates_failed} updates failed in this batch)")

    # --- Process Communities ---
    print(f"\nProcessing embeddings for {graph_prefix} communities...")
    in_community_rel = f"{graph_prefix}_IN_COMMUNITY"
    processed_community_count = 0
    while True: # Loop until no more unembedded communities are found
        comm_query = f"""
        MATCH (c:{community_label})
        WHERE c.embedding IS NULL
        OPTIONAL MATCH (c)<-[:{in_community_rel}]-(e:{entity_label})
        WITH c, collect(e.title)[0..5] AS entity_titles, collect(e.description)[0..5] AS entity_descriptions // Limit collected entity info
        RETURN c.id AS id, c.title AS title, c.summary AS summary, entity_titles, entity_descriptions
        LIMIT {batch_size}
        """
        try:
            result = driver.execute_query(comm_query, database_=NEO4J_DATABASE)
            communities_data = list(result.records)
        except Exception as e:
            print(f"ERROR fetching community batch: {e}. Stopping community processing.")
            break

        if not communities_data:
            print("No more unembedded communities found.")
            break

        print(f"  Processing batch of {len(communities_data)} communities...")
        communities_to_embed = []
        for record in communities_data:
            community_id = record["id"]
            community_title = record.get("title", "")
            community_summary = record.get("summary", "")
            entity_titles = record.get("entity_titles", []) # Already limited by query
            entity_descriptions = record.get("entity_descriptions", []) # Already limited

            content = []
            if community_title: content.append(community_title)
            if community_summary: content.append(community_summary)
            # Combine the limited entity titles/descriptions collected
            for i in range(len(entity_titles)):
                if entity_titles[i]: content.append(entity_titles[i])
                if i < len(entity_descriptions) and entity_descriptions[i]: content.append(entity_descriptions[i])

            text = " ".join(content).strip()
            if not text: text = "Empty community"
            communities_to_embed.append((community_id, text))

        # Prepare texts for embedding
        texts = [text for _, text in communities_to_embed]

        try:
            embeddings = model.encode(texts)
        except Exception as e:
            print(f"ERROR during community embedding generation: {e}. Skipping this batch.")
            time.sleep(2)
            continue

        updates_failed = 0
        for j, (comm_id, _) in enumerate(communities_to_embed):
            embedding = embeddings[j].tolist()
            update_query = f"MATCH (c:{community_label} {{id: $id}}) SET c.embedding = $embedding"
            try:
                driver.execute_query(update_query, id=comm_id, embedding=embedding, database_=NEO4J_DATABASE)
            except Exception as e:
                 print(f"  ERROR updating embedding for community {comm_id}: {e}")
                 updates_failed += 1

        processed_community_count += len(communities_to_embed) - updates_failed
        print(f"  Finished batch. Total communities processed so far: {processed_community_count}")
        if updates_failed > 0:
            print(f"  ({updates_failed} updates failed in this batch)")
        # Loop continues


    # --- Create Indexes ---
    print("\nManaging vector indexes...")
    try:
        # Drop existing indexes first (handle potential errors gracefully)
        index_names = [chunk_index_name, entity_index_name, community_index_name]
        for index_name in index_names:
            try:
                driver.execute_query(f"DROP INDEX {index_name} IF EXISTS", database_=NEO4J_DATABASE)
                print(f"Dropped {index_name} index (if it existed).")
            except Exception as e:
                # It's often okay if drop fails (e.g., index didn't exist)
                print(f"Note: Error dropping index {index_name}: {e} (Might be expected if index was not found)")

        # Create chunk vector index
        print(f"Creating vector index for {graph_prefix} text chunks...")
        chunk_index_stmt = f"""
        CREATE VECTOR INDEX {chunk_index_name} IF NOT EXISTS
        FOR (c:{chunk_label}) ON (c.embedding)
        OPTIONS {{indexConfig: {{ `vector.dimensions`: {EMBEDDING_DIMENSION}, `vector.similarity_function`: "cosine" }}}}
        """
        driver.execute_query(chunk_index_stmt, database_=NEO4J_DATABASE)
        print(f"{chunk_index_name} vector index created.")

        # Create entity vector index
        print(f"Creating vector index for {graph_prefix} entities...")
        entity_index_stmt = f"""
        CREATE VECTOR INDEX {entity_index_name} IF NOT EXISTS
        FOR (e:{entity_label}) ON (e.embedding)
        OPTIONS {{indexConfig: {{ `vector.dimensions`: {EMBEDDING_DIMENSION}, `vector.similarity_function`: "cosine" }}}}
        """
        driver.execute_query(entity_index_stmt, database_=NEO4J_DATABASE)
        print(f"{entity_index_name} vector index created.")

        # Create community vector index
        print(f"Creating vector index for {graph_prefix} communities...")
        comm_index_stmt = f"""
        CREATE VECTOR INDEX {community_index_name} IF NOT EXISTS
        FOR (c:{community_label}) ON (c.embedding)
        OPTIONS {{indexConfig: {{ `vector.dimensions`: {EMBEDDING_DIMENSION}, `vector.similarity_function`: "cosine" }}}}
        """
        driver.execute_query(comm_index_stmt, database_=NEO4J_DATABASE)
        print(f"{community_index_name} vector index created.")

        print("Vector index management complete.")

    except Exception as e:
        print(f"ERROR during vector index management: {e}")
        print("Note: Vector indexes require Neo4j Enterprise 5.0+ with vector features enabled.")

    print(f"\nFinished adding embeddings and managing indexes for {graph_prefix}")
    return True


def normalize_entity_name(name):
    """Basic normalization (can be expanded)."""
    if not isinstance(name, str): # Handle potential non-string inputs
        return ""
    name = name.strip().lower()
    name = re.sub(r"^(the|a|an|la|el|los|las)\s+", "", name)
    return name


def choose_canonical_form(cluster_mentions, mention_data):
    """Chooses the best representative for a cluster."""
    # Strategy: Prefer non-generic type, then higher frequency, then shortest length, then alphabetically
    def sort_key(mention_key):
        original_mention, mention_type = mention_key
        freq = mention_data[mention_key]['freq']
        # Check if the type associated *with this specific mention_key* is generic
        is_generic = mention_type in GENERIC_TYPES_FOR_CANONICAL
        # Prioritize non-generic (-is_generic makes False=0 become 0, True=1 become -1)
        # Then higher frequency (-freq)
        # Then shorter length (len)
        # Then alphabetically (original_mention)
        return (-is_generic, -freq, len(original_mention), original_mention)

    sorted_mentions = sorted(cluster_mentions, key=sort_key)
    # Return the original_mention string of the best one
    return sorted_mentions[0][0]


def robust_disambiguate_entities(triplets, similarity_threshold=SIMILARITY_THRESHOLD):
    """
    Performs robust entity disambiguation using fuzzy matching and clustering,
    without considering type compatibility for merging.

    Args:
        triplets (list): List of triplet dictionaries.
        similarity_threshold (int): The fuzz ratio threshold (0-100) for clustering.

    Returns:
        list: Disambiguated triplets.
    """
    unique_entities = defaultdict(lambda: {'types': set(), 'freq': 0})

    # 1. Extract unique entities, their types, and frequencies
    # We still track types to help choose a better canonical form later
    for triplet in triplets:
        head = triplet.get('head')
        head_type = triplet.get('head_type')
        tail = triplet.get('tail')
        tail_type = triplet.get('tail_type')

        if head:
            entity_key = (head, head_type)
            unique_entities[entity_key]['types'].add(head_type)
            unique_entities[entity_key]['freq'] += 1

        if tail:
            entity_key = (tail, tail_type)
            unique_entities[entity_key]['types'].add(tail_type)
            unique_entities[entity_key]['freq'] += 1

    if not unique_entities:
        return []

    entity_list = list(unique_entities.keys()) # List of (original_mention, type) tuples
    n_entities = len(entity_list)

    if n_entities <= 1:
        # Handle edge case of 0 or 1 unique entity-type pairs
        canonical_map = {key: key[0] for key in entity_list} # Map (mention, type) -> mention
        disambiguated_triplets = []
        for triplet in triplets:
            new_triplet = triplet.copy()
            head_key = (new_triplet.get('head'), new_triplet.get('head_type'))
            tail_key = (new_triplet.get('tail'), new_triplet.get('tail_type'))
            if new_triplet.get('head') and head_key in canonical_map:
                 new_triplet['head'] = canonical_map[head_key]
            if new_triplet.get('tail') and tail_key in canonical_map:
                 new_triplet['tail'] = canonical_map[tail_key]
            disambiguated_triplets.append(new_triplet)
        return disambiguated_triplets


    # 2. Calculate pairwise distances based *only* on string similarity
    distance_matrix = np.zeros((n_entities, n_entities))

    print(f"Calculating distances for {n_entities} unique (entity, type) pairs...")
    for i in range(n_entities):
        for j in range(i + 1, n_entities):
            mention1, type1 = entity_list[i] # Type is ignored for distance
            mention2, type2 = entity_list[j] # Type is ignored for distance

            norm_mention1 = normalize_entity_name(mention1)
            norm_mention2 = normalize_entity_name(mention2)

            # Use WRatio for a robust similarity score
            similarity = fuzz.WRatio(norm_mention1, norm_mention2)

            # Convert similarity (0-100) to distance (0-100)
            distance = 100 - similarity

            # Store distance symmetrically
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance
    print("Distance calculation complete.")

    # 3. Cluster entities based on distance
    clustering_distance_threshold = 100 - similarity_threshold

    print(f"Clustering with distance threshold: {clustering_distance_threshold}...")
    agg_cluster = AgglomerativeClustering(
        n_clusters=None,              # Use distance_threshold instead
        metric='precomputed',         # CORRECTED: Use 'metric' instead of 'affinity'
        linkage='average',
        distance_threshold=clustering_distance_threshold,
        compute_full_tree=True      # ADDED: Required when distance_threshold is not None
    )
    cluster_labels = agg_cluster.fit_predict(distance_matrix)
    print(f"Found {agg_cluster.n_clusters_} clusters.")

    # 4. Determine Canonical Forms for each cluster
    clusters = defaultdict(list)
    for i, label in enumerate(cluster_labels):
        clusters[label].append(entity_list[i]) # Add (original_mention, type)

    canonical_map = {} # Map: (original_mention, type) -> canonical_mention_string
    # Keep cluster details for potential debugging/inspection if needed internally
    # cluster_details = {}

    print("Determining canonical forms...")
    for label, members in clusters.items():
        mention_data_for_cluster = {
            member: unique_entities[member] for member in members
        }
        canonical_mention = choose_canonical_form(members, mention_data_for_cluster)

        # cluster_details[label] = {'members': members, 'canonical': canonical_mention} # Optional

        # Map all members ((mention, type) pairs) of this cluster to the canonical string
        for original_mention, m_type in members:
             canonical_map[(original_mention, m_type)] = canonical_mention
    print("Canonical forms determined.")

 # 5. Apply Mapping AND Filter Self-Loops
    filtered_triplets = [] # Changed variable name for clarity
    self_loops_removed = 0
    print("Applying mapping and filtering self-loops...")
    for triplet in triplets:
        new_triplet = triplet.copy()

        original_head = new_triplet.get('head')
        original_head_type = new_triplet.get('head_type')
        original_tail = new_triplet.get('tail')
        original_tail_type = new_triplet.get('tail_type')

        head_key = (original_head, original_head_type)
        tail_key = (original_tail, original_tail_type)

        final_head = new_triplet.get('head') # Default to original if not mapped
        final_tail = new_triplet.get('tail') # Default to original if not mapped

        # Apply mapping
        if original_head and head_key in canonical_map:
            final_head = canonical_map[head_key]
            new_triplet['head'] = final_head # Update the triplet

        if original_tail and tail_key in canonical_map:
            final_tail = canonical_map[tail_key]
            new_triplet['tail'] = final_tail # Update the triplet

        # --- Filter self-loops ---
        # Check if both head and tail are valid strings and are identical
        if isinstance(final_head, str) and isinstance(final_tail, str) and final_head == final_tail:
            self_loops_removed += 1
            continue # Skip adding this triplet

        # If not a self-loop, add it to the results
        filtered_triplets.append(new_triplet)

    print(f"Mapping complete. Removed {self_loops_removed} self-referential triplets.")

    # Corrected return variable name
    return filtered_triplets


def import_complete_graph_data(driver, entities_df, relationships_df, text_units_df, documents_df, 
                              communities_df=None, community_reports_df=None, add_embeddings=True, 
                              graph_prefix="DL"):
    """
    Import all columns from all dataframes into Neo4j with LLM/DL prefix and optionally add embeddings
    
    Args:
        driver: neo4j driver
        entities_df: DataFrame containing entity information
        relationships_df: DataFrame containing relationship information
        text_units_df: DataFrame containing text chunks
        documents_df: DataFrame containing documents
        communities_df: DataFrame containing communities (optional)
        community_reports_df: DataFrame containing community reports (optional)
        add_embeddings: Whether to add vector embeddings to entities and communities
        graph_prefix: Prefix to add to all node labels and relationships (default: "LLM")
    """
    import json
    import numpy as np
    
    print(f"Importing data with graph prefix: {graph_prefix}")
    print("Validating dataframes before import...")
    
    # Validate all dataframes for minimum required columns
    entity_issues = validate_dataframe_for_import(entities_df, ['id', 'title', 'type'])
    if entity_issues:
        print(f"Issues with entities dataframe: {entity_issues}")
    
    rel_issues = validate_dataframe_for_import(relationships_df, ['id', 'source', 'target'])
    if rel_issues:
        print(f"Issues with relationships dataframe: {rel_issues}")
    
    text_issues = validate_dataframe_for_import(text_units_df, ['id', 'text', 'document_ids'])
    if text_issues:
        print(f"Issues with text_units dataframe: {text_issues}")
    
    doc_issues = validate_dataframe_for_import(documents_df, ['id', 'title'])
    if doc_issues:
        print(f"Issues with documents dataframe: {doc_issues}")
    
    if communities_df is not None:
        comm_issues = validate_dataframe_for_import(communities_df, ['id', 'community', 'level'])
        if comm_issues:
            print(f"Issues with communities dataframe: {comm_issues}")
            
    if community_reports_df is not None:
        report_issues = validate_dataframe_for_import(community_reports_df, ['id'])
        if report_issues:
            print(f"Issues with community_reports dataframe: {report_issues}")
    
    # Merge ALL columns from community reports into communities if needed
    if communities_df is not None and community_reports_df is not None:
        print("Merging ALL columns from community reports into communities...")
        
        # Get all columns from community_reports_df except 'id' (which we use for matching)
        columns_to_transfer = [col for col in community_reports_df.columns if col != 'id']
        print(f"Transferring {len(columns_to_transfer)} columns from reports: {columns_to_transfer}")
        
        # Transfer each column
        for column in columns_to_transfer:
            # Skip columns that already exist in communities_df to avoid overwriting
            if column in communities_df.columns:
                print(f"Column '{column}' already exists in communities_df - will not overwrite")
                continue
                
            column_map = dict(zip(community_reports_df['id'], community_reports_df[column]))
            communities_df[column] = communities_df['id'].map(column_map)
            
            # Handle missing values based on column type
            if pd.api.types.is_string_dtype(community_reports_df[column]):
                communities_df[column] = communities_df[column].fillna("")
            elif pd.api.types.is_numeric_dtype(community_reports_df[column]):
                communities_df[column] = communities_df[column].fillna(0)
            elif pd.api.types.is_object_dtype(community_reports_df[column]):
                # For object types, check a sample value to decide how to fill NA
                sample_val = next((v for v in community_reports_df[column].dropna() if v is not None), None)
                if isinstance(sample_val, list):
                    communities_df[column] = communities_df[column].fillna([])
                elif isinstance(sample_val, dict):
                    communities_df[column] = communities_df[column].fillna({})
                else:
                    communities_df[column] = communities_df[column].fillna("")
                
            print(f"Added {column} to {communities_df[column].notnull().sum()} communities")
    
    # CRITICAL FIX: Convert complex objects to JSON strings BEFORE importing to Neo4j
    print("\nPreparing complex data types for Neo4j...")
    
    # First make a deep copy of communities_df to avoid modifying the original
    processed_communities_df = communities_df.copy() if communities_df is not None else None
    
    if processed_communities_df is not None:
        # Process each row to convert complex structures to JSON strings
        def convert_complex_objects(row):
            for key, value in row.items():
                # Handle numpy arrays
                if isinstance(value, np.ndarray):
                    # Convert numpy array to list first
                    list_value = value.tolist()
                    # If list contains complex structures (dicts), convert to JSON string
                    if any(isinstance(item, dict) for item in list_value):
                        row[key] = json.dumps(list_value)
                # Handle dictionaries directly
                elif isinstance(value, dict):
                    row[key] = json.dumps(value)
                # Handle lists that might contain dictionaries
                elif isinstance(value, list) and any(isinstance(item, dict) for item in value):
                    row[key] = json.dumps(value)
            return row
        
        # Apply the conversion to each row
        processed_communities_df = processed_communities_df.apply(convert_complex_objects, axis=1)
        
        # Check if findings column exists and force convert it to string
        if 'findings' in processed_communities_df.columns:
            print("Converting 'findings' column to JSON strings...")
            processed_communities_df['findings'] = processed_communities_df['findings'].apply(
                lambda x: json.dumps(x) if not isinstance(x, str) else x
            )
    
    print(f"Creating constraints with prefix {graph_prefix}...")
    if not create_constraints(driver, graph_prefix=graph_prefix):
        print("Warning: Some constraints failed to create. Continuing with import...")
    
    # Import documents - ALL COLUMNS
    print("\nImporting documents (all columns)...")
    doc_statement = """
    MERGE (d:__Document__ {id: value.id})
    SET d = value
    """
    doc_count = batched_import(driver, doc_statement, documents_df, graph_prefix=graph_prefix)
    
    # Import text units (chunks) - ALL COLUMNS
    print("\nImporting text units (all columns)...")
    text_statement = """
    MERGE (c:__Chunk__ {id: value.id})
    SET c = value
    WITH c, value
    UNWIND value.document_ids AS document_id
    MATCH (d:__Document__ {id: document_id})
    MERGE (c)-[:PART_OF]->(d)
    """
    text_count = batched_import(driver, text_statement, text_units_df, graph_prefix=graph_prefix)
    
    # Import entities - ALL COLUMNS
    print("\nImporting entities (all columns)...")
    try:
        # APOC version with prefixed type labels
        entity_statement = f"""
        MERGE (e:__Entity__ {{id: value.id}})
        SET e = value
        SET e.name = value.title  // Ensure name property exists for relationships
        WITH e, value
        CALL apoc.create.addLabels(e, 
            CASE WHEN coalesce(value.type,"") = "" THEN [] 
            ELSE ["{graph_prefix}_" + apoc.text.upperCamelCase(replace(value.type,'"',''))] END) 
        YIELD node
        RETURN count(*)
        """
        
        entity_count = batched_import(driver, entity_statement, entities_df, graph_prefix=graph_prefix)
    except Exception as e:
        print(f"Error with entity import using APOC: {e}")
        # Fallback without APOC
        entity_statement = """
        MERGE (e:__Entity__ {id: value.id})
        SET e = value
        SET e.name = value.title  // Ensure name property exists for relationships
        """
        entity_count = batched_import(driver, entity_statement, entities_df, graph_prefix=graph_prefix)
    
    # Connect entities to their text chunks
    print("\nConnecting entities to chunks...")
    connect_statement = """
    MATCH (e:__Entity__ {id: value.id})
    WITH e, value
    UNWIND value.text_unit_ids AS text_unit_id
    MATCH (c:__Chunk__ {id: text_unit_id})
    MERGE (c)-[:HAS_ENTITY]->(e)
    """
    connect_count = batched_import(driver, connect_statement, entities_df, graph_prefix=graph_prefix)
    
    # Import relationships - ALL COLUMNS
    print("\nImporting relationships (all columns)...")
    rel_statement = """
    MATCH (source:__Entity__ {name: value.source})
    MATCH (target:__Entity__ {name: value.target})
    MERGE (source)-[rel:RELATED {id: value.id}]->(target)
    SET rel = value
    """
    rel_count = batched_import(driver, rel_statement, relationships_df, graph_prefix=graph_prefix)
    
    # Import communities if provided - ALL COLUMNS
    if processed_communities_df is not None and not processed_communities_df.empty:
        print("\nImporting communities (all columns)...")
        comm_statement = """
        MERGE (c:__Community__ {id: value.id})
        SET c = value
        """
        comm_count = batched_import(driver, comm_statement, processed_communities_df, graph_prefix=graph_prefix)
        
        # Connect entities to communities
        print("\nConnecting entities to communities...")
        comm_connect_statement = """
        MATCH (c:__Community__ {id: value.id})
        WITH c, value
        UNWIND value.entity_ids AS entity_id
        MATCH (e:__Entity__ {id: entity_id})
        MERGE (e)-[:IN_COMMUNITY]->(c)
        """
        comm_connect_count = batched_import(driver, comm_connect_statement, processed_communities_df, graph_prefix=graph_prefix)
    
    print(f"\n{graph_prefix} graph import completed!")
    
    # Optionally add embeddings for vector search
    if add_embeddings:
        print(f"Adding embeddings to {graph_prefix} entities and communities...")
        add_embeddings_to_graph(driver=driver, graph_prefix=graph_prefix)
    
    return True


# Function to clear a specific graph
def clear_graph(driver, graph_prefix):
    """Delete all nodes and relationships with the specified prefix"""
    
    print(f"Clearing graph with prefix {graph_prefix}...")
    
    # Delete relationships first
    rel_types = [
        f"{graph_prefix}_PART_OF",
        f"{graph_prefix}_HAS_ENTITY",
        f"{graph_prefix}_RELATED",
        f"{graph_prefix}_IN_COMMUNITY"
    ]
    
    # Delete each relationship type
    for rel_type in rel_types:
        try:
            query = f"MATCH ()-[r:{rel_type}]->() DELETE r"
            result = driver.execute_query(query, database_=NEO4J_DATABASE)
            print(f"Deleted {result.summary.counters.relationships_deleted} {rel_type} relationships")
        except Exception as e:
            print(f"Error deleting relationships {rel_type}: {e}")
    
    # Delete nodes
    node_labels = [
        f"{graph_prefix}__Chunk__",
        f"{graph_prefix}__Document__",
        f"{graph_prefix}__Entity__",
        f"{graph_prefix}__Community__",
        f"{graph_prefix}__Covariate__"
    ]
    
    # Delete each node type
    for label in node_labels:
        try:
            query = f"MATCH (n:{label}) DELETE n"
            result = driver.execute_query(query, database_=NEO4J_DATABASE)
            print(f"Deleted {result.summary.counters.nodes_deleted} {label} nodes")
        except Exception as e:
            print(f"Error deleting nodes {label}: {e}")
            
    print(f"Graph with prefix {graph_prefix} cleared")
    return True


# Function to list available graphs
def list_graph_prefixes(driver):
    """List all graph prefixes in the database by checking node labels"""
    
    try:
        query = """
        CALL db.labels() YIELD label
        WITH label
        WHERE label CONTAINS '__'
        WITH split(label, '__')[0] AS prefix
        RETURN DISTINCT prefix
        """
        
        result = driver.execute_query(query, database_=NEO4J_DATABASE)
        prefixes = [record["prefix"] for record in result.records]
        
        print(f"Available graph prefixes: {prefixes}")
        return prefixes
    except Exception as e:
        print(f"Error listing graph prefixes: {e}")
        return []
    
    
def batched_import(driver, statement, df, batch_size=1000, graph_prefix=None):
    """
    Import a dataframe into Neo4j using a batched approach with better error handling.
    
    Args:
        statement: The Cypher statement to execute
        df: DataFrame to import
        batch_size: Number of rows to process in each batch
        graph_prefix: Optional prefix for node labels and relationship types
    """
    if df.empty:
        print("Warning: Empty dataframe, nothing to import")
        return 0
    
    # Apply prefix to statement if provided
    if graph_prefix:
        # Replace node labels
        statement = statement.replace(":__Chunk__", f":{graph_prefix}__Chunk__")
        statement = statement.replace(":__Document__", f":{graph_prefix}__Document__")
        statement = statement.replace(":__Community__", f":{graph_prefix}__Community__")
        statement = statement.replace(":__Entity__", f":{graph_prefix}__Entity__")
        statement = statement.replace(":__Covariate__", f":{graph_prefix}__Covariate__")
        
        # Replace relationship types
        statement = statement.replace(":PART_OF", f":{graph_prefix}_PART_OF")
        statement = statement.replace(":HAS_ENTITY", f":{graph_prefix}_HAS_ENTITY")
        statement = statement.replace(":RELATED", f":{graph_prefix}_RELATED")
        statement = statement.replace(":IN_COMMUNITY", f":{graph_prefix}_IN_COMMUNITY")
    
    total = len(df)
    start_s = time.time()
    imported = 0
    
    try:
        for start in range(0, total, batch_size):
            batch = df.iloc[start: min(start+batch_size, total)]
            
            try:
                # Convert dataframe to records and handle NaN values
                records = batch.replace({np.nan: None}).to_dict('records')
                
                # Execute query
                result = driver.execute_query(
                    "UNWIND $rows AS value " + statement, 
                    rows=records,
                    database_=NEO4J_DATABASE
                )
                
                # Print counters
                print(f"Batch {start//batch_size + 1}: {result.summary.counters}")
                imported += len(batch)
                
            except Exception as e:
                print(f"Error in batch starting at index {start}: {e}")
                
                # Try one by one for this batch to identify problematic records
                for i, row in batch.iterrows():
                    try:
                        single_result = driver.execute_query(
                            "UNWIND $rows AS value " + statement, 
                            rows=[row.replace({np.nan: None}).to_dict()],
                            database_=NEO4J_DATABASE
                        )
                        imported += 1
                    except Exception as inner_e:
                        print(f"  Error with row {i}: {inner_e}")
                        print(f"  Problematic data: {row.to_dict()}")
        
        print(f"Imported {imported}/{total} rows in {time.time() - start_s:.2f} seconds")
        return imported
        
    except Exception as e:
        print(f"Fatal error in import process: {e}")
        return imported
    


def create_constraints(driver, graph_prefix="DL"):
    """
    Create constraints with proper error handling and prefixed labels
    
    Args:
        graph_prefix: Prefix for node labels and relationship types (default: LLM)
    """
    # Apply prefix to labels
    chunk_label = f"{graph_prefix}__Chunk__"
    doc_label = f"{graph_prefix}__Document__"
    community_label = f"{graph_prefix}__Community__" 
    entity_label = f"{graph_prefix}__Entity__" 
    covariate_label = f"{graph_prefix}__Covariate__"
    related_rel = f"{graph_prefix}_RELATED"
    
    constraints = [
        f"CREATE CONSTRAINT {graph_prefix}_chunk_id IF NOT EXISTS FOR (c:{chunk_label}) REQUIRE c.id IS UNIQUE",
        f"CREATE CONSTRAINT {graph_prefix}_document_id IF NOT EXISTS FOR (d:{doc_label}) REQUIRE d.id IS UNIQUE",
        f"CREATE CONSTRAINT {graph_prefix}_community_id IF NOT EXISTS FOR (c:{community_label}) REQUIRE c.id IS UNIQUE",  
        f"CREATE CONSTRAINT {graph_prefix}_entity_id IF NOT EXISTS FOR (e:{entity_label}) REQUIRE e.id IS UNIQUE",
        f"CREATE CONSTRAINT {graph_prefix}_entity_name IF NOT EXISTS FOR (e:{entity_label}) REQUIRE e.name IS UNIQUE",
        f"CREATE CONSTRAINT {graph_prefix}_covariate_title IF NOT EXISTS FOR (e:{covariate_label}) REQUIRE e.title IS UNIQUE",
        f"CREATE CONSTRAINT {graph_prefix}_related_id IF NOT EXISTS FOR ()-[rel:{related_rel}]->() REQUIRE rel.id IS UNIQUE"
    ]
    
    success_count = 0
    for statement in constraints:
        try:
            print(f"Executing: {statement}")
            result = driver.execute_query(statement, database_=NEO4J_DATABASE)
            print(f"Success: {result.summary}")
            success_count += 1
        except Exception as e:
            print(f"Error executing constraint: {e}")
    
    print(f"Successfully created {success_count}/{len(constraints)} constraints")
    return success_count == len(constraints)


def detect_communities(entities_df, relationships_df, output_dir, min_community_size=3, visualize=True, graph_prefix='DL'):
    """Detect communities using Leiden algorithm with three hierarchy levels and visualization support"""
    print("Building directed network graph...")
    # Create a NetworkX graph
    G = nx.DiGraph()
    
    # Create a mapping from entity ID to title for visualization labels
    entity_id_to_title = {}
    
    # Add nodes (entities) with titles and types as attributes
    for _, entity in entities_df.iterrows():
        G.add_node(entity['id'], title=entity['title'], type=entity['type'])
        entity_id_to_title[entity['id']] = entity['title']
    
    # Add edges (relationships) with weights and descriptions
    for _, rel in relationships_df.iterrows():
        # Find entity IDs from titles
        source_entities = entities_df[entities_df['title'] == rel['source']]
        target_entities = entities_df[entities_df['title'] == rel['target']]
        
        if len(source_entities) > 0 and len(target_entities) > 0:
            source_id = source_entities['id'].iloc[0]
            target_id = target_entities['id'].iloc[0]
            G.add_edge(source_id, target_id, 
                      weight=rel['weight'], 
                      id=rel['id'], 
                      description=rel['description'],
                      type=rel['type'])
    
    # Save the original graph for visualization
    original_graph = G.copy()
    
    print("Converting to igraph...")
    # Convert to igraph for Leiden algorithm
    g_ig = ig.Graph.from_networkx(G)
    
    # Map back to original IDs
    id_mapping = {i: g_ig.vs[i]['_nx_name'] for i in range(len(g_ig.vs))}
    
    # Community detection the Leiden algorithm 
     
    print("Running Leiden community detection for Level 0...")
    if graph_prefix == 'DL':
        resolution_L0=0.00001
    else:
        resolution_L0 = 0.00035    
    # Level 0 communities (top level) - coarse resolution
    partition0 = la.find_partition(
        g_ig, 
        la.CPMVertexPartition, 
        weights='weight',
        resolution_parameter=resolution_L0,  # Lower resolution for broader communities
        n_iterations=-1,
        seed = SEED
    )
    
    # Store Level 0 community assignments
    communities_L0 = {}
    for i, membership in enumerate(partition0.membership):
        node_id = id_mapping[i]
        communities_L0[node_id] = membership
    
    # Create Level 1 communities (intermediate level)
    print("Running Leiden community detection for Level 1...")
    communities_L1 = {}
    parent_L0_to_L1 = {}
    children_L0_to_L1 = defaultdict(list)
    
    # Base offset for L1 communities to avoid ID conflicts
    L1_offset = 1000
    current_L1_id = L1_offset
    
    for community_id in range(max(partition0.membership) + 1):
        # Get nodes in this community
        community_nodes = [i for i, m in enumerate(partition0.membership) if m == community_id]
        children_L0_to_L1[community_id] = []
        
        if len(community_nodes) >= min_community_size:
            # Create subgraph
            subgraph = g_ig.subgraph(community_nodes)
            
            if graph_prefix == 'DL':
                resolution_L1=0.036
            else:
                resolution_L1 = 0.0013  
            
            # Run Leiden on subgraph with medium resolution
            subpartition = la.find_partition(
                subgraph, 
                la.CPMVertexPartition, 
                weights='weight',
                resolution_parameter=resolution_L1,  # Medium resolution
                n_iterations=-1,
                seed = SEED
            )
            
            # Store Level 1 community assignments
            for sub_i, sub_membership in enumerate(subpartition.membership):
                orig_node_idx = community_nodes[sub_i]
                node_id = id_mapping[orig_node_idx]
                
                # Assign L1 community ID
                L1_community_id = current_L1_id + sub_membership
                communities_L1[node_id] = L1_community_id
                
                # Record parent-child relationship
                parent_L0_to_L1[L1_community_id] = community_id
                if L1_community_id not in children_L0_to_L1[community_id]:
                    children_L0_to_L1[community_id].append(L1_community_id)
            
            # Update offset for next L0 community
            current_L1_id += max(subpartition.membership) + 1
    
    # Create Level 2 communities (most granular)
    print("Running Leiden community detection for Level 2...")
    communities_L2 = {}
    parent_L1_to_L2 = {}
    children_L1_to_L2 = defaultdict(list)
    
    # Base offset for L2 communities to avoid ID conflicts
    L2_offset = 10000
    current_L2_id = L2_offset
    
    # Process each L1 community to create L2 subcommunities
    L1_communities = set(communities_L1.values())
    for L1_community_id in L1_communities:
        # Get nodes in this L1 community
        L1_community_nodes_ids = [node_id for node_id, comm_id in communities_L1.items() 
                               if comm_id == L1_community_id]
        
        # Map node IDs back to igraph indices
        reverse_mapping = {v: k for k, v in id_mapping.items()}
        L1_community_nodes = [reverse_mapping[node_id] for node_id in L1_community_nodes_ids 
                             if node_id in reverse_mapping]
        
        children_L1_to_L2[L1_community_id] = []
        
        if len(L1_community_nodes) >= min_community_size:
            # Create subgraph
            subgraph = g_ig.subgraph(L1_community_nodes)
            if graph_prefix == 'DL':
                resolution_L2=0.6
            else:
                resolution_L2 = 0.8  
            # Run Leiden on subgraph with high resolution for fine-grained communities
            subpartition = la.find_partition(
                subgraph, 
                la.CPMVertexPartition, 
                weights='weight',
                resolution_parameter=resolution_L2,  # Higher resolution for more detailed communities
                n_iterations=-1,
                seed = SEED
            )
            
            # Store Level 2 community assignments
            for sub_i, sub_membership in enumerate(subpartition.membership):
                orig_node_idx = L1_community_nodes[sub_i]
                node_id = id_mapping[orig_node_idx]
                
                # Assign L2 community ID
                L2_community_id = current_L2_id + sub_membership
                communities_L2[node_id] = L2_community_id
                
                # Record parent-child relationship
                parent_L1_to_L2[L2_community_id] = L1_community_id
                if L2_community_id not in children_L1_to_L2[L1_community_id]:
                    children_L1_to_L2[L1_community_id].append(L2_community_id)
            
            # Update offset for next L1 community
            current_L2_id += max(subpartition.membership) + 1
    
    # Organize community relationships
    community_entity_mapping = defaultdict(list)
    community_relationship_mapping = defaultdict(list)
    
    # Create entity mappings for each level
    for node_id in G.nodes():
        # Level 0
        if node_id in communities_L0:
            comm_L0 = communities_L0[node_id]
            community_entity_mapping[(comm_L0, 0)].append(node_id)
        
        # Level 1
        if node_id in communities_L1:
            comm_L1 = communities_L1[node_id]
            community_entity_mapping[(comm_L1, 1)].append(node_id)
        
        # Level 2
        if node_id in communities_L2:
            comm_L2 = communities_L2[node_id]
            community_entity_mapping[(comm_L2, 2)].append(node_id)
    
    # Map relationships to communities at each level
    for _, rel in relationships_df.iterrows():
        source_entities = entities_df[entities_df['title'] == rel['source']]
        target_entities = entities_df[entities_df['title'] == rel['target']]
        
        if len(source_entities) > 0 and len(target_entities) > 0:
            source_id = source_entities['id'].iloc[0]
            target_id = target_entities['id'].iloc[0]
            
            # Level 0
            if source_id in communities_L0 and target_id in communities_L0:
                if communities_L0[source_id] == communities_L0[target_id]:
                    community_relationship_mapping[(communities_L0[source_id], 0)].append(rel['id'])
            
            # Level 1
            if source_id in communities_L1 and target_id in communities_L1:
                if communities_L1[source_id] == communities_L1[target_id]:
                    community_relationship_mapping[(communities_L1[source_id], 1)].append(rel['id'])
            
            # Level 2
            if source_id in communities_L2 and target_id in communities_L2:
                if communities_L2[source_id] == communities_L2[target_id]:
                    community_relationship_mapping[(communities_L2[source_id], 2)].append(rel['id'])
    
    # Create community data for dataframe
    community_data = []
    
    # Add Level 0 communities
    for (comm_id, level), entity_ids in [(k, v) for k, v in community_entity_mapping.items() if k[1] == 0]:
        community_data.append({
            'id': str(uuid.uuid4()),
            'human_readable_id': len(community_data),
            'community': int(comm_id),
            'level': level,
            'parent': -1,  # No parent for top level
            'children': children_L0_to_L1.get(comm_id, []),
            'title': f"Community {comm_id}",
            'entity_ids': entity_ids,
            'relationship_ids': community_relationship_mapping.get((comm_id, level), []),
            'text_unit_ids': [],  # Will populate below
            'period': datetime.now().strftime('%Y-%m-%d'),
            'size': len(entity_ids)
        })
    
    # Add Level 1 communities
    for (comm_id, level), entity_ids in [(k, v) for k, v in community_entity_mapping.items() if k[1] == 1]:
        community_data.append({
            'id': str(uuid.uuid4()),
            'human_readable_id': len(community_data),
            'community': int(comm_id),
            'level': level,
            'parent': parent_L0_to_L1.get(comm_id, -1),
            'children': children_L1_to_L2.get(comm_id, []),
            'title': f"Community {comm_id}",
            'entity_ids': entity_ids,
            'relationship_ids': community_relationship_mapping.get((comm_id, level), []),
            'text_unit_ids': [],  # Will populate below
            'period': datetime.now().strftime('%Y-%m-%d'),
            'size': len(entity_ids)
        })
    
    # Add Level 2 communities
    for (comm_id, level), entity_ids in [(k, v) for k, v in community_entity_mapping.items() if k[1] == 2]:
        community_data.append({
            'id': str(uuid.uuid4()),
            'human_readable_id': len(community_data),
            'community': int(comm_id),
            'level': level,
            'parent': parent_L1_to_L2.get(comm_id, -1),
            'children': [],  # Leaf nodes have no children
            'title': f"Community {comm_id}",
            'entity_ids': entity_ids,
            'relationship_ids': community_relationship_mapping.get((comm_id, level), []),
            'text_unit_ids': [],  # Will populate below
            'period': datetime.now().strftime('%Y-%m-%d'),
            'size': len(entity_ids)
        })
    
    # Create communities dataframe
    communities_df = pd.DataFrame(community_data)
    
    # Now populate the text_unit_ids for each community based on entity connections
    print("Connecting communities to text units...")
    
    # Create a mapping from entity_id to text_unit_ids
    entity_to_text_units = {}
    for _, entity in entities_df.iterrows():
        entity_id = entity['id']
        if 'text_unit_ids' in entity and isinstance(entity['text_unit_ids'], list):
            entity_to_text_units[entity_id] = entity['text_unit_ids']
    
    # Update each community's text_unit_ids
    for i, community in communities_df.iterrows():
        entity_ids = community['entity_ids']
        text_unit_ids = set()
        
        for entity_id in entity_ids:
            if entity_id in entity_to_text_units:
                text_unit_ids.update(entity_to_text_units[entity_id])
        
        communities_df.at[i, 'text_unit_ids'] = list(text_unit_ids)
    
    # Save communities
    communities_df.to_parquet(os.path.join(output_dir, 'communities.parquet'))
    
    print(f"Created {len(communities_df)} communities across 3 levels")
    if visualize:
        print("Creating visualizations of community structure...")
        community_graphs = {}
        
        # Copy original graph for each level
        G_L0 = original_graph.copy()
        G_L1 = original_graph.copy()
        G_L2 = original_graph.copy()
        
        # Add community attributes to nodes for each level
        for node_id in G.nodes():
            # Level 0
            if node_id in communities_L0:
                nx.set_node_attributes(G_L0, {node_id: {'community': communities_L0[node_id]}})
            
            # Level 1
            if node_id in communities_L1:
                nx.set_node_attributes(G_L1, {node_id: {'community': communities_L1[node_id]}})
            
            # Level 2
            if node_id in communities_L2:
                nx.set_node_attributes(G_L2, {node_id: {'community': communities_L2[node_id]}})
        
        community_graphs['level0'] = G_L0
        community_graphs['level1'] = G_L1
        community_graphs['level2'] = G_L2
    else:
        community_graphs = None
    
    # Return everything needed for visualization
    return {
        'communities': communities_df,
        'original_graph': original_graph,
        'community_graphs': community_graphs,
        'entity_id_to_title': entity_id_to_title,
        'communities_L0': communities_L0,
        'communities_L1': communities_L1,
        'communities_L2': communities_L2
    }
    
    
def plot_original_graph(graph, 
                        save_path: str,
                        prefix: str,
                        figsize=(15, 12),
                        node_size=1500,
                        node_alpha=DEFAULT_NODE_ALPHA,
                        font_size=DEFAULT_NODE_FONT_SIZE,
                        edge_attribute='type', 
                        edge_width=DEFAULT_EDGE_WIDTH,
                        edge_color=DEFAULT_EDGE_COLOR,
                        edge_alpha=DEFAULT_EDGE_ALPHA,
                        arrow_size=DEFAULT_ARROW_SIZE,
                        edge_font_size=DEFAULT_EDGE_FONT_SIZE,
                        edge_font_color=DEFAULT_EDGE_FONT_COLOR,
                        filename="original_graph_enhanced.png",
                        layout_type=DEFAULT_LAYOUT_TYPE,
                        k_layout=0.5,
                        iterations=50
                        ):
    """
    Visualize the original graph with enhanced styling, using node attributes
    directly from the graph object, and save it as an image file.

    Args:
        graph: The networkx graph object. Nodes should have 'title' and 'type' attributes.
               Edges should have the specified 'edge_attribute' (defaulting to 'type').
        save_path (str): Directory path where the plot image will be saved.
        prefix (str): Prefix for the plot title.
        figsize (tuple): Figure size for the plot.
        node_size (int): Size of the nodes.
        node_alpha (float): Opacity for nodes.
        font_size (int): Font size for node labels (taken from node 'title').
        edge_attribute (str): The key for the edge attribute holding the relation name.
        edge_width (float): Thickness of edge lines.
        edge_color (str): Color of edge lines.
        edge_alpha (float): Opacity for edges.
        arrow_size (int): Size of the edge arrowheads.
        edge_font_size (int): Font size for edge labels.
        edge_font_color (str): Color for edge labels.
        filename (str): Name for the saved image file.
        layout_type (str): NetworkX layout algorithm name.
        k_layout (float): Parameter for spring layout.
        iterations (int): Parameter for spring layout.
    """
    if not graph or graph.number_of_nodes() == 0:
        print("Graph is empty or not provided.")
        return

    print(f"Generating enhanced plot for {prefix} graph...")
    fig, ax = plt.subplots(figsize=figsize)

    print(f"Calculating '{layout_type}' layout...")
    try:
        # Calculate layout (same code as before)
        if layout_type == 'spring':
            pos = nx.spring_layout(graph, k=k_layout, iterations=iterations, seed=SEED)
        elif layout_type == 'kamada_kawai':
             if graph.number_of_nodes() > 1000: print("Warning: kamada_kawai layout can be slow for >1000 nodes.")
             pos = nx.kamada_kawai_layout(graph)
        else:
             print(f"Unknown layout_type '{layout_type}', using 'spring'.")
             pos = nx.spring_layout(graph, k=k_layout, iterations=iterations, seed=SEED)
    except Exception as e:
        print(f"Error during layout calculation: {e}. Falling back to spring layout.")
        pos = nx.spring_layout(graph, k=k_layout, iterations=iterations, seed=SEED)
    print("Layout calculation complete.")

    raw_node_types = [graph.nodes[node].get('type', DEFAULT_NODE_TYPE) for node in graph.nodes()]
    normalized_node_types = [str(t).strip().upper() for t in raw_node_types if t]
    unique_normalized_types = sorted(list(set(normalized_node_types)))
    type_to_color = {t: cm.tab20(i % 20) for i, t in enumerate(unique_normalized_types)}
    node_colors = [type_to_color.get(graph.nodes[node].get('type', DEFAULT_NODE_TYPE), (0.5, 0.5, 0.5)) for node in graph.nodes()]

    print("Drawing graph components...")
    # Draw nodes
    nx.draw_networkx_nodes(graph, pos, ax=ax, node_size=node_size, node_color=node_colors, alpha=node_alpha)

    # Draw edges (styled with arrows, curves)
    nx.draw_networkx_edges(graph, pos, ax=ax,
                           width=edge_width,
                           edge_color=edge_color,
                           alpha=edge_alpha,
                           arrows=True, 
                           arrowstyle="->",
                           arrowsize=arrow_size,
                           connectionstyle='arc3,rad=0.1')

    # Draw node labels (MODIFIED: Get from 'title' attribute)
    labels = {node: graph.nodes[node].get('title', str(node)) for node in graph.nodes()}
    nx.draw_networkx_labels(graph, pos, ax=ax, labels=labels, font_size=font_size)

    # Draw edge labels (Uses 'edge_attribute' parameter, default='description')
    try:
        edge_labels = nx.get_edge_attributes(graph, edge_attribute)
        if not edge_labels:
             print(f"Warning: No edge attributes found with key '{edge_attribute}'. Edge labels might not be drawn.")
        # Convert all edge labels to string just in case
        edge_labels_str = {k: str(v) for k, v in edge_labels.items()}
        nx.draw_networkx_edge_labels(graph, pos, ax=ax,
                                     edge_labels=edge_labels_str, 
                                     font_size=edge_font_size,
                                     font_color=edge_font_color,
                                     label_pos=0.5,
                                     bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.1'))
    except KeyError:
         print(f"Warning: Edge attribute key '{edge_attribute}' not found in graph edges. Edge labels will not be drawn.")
    except Exception as e:
         print(f"Error drawing edge labels: {e}")

    ax.set_title(f'{prefix}_Original Knowledge Graph (Enhanced)')
    ax.axis('off')

    # Add legend for entity types (same code as before)
    handles = [plt.scatter([], [], color=type_to_color.get(type_name, (0.5, 0.5, 0.5)), label=type_name)
               for type_name in unique_normalized_types]
    ax.legend(handles=handles, loc='upper right', bbox_to_anchor=(1.15, 1.0), title="Entity Types", fontsize='small')

    plt.tight_layout()
    print("Drawing complete.")

    # --- 5. Saving Logic ---
    try:
        os.makedirs(save_path, exist_ok=True)
        full_save_path = os.path.join(save_path, filename)
        plt.savefig(full_save_path, dpi=300, bbox_inches='tight')
        print(f"Enhanced original graph plot saved to: {full_save_path}")
        # logging.info(f"Enhanced original graph plot saved to: {full_save_path}")
    except Exception as e:
        print(f"Error saving graph to {full_save_path}: {e}")

    plt.close(fig)



def plot_community_graph(graph, communities,
                         level_name: str,
                         save_path: str,
                         figsize=(15, 12),
                         node_size=1500, 
                         node_alpha=DEFAULT_NODE_ALPHA,
                         font_size=DEFAULT_NODE_FONT_SIZE, 
                         edge_attribute='type', 
                         edge_width=DEFAULT_EDGE_WIDTH,
                         edge_color=DEFAULT_EDGE_COLOR,
                         edge_alpha=DEFAULT_EDGE_ALPHA,
                         arrow_size=DEFAULT_ARROW_SIZE,
                         edge_font_size=DEFAULT_EDGE_FONT_SIZE,
                         edge_font_color=DEFAULT_EDGE_FONT_COLOR,
                         filename_prefix="community_graph_enhanced", 
                         layout_type=DEFAULT_LAYOUT_TYPE,
                         k_layout=0.5,
                         iterations=50,
                         max_legend_items=15 
                         ):
    """
    Visualize the graph with community colors, directed edges, edge labels,
    and enhanced styling, then save it as an image file.

    Args:
        graph: The networkx graph object (nodes should have 'title' attribute).
               Edges should have the specified 'edge_attribute'.
        communities: Dictionary mapping node IDs (must match graph node IDs) to community IDs.
        level_name (str): Name of the community level (e.g., 'L0', 'Detailed'). Used in title/filename.
        save_path (str): Directory path where the plot image will be saved.
        figsize (tuple): Figure size for the plot.
        node_size (int): Size of the nodes.
        node_alpha (float): Opacity for nodes.
        font_size (int): Font size for node labels (taken from node 'title').
        edge_attribute (str): Key for the edge attribute holding the relation name.
        edge_width (float): Thickness of edge lines.
        edge_color (str): Color of edge lines.
        edge_alpha (float): Opacity for edges.
        arrow_size (int): Size of edge arrowheads.
        edge_font_size (int): Font size for edge labels.
        edge_font_color (str): Color for edge labels.
        filename_prefix (str): Prefix for the saved image filename.
        layout_type (str): NetworkX layout algorithm name.
        k_layout (float): Parameter for spring layout.
        iterations (int): Parameter for spring layout.
        max_legend_items (int): Max number of communities to show in the legend.
    """
    print("the graph has structure ", graph)    
    if not graph or graph.number_of_nodes() == 0:
        print("Graph is empty or not provided.")
        return
    if not communities:
        print("Community mapping is empty or not provided.")
        return

    print(f"Generating enhanced plot for community level: {level_name}...")
    fig, ax = plt.subplots(figsize=figsize)

    # --- 1. Layout ---
    print(f"Calculating '{layout_type}' layout...")
    try:
        # Calculate layout (same code as before)
        if layout_type == 'spring':
            pos = nx.spring_layout(graph, k=k_layout, iterations=iterations, seed=SEED)
        elif layout_type == 'kamada_kawai':
             if graph.number_of_nodes() > 1000: print("Warning: kamada_kawai layout can be slow for >1000 nodes.")
             pos = nx.kamada_kawai_layout(graph)
        # Add other layouts if needed...
        else:
             print(f"Unknown layout_type '{layout_type}', using 'spring'.")
             pos = nx.spring_layout(graph, k=k_layout, iterations=iterations, seed=SEED)
    except Exception as e:
        print(f"Error during layout calculation: {e}. Falling back to spring layout.")
        pos = nx.spring_layout(graph, k=k_layout, iterations=iterations, seed=SEED)
    print("Layout calculation complete.")


    # --- 2. Node Colors (Based on Community) ---
    valid_communities = {node: comm for node, comm in communities.items() if node in graph}
    if not valid_communities:
        print("Warning: No nodes from the community map found in the graph.")
        # Fallback: draw all nodes gray
        node_colors = [(0.5, 0.5, 0.5)] * graph.number_of_nodes()
        unique_communities = set()
        num_communities = 0
        color_map = {}
    else:
        unique_communities = set(valid_communities.values())
        num_communities = len(unique_communities)
        print(f"Coloring nodes based on {num_communities} communities...")

        # Create color map for communities
        cmap = plt.cm.rainbow # Or plt.cm.viridis, plt.cm.tab20 etc.
        color_map = {comm: cmap(i / max(1, num_communities - 1)) for i, comm in enumerate(sorted(unique_communities))}

        # Node colors based on community, default gray if node not in communities map
        node_colors = [color_map.get(valid_communities.get(node, -1), (0.5, 0.5, 0.5))
                    for node in graph.nodes()]


    # --- 3. Draw Network Components ---
    print("Drawing graph components...")
    # Draw nodes (colored by community)
    nx.draw_networkx_nodes(graph, pos, ax=ax, node_size=node_size, node_color=node_colors, alpha=node_alpha)

    # Draw edges (styled with arrows, curves)
    nx.draw_networkx_edges(graph, pos, ax=ax,
                           width=edge_width,
                           edge_color=edge_color,
                           alpha=edge_alpha,
                           arrows=True, 
                           arrowstyle="->",
                           arrowsize=arrow_size,
                           connectionstyle='arc3,rad=0.1')

    # Draw node labels (using node 'title' attribute)
    labels = {node: graph.nodes[node].get('title', str(node)) for node in graph.nodes()}
    nx.draw_networkx_labels(graph, pos, ax=ax, labels=labels, font_size=font_size)

    # Draw edge labels (using 'edge_attribute')
    try:
        edge_labels = nx.get_edge_attributes(graph, edge_attribute)
        if not edge_labels:
             print(f"Warning: No edge attributes found with key '{edge_attribute}'. Edge labels might not be drawn.")
        edge_labels_str = {k: str(v) for k, v in edge_labels.items()}
        nx.draw_networkx_edge_labels(graph, pos, ax=ax,
                                     edge_labels=edge_labels_str,
                                     font_size=edge_font_size,
                                     font_color=edge_font_color,
                                     label_pos=0.5,
                                     bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.1'))
    except KeyError:
         print(f"Warning: Edge attribute key '{edge_attribute}' not found in graph edges. Edge labels will not be drawn.")
    except Exception as e:
         print(f"Error drawing edge labels: {e}")


    # --- 4. Final Touches ---
    ax.set_title(f'Knowledge Graph with {level_name} Communities ({num_communities} detected)')
    ax.axis('off')

    # Add legend for communities (limited number shown for clarity)
    if num_communities > 0:
        communities_to_show = sorted(unique_communities)[:max_legend_items]
        handles = [plt.scatter([], [], color=color_map[comm_id], label=f'Comm. {comm_id}')
                   for comm_id in communities_to_show]

        legend_title = f'{num_communities} Communities'
        if num_communities > max_legend_items:
            legend_title += f' (Top {max_legend_items} shown)'

        ax.legend(handles=handles, loc='upper right', bbox_to_anchor=(1.15, 1.0), title=legend_title, fontsize='small')

    plt.tight_layout()
    print("Drawing complete.")

    # --- 5. Saving Logic ---
    try:
        os.makedirs(save_path, exist_ok=True)
        # Create a filename incorporating the level name (sanitize if needed)
        safe_level_name = "".join(c if c.isalnum() else "_" for c in level_name) # Basic sanitization
        filename = f"{filename_prefix}_{safe_level_name}.png"
        full_save_path = os.path.join(save_path, filename)

        plt.savefig(full_save_path, dpi=300, bbox_inches='tight')
        print(f"Community graph plot ({level_name}) saved to: {full_save_path}")
        # logging.info(f"Community graph plot ({level_name}) saved to: {full_save_path}")
    except Exception as e:
        print(f"Error saving graph to {full_save_path}: {e}")
        # plt.show()

    plt.close(fig)


def plot_hierarchical_communities(results, 
                                  save_path: str,
                                  figsize=(24, 8), 
                                  node_size=500, 
                                  node_alpha=DEFAULT_NODE_ALPHA,
                                  font_size=DEFAULT_NODE_FONT_SIZE, 
                                  edge_attribute='type', 
                                  edge_width=DEFAULT_EDGE_WIDTH,
                                  edge_color=DEFAULT_EDGE_COLOR,
                                  edge_alpha=DEFAULT_EDGE_ALPHA,
                                  arrow_size=DEFAULT_ARROW_SIZE,
                                  edge_font_size=DEFAULT_EDGE_FONT_SIZE,
                                  edge_font_color=DEFAULT_EDGE_FONT_COLOR,
                                  filename="hierarchical_communities_enhanced.png", 
                                  layout_type=DEFAULT_LAYOUT_TYPE,
                                  k_layout=0.5,
                                  iterations=50
                                  ):
    """
    Plot communities at each level of the hierarchy side by side with enhanced
    styling (directed edges, edge labels) and save the combined figure.

    Args:
        results (dict): Dict containing 'original_graph' and community assignments
                        for levels L0, L1, L2 (e.g., 'communities_L0'). Nodes in graph
                        should have 'title' attribute. Edges should have 'edge_attribute'.
        save_path (str): Directory path where the plot image will be saved.
        figsize (tuple): Figure size for the combined plot.
        node_size (int): Size of the nodes in the plots.
        node_alpha (float): Opacity for nodes.
        font_size (int): Font size for node labels (taken from node 'title').
        edge_attribute (str): Key for the edge attribute holding the relation name.
        edge_width (float): Thickness of edge lines.
        edge_color (str): Color of edge lines.
        edge_alpha (float): Opacity for edges.
        arrow_size (int): Size of edge arrowheads.
        edge_font_size (int): Font size for edge labels.
        edge_font_color (str): Color for edge labels.
        filename (str): Name for the saved image file.
        layout_type (str): NetworkX layout algorithm name.
        k_layout (float): Parameter for spring layout.
        iterations (int): Parameter for spring layout.
    """
    print("Generating enhanced combined plot for hierarchical communities...")

    # Graph reference from results dictionary
    original_graph = results.get('original_graph')

    if not original_graph or original_graph.number_of_nodes() == 0:
         print("Error: Missing or empty 'original_graph' in results dict.")
         return

    # Get community assignments (handle potential missing keys)
    communities_L0 = results.get('communities_L0', {})
    communities_L1 = results.get('communities_L1', {})
    communities_L2 = results.get('communities_L2', {})

    # --- 1. Layout (Calculated Once) ---
    print(f"Calculating '{layout_type}' layout...")
    try:
        # Calculate layout (same code as before)
        if layout_type == 'spring':
            pos = nx.spring_layout(original_graph, k=k_layout, iterations=iterations, seed=42)
        elif layout_type == 'kamada_kawai':
             if original_graph.number_of_nodes() > 1000: print("Warning: kamada_kawai layout can be slow for >1000 nodes.")
             pos = nx.kamada_kawai_layout(original_graph)
        # Add other layouts if needed...
        else:
             print(f"Unknown layout_type '{layout_type}', using 'spring'.")
             pos = nx.spring_layout(original_graph, k=k_layout, iterations=iterations, seed=42)
    except Exception as e:
        print(f"Error during layout calculation: {e}. Falling back to spring layout.")
        pos = nx.spring_layout(original_graph, k=k_layout, iterations=iterations, seed=42)
    print("Layout calculation complete.")


    # --- 2. Create Figure and Subplots ---
    fig, axs = plt.subplots(1, 3, figsize=figsize)
    # Ensure axs is always iterable even if only one plot was intended (though here it's 3)
    if not isinstance(axs, (list, tuple, np.ndarray)): axs = [axs]

    levels = [
        (communities_L0, "Level 0 (Broad)", axs[0]),
        (communities_L1, "Level 1 (Intermediate)", axs[1]),
        (communities_L2, "Level 2 (Detailed)", axs[2])
    ]

    print("Drawing subplots...")
    for communities, title, ax in levels:
        # --- 3a. Node Colors based on Community ---
        valid_communities = {node: comm for node, comm in communities.items() if node in original_graph}
        if not valid_communities:
            print(f"Warning: No nodes from communities map found in graph for level '{title}'. Using default color.")
            node_colors = [(0.7, 0.7, 0.7)] * original_graph.number_of_nodes()
            num_communities = 0
        else:
            unique_communities = set(valid_communities.values())
            num_communities = len(unique_communities)
            cmap = plt.cm.rainbow
            color_map = {comm: cmap(i / max(1, num_communities - 1)) for i, comm in enumerate(sorted(unique_communities))}
            node_colors = [color_map.get(valid_communities.get(node, -1), (0.7, 0.7, 0.7))
                        for node in original_graph.nodes()]


        # --- 3b. Draw Components on Subplot 'ax' ---
        # Draw nodes (colored by community)
        nx.draw_networkx_nodes(original_graph, pos, ax=ax, node_size=node_size, node_color=node_colors, alpha=node_alpha)

        # Draw edges (styled with arrows, curves)
        nx.draw_networkx_edges(original_graph, pos, ax=ax,
                               width=edge_width,
                               edge_color=edge_color,
                               alpha=edge_alpha,
                               arrows=True, 
                               arrowstyle="->",
                               arrowsize=arrow_size,
                               connectionstyle='arc3,rad=0.1')

        # Draw node labels (using 'title' attribute, drawing all - potentially cluttered)
        labels = {node: original_graph.nodes[node].get('title', str(node)) for node in original_graph.nodes()}
        nx.draw_networkx_labels(original_graph, pos, ax=ax, labels=labels, font_size=font_size)

        # Draw edge labels (using 'edge_attribute')
        try:
            edge_labels = nx.get_edge_attributes(original_graph, edge_attribute)
            if not edge_labels:
                 print(f"Warning: No edge attributes found with key '{edge_attribute}' for level '{title}'.")
            edge_labels_str = {k: str(v) for k, v in edge_labels.items()}
            nx.draw_networkx_edge_labels(original_graph, pos, ax=ax,
                                         edge_labels=edge_labels_str,
                                         font_size=edge_font_size,
                                         font_color=edge_font_color,
                                         label_pos=0.5,
                                         bbox=dict(facecolor='white', alpha=0.3, edgecolor='none', boxstyle='round,pad=0.1')) # Lighter bbox
        except KeyError:
             print(f"Warning: Edge attribute key '{edge_attribute}' not found for level '{title}'.")
        except Exception as e:
             print(f"Error drawing edge labels for level '{title}': {e}")

        # --- 3c. Subplot Final Touches ---
        ax.set_title(f'{title}\n({num_communities} communities detected)')
        ax.axis('off')

    # --- 4. Final Figure Adjustments ---
    plt.tight_layout(rect=[0, 0, 1, 0.97]) # Adjust layout, leave space at top if needed
    print("Drawing complete.")

    # --- 5. Saving Logic ---
    try:
        os.makedirs(save_path, exist_ok=True)
        full_save_path = os.path.join(save_path, filename)
        # Save the entire figure containing all subplots
        fig.savefig(full_save_path, dpi=300, bbox_inches='tight')
        print(f"Enhanced hierarchical communities plot saved to: {full_save_path}")
        # logging.info(f"Enhanced hierarchical communities plot saved to: {full_save_path}")
    except Exception as e:
        print(f"Error saving graph to {full_save_path}: {e}")
        # plt.show()

    plt.close(fig)


def validate_dataframe_for_import(df, required_columns=None):
    """Validate dataframe structure and content for Neo4j import"""
    issues = []
    
    # Check if df is not None
    if df is None:
        return ["Dataframe is None"]
    
    # Check if dataframe is empty
    if df.empty:
        issues.append("Dataframe is empty")
    
    # Check for required columns
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            issues.append(f"Missing required columns: {missing_columns}")
    
    # Check for columns with all None values
    for column in df.columns:
        if df[column].isna().all():
            issues.append(f"Column '{column}' contains all None values")
    
    # Special checks for Neo4j compatibility
    for column in df.columns:
        # Check for complex nested structures that might cause issues
        if df[column].dtype == 'object':
            sample = df[column].iloc[0] if len(df) > 0 else None
            if isinstance(sample, (dict, list)) and str(sample).count('{') > 10:
                issues.append(f"Column '{column}' contains deeply nested structures that may cause issues")
    
    return issues    

class TimeMeasurer:
    def __init__(self):
        self.timings: Dict[str, float] = {}
    @contextmanager
    def measure(self, key_name: str):
        start_time = time.perf_counter()
        try:
            yield
        finally:
            end_time = time.perf_counter()
            self.timings[key_name] = end_time - start_time
    def get_timing(self, key_name: str, default: float = 0.0) -> float:
        return self.timings.get(key_name, default)
    def get_all_timings(self) -> Dict[str, float]:
        return self.timings