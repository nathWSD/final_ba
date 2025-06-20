from src.ingestion_pipeline.helper_functions import gemini_llm, create_texts_data, clear_graph,robust_disambiguate_entities, list_graph_prefixes, import_complete_graph_data,detect_communities
import pandas as pd
import os
import hashlib
import re
from collections import defaultdict
from datetime import datetime
import json
import time
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
import tiktoken
import logging
import uuid
import random 
from typing import List, Optional, Callable

load_dotenv(dotenv_path=os.path.join(os.getcwd(), '.env'), override=True)


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DEFAULT_TOKEN_ENCODING = "cl100k_base" 

def count_tokens(text: str) -> int:
    """
    Counts the number of tokens in a text string using tiktoken.
    """
    if not text:
        return 0
    encoding = tiktoken.get_encoding(DEFAULT_TOKEN_ENCODING)
    return len(encoding.encode(text))

def parse_response_to_json(response_text):
    json_pattern = r'```json\s*([\s\S]*?)\s*```'
    json_match = re.search(json_pattern, response_text)
        
    if json_match:
       json_str = json_match.group(1).strip()
       return json.loads(json_str)
   

def generate_deterministic_id(text):
    """Generate a deterministic ID for content"""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def create_sampled_text_block(
    all_chunks: List[str],
    max_tokens: int,
    token_counter: Callable[[str], int], 
    random_seed: Optional[int] = None
) -> Optional[str]:
    """
    Randomly samples chunks and concatenates their text until max_tokens is approached.

    Args:
        all_chunks: List of all text chunk strings.
        max_tokens: The target maximum token count for the concatenated block.
        token_counter: Function to estimate token count of a string.
        random_seed: Optional seed for reproducible sampling.

    Returns:
        A single string containing concatenated sampled text, or None if input is empty.
    """
    if not all_chunks:
        return None

    if random_seed is not None:
        random.seed(random_seed)

    num_chunks = len(all_chunks)
    sampled_indices = list(range(num_chunks))
    random.shuffle(sampled_indices) 

    concatenated_text = ""
    current_tokens = 0
    chunks_used = 0

    for index in sampled_indices:
        chunk_text = all_chunks[index]
        chunk_tokens = token_counter(chunk_text)

        if current_tokens + chunk_tokens < max_tokens:
            if concatenated_text:
                concatenated_text += "\n\n---\n\n" 
                current_tokens += token_counter("\n\n---\n\n")

            concatenated_text += chunk_text
            current_tokens += chunk_tokens
            chunks_used += 1
        else:
            break

    logging.info(f"Created sampled text block using {chunks_used}/{num_chunks} chunks, "
                 f"estimated tokens: {current_tokens}/{max_tokens}")

    if not concatenated_text:
         logging.warning("Could not create sampled text block (maybe first chunk was too large?).")
         return None

    return concatenated_text

def generate_types_from_sampled_chunks(
    all_chunks: List[str],
    model: str = "gemini-2.0-flash", 
    temperature: float = 0.1,
    sample_max_tokens: int = 120000, 
    token_counter_func: Callable[[str], int] = count_tokens, 
    max_retries: int = 2,
    retry_delay: int = 5
) -> List[str]:
    """
    Generates and refines entity types by analyzing a sampled block of text chunks.

    Args:
        full_text: The complete input text.
        discovery_model: Model for the single discovery call on the sample.
        refinement_model: Model for the optional refinement call on the list.
        temperature: LLM temperature.
        chunk_size: Size used ONLY for initially splitting text into a list.
        chunk_overlap: Overlap used ONLY for initially splitting text.
        sample_max_tokens: Max tokens for the concatenated sample block sent to discovery LLM.
        token_counter_func: Function to estimate token count.
        max_retries: Retries for LLM calls.
        retry_delay: Delay between retries.

    Returns:
        A final, potentially refined list of entity types.
    """
    logging.info("--- Starting Dynamic Entity Type Generation (via Sampling) ---")
    if not all_chunks:
        logging.error("Text splitting resulted in no chunks.")
        return []

    sampled_text = create_sampled_text_block(
        all_chunks,
        sample_max_tokens,
        token_counter_func
    )
    if not sampled_text:
        logging.error("Failed to create a sampled text block.")
        return ["UNKNOWN"] 

    discovery_prompt_template = """
    Analyze the following text, which is a large sample concatenated from random parts of a bigger document.
    Your goal is to identify all relevant entity types or categories mentioned within THIS SAMPLE text.
    Focus on the *kinds* of things discussed (e.g., PLANET, COMPANY, PERSON, LAW, DISEASE, ASTRONOMICAL_EVENT, CHEMICAL_ELEMENT).
    Do not list specific entity names, only the types. Aim for a general overview for the content but not too general detailed enough to capture what is necessary,
    and be consistent.
    
    Output ONLY a JSON list of strings representing the unique types found in this sample. follow this Example output:
    
    ```json 
    ["PLANET", "STAR", "MISSION", "PERSON", ......]
    ```
    Sampled Text:
    {input_text}

    JSON list of types found in the sample:
    """
    discovery_prompt = ChatPromptTemplate.from_template(discovery_prompt_template)
    discovery_llm = gemini_llm(model, temperature)
    discovery_chain = discovery_prompt | discovery_llm | StrOutputParser()

    raw_types_list: Optional[List[str]] = None
    logging.info("Invoking LLM for type discovery on the sampled text...")
    for attempt in range(max_retries):
        try:
            response = discovery_chain.invoke({"input_text": sampled_text})
            print("response for types ", response)
            raw_types_list = parse_response_to_json(response) 

            if raw_types_list is not None:
                logging.info(f"Successfully discovered {len(raw_types_list)} raw types from sample.")
                logging.debug(f"Raw types from sample: {sorted(raw_types_list)}")
                break
            else:
                 if attempt < max_retries - 1: logging.warning(f"Discovery parsing failed (Attempt {attempt+1}), retrying...")
                 else: logging.error(f"Discovery parsing failed after {max_retries} attempts.")
                 import time; time.sleep(retry_delay)

        except Exception as e:
            logging.error(f"LLM invocation failed during discovery (Attempt {attempt+1}): {e}", exc_info=True)
            if attempt < max_retries - 1: logging.info(f"Retrying discovery after {retry_delay} seconds...")
            else: logging.error(f"Discovery LLM invocation failed after {max_retries} attempts.")
            import time; time.sleep(retry_delay)

    if raw_types_list is None:
        logging.error("Failed to discover any types from the sampled text.")
        return []

    refinement_prompt_template = """
    You are provided with a list of potential entity types discovered from a SAMPLE of a larger document. Your task is to refine this list into a concise, consistent, and useful set of types suitable for knowledge graph extraction from the full document.

    Follow these guidelines:
    1.  **Merge Synonyms & Related Concepts:** Combine types representing similar concepts (e.g., "PLANETARY_BODY", "CELESTIAL_BODY_TYPE" -> "CELESTIAL_OBJECT" or "PLANET").
    2.  **Ensure Consistent Granularity:** Aim for a balanced level suitable for the likely domain. Generalize slightly where appropriate (e.g., prefer "PERSON" over "ACTOR").
    3.  **Remove Redundancy:** Eliminate duplicates after merging.
    4.  **Remove Vague/Unhelpful Types:** Discard overly generic types like "THING".
    5.  **Format:** Output ONLY a single JSON list of strings containing the final, refined types, sorted alphabetically.

    Follow this Example output:
    
    ```json 
    ["PLANET", "STAR", "MISSION", "PERSON", ......]
    ```
    
    Input list of discovered types:
    {type_list}

    Refined JSON list of types:
    """
    refinement_prompt = ChatPromptTemplate.from_template(refinement_prompt_template)
    refinement_llm = gemini_llm(model, temperature)
    refinement_chain = refinement_prompt | refinement_llm | StrOutputParser()
    raw_types_set = set(raw_types_list)
    refinement_chain.invoke
    response = refinement_chain.invoke({
            "type_list": raw_types_set,
        })
    final_type_list = parse_response_to_json(response)
    final_types_set = set(final_type_list) 
    final_types_set.add("UNKNOWN")
    final_types_list_sorted = sorted(list(final_types_set)) 
    logging.info(f"--- Dynamic Entity Type Generation via Sampling Complete. Final types: {final_types_list_sorted} ---")
    return final_types_list_sorted


def extract_raw_triplets_with_llm( text:str, chunk_id: str, entity_types:List[str] , model:str="gemini-1.5-pro", temperature:float=0.1):
    """
    Extract entities and relationships from text using Google's Gemini LLM.
    
    Args:
        text (str): The text to analyze
        model (str): The Gemini model to use for extraction
        temperature (float): Temperature parameter for generation (0.0-1.0)
        
    Returns:
        dict: Dictionary with 'entities' and 'relationships' lists
    """
    
    prompt_template = """
      You are an expert graph information extractor. Your task is to extract key entities and their relationships from the provided text to form a consistent knowledge graph fragment, strictly adhering to the allowed entity types.

    Follow these steps internally to determine the final output:
    1.  **Identify Potential Entities:** Read the text and list all potential entities. For each, note its name (title), its likely type, and a brief description based on the text.
        **CRITICAL TYPE CONSTRAINT:** The 'type' assigned to an entity MUST be chosen strictly from the following allowed list: [{allowed_types_str}]. If an entity is found but does not clearly fit any type in this list, assign its 'type' as "UNKNOWN". Do NOT use synonyms, variations, plurals, or invent new types not present in the allowed list *or* the fallback "UNKNOWN". Select the *single best fit* from the list or use "UNKNOWN".
    2.  **Identify Relationships:** Find explicit relationships mentioned in the text *only between the entities identified in Step 1*. For each relationship found, note the source entity title, target entity title, a concise relationship label (verb phrase like 'orbits', 'discovered by', 'part of'), a weight score (0.1-1.0 reflecting clarity/importance), and a description of the relationship based on the text.
    3.  **Filter for Consistency:**
        a. Keep only the relationships identified in Step 2.
        b. Review the list of potential entities from Step 1. Keep ONLY those entities whose titles appear as *either* a 'source' *or* a 'target' in the relationships kept in Step 3a AND whose assigned type is valid (i.e., is either in the allowed list or is "UNKNOWN"). Discard any entities that are not part of any identified relationship or have an invalid type.
    4.  **Format Output:** Prepare the final JSON output using *only* the filtered entities (from Step 3b) and filtered relationships (from Step 3a). Ensure entity types strictly adhere to the allowed list provided in Step 1 *or* are exactly "UNKNOWN".

    Output Requirements:
    - Return ONLY a single JSON object. Do NOT include explanations, apologies, or the step-by-step reasoning in the output.
    - The JSON object must have EXACTLY two keys: "entities" and "relationships".
    - "entities": A list of JSON objects, each with "title", "type" (strictly only from the allowed list: [{allowed_types_str}] or "UNKNOWN"), and "description" for the filtered entities.
    - "relationships": A list of JSON objects, each with "source", "target", "label", "weight", and "description" for the filtered relationships.
    - Crucially, every "source" and "target" value in the "relationships" list MUST EXACTLY match a "title" value present in the "entities" list of your final JSON output.

    Example JSON Structure (using example allowed types + UNKNOWN):
    ```json
        {{
        "entities": [
            {{"title": "Mars", "type": "PLANET", "description": "The fourth planet from the Sun."}},
            {{"title": "Olympus Mons", "type": "VOLCANO", "description": "A large shield volcano on Mars."}},
            {{"title": "Curiosity Rover", "type": "UNKNOWN", "description": "A rover operating on Mars."}}
        ],
        "relationships": [
            {{"source": "Olympus Mons", "target": "Mars", "label": "located on", "weight": 0.9, "description": "Olympus Mons is explicitly stated to be located on Mars."}},
            {{"source": "Curiosity Rover", "target": "Mars", "label": "exploring", "weight": 0.8, "description": "The rover is exploring Mars."}}
        ]
        }}
    ```

        Text to analyze:
        {input_text}
    """
    
    raw_triplets_found = []
    max_retries = 4
    retry_delay = 4 

    for attempt in range(max_retries):
        try:
            prompt = ChatPromptTemplate.from_template(prompt_template)
            llm = gemini_llm(model, temperature)
            chain = prompt | llm | StrOutputParser()

            logging.debug(f"Invoking LLM (Attempt {attempt+1}) for chunk {chunk_id[:10]}...")
            response = chain.invoke({"input_text": text, "allowed_types_str": entity_types })
            extraction_result = parse_response_to_json(response)
            if not extraction_result or ('entities' not in extraction_result and 'relationships' not in extraction_result):
                 logging.warning(f"LLM parsing failed or returned empty structure for chunk {chunk_id}. Response: {response[:200]}")
                 raise ValueError("Failed to parse valid JSON from LLM response") 
            entity_details_map = {
                entity.get('title').strip(): {
                    'type': entity.get('type').strip().upper(),
                    'description': entity.get('description').strip()
                 }
                for entity in extraction_result.get('entities') if entity.get('title').strip()
            }
            for rel in extraction_result.get('relationships'):
                source = rel.get('source').strip()
                target = rel.get('target').strip()
                label = rel.get('label').strip() 
                weight = float(rel.get('weight')) 
                rel_description = rel.get('description').strip() 

                if not source or not target or not label or source == target:
                    logging.debug(f"Skipping invalid relationship: {rel}")
                    continue
                source_details = entity_details_map.get(source)
                target_details = entity_details_map.get(target)
                
                head_type = source_details['type']
                tail_type = target_details['type']
                head_description = source_details['description'] 
                tail_description = target_details['description'] 

                if source in entity_details_map or target in entity_details_map:
                    raw_triplets_found.append({
                        'head': source,
                        'tail': target,
                        'type': label,
                        'weight': max(0.1, min(1.0, weight)),
                        'chunk_id': chunk_id,
                        'head_type': head_type,
                        'tail_type': tail_type,
                        'head_description': head_description, 
                        'tail_description': tail_description, 
                        'description': rel_description 
                    })

            logging.info(f"LLM extracted {len(raw_triplets_found)} raw triplets for chunk {chunk_id[:10]}...")
            return raw_triplets_found

        except Exception as e:
            logging.error(f"Error during LLM extraction (Attempt {attempt+1}/{max_retries}) for chunk {chunk_id}: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                retry_delay *= 2 
            else:
                logging.error(f"LLM extraction failed permanently for chunk {chunk_id} after {max_retries} attempts.")
                return []
    return []


def process_text_to_graphrag_with_llm(dataset_path,
                                      output_dir,
                                      model="gemini-2.5-pro-preview-05-06",
                                      temperature=0.1,
                                      ): 
    """
    Process text chunks and create GraphRAG compatible parquet files using
    LLM extraction and BASIC entity normalization (strip/upper). The fuzzy
    matching/clustering normalization step is SKIPPED.

    Args:
        dataset_path (str): Path to the dataset file or directory.
        output_dir (str): Directory to save the output parquet files.
        model (str): The LLM model identifier to use.
        temperature (float): Temperature setting for LLM generation.

    Returns:
        dict: A dictionary containing the final pandas DataFrames:
              'entities', 'relationships', 'text_units', 'documents'.
              Returns an empty dict if processing fails at any critical stage.
    """
    start_time = datetime.now() 
    logging.info(f"Starting GraphRAG processing pipeline for path: {dataset_path}")

    texts = create_texts_data(dataset_path)
    if not texts:
        logging.error("Input 'texts' list is empty after loading. Cannot proceed.")
        return {}
    total_chunks = len(texts)
    print(f"there are {total_chunks} number of chunks for all the websites ")
    document_id = generate_deterministic_id(texts[0])
    collection_name = os.getenv('COLLECTION_BASE_NAME', 'DefaultCollection')
    logging.info("Starting Pass 1: Extracting raw triplets using LLM...")
    all_raw_triplets = []
    chunk_processing_info = []
    allowed_entity_types = generate_types_from_sampled_chunks(all_chunks = texts, max_retries = 5)
    for i, chunk_text in enumerate(texts):
        current_chunk_num = i + 1
        if not chunk_text or not chunk_text.strip():
            logging.warning(f"Skipping empty chunk {i+1}/{len(texts)}")
            continue
        
        chunk_id = generate_deterministic_id(chunk_text)
        chunk_processing_info.append({
            'id': chunk_id,
            'text': chunk_text,
            'human_readable_id': i + 1,
            'document_ids': [document_id]
        })
        logging.info(f"Processing chunk {current_chunk_num}/{total_chunks} for LLM extraction...")
        try:
            raw_chunk_triplets = extract_raw_triplets_with_llm(
                text=chunk_text,
                chunk_id=chunk_id,
                entity_types = allowed_entity_types,
                model=model,
                temperature=temperature
            )
            with open(os.path.join(os.getcwd(), 'src/ingestion_pipeline/graph_llm_ingestion/triplets.json'), 'w', encoding='utf-8') as f:
             json.dump(raw_chunk_triplets, f, indent=4, ensure_ascii=False)
                
            all_raw_triplets.extend(raw_chunk_triplets)
        except Exception as e:
            logging.error(f"Error extracting triplets from chunk {i+1}: {e}", exc_info=True)
            
    path_triplets = os.path.join(os.getcwd(), 'src/ingestion_pipeline/graph_llm_ingestion/triplets.json')      
    with open(path_triplets, 'w', encoding='utf-8') as f:
        json.dump(all_raw_triplets, f, indent=4)
    logging.info(f"Successfully saved graph analysis stats to: {path_triplets}") 
           
    logging.info(f"length of all triplets before ambiguity {len(all_raw_triplets)}")
    all_raw_triplets = robust_disambiguate_entities(triplets = all_raw_triplets)
    logging.info(f"Pass 1 Complete: Extracted {len(all_raw_triplets)} raw triplets via LLM. after ambiguity")

    if not all_raw_triplets:
        logging.warning("No raw triplets were extracted by the LLM. Check LLM responses or prompts.")
        return {}
    
    logging.info("Starting Pass 2: Basic Normalization and Aggregation...")
    entities_accumulator = {}
    normalized_relationships = []
    entity_frequency = defaultdict(int)
    entity_chunks = defaultdict(set)
    logging.info("Aggregating entities and relationships...")
    num_skipped_empty = 0
    num_skipped_self_loops = 0
    for raw_triplet in all_raw_triplets:
        original_head = raw_triplet.get('head')
        original_tail = raw_triplet.get('tail')
        chunk_id = raw_triplet.get('chunk_id')
        norm_head = str(original_head).strip() if original_head else ""
        norm_tail = str(original_tail).strip() if original_tail else ""

        if not norm_head or not norm_tail or not chunk_id:
            num_skipped_empty += 1
            continue

        norm_head_upper = norm_head.upper()
        norm_tail_upper = norm_tail.upper()
        if norm_head_upper == norm_tail_upper:
             num_skipped_self_loops += 1
             continue

        rel_type = raw_triplet.get('type', 'related to')
        rel_description = raw_triplet.get('description', '')
        if not rel_description: rel_description = f"{norm_head} {rel_type} {norm_tail}"
        for norm_entity_upper, _, entity_type_hint, entity_description_hint in [
                    (norm_head_upper, original_head, raw_triplet.get('head_type', 'UNKNOWN'), raw_triplet.get('head_description', '')),
                    (norm_tail_upper, original_tail, raw_triplet.get('tail_type', 'UNKNOWN'), raw_triplet.get('tail_description', ''))]:

            entity_frequency[norm_entity_upper] += 1
            entity_chunks[norm_entity_upper].add(chunk_id)

            if norm_entity_upper not in entities_accumulator:
                first_description = entity_description_hint if entity_description_hint else f"Entity: {norm_entity_upper}"
                first_type = entity_type_hint if entity_type_hint != 'UNKNOWN' else 'ENTITY' 
                entities_accumulator[norm_entity_upper] = {
                    'id': str(uuid.uuid4()),
                    'title': norm_entity_upper, 
                    'type': first_type,
                    'description': first_description,
                    'text_unit_ids': set(), 
                    'frequency': 0,        
                    'degree': 0,           
                 }

        normalized_relationships.append({
            'id': str(uuid.uuid4()),
            'source': norm_head_upper, 
            'target': norm_tail_upper,
            'description': rel_description,
            'type': rel_type,
            'weight': raw_triplet.get('weight', 0.5),
            'text_unit_ids': {chunk_id}, 
            'combined_degree': 0 
        })
    if num_skipped_empty > 0: logging.info(f"Skipped {num_skipped_empty} triplets due to empty head/tail after basic normalization.")
    if num_skipped_self_loops > 0: logging.info(f"Skipped {num_skipped_self_loops} self-loops after basic normalization.")
    final_dataframes = save_llm_graph_data(
        entities_accumulator=entities_accumulator,
        entity_frequency=entity_frequency,
        entity_chunks=entity_chunks,
        normalized_relationships=normalized_relationships,
        chunk_processing_info=chunk_processing_info,
        document_id=document_id,
        collection_name=collection_name,
        texts=texts, 
        output_dir=output_dir,
        start_time=start_time 
    )

    if not final_dataframes:
        logging.error("Saving/Finalization step failed.")
        return {} 
    return final_dataframes


def save_llm_graph_data(entities_accumulator, entity_frequency, entity_chunks,
                        normalized_relationships, chunk_processing_info, document_id,
                        collection_name, texts, output_dir, start_time):
    """
    Handles the final aggregation steps, DataFrame creation, and saving,
    assuming basic normalization happened before calling.

    Args:
        entities_accumulator (dict): Accumulated entity data.
        entity_frequency (defaultdict): Frequencies of entities.
        entity_chunks (defaultdict): Mapping of entities to chunk IDs.
        normalized_relationships (list): List of relationship dictionaries.
        chunk_processing_info (list): Information about each processed chunk.
        document_id (str): ID of the document.
        collection_name (str): Name of the collection.
        texts (list): Original text chunks.
        output_dir (str): Directory to save parquet files.
        start_time (datetime): Start time of the overall process.

    Returns:
        dict: Dictionary containing the final pandas DataFrames.
              Returns an empty dict or None on failure.
    """
    logging.info("Finalizing entity list...")
    final_entities_list = []
    if entities_accumulator:
        for norm_entity_upper, data in entities_accumulator.items():
            data['frequency'] = entity_frequency.get(norm_entity_upper, 0)
            data['text_unit_ids'] = list(entity_chunks.get(norm_entity_upper, set()))
            if 'degree' not in data: data['degree'] = 0 
            final_entities_list.append(data)
    else:
        logging.warning("entities_accumulator is empty. No entities to finalize.")

    logging.info("Finalizing relationship list...")
    if normalized_relationships:
        for rel in normalized_relationships:
            rel['text_unit_ids'] = list(rel.get('text_unit_ids', set()))
            if 'combined_degree' not in rel: rel['combined_degree'] = 0 
    else:
        logging.warning("normalized_relationships list is empty. No relationships to finalize.")
    logging.info("Calculating entity degrees...")
    entity_degrees = defaultdict(int)
    if final_entities_list: 
        final_entity_titles = {e['title'] for e in final_entities_list if 'title' in e}
        if normalized_relationships: 
            for rel in normalized_relationships:
                source = rel.get('source')
                target = rel.get('target')
                if source in final_entity_titles:
                    entity_degrees[source] += 1
                if target in final_entity_titles:
                    entity_degrees[target] += 1
        else:
            logging.warning("No relationships found to calculate degrees from.")
    else:
        logging.warning("No finalized entities found to calculate degrees for.")
    if final_entities_list:
        for entity_data in final_entities_list:
            entity_data['degree'] = entity_degrees.get(entity_data.get('title'), 0)
    if normalized_relationships:
        for rel in normalized_relationships:
             rel['combined_degree'] = entity_degrees.get(rel.get('source'), 0) + entity_degrees.get(rel.get('target'), 0)

    logging.info("Creating entities and relationships DataFrames...")
    entities_df = pd.DataFrame(final_entities_list)
    if not entities_df.empty:
        entities_df['human_readable_id'] = range(len(entities_df))
        if 'x' not in entities_df.columns: entities_df['x'] = 0.0
        if 'y' not in entities_df.columns: entities_df['y'] = 0.0
    else:
        entities_df = pd.DataFrame(columns=['id', 'title', 'type', 'description', 'text_unit_ids', 'frequency', 'degree', 'human_readable_id', 'x', 'y'])

    relationships_df = pd.DataFrame(normalized_relationships)
    if not relationships_df.empty:
        relationships_df['human_readable_id'] = range(len(relationships_df))
    else:
        relationships_df = pd.DataFrame(columns=['id', 'human_readable_id', 'source', 'target', 'description', 'type', 'weight', 'combined_degree', 'text_unit_ids'])

    logging.info("Creating text_units DataFrame...")
    text_units = []
    entity_title_to_id_map = entities_df.set_index('title')['id'].to_dict() if not entities_df.empty and 'title' in entities_df.columns else {}
    chunk_to_rel_ids = defaultdict(list)
    if normalized_relationships:
        for rel in normalized_relationships:
            rel_id = rel.get('id')
            rel_chunk_ids = rel.get('text_unit_ids', [])
            if rel_chunk_ids and rel_id:
                 chunk_id_for_rel = rel_chunk_ids[0] 
                 chunk_to_rel_ids[chunk_id_for_rel].append(rel_id)

    if chunk_processing_info:
        for chunk_info in chunk_processing_info:
            chunk_id = chunk_info.get('id')
            if not chunk_id: continue 
            chunk_canonical_entities = {title for title, chunks in entity_chunks.items() if chunk_id in chunks}
            chunk_entity_ids = [entity_title_to_id_map[title] for title in chunk_canonical_entities if title in entity_title_to_id_map]
            chunk_rel_ids = chunk_to_rel_ids.get(chunk_id, [])
            text_units.append({
                'id': chunk_id,
                'human_readable_id': chunk_info.get('human_readable_id', -1),
                'text': chunk_info.get('text', ''),
                'n_tokens': len(chunk_info.get('text', '').split()), 
                'document_ids': chunk_info.get('document_ids', []),
                'entity_ids': chunk_entity_ids,
                'relationship_ids': chunk_rel_ids,
                'covariate_ids': [] 
            })
    else:
        logging.warning("chunk_processing_info is empty. Cannot create text units.")

    text_units_df = pd.DataFrame(text_units)
    if text_units_df.empty:
         text_units_df = pd.DataFrame(columns=['id', 'human_readable_id', 'text', 'n_tokens', 'document_ids', 'entity_ids', 'relationship_ids', 'covariate_ids'])

    logging.info("Creating documents DataFrame...")
    all_final_text_unit_ids = list(text_units_df['id']) if not text_units_df.empty else []
    current_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S %z")

    documents_df = pd.DataFrame([{
        'id': document_id,
        'human_readable_id': 1,
        'title': collection_name,
        'text': "".join(texts) if texts else "", 
        'text_unit_ids': all_final_text_unit_ids,
        'creation_date': current_timestamp,
        'metadata': None 
    }])
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Saving parquet files to {output_dir}...")
    try:
        entities_df.to_parquet(os.path.join(output_dir, 'entities.parquet'), index=False)
        relationships_df.to_parquet(os.path.join(output_dir, 'relationships.parquet'), index=False)
        text_units_df.to_parquet(os.path.join(output_dir, 'text_units.parquet'), index=False)
        documents_df.to_parquet(os.path.join(output_dir, 'documents.parquet'), index=False)
    except Exception as e:
         logging.error(f"Failed to save parquet files to {output_dir}: {e}", exc_info=True)
         return None 

    end_time = datetime.now()
    logging.info(f"Aggregation and saving complete. Total time from start: {end_time - start_time}")
    logging.info(f"Created {len(entities_df)} final entities.")
    logging.info(f"Created {len(relationships_df)} final relationships.")
    logging.info(f"Created {len(text_units_df)} text units.")
    return {
        'entities': entities_df,
        'relationships': relationships_df,
        'text_units': text_units_df,
        'documents': documents_df
    }
    
def generate_community_reports(communities_df, entities_df, relationships_df, output_dir, model = "gemini-2.0-flash", temperature=0.7):
    """Generate summarized reports for each community with dynamic length handling"""
    print("Generating community reports...")
    community_reports = []    
    llm = gemini_llm(model, temperature)
    for idx, community in communities_df.iterrows():
        print(f"Processing community {idx+1}/{len(communities_df)}")
        community_id = community['id']
        entity_ids = community['entity_ids']
        relationship_ids = community['relationship_ids']        
        try:
            community_entities = entities_df[entities_df['id'].isin(entity_ids)]
        except:
            community_entities = entities_df.iloc[0:0].copy()  
        try:
            community_relationships = relationships_df[relationships_df['id'].isin(relationship_ids)]
        except:
            community_relationships = relationships_df.iloc[0:0].copy() 

        entity_context = []
        for _, entity in community_entities.iterrows():
            entity_type = entity.get('type', 'Unknown')
            entity_context.append({
                "title": entity['title'],
                "type": entity_type,
                "description": entity.get('description', ''),
                "degree": int(entity.get('degree', 0))
            })
        
        relationship_context = []
        for _, rel in community_relationships.iterrows():
            relationship_context.append({
                "source": rel['source'],
                "target": rel['target'],
                "type": rel.get('type', ''),
                "description": rel.get('description', ''),
                "weight": float(rel.get('weight', 1.0))
            })        
        entity_text = "\n".join([f"- {e['title']} ({e['type']}): {e['description']}" for e in entity_context[:20]])
        relationship_text = "\n".join([f"- {r['source']} → {r['target']}: {r['description']} (Weight: {r['weight']})" 
                                    for r in relationship_context[:20]])
        community_analysis_prompt = """
        You are an expert analyst tasked with summarizing and analyzing in details a community of related entities.
        Please analyze the following community of entities and their relationships:
        
        ENTITIES:
        {entity_text}
        
        RELATIONSHIPS:
        {relationship_text}
        
        Based on this information, please provide:
        
        1. A title for this community (focus on the key entities and their relationships)
        2. A concise summary (2-3 sentences) of what this community represents
        3. Three key findings about this community
        4. A rating of importance on a scale of 1.0 to 10.0 with explanation
        
        Format your response exactly as JSON:
        ```json
        {{
          "title": "Community Title",
          "summary": "Concise summary of the community",
          "findings": [
            "First detailed key finding",
            "Second detailed key finding", 
            "Third detailed key finding"
          ],
          "importance_rating": 7.5,
          "rating_explanation": "Explanation for the rating"
        }}
        ```
        """        
        prompt = ChatPromptTemplate.from_template(community_analysis_prompt)
        chain = prompt | llm | StrOutputParser()
        max_retries = 5
        for attempt in range(max_retries):
            try:
                response = chain.invoke({"entity_text": entity_text, "relationship_text": relationship_text})
                analysis = parse_response_to_json(response) 
                break
            except Exception as e:
                print(f"Attempt {attempt+1} failed: {e}")
                if attempt == max_retries - 1:
                    print(f"All attempts failed for community {community_id}. Using fallback.")
                    analysis = {
                        "title": f"Community {community['community']}",
                        "summary": f"This community contains {len(entity_ids)} entities and {len(relationship_ids)} relationships.",
                        "findings": [
                            "Contains multiple related entities",
                            "Shows interconnected relationships",
                            "May represent a key topic area"
                        ],
                        "importance_rating": 5.0,
                        "rating_explanation": "Medium importance based on limited analysis"
                    }
                else:
                    time.sleep(5)  
        community_title = analysis.get("title", f"Community {community['community']}")
        summary = analysis.get("summary", f"Community with {len(entity_ids)} entities.")
        findings_list = analysis.get("findings", [])
        importance_score = float(analysis.get("importance_rating", 5.0))
        rating_explanation = analysis.get("rating_explanation", "Rating based on entity and relationship analysis.")
        findings = [{"explanation": finding} for finding in findings_list]
        full_content = f"# {community_title}\n\n{summary}\n\n## Entities\n\n"
        for _, entity in community_entities.iterrows():
            entity_type = entity.get('type', 'Unknown')
            full_content += f"- **{entity['title']}** ({entity_type}): {entity.get('description', '')}\n"
        
        full_content += "\n## Key Relationships\n\n"
        for _, rel in community_relationships.head(10).iterrows():
            description = rel.get('description', 'related to')
            full_content += f"- {rel['source']} → {description} → {rel['target']}\n"
        try:
            full_content_json = {
                "title": community_title,
                "summary": summary,
                "entities": []
            }
            if len(community_entities) > 0:
                full_content_json["entities"] = []
                for _, entity in community_entities.iterrows():
                    entity_json = {
                        "title": entity['title'],
                        "type": entity.get('type', 'Unknown')
                    }
                    if 'degree' in entity:
                        entity_json["degree"] = int(entity['degree'])
                    full_content_json["entities"].append(entity_json)  
            full_content_json["findings"] = findings_list
            json_str = json.dumps(full_content_json, indent=1)
        except Exception as e:
            print(f"Error creating JSON content: {e}")
            json_str = json.dumps({"title": community_title, "summary": summary})
        community_reports.append({
            'id': community_id,
            'human_readable_id': len(community_reports),
            'community': community['community'],
            'level': community['level'],
            'parent': community.get('parent', -1),
            'children': community.get('children', []),
            'title': community_title,
            'summary': summary,
            'full_content': full_content,
            'rank': importance_score,
            'rating_explanation': rating_explanation,
            'findings': findings,
            'full_content_json': json_str,
            'period': datetime.now().strftime('%Y-%m-%d'),
            'size': community.get('size', len(entity_ids))
        })
        if idx < len(communities_df) - 1:
            time.sleep(0.5)
    try:
        community_title_map = {report['community']: report['title'] for report in community_reports}
        
        for i, row in communities_df.iterrows():
            if row['community'] in community_title_map:
                communities_df.at[i, 'title'] = community_title_map[row['community']]
    except Exception as e:
        print(f"Error updating community titles: {e}")
    community_reports_df = pd.DataFrame(community_reports)
    communities_df.to_parquet(os.path.join(output_dir, 'communities.parquet'))
    community_reports_df.to_parquet(os.path.join(output_dir, 'community_reports.parquet'))
    print(f"Generated {len(community_reports)} community reports")
    return community_reports_df   
   
   
def run_graph_analysis(results, entities_df, relationships_df, text_units_df, output_dir):
    """Run complete graph analysis workflow with fixed hierarchy and text connections"""

    communities_df = results['communities']
    community_reports_df = generate_community_reports(communities_df, entities_df, relationships_df, output_dir)
    print("Generating entity embeddings and positions...")
    try:
        entity_texts = [f"{row['title']}: {row['description']}" for _, row in entities_df.iterrows()]
        entity_ids = entities_df['id'].tolist()
        embedding_model = SentenceTransformer(os.getenv('DENSE_MODEL_KEY'))
        embeddings = embedding_model.encode(entity_texts)
        if len(embeddings) > 1:
            pca = PCA(n_components=2)
            positions_2d = pca.fit_transform(embeddings)
            for i, entity_id in enumerate(entity_ids):
                idx = entities_df[entities_df['id'] == entity_id].index
                if len(idx) > 0:
                    entities_df.at[idx[0], 'x'] = float(positions_2d[i][0])
                    entities_df.at[idx[0], 'y'] = float(positions_2d[i][1])
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        for i in range(len(entities_df)):
            entities_df.at[i, 'x'] = float(i % 10)
            entities_df.at[i, 'y'] = float(i // 10)

    entities_df.to_parquet(os.path.join(output_dir, 'entities.parquet'))
    print(f"Created {len(communities_df)} communities")
    print(f"Generated {len(community_reports_df)} community reports")
    return {
        'communities': communities_df,
        'community_reports': community_reports_df,
        'entities': entities_df
    }


def llm_model_run_graph_ingestion(driver, dataset_path):

    output_dir = os.path.join(os.getcwd(), 'src/ingestion_pipeline/graph_llm_ingestion/rag_files')
    results = process_text_to_graphrag_with_llm(dataset_path=dataset_path, output_dir= output_dir)
    entities_df = pd.read_parquet(os.path.join(output_dir, 'entities.parquet'))
    relationships_df = pd.read_parquet(os.path.join(output_dir, 'relationships.parquet'))
    text_units_df = pd.read_parquet(os.path.join(output_dir, 'text_units.parquet'))
    results = detect_communities(entities_df, relationships_df, output_dir, visualize=True, graph_prefix='LLM')
    communities_df = results['communities']
    communities_df.to_parquet(os.path.join(output_dir, 'communities.parquet'))
    print(f"Created {len(communities_df)} communities across 3 levels")
 
    entities_df = pd.read_parquet(os.path.join(output_dir, 'entities.parquet'))
    relationships_df = pd.read_parquet(os.path.join(output_dir, 'relationships.parquet'))
    text_units_df = pd.read_parquet(os.path.join(output_dir, 'text_units.parquet'))
    
    results_g = run_graph_analysis(results, entities_df, relationships_df, text_units_df, output_dir)

    if results_g:
        clear_graph(driver, graph_prefix="LLM")
        list_graph_prefixes(driver)
        try:
            entities_df = pd.read_parquet(os.path.join(output_dir, 'entities.parquet'))
            relationships_df = pd.read_parquet(os.path.join(output_dir, 'relationships.parquet')) 
            text_units_df = pd.read_parquet(os.path.join(output_dir, 'text_units.parquet')) 
            documents_df = pd.read_parquet(os.path.join(output_dir, 'documents.parquet')) 
            communities_df = pd.read_parquet(os.path.join(output_dir, 'communities.parquet')) 
            community_reports_df =  pd.read_parquet(os.path.join(output_dir, 'community_reports.parquet')) 

            success = import_complete_graph_data(
                driver,
                entities_df, 
                relationships_df, 
                text_units_df, 
                documents_df, 
                communities_df,
                community_reports_df,
                add_embeddings=True,
                graph_prefix="LLM"  
            )
            
            if success:
                print("LLM-base Successful import of all data to Neo4j")
            else:
                print("Import completed with some issues")
                
        except Exception as e:
            print(f"Error during import process: {e}")        
    results['number_of_nodes'] = len(entities_df)
    results['number_of_relationships'] = len(relationships_df)  
    return results         
    
    
    
    