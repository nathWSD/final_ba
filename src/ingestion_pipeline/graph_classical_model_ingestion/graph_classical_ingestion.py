from src.ingestion_pipeline.helper_functions import create_texts_data, robust_disambiguate_entities, clear_graph, list_graph_prefixes, detect_communities, import_complete_graph_data
import pandas as pd
import numpy as np
import uuid
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForMaskedLM, pipeline
import os
import hashlib
from collections import defaultdict
from datetime import datetime
from sklearn.decomposition import PCA
import json
import re
from typing import List, Dict, Any, Tuple, Set
import logging
import difflib
from transformers import Pipeline 
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv


load_dotenv(dotenv_path=os.path.join(os.getcwd(), '.env'), override=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
output_dir = os.path.join(os.getcwd(), 'src/ingestion_pipeline/graph_classical_model_ingestion/rag_files')

def generate_deterministic_id(text):
    """Generate a deterministic ID for content"""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def get_bert_contextual_description(text, entity, start_char, end_char, entity_type, tokenizer, model, context_window=100):
    """Generate a contextual description of an entity using BERT masked language modeling"""
    start_context = max(0, start_char - context_window)
    end_context = min(len(text), end_char + context_window)
    context = text[start_context:end_context]
    entity_start_in_context = start_char - start_context
    entity_end_in_context = entity_start_in_context + (end_char - start_char)
    context_with_mask = (
        context[:entity_start_in_context] + 
        tokenizer.mask_token + 
        context[entity_end_in_context:]
    )
    inputs = tokenizer(context_with_mask, return_tensors="pt", truncation=True)
    mask_idx = torch.where(inputs["input_ids"][0] == tokenizer.mask_token_id)[0]
    
    if len(mask_idx) == 0:
        return f"{entity_type} entity: {entity}. Found in context: \"{trim_to_complete_words(context, 50)}...\""
    
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    mask_token_logits = logits[0, mask_idx, :]
    top_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()
    predicted_tokens = [tokenizer.decode([token]).strip() for token in top_tokens]
    context_before = context[:entity_start_in_context].strip()
    context_after = context[entity_end_in_context:].strip()
    
    if len(context_before) > 60:
        context_before = " " + trim_to_complete_words(context_before, 60, trim_start=True)
    if len(context_after) > 60:
        context_after = trim_to_complete_words(context_after, 60) + " "
    
    description = f"{entity_type} entity: {entity}. Context: \"{context_before} [ENTITY] {context_after}\". "
    description += f"Based on context, this likely refers to: {', '.join(predicted_tokens)}."
    return description

def trim_to_complete_words(text, max_length, trim_start=False):
    """
    Trim text to the nearest complete word within max_length
    
    Args:
        text (str): The text to trim
        max_length (int): Maximum length to trim to
        trim_start (bool): If True, trim from the start; otherwise trim from the end
        
    Returns:
        str: Trimmed text ending at a word boundary
    """
    if len(text) <= max_length:
        return text
        
    if trim_start:
        trimmed_text = text[-max_length:]
        space_pos = trimmed_text.find(' ')
        
        if space_pos != -1:
            return trimmed_text[space_pos+1:]
        return trimmed_text
    else:
        trimmed_text = text[:max_length]
        space_pos = trimmed_text.rfind(' ')
        
        if space_pos != -1:
            return trimmed_text[:space_pos]
        return trimmed_text


def split_into_sentences(text:str)->List[str]:
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    return sentences

def parse_rebel_output(text):
    """Parse REBEL output to extract triplets"""
    triplets = []
    relation, subject, relation, object_ = '', '', '', ''
    text = text.strip()
    current = 'x'
    
    # Clean up the text
    text = text.replace("<s>", "").replace("<pad>", "").replace("</s>", "")
    
    for token in text.split():
        if token == "<triplet>":
            current = 't'
            if relation != '':
                triplets.append({'head': subject.strip(), 
                                'type': relation.strip(),
                                'tail': object_.strip()})
                relation = ''
            subject = ''
        elif token == "<subj>":
            current = 's'
            if relation != '':
                triplets.append({'head': subject.strip(), 
                                'type': relation.strip(),
                                'tail': object_.strip()})
            object_ = ''
        elif token == "<obj>":
            current = 'o'
            relation = ''
        else:
            if current == 't':
                subject += ' ' + token
            elif current == 's':
                object_ += ' ' + token
            elif current == 'o':
                relation += ' ' + token
    
    if subject != '' and relation != '' and object_ != '':
        triplets.append({'head': subject.strip(), 
                        'type': relation.strip(),
                        'tail': object_.strip()})
    
    return triplets

def filter_triplets(
    triplets: List[Dict[str, Any]],
    min_entity_len: int = 2 
) -> List[Dict[str, Any]]:
    """
    Filters a list of triplets based on multiple criteria:
    1. Removes triplets with missing 'head' or 'tail'.
    2. Removes triplets where the stripped 'head' or 'tail' text length
       is less than `min_entity_len`.
    3. Removes self-referential triplets where head and tail are the same
       (case-insensitive, after stripping).
    4. Keeps only the *first* occurrence of each unique (head, tail) pair
       (case-insensitive, after stripping).

    Args:
        triplets: A list of dictionaries, where each dictionary represents a triplet
                  and is expected to have 'head' and 'tail' keys.
        min_entity_len: The minimum character length required for both the head
                        and tail entity text after stripping whitespace. Defaults to 3.

    Returns:
        A new list containing the filtered triplets, preserving the order
        of the first valid occurrences.
    """
    filtered_list: List[Dict[str, Any]] = []
    seen_head_tail_pairs: Set[Tuple[str, str]] = set()

    if not isinstance(triplets, list):
        logging.error(f"Invalid input: 'triplets' must be a list, got {type(triplets)}")
        return []

    for i, triplet in enumerate(triplets):
        if not isinstance(triplet, dict):
            logging.warning(f"Skipping item at index {i}: not a dictionary. Item: {triplet}")
            continue

        head = triplet.get('head')
        tail = triplet.get('tail')

        # --- Filter 1: Missing essential parts ---
        if head is None or tail is None:
            logging.debug(f"Skipping triplet due to missing head/tail: {triplet}")
            continue

        # Ensure head and tail are strings, strip whitespace
        head_str = str(head).strip()
        tail_str = str(tail).strip()

        # --- Filter 2: Entity length ---
        if len(head_str) < min_entity_len or len(tail_str) < min_entity_len:
            logging.debug(f"Skipping triplet due to short entity (min_len={min_entity_len}): Head='{head_str}', Tail='{tail_str}'")
            continue

        # --- Filter 3: Self-referential (Head == Tail, Case-Insensitive) ---
        # Compare lowercased versions
        if head_str.lower() == tail_str.lower():
            logging.debug(f"Skipping self-referential triplet: Head='{head_str}', Tail='{tail_str}'")
            continue

        # --- Filter 4: Duplicate (Head, Tail) pair (Case-Insensitive) ---
        pair = (head_str.lower(), tail_str.lower())

        if pair not in seen_head_tail_pairs:
            seen_head_tail_pairs.add(pair)
            filtered_list.append(triplet)
        else:
            logging.debug(f"Skipping duplicate case-insensitive (head, tail) pair: {pair}")
            pass 
    return filtered_list



def get_ner_type_for_entity(
    entity_text: str,
    original_text: str,
    ner_pipeline: Pipeline,
    similarity_threshold: float = 0.85, 
    debug: bool = False 
) -> str:
    """
    Determines the NER type for a given entity text extracted by REBEL.

    It first runs NER on the original text. Then, it tries to find an NER
    prediction that exactly matches the entity_text. If found, its type is returned.
    If no exact match, it finds the NER prediction with the highest string
    similarity to the entity_text, above a given threshold.
    If no suitable match is found, it returns "REBEL_ENTITY".

    Args:
        entity_text: The text of the head or tail entity from REBEL.
        original_text: The full text from which the entity was extracted.
        similarity_threshold: The minimum SequenceMatcher ratio for a
                              similarity-based match (default: 0.85).
        debug: If True, prints matching attempts and results.

    Returns:
        The determined entity type (e.g., "ORG", "PER", "LOC") or
        "REBEL_ENTITY" as a fallback.
    """
    default_type = "REBEL_ENTITY"
    
    if not entity_text or not entity_text.strip():
        if debug: print("Debug: Empty entity_text provided.")
        return default_type

    entity_text = entity_text.strip()

    try:
        if debug: print(f"\nDebug: Running NER for entity '{entity_text}'...")
        ner_results = ner_pipeline(original_text)
        if debug: print(f"Debug: NER found {len(ner_results)} entities.")
        if not ner_results:
             if debug: print("Debug: NER returned no entities.")
             return default_type

    except Exception as e:
        print(f"Error during NER inference: {e}")
        return default_type 

    # --- 1. Exact Match Check ---
    exact_matches = []
    for ner_entity in ner_results:
        ner_word = ner_entity.get("word", "").strip()
        ner_type = ner_entity.get("entity_group")

        if ner_word and ner_type and ner_word == entity_text:
            exact_matches.append(ner_entity)

    if exact_matches:
        matched_entity = exact_matches[0]
        entity_type = matched_entity.get("entity_group", default_type)
        if debug:
            print(f"Debug: Exact match found for '{entity_text}'. Type: {entity_type}, Score: {matched_entity.get('score'):.4f}")
        return entity_type

    if debug: print(f"Debug: No exact text match found for '{entity_text}'. Checking similarity...")

    # --- 2. Similarity Match Check ---
    best_similarity = -1.0
    matched_type = default_type
    best_match_details = None

    for ner_entity in ner_results:
        ner_word = ner_entity.get("word", "").strip()
        ner_type = ner_entity.get("entity_group")

        if not ner_word or not ner_type:
            continue
        similarity = difflib.SequenceMatcher(None, ner_word, entity_text).ratio()

        if similarity > best_similarity:
            best_similarity = similarity
            if similarity >= similarity_threshold:
                 matched_type = ner_type 
                 best_match_details = ner_entity  
    if debug:
        if matched_type != default_type and best_match_details:
            print(f"Debug: Best similarity match for '{entity_text}': NER Word='{best_match_details.get('word')}', Type='{matched_type}', Similarity={best_similarity:.4f} (Threshold={similarity_threshold})")
        elif best_similarity >= 0:
             print(f"Debug: Highest similarity found for '{entity_text}' was {best_similarity:.4f}, which is below threshold {similarity_threshold}. Falling back to '{default_type}'.")
        else:
             print("Debug: No potential similarity matches found.")
    return matched_type


def extract_raw_triplets_from_sentence(
    sentence_text: str,
    rebel_model, rebel_tokenizer,
    ner_pipeline, 
    chunk_id: str
    ) -> list[dict]:
    """
    Extracts raw triplets from a single sentence using REBEL, performs basic filtering,
    and optionally adds NER types. Returns simple dictionaries.
    """
    raw_triplets_for_sentence = []

    # --- REBEL Extraction ---
    inputs = rebel_tokenizer(sentence_text, return_tensors="pt", max_length=512, truncation=True)
    try:
        with torch.no_grad():
             outputs = rebel_model.generate(
                 inputs["input_ids"],
                 attention_mask=inputs["attention_mask"],
                 max_length=512, 
                 num_beams=5,
                 num_return_sequences=1, 
                 output_scores=True,
                 return_dict_in_generate=True,
                 early_stopping=True 
             )
        sequence_scores = outputs.sequences_scores.tolist()
        decoded_preds = rebel_tokenizer.batch_decode(outputs.sequences, skip_special_tokens=False)
    except Exception as e:
        logging.error(f"REBEL generation failed for sentence: {sentence_text[:100]}... Error: {e}")
        return [] 

    # --- Parsing and Filtering ---
    for i, decoded_sentence in enumerate(decoded_preds):
        # Parse the <triplet> <subj> ... structure
        triplets = parse_rebel_output(decoded_sentence) 
        filtered_triplets = filter_triplets(triplets, min_entity_len=2)
        weight = float(sequence_scores[i]) if i < len(sequence_scores) else 0.0
        normalized_weight = round(min(1.0, max(0.0, float(np.exp(weight)))), 2) 

        for triplet in filtered_triplets:
            head = triplet.get('head')
            tail = triplet.get('tail')
            rel_type = triplet.get('type')

            if not head or not tail or not rel_type:
                continue

            #  Get NER types 
            head_type = get_ner_type_for_entity(entity_text=head, original_text=sentence_text, ner_pipeline=ner_pipeline, similarity_threshold=0.60)
            tail_type = get_ner_type_for_entity(entity_text=tail, original_text=sentence_text, ner_pipeline=ner_pipeline, similarity_threshold=0.60)

            # Add the raw triplet with context
            raw_triplets_for_sentence.append({
                'head': head,
                'tail': tail,
                'type': rel_type,
                'weight': normalized_weight,
                'chunk_id': chunk_id,
                'head_type': head_type, 
                'tail_type': tail_type  
            })
    return raw_triplets_for_sentence


def aggregate_and_finalize_graph(
    all_raw_triplets: List[Dict[str, Any]],
    chunk_processing_info: List[Dict[str, Any]],
    document_id: str,
    collection_name: str,
    output_dir: str,
    bert_tokenizer: Any, 
    bert_model: Any,     
    chunk_id_to_text_map: Dict[str, str]
) -> Dict[str, pd.DataFrame]:
    """
    Aggregates raw triplets, generates descriptions, finalizes entities and
    relationships, creates GraphRAG DataFrames, and saves them to parquet files.
    This version does NOT perform automatic fuzzy normalization.

    Args:
        all_raw_triplets: List of raw triplet dictionaries extracted in Pass 1.
        chunk_processing_info: List of dictionaries containing info about each chunk.
        document_id: The unique ID for the document being processed.
        collection_name: The base name for the collection/document title.
        output_dir: The directory to save the final parquet files.
        bert_tokenizer: Loaded BERT tokenizer.
        bert_model: Loaded BERT model.
        chunk_id_to_text_map: A mapping from chunk ID to chunk text.

    Returns:
        A dictionary containing the final pandas DataFrames:
        {'entities': entities_df, 'relationships': relationships_df,
         'text_units': text_units_df, 'documents': documents_df}
    """
    logging.info("Starting Graph Aggregation and Finalization...")

    entities_accumulator = {}
    normalized_relationships = []
    entity_frequency = defaultdict(int)
    entity_chunks = defaultdict(set)
    logging.info("Aggregating entities and relationships...")

    for raw_triplet in all_raw_triplets:
        original_head = raw_triplet.get('head')
        original_tail = raw_triplet.get('tail')
        chunk_id = raw_triplet.get('chunk_id')

        if not original_head or not original_tail or not chunk_id:
            logging.debug(f"Skipping incomplete raw triplet: {raw_triplet}")
            continue
        
        norm_head = original_head
        norm_tail = original_tail

        if norm_head == norm_tail: 
            logging.debug(f"Skipping self-loop: {norm_head} -> {norm_tail}")
            continue

        norm_head_upper = norm_head.upper()
        norm_tail_upper = norm_tail.upper()
        rel_type = raw_triplet.get('type', 'related to')
        rel_weight = raw_triplet.get('weight', 0.5) 

        for entity_title_upper, original_entity_text, entity_type_hint in [
                (norm_head_upper, original_head, raw_triplet.get('head_type', 'UNKNOWN')),
                (norm_tail_upper, original_tail, raw_triplet.get('tail_type', 'UNKNOWN'))]:

            entity_frequency[entity_title_upper] += 1
            entity_chunks[entity_title_upper].add(chunk_id)

            if entity_title_upper not in entities_accumulator:
                context_chunk_id = chunk_id 
                context_chunk_text = chunk_id_to_text_map.get(context_chunk_id, "")
                first_original_mention = original_entity_text
                bert_description = f"Entity representing '{entity_title_upper}'" 

                if context_chunk_text:
                    search_mention_lower = first_original_mention.lower()
                    context_text_lower = context_chunk_text.lower()
                    start_char_lower = context_text_lower.find(search_mention_lower)

                    if start_char_lower != -1:
                        start_char = start_char_lower
                        end_char = start_char + len(first_original_mention) 
                        try:
                            bert_description = get_bert_contextual_description(
                                text=context_chunk_text,
                                entity=first_original_mention,
                                start_char=start_char,
                                end_char=end_char,
                                entity_type=entity_type_hint,
                                tokenizer=bert_tokenizer,
                                model=bert_model
                            )
                        except NameError:
                             logging.error("`get_bert_contextual_description` function not found!")
                             bert_description = f"Description generation error for '{entity_title_upper}'"
                        except Exception as e:
                            logging.error(f"Failed to generate BERT description for '{first_original_mention}' (approx context) in chunk {context_chunk_id}: {e}")
                            bert_description = f"Error generating description for '{entity_title_upper}'"
                    else:                       
                        bert_description = f"Contextual description unavailable for '{entity_title_upper}' (mention not found)"
                else:
                    logging.warning(f"Could not find chunk text for chunk_id {context_chunk_id} to generate description.")
                    bert_description = f"Contextual description unavailable for '{entity_title_upper}' (chunk text missing)"

                entities_accumulator[entity_title_upper] = {
                    'id': str(uuid.uuid4()),
                    'title': entity_title_upper,
                    'type': entity_type_hint,
                    'description': bert_description,
                    'text_unit_ids': set(), 
                    'frequency': 0,         
                    'degree': 0,            
                 }
        # --- Create Relationship ---
        relationship = {
            'id': str(uuid.uuid4()),
            'source': norm_head_upper,
            'target': norm_tail_upper,
            'description': rel_type,
            'type': rel_type,
            'weight': rel_weight,
            'text_unit_ids': {chunk_id}, 
            'combined_degree': 0 
        }
        normalized_relationships.append(relationship)

    # ---  Entities ---
    logging.info("Finalizing entity list...")
    final_entities_list = []
    for entity_title_upper, data in entities_accumulator.items():
        data['frequency'] = entity_frequency[entity_title_upper]
        data['text_unit_ids'] = list(entity_chunks[entity_title_upper])
        final_entities_list.append(data)

    # ---  Relationships ---
    logging.info("Finalizing relationship list...")
    for rel in normalized_relationships:
        rel['text_unit_ids'] = list(rel['text_unit_ids'])

    logging.info("Calculating entity degrees...")
    entity_degrees = defaultdict(int)
    if final_entities_list: 
        final_entity_titles = {e['title'] for e in final_entities_list}
        for rel in normalized_relationships:
            if rel['source'] in final_entity_titles:
                 entity_degrees[rel['source']] += 1
            if rel['target'] in final_entity_titles:
                 entity_degrees[rel['target']] += 1

    for entity_data in final_entities_list:
        entity_data['degree'] = entity_degrees.get(entity_data['title'], 0)

    for rel in normalized_relationships:
        rel['combined_degree'] = entity_degrees.get(rel['source'], 0) + entity_degrees.get(rel['target'], 0)

    # --- Create DataFrames ---
    logging.info("Creating entities and relationships DataFrames...")
    entities_df = pd.DataFrame(final_entities_list)
    if not entities_df.empty:
        entities_df['human_readable_id'] = range(len(entities_df))
        entities_df['x'] = 0.0
        entities_df['y'] = 0.0
    else:
        entities_df = pd.DataFrame(columns=['id', 'title', 'type', 'description', 'text_unit_ids', 'frequency', 'degree', 'human_readable_id', 'x', 'y'])

    relationships_df = pd.DataFrame(normalized_relationships)
    if not relationships_df.empty:
        relationships_df['human_readable_id'] = range(len(relationships_df))
        if 'type' not in relationships_df.columns:
             relationships_df['type'] = relationships_df['description']
    else:
        relationships_df = pd.DataFrame(columns=['id', 'human_readable_id', 'source', 'target', 'description', 'type', 'weight', 'combined_degree', 'text_unit_ids'])

    logging.info("Creating text_units DataFrame...")
    text_units = []
    entity_title_to_id_map = entities_df.set_index('title')['id'].to_dict() if not entities_df.empty else {}

    chunk_to_final_rel_ids = defaultdict(list)
    for _, rel_row in relationships_df.iterrows():
        rel_id = rel_row['id']
        for chunk_id in rel_row['text_unit_ids']:
             chunk_to_final_rel_ids[chunk_id].append(rel_id)

    final_entity_chunks_in_df = defaultdict(set)
    for _, entity_row in entities_df.iterrows():
         title = entity_row['title']
         for chunk_id in entity_row['text_unit_ids']:
              final_entity_chunks_in_df[title].add(chunk_id)


    for chunk_info in chunk_processing_info:
        chunk_id = chunk_info['id']
        chunk_canonical_entities = {title for title, chunks in final_entity_chunks_in_df.items() if chunk_id in chunks}
        chunk_entity_ids = [entity_title_to_id_map[title] for title in chunk_canonical_entities if title in entity_title_to_id_map]
        chunk_rel_ids = chunk_to_final_rel_ids.get(chunk_id, [])

        text_units.append({
            'id': chunk_id,
            'human_readable_id': chunk_info['human_readable_id'],
            'text': chunk_info['text'],
            'n_tokens': len(chunk_info['text'].split()),
            'document_ids': chunk_info['document_ids'],
            'entity_ids': chunk_entity_ids,
            'relationship_ids': chunk_rel_ids,
            'covariate_ids': [] 
        })

    text_units_df = pd.DataFrame(text_units)
    if text_units_df.empty:
         text_units_df = pd.DataFrame(columns=['id', 'human_readable_id', 'text', 'n_tokens', 'document_ids', 'entity_ids', 'relationship_ids', 'covariate_ids'])

    # --- Create documents_df ---
    logging.info("Creating documents DataFrame...")
    all_final_text_unit_ids = list(text_units_df['id'])
    current_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S %z")
    full_text = "".join([info['text'] for info in chunk_processing_info])

    documents_df = pd.DataFrame([{
        'id': document_id,
        'human_readable_id': 1, 
        'title': collection_name,
        'text': full_text,
        'text_unit_ids': all_final_text_unit_ids,
        'creation_date': current_timestamp,
        'metadata': None 
    }])

    # --- Save Parquet Files ---
    logging.info(f"Saving parquet files to {output_dir}...")
    entities_df.to_parquet(os.path.join(output_dir, 'entities.parquet'), index=False)
    relationships_df.to_parquet(os.path.join(output_dir, 'relationships.parquet'), index=False)
    text_units_df.to_parquet(os.path.join(output_dir, 'text_units.parquet'), index=False)
    documents_df.to_parquet(os.path.join(output_dir, 'documents.parquet'), index=False)

    logging.info(f"Created {len(entities_df)} final entities.")
    logging.info(f"Created {len(relationships_df)} final relationships.")
    logging.info(f"Created {len(text_units_df)} text units.")
    return {
        'entities': entities_df,
        'relationships': relationships_df,
        'text_units': text_units_df,
        'documents': documents_df
    }
    

def process_text_to_graphrag(texts: List[str], output_dir: str) -> Dict[str, pd.DataFrame]:
    """
    Processes text chunks to extract raw triplets using REBEL/NER, then calls
    a separate function to aggregate triplets, generate BERT descriptions,
    create final GraphRAG compatible parquet files.
    Does NOT perform automatic fuzzy normalization.

    Args:
        texts: A list of strings, where each string is a text chunk.
        output_dir: The directory where the final parquet files will be saved.

    Returns:
        A dictionary containing the final pandas DataFrames:
        {'entities': ..., 'relationships': ..., 'text_units': ..., 'documents': ...}
        Returns an empty dictionary if no raw triplets are extracted.
    """
    os.makedirs(output_dir, exist_ok=True)
    start_time = datetime.now()

    # --- Load Models ---
    logging.info("Loading BERT model for descriptions...")
    try:
        bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        bert_model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")
        bert_model.eval()
    except Exception as e:
        logging.error(f"Failed to load BERT models: {e}")
        raise
    logging.info("Loading REBEL models...")
    try:
        rebel_tokenizer = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
        rebel_model = AutoModelForSeq2SeqLM.from_pretrained("Babelscape/rebel-large")
    except Exception as e:
        logging.error(f"Failed to load REBEL models: {e}")
        raise 

    logging.info("Loading NER pipeline...")
    try:
        ner_pipeline = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
    except Exception as e:
        logging.error(f"Failed to load NER pipeline: {e}")
        raise
    # --- Document Setup ---
    if not texts:
        logging.error("Input 'texts' list is empty. Cannot proceed.")
        return {}
    try:
        document_id = generate_deterministic_id(texts[0])
    except NameError:
        logging.error("`generate_deterministic_id` function not found!")
        document_id = "fallback_doc_id_" + datetime.now().isoformat()
    collection_name = os.getenv('COLLECTION_BASE_NAME', 'DefaultCollection') 

    # --- Pass 1: Raw Extraction ---
    logging.info("Starting Pass 1: Extracting raw triplets...")
    all_raw_triplets = []
    chunk_processing_info = []

    for i, chunk_text in enumerate(texts):
        if not chunk_text or not chunk_text.strip():
            logging.warning(f"Skipping empty chunk {i+1}/{len(texts)}")
            continue

        logging.info(f"Processing chunk {i+1}/{len(texts)}")
        try:
            chunk_id = generate_deterministic_id(chunk_text)
        except NameError:
            logging.error("`generate_deterministic_id` function not found!")
            chunk_id = f"fallback_chunk_{i+1}_" + datetime.now().isoformat()
        chunk_processing_info.append({
            'id': chunk_id,
            'text': chunk_text,
            'human_readable_id': i + 1,
            'document_ids': [document_id]
        })

        try:
            sentences = split_into_sentences(chunk_text)
        except NameError:
            logging.error("`split_into_sentences` function not found! Processing chunk as single sentence.")
            sentences = [chunk_text]

        for sentence in sentences:
            if not sentence or not sentence.strip(): continue

            try:
                raw_sentence_triplets = extract_raw_triplets_from_sentence(
                    sentence_text=sentence,
                    rebel_model=rebel_model,
                    rebel_tokenizer=rebel_tokenizer,
                    ner_pipeline=ner_pipeline,
                    chunk_id=chunk_id
                )
                all_raw_triplets.extend(raw_sentence_triplets)
            except NameError:
                logging.error("`extract_raw_triplets_from_sentence` function not found!")
                break 
            except Exception as e:
                logging.error(f"Error extracting triplets from sentence in chunk {i+1}: {e}")

    logging.info(f"Pass 1 Complete: Extracted {len(all_raw_triplets)} raw triplets.")

    all_raw_triplets = robust_disambiguate_entities(triplets = all_raw_triplets)
    
    if not all_raw_triplets:
        logging.warning("No raw triplets were extracted after Pass 1. Cannot proceed.")
        return {}
    chunk_id_to_text_map = {info['id']: info['text'] for info in chunk_processing_info}
    logging.info("Calling graph aggregation and finalization process...")
    try:
        final_graph_data = aggregate_and_finalize_graph(
            all_raw_triplets=all_raw_triplets,
            chunk_processing_info=chunk_processing_info,
            document_id=document_id,
            collection_name=collection_name,
            output_dir=output_dir,
            bert_tokenizer=bert_tokenizer,
            bert_model=bert_model,
            chunk_id_to_text_map=chunk_id_to_text_map
        )
    except NameError:
         logging.error("`aggregate_and_finalize_graph` function not found! Cannot finalize graph.")
         return {}
    except Exception as e:
        logging.error(f"Error during graph aggregation and finalization: {e}")
        return {} 

    end_time = datetime.now()
    logging.info(f"Full processing complete in {end_time - start_time}")
    return final_graph_data


def generate_community_reports(communities_df, entities_df, relationships_df, text_units_df, output_dir):
    """Generate summarized reports for each community with dynamic length handling"""
    print("Generating community reports...")
    summarizer_model = "sshleifer/distilbart-cnn-12-6"
    tokenizer = AutoTokenizer.from_pretrained(summarizer_model)
    # Load summarization model
    summarizer = pipeline("summarization", model=summarizer_model,tokenizer=tokenizer)
    
    community_reports = []
    
    for _, community in communities_df.iterrows():
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
            community_relationships = relationships_df.iloc[0:0].copy()  # Empty DataFrame
        
        context_parts = []
        
        context_parts.append("Entities in this community:")
        for _, entity in community_entities.iterrows():
            context_parts.append(f"- {entity['title']}: {entity['description']}")
        
        context_parts.append("\nRelationships in this community:")
        for _, rel in community_relationships.iterrows():
            context_parts.append(f"- {rel['source']} {rel['description']} {rel['target']}")
        
        context = "\n".join(context_parts)
        try:
            if len(community_entities) > 0 and 'degree' in community_entities.columns:
                central_entities = community_entities.nlargest(min(3, len(community_entities)), 'degree')
                if len(central_entities) > 0:
                    title_entities = central_entities['title'].tolist()
                    community_title = " and ".join(title_entities)
                else:
                    community_title = f"Community {community['community']}"
            else:
                community_title = f"Community {community['community']}"
        except Exception as e:
            print(f"Error creating title for community {community['community']}: {e}")
            community_title = f"Community {community['community']}"
        try:
            input_length = len(context.split())
            
            if input_length > 300: 
                max_output_length = min(150, max(50, input_length // 2))
                min_output_length = max(20, max_output_length // 3)
                
                summary = summarizer(
                    context, 
                    max_length=max_output_length, 
                    min_length=min_output_length, 
                    do_sample=False,
                    truncation=True 
                )[0]['summary_text']
            else:
                entity_count = len(community_entities)
                rel_count = len(community_relationships)
                if entity_count > 0 and 'type' in community_entities.columns:
                    entity_types_list = community_entities['type'].tolist()
                    type_counts = {}
                    for t in entity_types_list:
                        if t in type_counts:
                            type_counts[t] += 1
                        else:
                            type_counts[t] = 1
                    sorted_types = sorted(type_counts.items(), key=lambda x: x[1], reverse=True)[:3]
                    main_types = [f"{count} {type_name}" for type_name, count in sorted_types]
                    type_description = ", ".join(main_types)
                    summary = f"This community contains {entity_count} entities ({type_description}) with {rel_count} relationships. The main entities include {community_title}."
                else:
                    summary = f"This small community contains entities related to {community_title}."
        except Exception as e:
            print(f"Error summarizing community {community['community']}: {e}")
            summary = f"This community contains {len(entity_ids)} entities and {len(relationship_ids)} relationships."
        findings = []
        if len(community_entities) > 0 and 'degree' in community_entities.columns:
            try:
                max_degree_idx = community_entities['degree'].idxmax()
                central_entity = community_entities.loc[max_degree_idx]
                findings.append({
                    'explanation': f"{central_entity['title']} is a key entity in this community with {central_entity['degree']} connections."
                })
            except Exception as e:
                print(f"Error finding central entity: {e}")
        if len(community_relationships) > 0 and 'weight' in community_relationships.columns:
            try:
                max_weight_idx = community_relationships['weight'].idxmax()
                strongest_rel = community_relationships.loc[max_weight_idx]
                findings.append({
                    'explanation': f"The strongest relationship is between {strongest_rel['source']} and {strongest_rel['target']} ({strongest_rel['description']})."
                })
            except Exception as e:
                print(f"Error finding strongest relationship: {e}")
        if len(community_entities) > 0 and 'type' in community_entities.columns:
            try:
                entity_types_list = community_entities['type'].tolist()
                type_counts = {}
                for t in entity_types_list:
                    if t in type_counts:
                        type_counts[t] += 1
                    else:
                        type_counts[t] = 1
                
                if type_counts:
                    most_common_type = max(type_counts.items(), key=lambda x: x[1])
                    findings.append({
                        'explanation': f"The most common entity type is {most_common_type[0]} with {most_common_type[1]} instances."
                    })
            except Exception as e:
                print(f"Error finding most common entity type: {e}")
        
        full_content = f"# {community_title}\n\n{summary}\n\n## Entities\n\n"
        for _, entity in community_entities.iterrows():
            entity_type = entity.get('type', 'Unknown')
            full_content += f"- **{entity['title']}** ({entity_type}): {entity['description']}\n"
        
        full_content += "\n## Key Relationships\n\n"
        for _, rel in community_relationships.head(10).iterrows():
            full_content += f"- {rel['source']} → {rel['description']} → {rel['target']}\n"
        try:
            avg_entity_degree = community_entities['degree'].mean() if len(community_entities) > 0 and 'degree' in community_entities.columns else 0
            avg_relationship_weight = community_relationships['weight'].mean() if len(community_relationships) > 0 and 'weight' in community_relationships.columns else 0
            
            importance_score = (0.7 * min(10, avg_entity_degree / 2)) + (0.3 * min(10, avg_relationship_weight * 10))
            importance_score = round(max(1, min(10, importance_score)), 1) 
        except Exception as e:
            print(f"Error calculating importance score: {e}")
            importance_score = 5.0  
        
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
            
            if len(community_relationships) > 0:
                full_content_json["relationships"] = []
                for _, rel in community_relationships.iterrows():
                    rel_json = {
                        "source": rel['source'],
                        "target": rel['target'],
                        "type": rel['description']
                    }
                    if 'weight' in rel:
                        rel_json["weight"] = float(rel['weight'])
                    full_content_json["relationships"].append(rel_json)
            
            full_content_json["findings"] = [finding['explanation'] for finding in findings]
            
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
            'rating_explanation': f"The impact severity rating is {importance_score >= 7.5 and 'high' or 'medium'} due to the {len(entity_ids)} entities and {len(relationship_ids)} relationships in this community.",
            'findings': findings,
            'full_content_json': json_str,
            'period': datetime.now().strftime('%Y-%m-%d'),
            'size': community.get('size', len(entity_ids))
        })
    
    try:
        community_title_map = {row['community']: row['title'] for row in community_reports}
        
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
    community_reports_df = generate_community_reports(communities_df, entities_df, relationships_df, text_units_df, output_dir)
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


def classical_model_run_graph_ingestion(driver, dataset_path):
    print(" starting classical ingestion to neo4j.....")
    dataset_text = create_texts_data(dataset_path)
    results = process_text_to_graphrag(dataset_text, output_dir)
    print("====== End of text process analysis =======")
    entities_df = pd.read_parquet(os.path.join(output_dir, 'entities.parquet'))
    relationships_df=pd.read_parquet(os.path.join(output_dir, 'relationships.parquet'))
    results = detect_communities(entities_df, relationships_df, output_dir, visualize=True, graph_prefix='DL')
    communities_df = results['communities']
    communities_df.to_parquet(os.path.join(output_dir, 'communities.parquet'))
    logging.info(f"Created {len(communities_df)} communities across 3 levels")

        # Load generated data
    entities_df = pd.read_parquet(os.path.join(output_dir, 'entities.parquet'))
    relationships_df = pd.read_parquet(os.path.join(output_dir, 'relationships.parquet'))
    text_units_df = pd.read_parquet(os.path.join(output_dir, 'text_units.parquet'))
    
    logging.info("================= Community Reports done ===========:")
        # Run graph analysis
    results_g = run_graph_analysis(results, entities_df, relationships_df, text_units_df, output_dir)
    logging.info("================= DL graph analysis done ===========:")

    if results_g:
        clear_graph(driver, graph_prefix="DL")
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
                graph_prefix="DL"  
            )
            
            if success:
                logging.info("Successfully imported all data to Neo4j")
            else:
                print("Import completed with some issues")
                
        except Exception as e:
            print(f"Error during import process: {e}")   
    results['number_of_nodes'] = len(entities_df)
    results['number_of_relationships'] = len(relationships_df)
    return results    
        
        