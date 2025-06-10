from src.ingestion_pipeline.helper_functions import create_texts_data, gemini_llm
import os
import re
import json
import math
from typing import List, Dict 
import logging
import time
import traceback
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(os.getcwd(), '.env'), override=True)


def parse_response_to_json(response_text):
        # Look for JSON pattern
    json_pattern = r'```json\s*([\s\S]*?)\s*```'
    json_match = re.search(json_pattern, response_text)
        
    if json_match:
            # Extract JSON from code block
       json_str = json_match.group(1).strip()
       return json.loads(json_str)

def generate_question_answer_levels(text_list: List[str], 
                                    model: str = "gemini-2.5-pro-preview-03-25", 
                                    temperature: float = 0.7, 
                                    questions_per_level: int = 3,
                                    max_retries: int = 3,
                                    retry_delay_seconds: int = 2) -> Dict[str, List[Dict[str, str]]]:
    """
    Generates questions and detailed answers at three complexity levels based on a list of text chunks.

    Args:
        text_list (List[str]): A list of text chunks to analyze.
        model (str): The Gemini model to use.
        temperature (float): Temperature parameter for generation (0.0-1.0).
        questions_per_level (int): Number of question-answer pairs per level.

    Returns:
        Dict[str, List[Dict[str, str]]]: Dictionary containing lists of question-answer pairs
                                         for 'level_1', 'level_2', and 'level_3'.
    """


    num_chunks = len(text_list)
    # Create a descriptive string for the Level 2 range
    level_2_range_description = f"e.g., requiring integration of {max(2, num_chunks // 3)} to {max(3, num_chunks // 2 + 1)} specific chunks"

    # Format the input text chunks
    formatted_text = "\n\n---\n\n".join([f"Chunk {i+1}:\n{chunk}" for i, chunk in enumerate(text_list)])


    prompt_template = """
        You are an AI expert tasked with creating evaluation questions to differentiate between standard vector-based RAG and more advanced graph-based RAG systems. You are provided with **{num_chunks} distinct text chunks** separated by '---'. Your objective is to generate question-answer pairs at three complexity levels. Level 1 should be answerable by simple vector retrieval from a single chunk. **Levels 2 and 3 MUST be designed to be difficult for standard vector RAG** because they require understanding relationships, sequences, or comparisons across multiple, potentially non-adjacent or semantically dissimilar chunks, or require a holistic grasp of the entire dataset – tasks where graph RAG's ability to model connections explicitly is advantageous.

        ### Instructions:

        1.  **Level 1 QA Pairs (Easy for Vector RAG):**
            *   Generate exactly {questions_per_level} pair(s).
            *   Question requires information **explicitly stated within one single chunk**.
            *   Answer is a direct, detailed extraction from that *one* chunk. Vector search based on question keywords should easily find the relevant chunk.

        2.  **Level 2 QA Pairs (Challenging for Vector RAG / Target for Graph RAG - Requires Subset Synthesis/Relationships):**
            *   Generate exactly {questions_per_level} pair(s).
            *   Question **MUST** require integrating, comparing, tracing, or finding relationships (e.g., cause/effect, sequence, comparison) between information located in **SEVERAL distinct chunks ({level_2_range_description}), but NOT ALL {num_chunks} chunks**.
            *   **Focus on questions where simple semantic similarity of the *entire* chunks might fail.** For example, ask about a connection between a detail in Chunk A and a detail in Chunk F, where Chunks A and F discuss different sub-topics overall but contain related entities or events. Require *multi-step reasoning* or *synthesis* across these specific chunks.
            *   *Examples of target structures:* "How did the decision mentioned regarding [Concept X] influence the outcome reported for [Concept Y]?", "Compare the methodologies used for [Task A] and [Task B], based on their descriptions.", "Trace the development steps of [Product Z] as outlined across the updates."
            *   Answer must demonstrate this synthesis, combining information *only* from the necessary subset of chunks.

        3.  **Level 3 QA Pairs (Very Challenging for Vector RAG / Target for Graph RAG - Requires Holistic/All-Chunk Synthesis):**
            *   Generate exactly {questions_per_level} pair(s).
            *   Question **MUST** require a **holistic understanding, analysis of overarching themes, complex inference, or synthesis across ALL {num_chunks} provided chunks**. These questions probe the bigger picture, the system described, or conclusions that only emerge when *all* pieces are considered together. Simple vector retrieval of top-k chunks would likely miss necessary context or connections.
            *   **Focus on questions about emergent properties, complex interdependencies, or the overall narrative/structure.**
            *   *Examples of target structures:* "What is the central conflict or primary driving force described throughout the entire set of updates?", "Synthesize the complete lifecycle or process flow implied by combining information from all chunks.", "Based on all provided details, what underlying assumption or principle connects the various activities described?", "What are the most significant systemic risks or opportunities highlighted when considering all the information together?"
            *   Answer must be detailed, demonstrating this comprehensive synthesis across *all* chunks, grounded strictly in the provided text.

        ### Output Format:
        Strictly return ONLY a valid JSON object. No explanations, apologies, or text before or after the JSON object. Use double quotes for all keys and string values in the JSON.

        ```json
        {{
        "level_1": [
            {{
            "question": "Level 1 question...",
            "answer": "Detailed answer from one chunk..."
            }}
            // ... (repeat for {questions_per_level} total level 1 pairs)
        ],
        "level_2": [
            {{
            "question": "Level 2 question requiring synthesis/relationship across a subset of chunks...",
            "answer": "Detailed answer synthesizing info from specific multiple chunks..."
            }}
            // ... (repeat for {questions_per_level} total level 2 pairs)
        ],
        "level_3": [
            {{
            "question": "Level 3 question requiring holistic analysis across ALL chunks...",
            "answer": "Detailed answer demonstrating comprehensive synthesis based on ALL chunks..."
            }}
            // ... (repeat for {questions_per_level} total level 3 pairs)
        ]
        }}
        ```

        ### Additional Notes:
        *   Ensure all generated questions and answers are **relevant, accurate, and strictly based on the provided text chunks**.
        *   **Crucially: Avoid using any external knowledge whatsoever. All questions and answers must be fully derivable and grounded *only* within the provided text chunks.**
        *   Adhere exactly to the number of question-answer pairs requested per level ({questions_per_level}).
        *   The answers must be detailed and comprehensive, reflecting the complexity level and specified scope (single chunk, several chunks, all chunks).
        *   Do not include any explanations, introductions, or text outside the specified JSON structure.
        *   Avoid question structures like according to chunk (x - y) what are .... or which is .... or something similar, just go straight to the question it self(same thing applies for the answers) eg what are ... compare Concept A and B ... and so on.
        *   Do not give references of the chunks( either their number or position eg according to chunk 4 -6 what is ...., this is to be avoided in the both questions or answers), the chunk separation are for your internal analysis

        Text Chunks to Analyze:
        {input_text}
    """
    # Initialize the result structure
    result: Dict[str, List[Dict[str, str]]] = {
        "level_1": [],
        "level_2": [],
        "level_3": []
    }
    parsed_data = None 

    # Set up the LLM chain components
    try:
        prompt = ChatPromptTemplate.from_template(prompt_template)
        llm = gemini_llm(model_name=model, temperature=temperature)
        chain = prompt | llm | StrOutputParser()
    except Exception as e:
        logging.error(f"Failed to initialize LLM chain: {e}")
        traceback.print_exc()
        return result # Cannot proceed without the chain

    for attempt in range(max_retries):
        logging.info(f"Attempt {attempt + 1} of {max_retries} to generate QA levels.")
        response = None # Reset response for each attempt
        try:
            # --- Step 1: Invoke LLM Chain ---
            # This call might raise its own exceptions (network, API key, etc.)
            # which will *not* trigger our specific JSON retry, but will exit the attempt.
            response = chain.invoke({
                "input_text": formatted_text,
                "questions_per_level": questions_per_level,
                "num_chunks": num_chunks,
                "level_2_range_description": level_2_range_description
            })
            logging.debug(f"LLM Raw Response (Attempt {attempt + 1}):\n{response}")

            # --- Step 2: Attempt to Parse JSON ---
            # This is the specific point where failure triggers a retry
            parsed_data = parse_response_to_json(response)
            logging.info(f"Successfully parsed JSON on attempt {attempt + 1}.")
            break # Exit the loop on successful parsing

        except json.JSONDecodeError as e:
            logging.warning(f"Attempt {attempt + 1} failed: JSON Decode Error - {e}")
            #logging.debug(f"Failed response content: {response}") 
            if attempt < max_retries - 1:
                logging.info(f"Waiting {retry_delay_seconds} seconds before next attempt...")
                time.sleep(retry_delay_seconds)
            else:
                logging.error(f"Max retries ({max_retries}) reached. Failed to parse JSON response.")
                logging.error(f"Final failing response: {response}")

        except Exception as e:
            # Catch other potential errors during LLM call or unexpected issues
            logging.error(f"Attempt {attempt + 1} failed due to an unexpected error: {e}")
            traceback.print_exc()

            break

    # --- Step 3: Populate results if parsing succeeded ---
    if parsed_data is not None:
        result["level_1"] = parsed_data.get("level_1", []) if isinstance(parsed_data.get("level_1"), list) else []
        result["level_2"] = parsed_data.get("level_2", []) if isinstance(parsed_data.get("level_2"), list) else []
        result["level_3"] = parsed_data.get("level_3", []) if isinstance(parsed_data.get("level_3"), list) else []


        for level_key in ["level_1", "level_2", "level_3"]:
             if level_key not in parsed_data:
                 logging.warning(f"Parsed data is missing key: {level_key}")

    else:
        logging.error("Failed to obtain valid JSON data after all retries.")

    return result

def process_batched_qa_pairs(
    all_text_list: List[str],
    set_size: int,
    model: str = "gemini-2.5-pro-preview-03-25", 
    temperature: float = 0.7,     
    questions_per_level: int = 3
) -> Dict[str, List[Dict[str, str]]]:
    """
    Processes text chunks in batched sets to generate question-answer pairs
    at different complexity levels and accumulates the results.

    Args:
        all_text_list (List[str]): List of text chunks to process.
        set_size (int): Number of chunks per batch.
        model (str): LLM model name.
        temperature (float): Generation temperature.
        questions_per_level (int): The number of question-answer pairs for each level per batch.

    Returns:
        Dict[str, List[Dict[str, str]]]: Aggregated results containing lists of
                                         question-answer pairs, deduplicated by question text.
    """
    # Calculate number of batches needed
    num_batches = math.ceil(len(all_text_list) / set_size)
    if not all_text_list:
        print("Input text list is empty. No processing needed.")
        return {"level_1": [], "level_2": [], "level_3": []}
        
    print(f"Processing {len(all_text_list)} chunks in {num_batches} batches of up to {set_size} chunks each.")

    # Initialize aggregated results structure for question-answer pairs
    aggregated_results: Dict[str, List[Dict[str, str]]] = {
        "level_1": [],
        "level_2": [],
        "level_3": []
    }
    # Keep track of keys for iteration
    level_keys = list(aggregated_results.keys())

    # Process in batches
    for batch_idx in range(num_batches):
        start_idx = batch_idx * set_size
        end_idx = min(start_idx + set_size, len(all_text_list)) # Handle last batch size correctly
        batch = all_text_list[start_idx:end_idx]

        if not batch: 
            continue

        print(f"\n--- Processing Batch {batch_idx+1}/{num_batches} ({len(batch)} chunks) ---")

        # Generate question-answer pairs for the current batch
        # **** Call the NEW function ****
        batch_results = generate_question_answer_levels(
            text_list=batch,
            model=model,
            temperature=temperature,
            questions_per_level=questions_per_level
        )

        # Aggregate results
        print(f"Batch {batch_idx+1} results obtained:")
        for level_key in level_keys:
            batch_qa_pairs = batch_results.get(level_key, [])
            print(f"  - {level_key}: Found {len(batch_qa_pairs)} pairs")
            aggregated_results[level_key].extend(batch_qa_pairs)

    print("\n--- Aggregation Complete. Starting Deduplication... ---")

    # Deduplicate results across all batches based on the 'question' text
    deduplicated_results: Dict[str, List[Dict[str, str]]] = {level: [] for level in level_keys}
    #seen_questions_by_level: Dict[str, Set[str]] = {level: set() for level in level_keys}

    total_before_dedup = sum(len(qa_list) for qa_list in aggregated_results.values())
    print(f"Total pairs before deduplication: {total_before_dedup}")

    for level_key in level_keys:
        unique_pairs_for_level = []
        seen_questions_for_level = set() # Track questions seen *within this level*
        
        for qa_pair in aggregated_results[level_key]:
            # Basic validation of the pair structure
            if isinstance(qa_pair, dict) and "question" in qa_pair and "answer" in qa_pair:
                question_text = qa_pair["question"]
                # Add the pair only if its question hasn't been seen for this level yet
                if question_text not in seen_questions_for_level:
                    unique_pairs_for_level.append(qa_pair)
                    seen_questions_for_level.add(question_text)
            else:
                print(f"Warning: Skipping invalid/malformed item in '{level_key}' during deduplication: {qa_pair}")
        
        deduplicated_results[level_key] = unique_pairs_for_level
        print(f"  - {level_key}: Kept {len(unique_pairs_for_level)} unique pairs (based on question).")


    print("\n--- Processing Complete. Final Unique Counts: ---")
    for level_key in level_keys:
        print(f"- {level_key}: {len(deduplicated_results[level_key])} unique question-answer pairs")

    return deduplicated_results

def generate_eval_questions_answers(dataset_path, set_size, questions_per_level):
    all_text_list = create_texts_data(dataset_path)
    all_questions_answers = process_batched_qa_pairs(all_text_list=all_text_list, set_size=set_size, model="gemini-2.5-pro-preview-03-25", temperature=0.9, questions_per_level=questions_per_level)
    
    path_qa = os.path.join(os.getcwd(), 'src/evaluation/question_answer_generator/qa_data.json')
    
    with open(path_qa, 'w', encoding='utf-8') as json_file:
        json.dump(all_questions_answers, json_file, indent=4, ensure_ascii=False)
    
    print(f"Successfully saved QA data to: {path_qa}")
    
    return path_qa #,all_questions_answers,
    
    
    