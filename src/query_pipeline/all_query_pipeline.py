from src.query_pipeline.vector_retriever.vector_retriever import all_vector_retrieve
from src.query_pipeline.graph_retriever.graph_retriever import classical_and_llm_run_graph_search
from src.ingestion_pipeline.helper_functions import gemini_llm
from src.evaluation.generate_eveluation import run_evaluation_metric
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import logging
from typing import Any, Dict, List
import json
from langchain_core.outputs import LLMResult, ChatGeneration
from sentence_transformers import SentenceTransformer

# LangChain callback system
from langchain_core.callbacks import BaseCallbackHandler
from src.ingestion_pipeline.helper_functions import TimeMeasurer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# --- Callback Handler to Capture Token Usage ---
class GeminiTokenUsageCallbackHandler(BaseCallbackHandler):
    """Captures token usage (input, output, total) from ChatGoogleGenerativeAI."""
    def __init__(self):
        super().__init__()
        self.prompt_tokens = 0
        self.completion_tokens = 0 # Maps to output_tokens
        self.total_tokens = 0

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Reset counters when a new LLM call starts with this handler."""
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Collect token usage from AIMessage's usage_metadata within generations."""
        if not response.generations or not response.generations[0]:
            logging.warning("No generations found in LLMResult.")
            return

        usage_found = False
        for gen in response.generations[0]:
            if not isinstance(gen, ChatGeneration):
                continue

            # Access usage_metadata on the message attribute
            if hasattr(gen, 'message') and hasattr(gen.message, 'usage_metadata'):
                usage_metadata = gen.message.usage_metadata
                if usage_metadata and isinstance(usage_metadata, dict):
                    prompt_tokens = usage_metadata.get("input_tokens")
                    completion_tokens = usage_metadata.get("output_tokens")
                    total_tokens = usage_metadata.get("total_tokens")

                    self.prompt_tokens += prompt_tokens
                    self.completion_tokens += completion_tokens
                    self.total_tokens += total_tokens

                    logging.info(f"LLM Token Usage recorded: Prompt={prompt_tokens}, Completion={completion_tokens}, Total={total_tokens}")
                    usage_found = True
                    break # Exit loop once usage is found

        if not usage_found:
             logging.warning(f"Could not find usage_metadata in any generation message within LLMResult: {response.generations}")


    def get_usage_dict(self) -> Dict[str, int]:
        """Returns the collected token usage (input, output, total)."""
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }
        
def generate_response_langchain(query, context_, context_name, model):
    
    template = """
        Your task is to answer the question using ONLY the information explicitly present in the provided context below.

        **Constraints:**
        1.  **Strict Context Grounding:** Base your entire answer strictly on the text provided in the 'Context' section. Do not add external facts, assumptions, or use any pre-existing knowledge.
        2.  **Limited Synthesis:** Do not synthesize broad conclusions or infer relationships that are not directly stated or demonstrated within the provided text snippets. Stick closely to the information given. Combining directly stated facts from the context to answer the question is allowed, but avoid making leaps in logic or filling gaps.
        3.  **Partial Answers Required:** Answer the question as completely as possible based *solely* on the context. If the context only provides enough information to answer *part* of the question, then you MUST provide that partial answer.
        4.  **Handle Complete Absence:** If the context contains NO relevant information to answer ANY part of the question or the entire question, respond ONLY with: "The provided context does not contain relevant information to answer this question."

        Context:
        ====== context ======
        {context}
        ====== context ======

        Question:
        ====== user question ======
        {question}
        ====== user question ======

        Answer based strictly on context:
    """
    measurer = TimeMeasurer()

    custom_rag_prompt = PromptTemplate.from_template(template)

    token_callback = GeminiTokenUsageCallbackHandler()

    llm = gemini_llm(model_name=model, temperature=0.0)

    rag_chain = (
        {"context": lambda _: context_, "question": RunnablePassthrough()}
        | custom_rag_prompt
        | llm
        | StrOutputParser()
    )
     
    response = ""
    usage_data = {}

    try:
        # Use the context manager to time the invoke call
        with measurer.measure('llm_invocation_time'):
            response = rag_chain.invoke(
                query,
                config={"callbacks": [token_callback]}
            )
        logging.info("RAG chain invocation complete.")
        usage_data = token_callback.get_usage_dict()

    except Exception as e:
        logging.error(f"Error during RAG chain invocation: {e}", exc_info=True)
        response = f"Error: Failed to generate response due to: {e}"
        usage_data = token_callback.get_usage_dict() # Get potentially partial token data

    
    invocation_time = measurer.get_timing('llm_invocation_time')
    usage_data['time_taken'] = invocation_time

    logging.info(f"{context_name} Usage Data (with response time): {usage_data}")

    return response, usage_data



def create_evaluation_object(
    qa_level: str,
    query: str,
    expected_output: str,
    retrieval_package: Dict[str, Any], 
    llm_response: str,
    llm_usage_data: Dict[str, Any] 
    ) -> Dict[str, Any]:
    """
    Assembles the final evaluation object by combining retrieval and generation results.

    Args:
        query: The original user query.
        expected_output: The target answer for evaluation.
        retrieval_package: The dictionary returned by a retrieval function
                           (e.g., dense_package, global_DL_results). Must contain
                           'context' and 'retrieval_time_taken' keys.
        llm_response: The actual response generated by the LLM for the given context.
        llm_usage_data: The dictionary returned by generate_response_langchain,
                        containing token counts and LLM execution time ('time_taken').

    Returns:
        A dictionary structured for evaluation ('response_obj').
    """
    response_obj = {}

    # --- Basic Info ---
    response_obj['query'] = query
    response_obj['level'] = qa_level
    response_obj['expected_output'] = expected_output
    response_obj['llm_response'] = llm_response

    # --- Context ---
    # Extract context used for this specific generation run
    retrieved_context_str = retrieval_package.get('context')
    # Store context as a list, as per original structure request
    response_obj['retrieval_context'] = [retrieved_context_str]

    # --- Token Usage ---
    # Extract from the LLM usage data dictionary
    response_obj['num_input_token'] = llm_usage_data.get('prompt_tokens') # input tokens
    response_obj['num_output_token'] = llm_usage_data.get('completion_tokens') # output tokens
    response_obj['total_tokens'] = llm_usage_data.get('total_tokens') # total_tokens tokens

    # --- Timing ---
    retrieval_time = retrieval_package.get('retrieval_time_taken', 0.0)
    llm_generation_time = llm_usage_data.get('time_taken', 0.0) # Time measured inside generate_response

    # Calculate total time for this specific retrieval+generation path
    total_time = retrieval_time + llm_generation_time
    response_obj['time_taken'] = total_time # Store the combined time

    logging.info(f"Created evaluation object. Retrieval: {retrieval_time:.4f}s, LLM: {llm_generation_time:.4f}s, Total: {total_time:.4f}s")

    return response_obj

def read_and_flatten_qa(json_path: str) -> List[Dict[str, str]]:
    """
    Reads a JSON file containing QA pairs structured by levels and flattens
    it into a single list of QA dictionaries.

    Args:
        json_path: Path to the JSON file. The file should contain a dictionary
                   where keys are level names (e.g., 'level_1', 'level_2') and
                   values are lists of QA pair dictionaries [{'question': ..., 'answer': ...}].

    Returns:
        A single list containing all valid QA pair dictionaries found across all levels.
        Each dictionary in the list will have 'question' and 'answer' keys.
        Returns an empty list if the file is not found, is invalid JSON,
        or contains no valid QA pairs in the expected structure.
    """
    flat_qa_list: List[Dict[str, str]] = []
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not isinstance(data, dict):
            logging.error(f"Error: Content in '{json_path}' is not a JSON object (dictionary).")
            return []

        logging.info(f"Reading QA data from '{json_path}'...")
        # Iterate through the values of the top-level dictionary (which should be lists)
        for level_key, list_of_pairs in data.items():
            if isinstance(list_of_pairs, list):
                # Iterate through the dictionaries within the list
                for qa_pair in list_of_pairs:
                    # Check if it's a dictionary and has the required keys with non-empty string values
                    if (isinstance(qa_pair, dict) and
                            'question' in qa_pair and isinstance(qa_pair['question'], str) and qa_pair['question'] and
                            'answer' in qa_pair and isinstance(qa_pair['answer'], str) and qa_pair['answer']):
                        # Append a new dictionary containing only the essential keys
                        flat_qa_list.append({
                            'question': qa_pair['question'],
                            'answer': qa_pair['answer'],
                            'level': level_key
                        })
                    else:
                        logging.warning(f"Skipping invalid or incomplete QA pair in level '{level_key}': {qa_pair}")
            else:
                logging.warning(f"Skipping level '{level_key}' because its value is not a list.")

        logging.info(f"Successfully flattened {len(flat_qa_list)} QA pairs.")

    except FileNotFoundError:
        logging.error(f"Error: QA file not found at '{json_path}'.")
        return []
    except json.JSONDecodeError as e:
        logging.error(f"Error: Could not decode JSON from '{json_path}'. Invalid JSON format. Details: {e}")
        return []
    except Exception as e:
        logging.error(f"An unexpected error occurred while processing '{json_path}': {e}", exc_info=True)
        return []

    return flat_qa_list
    
def retriever_and_metrics_analysis_pipeline(driver, json_path):    
    num_results = 4  
    save_path_all_evaluation = os.path.join(os.getcwd(), 'src/evaluation/pipeline_evaluation_data')
    logging.info(f"loading embedding model {os.getenv('DENSE_MODEL_KEY')} ........")        
    embedding_model = SentenceTransformer(os.getenv('DENSE_MODEL_KEY'))
    logging.info(f"loading embedding model {os.getenv('DENSE_MODEL_KEY')} done")        

    # Empty the folder
    for filename in os.listdir(save_path_all_evaluation):
        file_path = os.path.join(save_path_all_evaluation, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # Remove the file
            elif os.path.isdir(file_path):
                os.rmdir(file_path)  # Remove the directory if empty
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")
            
    all_evaluation_objects = []

    for qa_pair in read_and_flatten_qa(json_path):
        search_query = qa_pair['question']
        expected_output = qa_pair['answer']
        qa_pair_level = qa_pair['level']
        
        # Retrieve contexts
        vector_dense_context, vector_sparse_context, vector_hybrid_context = all_vector_retrieve(search_query, num_results)
        global_DL_context, local_DL_context, drift_DL_context = classical_and_llm_run_graph_search(driver, search_query, graph_prefix='DL', embedding_model = embedding_model)
        global_LLM_context, local_LLM_context, drift_LLM_context = classical_and_llm_run_graph_search(driver, search_query, graph_prefix='LLM', embedding_model = embedding_model)
        
        # Combine all retrievers into a list
        total_retriever_for_query = [
            ("vector_dense_search_metrics", vector_dense_context),#vector_dense_search_metrics , vector_dense_context
            ("vector_sparse_search_metrics", vector_sparse_context),# vector_sparse_search_metrics, vector_sparse_context
            ("vector_hybrid_search_metrics", vector_hybrid_context),# vector_hybrid_search_metrics, vector_hybrid_context
            ("graph_classical_global_search_metrics", global_DL_context),# graph_classical_global_search_metrics,  global_DL_context
            ("graph_classical_local_search_metrics", local_DL_context), # graph_classical_local_search_metrics, local_DL_context
            ("graph_classical_drift_search_metrics", drift_DL_context), #graph_classical_drift_search_metrics, drift_DL_context
            ("graph_llm_global_search_metrics", global_LLM_context), # graph_llm_global_search_metrics, global_LLM_context
            ("graph_llm_local_search_metrics", local_LLM_context), # graph_llm_local_search_metrics, local_LLM_context
            ("graph_llm_drift_search_metrics", drift_LLM_context) # graph_llm_drift_search_metrics, drift_LLM_context
        ]
        
        for context_name, context_value in total_retriever_for_query:
            # Generate response
            response, usage_data = generate_response_langchain(query= search_query,
                                                               context_ = context_value.get('context'),
                                                               context_name = context_name, 
                                                               model= "gemini-1.5-pro")
            
            # Create evaluation object
            response_obj_to_evaluate = create_evaluation_object(
                qa_level = qa_pair_level,
                query=search_query,
                expected_output=expected_output,
                retrieval_package=context_value,
                llm_response=response,
                llm_usage_data=usage_data
            )
            
            all_evaluation_objects.append(response_obj_to_evaluate)
                        
            # Run evaluation and save metrics
            run_evaluation_metric(
                file_name=context_name,  # Dynamically set the dedicated file name based on the context name
                save_directory=save_path_all_evaluation,
                response_obj=response_obj_to_evaluate,
                embedding_model = embedding_model,
                model = "gemini-1.5-pro", #"gemini-2.5-pro-preview-03-25", gemini-2.5-flash-preview-04-17
                fallback_model_name = "gemini-2.5-pro-preview-03-25",
                max_retries = 4,
                retry_delay_seconds = 5
            )
            
    logging.info("Finished processing all QA pairs.")        






