from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    HarmBlockThreshold,
    HarmCategory
)
from deepeval.models.base_model import DeepEvalBaseLLM
from dotenv import load_dotenv
import os
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualPrecisionMetric
from deepeval.metrics import ContextualRecallMetric
from deepeval.scorer import Scorer
from deepeval.metrics import BaseMetric
from typing import Dict, Any, Optional, Tuple, List
import json
import time
load_dotenv(dotenv_path=os.path.join(os.getcwd(), '.env'), override=True)


class GoogleVertexAI(DeepEvalBaseLLM):
    """Class to implement Vertex AI for DeepEval"""
    def __init__(self, model):
        self.model = model

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        return chat_model.invoke(prompt).content

    async def a_generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        res = await chat_model.ainvoke(prompt)
        return res.content

    def get_model_name(self):
        return "Vertex AI Model"


class RougeMetric(BaseMetric):
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.scorer = Scorer()

    def measure(self, test_case: LLMTestCase):
        self.score = self.scorer.rouge_score(
            prediction=test_case.actual_output,
            target=test_case.expected_output,
            score_type="rouge1"
        )
        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case: LLMTestCase):
        return self.measure(test_case)

    def is_successful(self):
        return self.success

    @property
    def __name__(self):
        return "Rouge Metric"


safety_settings = {
    HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
}

def calculate_cosine_similarity(
    actual_output: Optional[str],
    expected_output: Optional[str],
    embedding_model
) -> float: 
    """
    Calculates the cosine similarity between the embeddings of two text strings.

    Args:
        actual_output: The generated text output.
        expected_output: The reference/expected text output.

    Returns:
        The cosine similarity score (0.0 to 1.0), or 0.0 if calculation
        cannot be performed (missing input, model error, etc.).
    """
    metric_name = "Cosine Similarity"
    print(f"\nAttempting {metric_name} calculation...")
    
    if embedding_model is None:
        print(f"  Skipping {metric_name}: Embedding model is not available.")
        return 0.0
    if not actual_output or not expected_output:
        print(f"  Skipping {metric_name}: Missing actual_output or expected_output.")
        return 0.0
    try:
        embedding_actual = embedding_model.encode(actual_output)
        embedding_expected = embedding_model.encode(expected_output)

        if embedding_actual is None or embedding_expected is None:
             print(f"  Skipping {metric_name}: Failed to generate one or both embeddings.")
             return 0.0

        embedding_actual_2d = embedding_actual.reshape(1, -1)
        embedding_expected_2d = embedding_expected.reshape(1, -1)
        score = sklearn_cosine_similarity(embedding_actual_2d, embedding_expected_2d)[0][0]
        score = max(0.0, min(1.0, score))
        print(f"  {metric_name} calculated successfully. Score: {score:.4f}")
        return score

    except Exception as e:
        print(f"  Error calculating {metric_name}: {e}")
        return 0.0 
    

def calculate_answer_relevancy(
    query: Optional[str],
    actual_output: Optional[str],
    llm_model: Any, 
    max_retries: int,
    retry_delay_seconds: int,
    threshold: float = 0.7
) -> Tuple[Optional[float], bool]:
    """
    Calculates Answer Relevancy with retries.

    Returns:
        Tuple (score, failed_after_retries):
          - score (float | None): The calculated score, or None if calculation failed/skipped.
          - failed_after_retries (bool): True if all retries failed, False otherwise.
    """
    if llm_model is None:
        print("  Skipping Answer Relevancy: LLM model not provided.")
        return None, False 
    if query is None or actual_output is None:
        print("  Skipping Answer Relevancy: query or actual_output is missing.")
        return None, False 

    metric_name = "Answer Relevancy"
    score = None
    failed = False

    for attempt in range(max_retries + 1):
        try:
            metric = AnswerRelevancyMetric(threshold=threshold, model=llm_model, include_reason=False)
            test_case = LLMTestCase(input=query, actual_output=actual_output)
            metric.measure(test_case)
            score = metric.score 
            print(f"  {metric_name} calculated successfully on attempt {attempt + 1}. Score: {score}")
            failed = False
            break
        except Exception as e:
            print(f"  Error calculating {metric_name} on attempt {attempt + 1}: {e}")
            if attempt < max_retries:
                print(f"  Retrying in {retry_delay_seconds} seconds...")
                time.sleep(retry_delay_seconds)
    else: 
        print(f"  Max retries ({max_retries + 1}) reached for {metric_name}")
        failed = True 
        score = None
    return score, failed


def calculate_contextual_precision(
    query: Optional[str],
    actual_output: Optional[str],
    expected_output: Optional[str],
    retrieval_context: Optional[List[str]],
    llm_model: Any,
    max_retries: int,
    retry_delay_seconds: int,
    threshold: float = 0.7
) -> Tuple[Optional[float], bool]:
    """
    Calculates Contextual Precision with retries.

    Returns:
        Tuple (score, failed_after_retries)
    """
    if llm_model is None:
        print("  Skipping Contextual Precision: LLM model not provided.")
        return None, False
    missing_data = [k for k, v in {
        'query': query, 'actual_output': actual_output,
        'expected_output': expected_output, 'retrieval_context': retrieval_context
        }.items() if v is None]
    if missing_data:
         print(f"  Skipping Contextual Precision: Missing required data: {missing_data}")
         return None, False

    metric_name = "Contextual Precision"
    score = None
    failed = False

    for attempt in range(max_retries + 1):
        try:
            metric = ContextualPrecisionMetric(threshold=threshold, model=llm_model, include_reason=False)
            test_case = LLMTestCase(input=query, actual_output=actual_output, expected_output=expected_output, retrieval_context=retrieval_context)
            metric.measure(test_case)
            score = metric.score
            print(f"  {metric_name} calculated successfully on attempt {attempt + 1}. Score: {score}")
            failed = False
            break
        except Exception as e:
            print(f"  Error calculating {metric_name} on attempt {attempt + 1}: {e}")
            if attempt < max_retries:
                print(f"  Retrying in {retry_delay_seconds} seconds...")
                time.sleep(retry_delay_seconds)
    else: 
        print(f"  Max retries ({max_retries + 1}) reached for {metric_name}.")
        failed = True
        score = None
    return score, failed


def calculate_contextual_recall(
    query: Optional[str],
    actual_output: Optional[str],
    expected_output: Optional[str],
    retrieval_context: Optional[List[str]],
    llm_model: Any,
    max_retries: int,
    retry_delay_seconds: int,
    threshold: float = 0.7
) -> Tuple[Optional[float], bool]:
    """
    Calculates Contextual Recall with retries.

    Returns:
        Tuple (score, failed_after_retries)
    """
    if llm_model is None:
        print("  Skipping Contextual Recall: LLM model not provided.")
        return None, False
    missing_data = [k for k, v in {
        'query': query, 'actual_output': actual_output,
        'expected_output': expected_output, 'retrieval_context': retrieval_context
        }.items() if v is None]
    if missing_data:
         print(f"  Skipping Contextual Recall: Missing required data: {missing_data}")
         return None, False

    metric_name = "Contextual Recall"
    score = None
    failed = False

    for attempt in range(max_retries + 1):
        try:
            metric = ContextualRecallMetric(threshold=threshold, model=llm_model, include_reason=False)
            test_case = LLMTestCase(input=query, actual_output=actual_output, expected_output=expected_output, retrieval_context=retrieval_context)
            metric.measure(test_case)
            score = metric.score
            print(f"  {metric_name} calculated successfully on attempt {attempt + 1}. Score: {score}")
            failed = False
            break
        except Exception as e:
            print(f"  Error calculating {metric_name} on attempt {attempt + 1}: {e}")
            if attempt < max_retries:
                print(f"  Retrying in {retry_delay_seconds} seconds...")
                time.sleep(retry_delay_seconds)
    else: 
        print(f"  Max retries ({max_retries + 1}) reached for {metric_name}")
        failed = True
        score = None
    return score, failed


def calculate_rouge_score(
    query:str,
    actual_output: Optional[str],
    expected_output: Optional[str],
    rouge_type: str = "rouge1", 
    threshold: float = 0.5 
) -> Optional[float]:
    """
    Calculates the ROUGE score using the RougeMetric class.

    Returns:
        The calculated ROUGE score, or 0.0 if calculation skipped/failed.
    """
    if actual_output is None or expected_output is None:
        print(f"  Skipping ROUGE-{rouge_type[5:].upper()}: actual_output or expected_output is missing.")
        return 0.0 

    metric_name = f"ROUGE-{rouge_type[5:].upper()}"
    print(f"\nAttempting {metric_name} calculation...")

    try:
        metric = RougeMetric(threshold=threshold)
        test_case = LLMTestCase(input = query, actual_output=actual_output, expected_output=expected_output)
        score = metric.measure(test_case)
        print(f"  {metric_name} calculated successfully. Score: {score}")
        return score if score is not None else 0.0
    except Exception as e:
        print(f"  Error calculating {metric_name}: {e}")
        return 0.0 

def run_evaluation_metric(
    file_name: str,
    save_directory: str,
    response_obj: Dict[str, Any],
    embedding_model ,
    model: str,  
    fallback_model_name = "gemini-1.5-pro", 
    max_retries: int = 2,
    retry_delay_seconds: int = 3,
):
    """
    Calculates evaluation metrics (Relevancy, Precision, Recall, ROUGE-1)
    with retries and fallback model logic for LLM-based metrics,
    and appends the results to a JSON file.

    Args:
        file_name (str): Identifier for the search type, used for naming the output file.
        save_directory (str): Directory path where the output JSON file will be saved.
        response_obj (dict): Dictionary containing evaluation data.
        model (str): Name of the primary evaluation LLM to use (e.g., "gemini-1.5-pro").
        max_retries (int): Max retries for failing LLM-based metric calculation with the primary model.
        retry_delay_seconds (int): Seconds to wait between retries.
    """
    print(f"\n--- Evaluating for search_type: {file_name} (Primary Model: {model}) ---")

    primary_googleai_gemini_model = None
    fallback_googleai_gemini_model = None

    try:
        if not os.getenv('GOOGLE_API_KEY'):
             print("Warning: GOOGLE_API_KEY not set, using dummy value for demo.")
             os.environ['GOOGLE_API_KEY'] = 'dummy-key-for-testing'

        primary_custom_model_gemini = ChatGoogleGenerativeAI(
            model=model, safety_settings=safety_settings,
            google_api_key=os.getenv('GOOGLE_API_KEY'), temperature=0.2
        )
        primary_googleai_gemini_model = GoogleVertexAI(model=primary_custom_model_gemini)
        print(f"Primary model '{model}' initialized successfully.")
    except Exception as e:
        print(f"  CRITICAL ERROR during primary model setup for '{file_name}' ({model}): {e}. Aborting evaluation.")
        return 

    query = response_obj.get('query')
    actual_output = response_obj.get('llm_response')
    expected_output = response_obj.get('expected_output')
    retrieval_context = response_obj.get('retrieval_context')
    if isinstance(retrieval_context, str):
        retrieval_context = [retrieval_context]

    relevancy_score, relevancy_failed = calculate_answer_relevancy(
        query, actual_output, primary_googleai_gemini_model, max_retries, retry_delay_seconds
    )
    precision_score, precision_failed = calculate_contextual_precision(
        query, actual_output, expected_output, retrieval_context, primary_googleai_gemini_model, max_retries, retry_delay_seconds
    )
    recall_score, recall_failed = calculate_contextual_recall(
        query, actual_output, expected_output, retrieval_context, primary_googleai_gemini_model, max_retries, retry_delay_seconds
    )
  
    rouge1_score = calculate_rouge_score(
        query, actual_output, expected_output, rouge_type="rouge1"
    )
    cosine_sim_score = calculate_cosine_similarity(
        actual_output, expected_output , embedding_model 

    )

    needs_fallback = relevancy_failed or precision_failed or recall_failed
    fallback_attempted = False
    fallback_succeeded_relevancy = False
    fallback_succeeded_precision = False
    fallback_succeeded_recall = False

    if needs_fallback and model != fallback_model_name:
        print(f"\n--- Primary model '{model}' failed for some LLM metrics. Attempting fallback with '{fallback_model_name}' ---")
        fallback_attempted = True
        try:
            if not os.getenv('GOOGLE_API_KEY'): 
                 raise ValueError("GOOGLE_API_KEY environment variable not set (checked before fallback).")

            fallback_custom_model_gemini = ChatGoogleGenerativeAI(
                model=fallback_model_name, safety_settings=safety_settings,
                google_api_key=os.getenv('GOOGLE_API_KEY'), temperature=0.2
            )
            fallback_googleai_gemini_model = GoogleVertexAI(model=fallback_custom_model_gemini)
            print(f"Fallback model '{fallback_model_name}' initialized successfully.")

        except Exception as e:
            print(f"  ERROR during fallback model setup ('{fallback_model_name}'): {e}. Skipping fallback attempts.")
            fallback_googleai_gemini_model = None 

        if fallback_googleai_gemini_model:
            if relevancy_failed:
                relevancy_score_fb, failed_fb = calculate_answer_relevancy(
                    query, actual_output, fallback_googleai_gemini_model, max_retries=3, retry_delay_seconds=7
                )
                if not failed_fb: 
                    relevancy_score = relevancy_score_fb 
                    relevancy_failed = False 
                    fallback_succeeded_relevancy = True

            if precision_failed:
                 precision_score_fb, failed_fb = calculate_contextual_precision(
                    query, actual_output, expected_output, retrieval_context, fallback_googleai_gemini_model, max_retries=3, retry_delay_seconds=7
                 )
                 if not failed_fb:
                     precision_score = precision_score_fb
                     precision_failed = False
                     fallback_succeeded_precision = True

            if recall_failed:
                recall_score_fb, failed_fb = calculate_contextual_recall(
                    query, actual_output, expected_output, retrieval_context, fallback_googleai_gemini_model, max_retries=3, retry_delay_seconds=7
                )
                if not failed_fb:
                    recall_score = recall_score_fb
                    recall_failed = False
                    fallback_succeeded_recall = True
        else:
             print("  Fallback model could not be initialized. Scores for failed metrics will remain None/default.")

    elif needs_fallback and model == fallback_model_name:
         print(f"\n--- Primary model '{model}' failed for some LLM metrics, but it is already the fallback model. No further fallback attempted. ---")
  
    final_relevancy_score = 0.000 if relevancy_score is None else relevancy_score
    final_precision_score = 0.000 if precision_score is None else precision_score
    final_recall_score = 0.000 if recall_score is None else recall_score
    final_rouge1_score = 0.000 if rouge1_score is None else float(rouge1_score)
    final_cosine_sim_score = 0.000 if cosine_sim_score is None else float(cosine_sim_score)
    num_input_token = response_obj.get('num_input_token')
    num_output_token = response_obj.get('num_output_token')
    time_taken = response_obj.get('time_taken')
    qa_level = response_obj.get('level')
    
    result_data = {
        "search_type": file_name,
        "precision": final_precision_score,
        "recall": final_recall_score,
        "relevancy": final_relevancy_score,
        "rouge1": final_rouge1_score, 
        "cosine_similarity": final_cosine_sim_score, 
        "time_taken": time_taken,
        "num_input_token": num_input_token,
        "num_output_token": num_output_token,
        "query" : query,
        "qa_level": qa_level,
        "actual_output" : actual_output,
        "expected_output" : expected_output,
        "retrieval_context" : retrieval_context[0], 
        "primary_llm_used": model,
        "fallback_llm_attempted": fallback_model_name if fallback_attempted else None,
        "fallback_used_and_succeeded_relevancy": fallback_succeeded_relevancy,
        "fallback_used_and_succeeded_precision": fallback_succeeded_precision,
        "fallback_used_and_succeeded_recall": fallback_succeeded_recall,
        "final_state_failed_relevancy": relevancy_failed,
        "final_state_failed_precision": precision_failed,
        "final_state_failed_recall": recall_failed,
    }

    try:
        if not file_name.lower().endswith('.json'):
            output_json_filename = f"{file_name}.json"
        else:
            base = os.path.splitext(file_name)[0]
            output_json_filename = f"{base}.json"

        os.makedirs(save_directory, exist_ok=True)
        output_json_path = os.path.join(save_directory, output_json_filename)
        append_to_json(result_data, output_json_path)
        print(f"Results for '{file_name}' appended to {output_json_path}")

    except Exception as e:
        print(f"  Error saving results for '{file_name}' to {output_json_path}: {e}")
        

def append_to_json(data_to_append: dict, json_file_path: str):
    """Reads a JSON file, appends new data as a dictionary to a list, and writes it back."""
    results_list = []
    try:
        if os.path.exists(json_file_path) and os.path.getsize(json_file_path) > 0:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                try:
                    results_list = json.load(f)
                    # Ensure it's a list
                    if not isinstance(results_list, list):
                        print(f"Warning: Existing file {json_file_path} does not contain a list. Overwriting with a new list.")
                        results_list = []
                except json.JSONDecodeError:
                    print(f"Warning: Could not decode JSON from {json_file_path}. Starting with a new list.")
                    results_list = []

        results_list.append(data_to_append)
        output_dir = os.path.dirname(json_file_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(json_file_path, 'w', encoding='utf-8') as f:
            json.dump(results_list, f, indent=4, ensure_ascii=False)

    except IOError as e:
        print(f"Error reading/writing file {json_file_path}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during JSON handling for {json_file_path}: {e}")

