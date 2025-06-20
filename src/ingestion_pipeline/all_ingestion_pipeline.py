import os
import json
import networkx as nx
import logging
from src.ingestion_pipeline.vector_ingestion.vector_ingestion import run_vector_ingestion
from src.ingestion_pipeline.graph_classical_model_ingestion.graph_classical_ingestion import classical_model_run_graph_ingestion
from src.ingestion_pipeline.graph_llm_ingestion.graph_llm_data_ingestion import llm_model_run_graph_ingestion

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def _calculate_graph_metrics(G):
    """
    Helper function to calculate specified metrics for a NetworkX DiGraph.
    'diameter' and 'radius' are calculated based on the undirected version
    of the largest weakly connected component (LWCC).
    'mean_distance' is attempted on the largest strongly connected component (LSCC).
    """
    metrics = {
        'mean_distance': None,
        'diameter': None,
        'radius': None,
        'num_wcc': 0,  
        'largest_wcc_size': 0,
        'num_scc': 0,
        'largest_scc_size': 0,
    }

    if not isinstance(G, nx.DiGraph):
        logging.error("Input graph is not a NetworkX DiGraph. Metrics calculation requires a DiGraph.")
        return metrics

    if G is None or G.number_of_nodes() == 0:
        logging.warning("Graph is None or empty, cannot calculate metrics.")
        return metrics

    G_lwcc = None
    largest_wcc_nodes = set()
    try:
        wccs = list(nx.weakly_connected_components(G))
        metrics['num_wcc'] = len(wccs)
        if wccs:
            largest_wcc_nodes = max(wccs, key=len)
            metrics['largest_wcc_size'] = len(largest_wcc_nodes)
            if metrics['largest_wcc_size'] > 1:
                 G_lwcc = G.subgraph(largest_wcc_nodes)

    except Exception as e:
        logging.error(f"Error calculating weak connectivity: {e}")
    G_lscc = None 
    largest_scc_nodes = set()
    try:
        sccs = list(nx.strongly_connected_components(G))
        metrics['num_scc'] = len(sccs)
        if sccs:
            largest_scc_nodes = max(sccs, key=len)
            metrics['largest_scc_size'] = len(largest_scc_nodes)
            if metrics['largest_scc_size'] > 1:
                 G_lscc = G.subgraph(largest_scc_nodes) 
    except Exception as e:
        logging.error(f"Error calculating strong connectivity: {e}")
    if G_lscc: 
        try:
            metrics['mean_distance'] = nx.average_shortest_path_length(G_lscc)
            logging.info(f"Calculated 'mean_distance' on the largest SCC ({metrics['largest_scc_size']} nodes).")
        except nx.NetworkXError as e:
            logging.warning(f"Could not calculate 'mean_distance' on LSCC (NetworkXError: {e}). It might not be strongly connected internally or have issues.")
            metrics['mean_distance'] = None 
        except Exception as e:
            logging.error(f"Unexpected error calculating 'mean_distance' on LSCC: {e}")
            metrics['mean_distance'] = None
    elif metrics['largest_scc_size'] <= 1:
         logging.info("Skipping 'mean_distance' calculation (LSCC size <= 1).")
         metrics['mean_distance'] = None 

    if G_lwcc: 
        try:
            G_lwcc_undirected = G_lwcc.to_undirected(as_view=False)
            if nx.is_connected(G_lwcc_undirected):
                 metrics['diameter'] = nx.diameter(G_lwcc_undirected)
                 metrics['radius'] = nx.radius(G_lwcc_undirected)
                 logging.info(f"Assigned 'diameter' and 'radius' based on the undirected version of the largest WCC ({metrics['largest_wcc_size']} nodes).")
            else:
                 logging.warning("Undirected version of LWCC is unexpectedly not connected. Setting diameter/radius to None.")
                 metrics['diameter'] = None 
                 metrics['radius'] = None   
        except nx.NetworkXError as e:
             logging.warning(f"Could not calculate diameter/radius on undirected LWCC (NetworkXError: {e}).")
             metrics['diameter'] = None
             metrics['radius'] = None
        except Exception as e:
            logging.error(f"Unexpected error calculating diameter/radius on undirected LWCC: {e}")
            metrics['diameter'] = None
            metrics['radius'] = None
    elif metrics['largest_wcc_size'] == 1:
        logging.info("Largest WCC has size 1. Setting diameter=0, radius=0.")
        metrics['diameter'] = 0
        metrics['radius'] = 0
    else: 
        logging.warning("Skipping 'diameter' and 'radius' calculation (no suitable LWCC found).")
        metrics['diameter'] = None
        metrics['radius'] = None
    return metrics

def analyse_and_store_graph_data(classical_graph_stats, 
                                 llm_graph_stats):
    """
    Analyzes graph statistics from classical and LLM results, stores them in a JSON file.

    Args:
        classical_graph_stats (dict): The results dictionary from the classical graph process.
                                      Expected keys: 'original_graph', 'communities_L0',
                                      'communities_L1', 'communities_L2', 'number_of_nodes',
                                      'number_of_relationships'.
        llm_graph_stats (dict): The results dictionary from the LLM graph process (similar structure).
        output_dir (str): The directory path where the JSON stats file will be saved.

    Returns:
        str: The full path to the saved JSON file, or None if an error occurred.
    """
    all_graph_data = []
    output_dir = os.path.join(os.getcwd(), 'src/ingestion_pipeline')
    
    for graph_type, graph_results in [('Classical', classical_graph_stats),
                                      ('LLM', llm_graph_stats)]:
        if graph_results is None:
            logging.warning(f"Input stats for '{graph_type}' graph are None. Skipping analysis.")
            continue

        logging.info(f"Analyzing '{graph_type}' graph...")
        stats = {'graph_type': graph_type}
        if 'number_of_nodes' in graph_results:
             stats['number_of_nodes'] = graph_results.get('number_of_nodes', 0)
        elif 'original_graph' in graph_results and graph_results['original_graph'] is not None:
             stats['number_of_nodes'] = graph_results['original_graph'].number_of_nodes()
        else:
             stats['number_of_nodes'] = 0
             logging.warning(f"Could not determine number of nodes for {graph_type} graph.")

        if 'number_of_relationships' in graph_results:
            stats['number_of_relationships'] = graph_results.get('number_of_relationships', 0)
        elif 'original_graph' in graph_results and graph_results['original_graph'] is not None:
            stats['number_of_relationships'] = graph_results['original_graph'].number_of_edges()
        else:
            stats['number_of_relationships'] = 0
            logging.warning(f"Could not determine number of relationships for {graph_type} graph.")

        comm_l0 = graph_results.get('communities_L0')
        stats['communities_level_1'] = len(set(comm_l0.values())) if comm_l0 else 0

        comm_l1 = graph_results.get('communities_L1')
        stats['communities_level_2'] = len(set(comm_l1.values())) if comm_l1 else 0

        comm_l2 = graph_results.get('communities_L2')
        stats['communities_level_3'] = len(set(comm_l2.values())) if comm_l2 else 0

        graph = graph_results.get('original_graph')
        if graph is not None and isinstance(graph, nx.Graph):
            advanced_metrics = _calculate_graph_metrics(graph)
            stats.update(advanced_metrics)
        else:
            logging.warning(f"'original_graph' not found or not a NetworkX graph for '{graph_type}'. Skipping advanced metrics.")
            stats.update({
                'mean_distance': None, 'diameter': None, 'radius': None, 'num_wcc': None, 'largest_wcc_size': None,'num_scc': None, 'largest_scc_size': None,
            })

        all_graph_data.append(stats)
        logging.info(f"Finished analyzing '{graph_type}' graph.")

    if not all_graph_data:
         logging.error("No graph data was analyzed. Cannot save JSON.")
         return None
    output_filename = "graph_analysis_stats.json"
    graph_json_path = os.path.join(output_dir, output_filename)

    try:
        os.makedirs(output_dir, exist_ok=True) 
        with open(graph_json_path, 'w', encoding='utf-8') as f:
            json.dump(all_graph_data, f, indent=4)
        logging.info(f"Successfully saved graph analysis stats to: {graph_json_path}")
        return graph_json_path
    except IOError as e:
        logging.error(f"Error writing graph stats to JSON file '{graph_json_path}': {e}")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred while saving JSON: {e}")
        return None

def ingestion_complete_data(driver, dataset_path):
    print(f"crawling done and text saved to {dataset_path}")
    classical_graph_stats_data = classical_model_run_graph_ingestion(driver, dataset_path)
    
    print(f"crawling done and text saved to {dataset_path}")
    llm_graph_stats_data = llm_model_run_graph_ingestion(driver, dataset_path)
    
    graph_json_path = analyse_and_store_graph_data(classical_graph_stats_data, llm_graph_stats_data) 
    
    print(">>> Finished classical_model_run_graph_ingestion") 
    run_vector_ingestion(dataset_path= dataset_path, model="gemini-2.0-flash-lite", temperature = 0.7) 
    
    print(">>> Exiting ingestion_complete_data")
    return graph_json_path
    