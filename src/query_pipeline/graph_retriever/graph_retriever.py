import os
from src.ingestion_pipeline.helper_functions import TimeMeasurer, gemini_llm
from sentence_transformers import SentenceTransformer
from typing import Dict, Any, Tuple, List, Optional, Set
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv 
import numpy as np
import json
import tiktoken
import asyncio 
load_dotenv(dotenv_path=os.path.join(os.getcwd(), '.env'), override=True)

NEO4J_DATABASE=os.getenv('NEO4J_DB') 
DEFAULT_TOKEN_ENCODING = "cl100k_base" 

def count_tokens(text: str) -> int:
    """
    Counts the number of tokens in a text string using tiktoken.
    """
    if not text:
        return 0
    encoding = tiktoken.get_encoding(DEFAULT_TOKEN_ENCODING)
    return len(encoding.encode(text))


def _extract_entities_from_query_llm( llm, query_text: str, max_entities: int = 5) -> List[str]:
        """
        Uses the configured LLM client (via a LangChain-style chain)
        to extract key entities from the query text.
        Returns a list of lowercase entity strings.
        """
        if not llm:
            print("Info: Skipping LLM entity extraction - no LLM client configured.")
            return []
        if not query_text:
            return []

        prompt_template_str = """
                Extract the main real-world entities (like people, places, organizations, concepts, topics etc) mentioned in the following query. 
                Focus on the most important {max_entities} entities relevant for searching related information.
                Return ONLY a valid JSON list of strings, with each string being a lowercase entity name. 
                Do not add any explanation or commentary before or after the JSON list.

                Query:
                "{query_input}"

                JSON List:
                """
        try:
            prompt = ChatPromptTemplate.from_template(prompt_template_str)
            chain = prompt | llm | StrOutputParser()
            response_str = chain.invoke({
                "query_input": query_text,
                "max_entities": max_entities
                })

            response_str = response_str.strip().removeprefix("```json").removesuffix("```").strip()

            entities = json.loads(response_str)
            if isinstance(entities, list) and all(isinstance(e, str) for e in entities):
                 lowercase_entities = [e.lower() for e in entities]
                 print("the number of entities found by llm, ", len(lowercase_entities))
                 print("the extracted entities ", lowercase_entities)
                 return lowercase_entities
            else:
                 print(f"Warning: LLM entity extraction returned invalid format: {entities}")
                 return []

        except json.JSONDecodeError as e:
            print(f"Error decoding LLM JSON for entities: {e}")
            print(f"LLM Output String was: '{response_str}'")
            return []
        except Exception as e:
            print(f"Error during LLM entity extraction chain execution: {e}")
            return []        

class DriftSearch:
    """
        Knowledge graph search class for retrieving information from Neo4j graph database.
        Supports multiple search strategies across different prefixed graphs.
    """
    
    def __init__(self, 
                 driver,
                 database=None, 
                 llm=None,
                 embedding_model= None):
        """
        Initialize the search class with Neo4j driver and configuration.
        
        Args:
            driver: Neo4j driver instance
            database: Name of Neo4j database to use (or None for default)
            embedding_model_name: Name of the embedding model to use
        """
        self.driver = driver
        self.database = database
        self.model = embedding_model
        self.llm = llm
    
    def _get_embedding(self, text):
        """Get embedding vector for a text string"""
        if self.model is None:
            self._load_embedding_model()
            if self.model is None:
                return None
                
        try:
            return self.model.encode(text).tolist()
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return None
            
    def build_drift_context(self,
                            relevant_communities: List[Dict[str, Any]],
                            entities_in_focus: List[Dict[str, Any]], 
                            entity_texts: Dict[str, List[Dict[str, Any]]], 
                            seed_entity_ids: Set[str], 
                            relationships_found: Optional[List[Dict[str, Any]]] = None,
                            max_communities: int = 3,
                            max_context_chunks: int = 8, 
                            seed_boost: float = 0.02, 
                            traversal_boost: float = 0.01 
                            ) -> Dict[str, Any]:
        """
        Builds a context for DRIFT search, combining community overviews with
        structurally-aware selection of relevant text chunks using boosting.

        Args:
            relevant_communities: List of top relevant community dictionaries.
            entities_in_focus: List of all entity dictionaries considered (seed + traversed).
            entity_texts: Dictionary mapping entity IDs to lists of their associated text chunks
                          (expected to have 'text' and 'relevance' keys).
            seed_entity_ids: Set of IDs for entities used as the starting point for traversal/expansion.
            relationships_found: Optional list of relationship dictionaries found.
            max_communities: Max number of community summaries to include.
            max_context_chunks: Max number of text chunks to include in the context.
            seed_boost (float): Value added to the score for chunks linked to seed entities. Needs tuning.
            traversal_boost (float): Value added to the score for chunks linked to non-seed (traversed) entities. Needs tuning.

        Returns:
            Dictionary with "text" (the built context) and "approx_token_count".
        """
        context = []
        seen_source_texts = set()
        added_any_relevant_context = False

        all_chunks_with_details = []
        entity_ids_in_context = {e['id'] for e in entities_in_focus}
        entity_name_map = {e['id']: e.get('name', 'Unknown Entity') for e in entities_in_focus}

        if entity_texts:
            for entity_id, chunks in entity_texts.items():
                if entity_id not in entity_ids_in_context:
                    continue

                entity_name = entity_name_map.get(entity_id, "Unknown Entity")

                for chunk in chunks:
                    chunk_text = chunk.get('text')
                    if not chunk_text:
                        continue 

                    base_relevance = chunk.get('relevance', 0.0)
                    if base_relevance is None: base_relevance = 0.0 

                    boost = 0.0
                    if entity_id in seed_entity_ids:
                        boost += seed_boost
                    elif entity_id in entity_ids_in_context:
                        boost += traversal_boost

                    final_score = base_relevance + boost 

                    all_chunks_with_details.append({
                        'entity_id': entity_id,
                        'entity_name': entity_name,
                        'text': chunk_text,
                        'final_score': final_score, 
                        'base_relevance': base_relevance 
                    })
        all_chunks_with_details.sort(key=lambda x: x['final_score'], reverse=True)
        top_chunks_to_add = all_chunks_with_details[:max_context_chunks]

        if top_chunks_to_add:
            context.append("## Detailed Evidence from Sources\n")
            added_entity_headers = set()

            for i, chunk_data in enumerate(top_chunks_to_add): 
                chunk_text = chunk_data['text']
                entity_name = chunk_data['entity_name']
                final_score = chunk_data['final_score']

                if chunk_text in seen_source_texts:
                    continue
                seen_source_texts.add(chunk_text)

                try:
                    chunk_tokens = count_tokens(chunk_text) 
                    print(f"DEBUG build_drift_context: Adding Chunk {i+1}/{len(top_chunks_to_add)} - Entity: '{entity_name}' - Score: {final_score:.4f} - Tokens: {chunk_tokens}")
                except Exception as e:
                    print(f"DEBUG build_drift_context: Error counting tokens for chunk {i+1}: {e}")

                if entity_name not in added_entity_headers:
                    context.append(f"\n### Context related to: {entity_name} (Top Chunk Score: {final_score:.3f})\n")
                    added_entity_headers.add(entity_name)
                context.append(f"{chunk_text}\n") 
                added_any_relevant_context = True

        if relationships_found:
             context.append("## Identified Connections (Illustrative)\n")
             added_rels = 0
             for rel in relationships_found[:5]: 
                 source = rel.get('source_name', rel.get('source', '?'))
                 target = rel.get('target_name', rel.get('target', '?'))
                 desc = rel.get('description', rel.get('relationship_description', 'related to')) 
                 if source != '?' and target != '?':
                     context.append(f"- {source} --[{desc}]--> {target}")
                     added_rels += 1
             if added_rels == 0:
                 context.append(" (No direct relationships surfaced in this view)")
             context.append("\n")
             added_any_relevant_context = True 

        if not added_any_relevant_context:
                pass
        context_text = "\n".join(context)
        context_lines = [line for line in context_text.split('\n') if line.strip()]
        context_text = "\n".join(context_lines) 
        final_approx_token_count = count_tokens(context_text)

        return {
            "text": context_text,
            "approx_token_count": final_approx_token_count
        }
 
      
    def get_entity_text_chunks(self, entity_ids, graph_prefix, query_embedding, max_chunks_per_entity=2):
        """Get source text chunks for entities, ranked by relevance to query embedding."""
        if not entity_ids or not query_embedding:
            return {} 

        entity_label = f"{graph_prefix}__Entity__"
        chunk_label = f"{graph_prefix}__Chunk__"
        has_entity_rel = f"{graph_prefix}_HAS_ENTITY"

        query = f"""
        MATCH (e:{entity_label})<-[:{has_entity_rel}]-(c:{chunk_label})
        WHERE e.id IN $entity_ids AND c.embedding IS NOT NULL
        WITH e, c, vector.similarity.cosine(c.embedding, $query_embedding) AS score
        ORDER BY score DESC 
        WITH e, collect({{chunk: c, score: score}}) AS chunks_with_scores
        UNWIND chunks_with_scores[0..$max_per_entity] AS chunk_data

        RETURN e.id AS entity_id,
            e.name AS entity_name,
            chunk_data.chunk.id AS chunk_id,
            chunk_data.chunk.text AS chunk_text,
            chunk_data.chunk.n_tokens AS chunk_tokens,
            chunk_data.score AS relevance_score
        """
        params = {
            'entity_ids': entity_ids,
            'max_per_entity': max_chunks_per_entity,
            'query_embedding': query_embedding 
        }

        result = self._execute_query(query, params) 

        entity_texts = {}
        for record in result:
            entity_id = record['entity_id']
            if entity_id not in entity_texts:
                entity_texts[entity_id] = []
            entity_texts[entity_id].append({
                'chunk_id': record['chunk_id'],
                'text': record['chunk_text'],
                'tokens': record.get('chunk_tokens', 0),
                'relevance': record.get('relevance_score', 0.0)
            })        
        return entity_texts
 
    def traverse_entity_relationships(self,
                                      entity_ids: List[str],
                                      graph_prefix: str,
                                      max_depth: int = 2,
                                      use_apoc: bool = True,
                                      limit: int = 50,
                                      relationship_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Traverse relationships from seed entities to find related entities,
        allowing for deeper traversal and more flexible relationship types.

        Args:
            entity_ids: List of starting entity IDs.
            graph_prefix: The graph prefix.
            max_depth: Maximum traversal depth (default: 2).
            use_apoc: Whether to attempt using APOC procedures (default: True).
            limit: Maximum number of related entities to return.
            relationship_filter: Optional APOC relationship filter string (e.g., "REL_TYPE>|REL_TYPE2<").
                                 If None, uses a default broad set.

        Returns:
            List of dictionaries representing the related entities found.
        """
        if not entity_ids:
            return []

        entity_label = f"{graph_prefix}__Entity__"
        related_rel = f"{graph_prefix}_RELATED"
        in_community_rel = f"{graph_prefix}_IN_COMMUNITY"
      
        if relationship_filter is None:
            rels_to_include = [related_rel, in_community_rel]
            relationship_filter_str = "|".join([f"{rel}>" for rel in rels_to_include] + [f"{rel}<" for rel in rels_to_include])
            cypher_rels_str = "|:".join(rels_to_include) 
        else:
            relationship_filter_str = relationship_filter
            cypher_rels_str = ":"+related_rel 
        apoc_available = use_apoc
        if use_apoc:
            try:
                self._execute_query("CALL apoc.path.expandConfig(null, {}) YIELD path RETURN count(path)")
            except Exception:
                apoc_available = False

        results = []
        if apoc_available:
            query = f"""
            MATCH (e:{entity_label})
            WHERE e.id IN $entity_ids

            CALL apoc.path.expandConfig(e, {{
                relationshipFilter: $relationship_filter_str,
                labelFilter: '+{entity_label}', 
                uniqueness: "NODE_GLOBAL",
                minLevel: 1, 
                maxLevel: $max_depth
            }})
            YIELD path

            WITH nodes(path)[-1] AS related_node
            WHERE related_node:{entity_label} AND NOT related_node.id IN $entity_ids

            RETURN DISTINCT related_node.id AS id,
                   related_node.name AS name,
                   related_node.type AS type,
                   related_node.description AS description
            LIMIT $limit
            """
            params = {
                'entity_ids': entity_ids,
                'relationship_filter_str': relationship_filter_str,
                'max_depth': max_depth,
                'limit': limit
            }
            results = self._execute_query(query, params)

        else:
            depth_str = f"*1..{max_depth}" if max_depth > 0 else "" 
            query = f"""
            MATCH (e:{entity_label})
            WHERE e.id IN $entity_ids
            MATCH path = (e)-[:{cypher_rels_str}{depth_str}]-(related:{entity_label})
            WHERE NOT related.id IN $entity_ids

            RETURN DISTINCT related.id AS id,
                   related.name AS name,
                   related.type AS type,
                   related.description AS description
            LIMIT $limit
            """
            params = {
                'entity_ids': entity_ids,
                'limit': limit
            }
            results = self._execute_query(query, params)
        return results

   
    def _find_communities_by_entity_names(self,
                                          graph_prefix: str,
                                          entity_names: List[str],
                                          level: int = 0,
                                          limit_per_entity: int = 2 
                                          ) -> List[Dict[str, Any]]:
        """
        Finds communities that contain entities matching the provided names at a specific level.
        Returns community details, assigning a simple presence score.
        """
        if not entity_names: return []

        community_label = f"{graph_prefix}__Community__"
        entity_label = f"{graph_prefix}__Entity__"
        in_community_rel = f"{graph_prefix}_IN_COMMUNITY"
        lowercase_names = [name.lower() for name in entity_names]
        query = f"""
        MATCH (e:{entity_label})
        WHERE toLower(e.name) IN $names
        WITH collect(e) AS matched_entities 
        UNWIND matched_entities AS entity_node
        MATCH (c:{community_label} {{level: $level}})<-[:{in_community_rel}]-(entity_node)
        WHERE c.summary IS NOT NULL
        RETURN DISTINCT c.id AS id,
            c.community AS community_id,
            c.title AS title,
            c.summary AS summary,
            c.rank AS rank,
            c.full_content AS full_content,
            c.period AS period,
            c.size AS size,
            1.0 AS entity_presence_score
        ORDER BY c.rank DESC 
        LIMIT $overall_limit 
        """
        overall_limit = len(lowercase_names) * limit_per_entity
        params = {'names': lowercase_names, 'level': level, 'overall_limit': overall_limit}
        results = self._execute_query(query, params)
        print(f"DEBUG: Found {len(results)} communities via entity names: {[r['title'][:30] for r in results]}")
        return results     
 
    def _execute_query(self, query, params=None):
        """Execute a Cypher query and return results as a list of dictionaries"""
        if params is None:
            params = {}
            
        try:
            result = self.driver.execute_query(query, params, database_=self.database)
            return [record.data() for record in result.records]
        except Exception as e:
            print(f"Query execution error: {e}")
            print(f"Query: {query}")
            print(f"Parameters: {params}")
            return []
    
          
    def find_entities_by_names(self, entity_names: List[str], graph_prefix: str, limit: int = 5) -> List[Dict[str, Any]]:
        """ Finds entities matching names, ordered by degree. """
        if not entity_names: return []
        entity_label = f"{graph_prefix}__Entity__"
        lowercase_names = [name.lower() for name in entity_names]

        query = f"""
        MATCH (e:{entity_label})
        WHERE toLower(e.name) IN $names AND e.description IS NOT NULL
        WITH e
        OPTIONAL MATCH (e)-[r]-()
        WITH e, count(r) as degree

        RETURN e.id AS id,
               e.name AS name,
               e.type AS type,
               e.description AS description,
               degree 
        ORDER BY degree DESC 
        LIMIT $limit
        """
        params = {'names': lowercase_names, 'limit': limit}
        query_entities_graph = self._execute_query(query, params)
        return query_entities_graph

    def find_entities_by_vector(self, query_text, graph_prefix, max_entities=5):
        """Find entities using vector search"""
        embedding = self._get_embedding(query_text)
        if not embedding:
            return []
            
        entity_label = f"{graph_prefix}__Entity__"
        query = f"""
        MATCH (node:{entity_label})
        WHERE node.embedding IS NOT NULL
        WITH node, vector.similarity.cosine(node.embedding, $query_embedding) AS score
        WHERE score > 0.6 AND node.description IS NOT NULL
        
        RETURN node.id AS id,
               node.name AS name,
               node.type AS type, 
               node.description AS description,
               score AS similarity_score
        ORDER BY similarity_score DESC
        LIMIT $limit
        """
        params = {
            'query_embedding': embedding,
            'limit': max_entities
        }
        
        result = self._execute_query(query, params)
        return result
      
    
    def calculate_community_relevance(self, query_embedding, communities, entity_weights=None, search_type='drift'):
        for community in communities:
            base_rank = float(community.get('rank', 0.0)) 
            semantic_score = 0.0
            if query_embedding and community.get('embedding'):
                try:
                    comm_emb = np.array(community['embedding'])
                    query_emb_np = np.array(query_embedding)
                    norm_product = np.linalg.norm(comm_emb) * np.linalg.norm(query_emb_np)
                    if norm_product > 1e-9:
                        semantic_score = float(np.dot(query_emb_np, comm_emb) / norm_product)
                    else:
                        semantic_score = 0.0
                except Exception as e:
                       print(e)
                       
            entity_score = 0.0
            if entity_weights and 'entity_ids' in community:
                community_entities = set(community.get('entity_ids', []))
                weighted_entities_in_community = community_entities.intersection(entity_weights.keys())
                if weighted_entities_in_community:
                    sum_weights = sum(entity_weights[e_id] for e_id in weighted_entities_in_community)
                    entity_score = sum_weights
            if search_type=='drift':
                final_score = (0.1 * semantic_score) + (0.7 * entity_score) + (0.2 * base_rank)
            else:
                final_score = (0.5 * semantic_score) + (0.2 * entity_score) + (0.2 * base_rank)
            community['relevance_score'] = final_score
            community['score_components'] = {
                'semantic_score': semantic_score,
                'entity_score': entity_score,
                'base_rank': base_rank,
            }
        communities.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        return communities

    def get_entities_for_query(self, query_embedding, graph_prefix, max_entities=5):
        """
        Find entities relevant to the query for weighting communities
        """
        entity_weights = {}
        
        if query_embedding:
            entity_label = f"{graph_prefix}__Entity__"
            query = f"""
            MATCH (e:{entity_label})
            WHERE e.embedding IS NOT NULL
            WITH e, vector.similarity.cosine(e.embedding, $query_embedding) AS score
            WHERE score > 0.7
            RETURN e.id AS id, e.name AS name, score AS relevance
            ORDER BY relevance DESC
            LIMIT $limit
            """
            
            results = self._execute_query(query, {
                'query_embedding': query_embedding,
                'limit': max_entities
            })
            for entity in results:
                entity_weights[entity['id']] = float(entity['relevance'])
                
        return entity_weights
  

             
    def get_community_entities_map(self, community_ids, graph_prefix):
        """
        Get entity IDs for specific communities
        
        Returns:
            Dict mapping community IDs to lists of entity IDs
        """
        if not community_ids:
            return {}
            
        community_label = f"{graph_prefix}__Community__"
        entity_label = f"{graph_prefix}__Entity__"
        in_community_rel = f"{graph_prefix}_IN_COMMUNITY"
        
        query = f"""
        MATCH (c:{community_label})<-[:{in_community_rel}]-(e:{entity_label})
        WHERE c.id IN $community_ids
        RETURN c.id AS community_id, collect(e.id) AS entity_ids
        """
        
        result = self._execute_query(query, {'community_ids': community_ids})
        community_entity_map = {}
        for record in result:
            community_entity_map[record['community_id']] = record['entity_ids']
        return community_entity_map  
  
  
            
    def find_community_reports(self, graph_prefix, llm_entities, query_embedding, level=0, limit=30, min_similarity=0.2):
        """
        Finds relevant communities by combining results from vector similarity search
        and a search based on LLM-extracted entity names. Returns a de-duplicated list.

        Args:
            graph_prefix: Graph prefix.
            query_text: User query text (used if entities are not precomputed).
            query_embedding: Embedding of the user query.
            precomputed_query_entities: Optional list of lowercase entities from LLM.
            level: Community level.
            limit: Approximate final number of unique communities to return.
            min_vector_similarity: Min vector score for vector search candidates.
            vector_limit_factor: Fetch limit*factor candidates via vector search.
            max_llm_entities: Max entities to extract if not precomputed.
            entity_community_limit_per_entity: Max communities to fetch per matched entity name.

        Returns:
            List of unique community report dictionaries, potentially more than 'limit' initially.
            Further ranking might be needed downstream.
        """
        if not query_embedding:
            return []
        vector_search_limit = int(limit * limit)
        community_label = f"{graph_prefix}__Community__"
        vector_query = f"""
        MATCH (c:{community_label} {{level: $level}})
        WHERE c.summary IS NOT NULL AND c.embedding IS NOT NULL
        WITH c, vector.similarity.cosine(c.embedding, $query_embedding) AS vector_score
        WHERE vector_score >= $min_vector_similarity
        RETURN c.id AS id, c.community AS community_id, c.title AS title, c.summary AS summary,
               c.rank AS rank, c.full_content AS full_content, c.period AS period,
               c.size AS size, vector_score
        ORDER BY vector_score DESC LIMIT $vector_search_limit
        """
        vector_params = {
            'level': level, 'vector_search_limit': vector_search_limit,
            'query_embedding': query_embedding, 'min_vector_similarity': min_similarity
        }
        vector_results = self._execute_query(vector_query, vector_params)
        entity_results = self._find_communities_by_entity_names(
            graph_prefix=graph_prefix,
            entity_names=llm_entities,
            level=level,
            limit_per_entity=2
        )
        print(f"DEBUG: Entity search found {len(entity_results)} communities.")
        combined_results_map: Dict[str, Dict[str, Any]] = {}
        for comm in entity_results:
            comm_id = comm['id']
            if comm_id not in combined_results_map:
                 comm_data = comm.copy()
                 comm_data['retrieval_source'] = 'entity'
                 comm_data['relevance_score'] = comm.get('entity_presence_score', 0.5) 
                 combined_results_map[comm_id] = comm_data
        for comm in vector_results:
            comm_id = comm['id']
            if comm_id not in combined_results_map:
                comm_data = comm.copy()
                comm_data['retrieval_source'] = 'vector'
                comm_data['relevance_score'] = comm.get('vector_score', 0.0)
                combined_results_map[comm_id] = comm_data
            else:
                 combined_results_map[comm_id]['vector_score_if_overlap'] = comm.get('vector_score')

        print(f"DEBUG: Entity search found vector_results {len(vector_results)} communities.")
        final_list = list(combined_results_map.values())
        final_list.sort(key=lambda x: x.get('relevance_score', 0.0), reverse=True)
        print(f"DEBUG: Entity search found final_list {len(final_list)} communities.")

        return final_list[:limit]
    
    def _get_embedding(self, text):
        """Get embedding vector for a text string"""
        if self.model is None:
            self._load_embedding_model()
            if self.model is None:
                return None
                
        try:
            return self.model.encode(text).tolist()
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return None
   
            
    def drift_search(self,
                     query_text: str,
                     graph_prefix: str = "DL",
                     llm_entities=None,
                     max_entities_per_community: int = 10, 
                     max_seed_entities_total: int = 15,
                     max_relevant_communities: int = 10, 
                     max_depth: int = 12, 
                     max_context_chunks: int = 30, 
                     community_min_similarity: float = 0.35 
                     ) -> Dict[str, Any]:
            """
            Enhanced DRIFT search: Combines community search with broader entity
            seeding and deeper graph traversal for complex query synthesis.

            Args:
                query_text: The user's query.
                graph_prefix: Graph identifier prefix.
                max_entities_per_community: Max entities to select from each top community.
                max_seed_entities_total: Overall limit on entities starting traversal.
                max_relevant_communities: How many top communities to focus on.
                max_depth: Max depth for relationship traversal (use >= 2 for L2/L3).
                max_context_chunks: Number of text chunks in the final context.
                community_search_limit_factor: Multiplier for initial community fetch limit.
                community_min_similarity: Minimum vector similarity for initial community retrieval.

            Returns:
                Comprehensive search results dictionary.
            """
            query_embedding = self._get_embedding(query_text)
            if not query_embedding:
                return {'error': 'Failed to generate query embedding', 'context': '', 'approx_token_count': 0}

            community_reports = self.find_community_reports(
                                        graph_prefix=graph_prefix,
                                        query_embedding=query_embedding,
                                        llm_entities = llm_entities,
                                        level=0,
                                        min_similarity=community_min_similarity
                                    )
            community_ids = [report['id'] for report in community_reports]
            if community_ids:
                community_entity_map = self.get_community_entities_map(community_ids, graph_prefix)
                for report in community_reports:
                    report['entity_ids'] = community_entity_map.get(report['id'], [])
            entity_weights = self.get_entities_for_query(query_embedding, graph_prefix, max_entities=max_seed_entities_total) # Get weights for potential seeds
            relevant_communities = self.calculate_community_relevance(
                query_embedding, 
                community_reports,
                entity_weights, 
                search_type='drift' 
            )
            top_relevant_communities = sorted(relevant_communities, key=lambda x: x.get('relevance_score', 0), reverse=True)
            print(" top_relevant_communities ---->", len(top_relevant_communities))
            seed_entity_ids: Set[str] = set()
            for community in top_relevant_communities:
                community_entity_ids = community.get('entity_ids', [])
                weighted_ids_in_comm = sorted(
                    [eid for eid in community_entity_ids if eid in entity_weights],
                    key=lambda eid: entity_weights[eid],
                    reverse=True
                )
                ids_to_add = weighted_ids_in_comm[:max_entities_per_community]
                non_weighted_ids = [eid for eid in community_entity_ids if eid not in entity_weights]
                ids_to_add.extend(non_weighted_ids[:max_entities_per_community])
                seed_entity_ids.update(ids_to_add)
            direct_query_entities = self.find_entities_by_vector(query_text, graph_prefix, max_entities=max_seed_entities_total // 2)
            entities_llm_names = self.find_entities_by_names(llm_entities, graph_prefix, max_seed_entities_total)
            direct_query_entities.extend(entities_llm_names)
            print(" direct_query_entities ", len(direct_query_entities))
            seed_entity_ids.update([e['id'] for e in direct_query_entities])
            final_seed_entity_ids = list(seed_entity_ids)[:max_seed_entities_total]
            if not final_seed_entity_ids:
                seed_entities = []
            else:
                entity_detail_query = f"""
                MATCH (e:{graph_prefix}__Entity__)
                WHERE e.id IN $entity_ids
                RETURN e.id AS id,
                    e.name AS name,
                    e.type AS type,
                    e.description AS description
                """
                seed_entities = self._execute_query(entity_detail_query, {'entity_ids': final_seed_entity_ids})
            if final_seed_entity_ids and max_depth > 0:
                related_entities = self.traverse_entity_relationships(
                    entity_ids=final_seed_entity_ids,
                    graph_prefix=graph_prefix,
                    max_depth=max_depth,
                    use_apoc=True, 
                    limit=max_seed_entities_total * 2
                )
            else:
                related_entities = []
            all_entities_in_focus = seed_entities + related_entities
            all_entities_in_focus_map = {e['id']: e for e in all_entities_in_focus}
            final_entities_in_focus = list(all_entities_in_focus_map.values())
            final_entity_ids_in_focus = list(all_entities_in_focus_map.keys())
            entity_texts = {}
            if final_entity_ids_in_focus:
                entity_texts = self.get_entity_text_chunks(
                    entity_ids=final_entity_ids_in_focus,
                    graph_prefix=graph_prefix,
                    query_embedding=query_embedding, 
                    max_chunks_per_entity=2 
                )
            context_data = self.build_drift_context(
                relevant_communities=top_relevant_communities,
                entities_in_focus=final_entities_in_focus, 
                entity_texts=entity_texts,
                seed_entity_ids=set(final_seed_entity_ids), 
                max_communities=max_relevant_communities,
                max_context_chunks=max_context_chunks
            )
            result = {
                'top_communities': top_relevant_communities,
                'seed_entities': seed_entities,
                'related_entities': related_entities,
                'final_entities_in_focus': final_entities_in_focus,
                'context': context_data['text'],
                'approx_token_count': context_data['approx_token_count'],
                'query': query_text,
                'graph_prefix': graph_prefix,
            }
            return result
   
     
class LocalSearch:
    """
        Knowledge graph search class for retrieving information from Neo4j graph database.
        Supports multiple search strategies across different prefixed graphs.
    """
    def __init__(self, 
                 driver,
                 database=None, 
                 llm=None,
                 embedding_model = None,
                ):
        """
        Initialize the search class with Neo4j driver and configuration.
        
        Args:
            driver: Neo4j driver instance
            database: Name of Neo4j database to use (or None for default)
            embedding_model_name: Name of the embedding model to use
        """
        self.driver = driver
        self.database = database
        self.model = embedding_model
        self.llm = llm
    
    def _get_embedding(self, text):
        """Get embedding vector for a text string"""
        if self.model is None:
            self._load_embedding_model()
            if self.model is None:
                return None
                
        try:
            return self.model.encode(text).tolist()
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return None
        
    def _execute_query(self, query, params=None):
        """Execute a Cypher query and return results as a list of dictionaries"""
        if params is None:
            params = {}
            
        try:
            result = self.driver.execute_query(query, params, database_=self.database)
            return [record.data() for record in result.records]
        except Exception as e:
            print(f"Query execution error: {e}")
            print(f"Query: {query}")
            print(f"Parameters: {params}")
            return []
            
    def build_local_context(self, entities, related_entities, relationships, chunks, graph_prefix):
        """Build a comprehensive context from search results"""
        context = []
        if chunks:
            context.append(f"# Relevant Information from Source Documents ({graph_prefix} Graph)\n")
            for i, chunk in enumerate(chunks): 
                doc_title = chunk.get('document_title', 'Source document')
                context.append(f"## From: {doc_title}")
                text = chunk['chunk_text']
                context.append(f"{text}\n")

        context.append(f"# Key Entities ({graph_prefix} Graph)\n")
        for entity in entities:
            context.append(f"## {entity['name']} ({entity.get('type', 'Unknown')})")
            context.append(f"{entity.get('description', 'No description available')}\n")
        
        if relationships:
            context.append(f"# Key Relationships ({graph_prefix} Graph)\n")
            for rel in relationships:  
                weight_str = f" (strength: {rel.get('weight', 1.0):.1f})" if 'weight' in rel else ""
                context.append(f"- {rel['source']} → {rel.get('description', 'related to')} → {rel['target']}{weight_str}")
        shown_entities = set(entity['name'] for entity in entities)
        additional_entities = [entity for entity in related_entities 
                             if entity.get('name') and entity['name'] not in shown_entities]
        
        if additional_entities:
            context.append(f"\n# Related Entities ({graph_prefix} Graph)\n")
            for entity in additional_entities:  
                if entity.get('description'):
                    context.append(f"- **{entity['name']}** ({entity.get('type', 'Unknown')}): {entity['description']}")
                else:
                    context.append(f"- **{entity['name']}** ({entity.get('type', 'Unknown')})")
        context_text = "\n".join(context)
        approx_token_count = len(context_text.split())
        
        return {
            "text": context_text,
            "approx_token_count": approx_token_count
        }
   
        
    def rerank_chunks_by_similarity(self,
                                    query_text: str,
                                    chunks: List[Dict[str, Any]],
                                    max_chunks: int) -> List[Dict[str, Any]]:
        """
        Reranks a list of chunks based on their cosine similarity to the query text
        (using precomputed embeddings if available) and returns the top N chunks.

        Args:
            query_text (str): The user's query text.
            chunks (List[Dict[str, Any]]): A list of chunk dictionaries. Each dict
                                            is expected to contain 'chunk_text'
                                            and preferably an 'embedding' key (list or numpy).
            max_chunks (int): The maximum number of top chunks to return.

        Returns:
            List[Dict[str, Any]]: A new list containing the top max_chunks from the
                                  input list, sorted by similarity score descending.
                                  Returns empty list if no chunks or max_chunks is 0.
        """
        if not chunks or max_chunks <= 0:
            return []
        raw_query_embedding = self._get_embedding(query_text)

        if raw_query_embedding is None:
            print("Warning: Failed to generate query embedding for reranking. Returning first max_chunks without ranking.")
            return chunks[:max_chunks]
        try:
            query_embedding_np = np.asarray(raw_query_embedding).reshape(1, -1)
        except Exception as e:
             print(f"Critical Error: Failed to prepare query embedding ({type(raw_query_embedding)}). Cannot rerank. Error: {e}")
             return chunks[:max_chunks] 


        scored_chunks = []
        for chunk in chunks:
            chunk_id = chunk.get('chunk_id', 'N/A')
            chunk_text = chunk.get('chunk_text', '')
            chunk_precomputed_embedding = chunk.get('embedding')

            raw_chunk_embedding = None
            if chunk_precomputed_embedding is not None:
                raw_chunk_embedding = chunk_precomputed_embedding
            elif chunk_text:
                 print("chunk does not have embedding")
                 raw_chunk_embedding = self._get_embedding(chunk_text)

            if raw_chunk_embedding is None:
                print(f"Warning: No valid embedding found or generated for chunk {chunk_id}. Skipping chunk.")
                continue
            try:
                 chunk_embedding_np = np.asarray(raw_chunk_embedding).reshape(1, -1)
            except Exception as e:
                 print(f"Error preparing chunk embedding {chunk_id} ({type(raw_chunk_embedding)}). Skipping chunk. Error: {e}")
                 continue 

            try:
                 similarity = cosine_similarity(query_embedding_np, chunk_embedding_np)[0][0]
                 scored_chunks.append({'chunk': chunk, 'score': similarity})
            except Exception as e:
                 print(f"Unexpected Error calculating similarity for chunk {chunk_id}: {e}. Skipping chunk.")
        scored_chunks.sort(key=lambda x: x['score'], reverse=True)
        top_chunks_data = [item['chunk'] for item in scored_chunks[:max_chunks]]
        return top_chunks_data  
   
    def find_chunks_by_entities(self, entity_ids, graph_prefix, max_chunks_per_entity=7):
        """Find text chunks related to specific entities"""
        if not entity_ids:
            return []
            
        chunk_label = f"{graph_prefix}__Chunk__"
        document_label = f"{graph_prefix}__Document__"
        entity_label = f"{graph_prefix}__Entity__"
        has_entity_rel = f"{graph_prefix}_HAS_ENTITY"
        part_of_rel = f"{graph_prefix}_PART_OF"
        
        query = f"""
        MATCH (e:{entity_label})<-[:{has_entity_rel}]-(c:{chunk_label})
        WHERE e.id IN $entity_ids
        WITH e, c
        ORDER BY c.n_tokens ASC  
        WITH e, collect(c) AS chunks
        UNWIND chunks[0..{max_chunks_per_entity}] AS chunk  
        OPTIONAL MATCH (chunk)-[:{part_of_rel}]->(d:{document_label})
        
        RETURN e.id AS entity_id,
               e.name AS entity_name,
               chunk.id AS chunk_id,
               chunk.text AS chunk_text,
               chunk.embedding As embedding,
               chunk.n_tokens AS chunk_tokens,
               d.title AS document_title
        LIMIT $limit
        """
        result = self._execute_query(query, {
            'entity_ids': entity_ids,
            'limit': len(entity_ids) * max_chunks_per_entity
        })
        return result
        
    def find_chunks_by_vector(self, query_text, graph_prefix, limit=5):
        """Find text chunks directly relevant to the query using vector search"""
        embedding = self._get_embedding(query_text)
        if not embedding:
            print("\n ================ embeding is none \n =======================")
            return []
            
        chunk_label = f"{graph_prefix}__Chunk__"
        document_label = f"{graph_prefix}__Document__"
        part_of_rel = f"{graph_prefix}_PART_OF"
        
        query = f"""
        MATCH (c:{chunk_label})
        WHERE c.embedding IS NOT NULL
        WITH c, vector.similarity.cosine(c.embedding, $query_embedding) AS score
        WHERE score > 0.2
        OPTIONAL MATCH (c)-[:{part_of_rel}]->(d:{document_label})  
        RETURN c.id AS chunk_id,
               c.text AS chunk_text,
               c.n_tokens AS chunk_tokens,
               c.embedding As embedding,
               d.title AS document_title,
               score AS similarity_score
        ORDER BY similarity_score DESC
        LIMIT $limit
        """
        result = self._execute_query(query, {
            'query_embedding': embedding,
            'limit': limit
        })

        return result
    
        
    def find_relationships(self, entity_ids, all_entity_ids, graph_prefix):
        """Find relationships between entities"""
        if not entity_ids or not all_entity_ids:
            return []
            
        entity_label = f"{graph_prefix}__Entity__"
        related_rel = f"{graph_prefix}_RELATED"
        
        query = f"""
        MATCH (e1:{entity_label})-[r:{related_rel}]->(e2:{entity_label})
        WHERE e1.id IN $entity_ids AND e2.id IN $all_entity_ids
        RETURN DISTINCT r.id AS id,
               e1.name AS source,
               e2.name AS target,
               r.description AS description,
               r.weight AS weight
        """
        result = self._execute_query(query, {
            'entity_ids': entity_ids, 
            'all_entity_ids': all_entity_ids
        })
        return result
    
        
    def find_related_entities(self, entity_ids, graph_prefix, max_hops=2, limit=20):
        """Find entities related to the given entities"""
        if not entity_ids:
            return []
            
        entity_label = f"{graph_prefix}__Entity__"
        related_rel = f"{graph_prefix}_RELATED"
        
        query = f"""
        MATCH (e:{entity_label})
        WHERE e.id IN $entity_ids
        MATCH path = (e)-[:{related_rel}*1..{max_hops}]-(related:{entity_label})
        WHERE related.id <> e.id  
        RETURN DISTINCT related.id AS id,
               related.name AS name,
               related.type AS type,
               related.description AS description
        LIMIT $limit
        """
        result = self._execute_query(query, {'entity_ids': entity_ids, 'limit': limit})
        return result
   
      
    def find_entities_by_names(self, entity_names: List[str], graph_prefix: str, limit: int = 5) -> List[Dict[str, Any]]:
        """ Finds entities matching names, ordered by degree. """
        if not entity_names: return []
        entity_label = f"{graph_prefix}__Entity__"
        lowercase_names = [name.lower() for name in entity_names]

        query = f"""
        MATCH (e:{entity_label})
        WHERE toLower(e.name) IN $names AND e.description IS NOT NULL
        WITH e
        OPTIONAL MATCH (e)-[r]-()
        WITH e, count(r) as degree

        RETURN e.id AS id,
               e.name AS name,
               e.type AS type,
               e.description AS description,
               degree
        ORDER BY degree DESC
        LIMIT $limit
        """
        params = {'names': lowercase_names, 'limit': limit}
        query_entities_graph = self._execute_query(query, params)
        return query_entities_graph
        
    def find_entities_by_vector(self, query_text, graph_prefix, max_entities=5):
        """Find entities using vector search"""
        embedding = self._get_embedding(query_text)
        if not embedding:
            return []
            
        entity_label = f"{graph_prefix}__Entity__"
        query = f"""
        MATCH (node:{entity_label})
        WHERE node.embedding IS NOT NULL
        WITH node, vector.similarity.cosine(node.embedding, $query_embedding) AS score
        WHERE score > 0.6 AND node.description IS NOT NULL

        RETURN node.id AS id,
               node.name AS name,
               node.type AS type, 
               node.description AS description,
               score AS similarity_score
        ORDER BY similarity_score DESC
        LIMIT $limit
        """
        params = {
            'query_embedding': embedding,
            'limit': max_entities
        }
        result = self._execute_query(query, params)
        return result
             
    def local_search(self,
                     query_text: str,
                     graph_prefix: str = "DL",
                     llm_entities: Optional[List[str]] = None, 
                     max_total_seed_entities: int = 8, 
                     max_hops: int = 4,
                     max_chunks: int = 7,
                     max_vector_candidates: int = 8,
                     max_llm_name_candidates: int = 8
                     ) -> Dict[str, Any]:
        """
        Local search combining seed entities from vector search and LLM entity names.
        Prioritizes locally important (high-degree) entities in the combined seed list.

        Args:
            query_text (str): User query text.
            graph_prefix (str): Graph prefix.
            precomputed_query_entities (Optional[List[str]]): Entities extracted by LLM beforehand.
            max_total_seed_entities (int): Max number of unique seed entities AFTER combining sources.
            max_hops (int): Max traversal hops.
            max_chunks (int): Max text chunks to fetch.
            max_vector_candidates (int): Initial limit for vector search results.
            max_llm_name_candidates (int): Initial limit for name matching results.


        Returns:
            dict: Search results including combined entities, relationships, context.
        """
        print("--- local_search: Finding vector candidates ---")
        entities_vector = self.find_entities_by_vector(query_text, graph_prefix, max_vector_candidates)
        print(f"--- local_search: Found {len(entities_vector)} vector candidates ---")

        entities_llm_names = []
        if llm_entities:
            entities_llm_names = self.find_entities_by_names(llm_entities, graph_prefix, max_llm_name_candidates)
            print(f"--- local_search: Found {len(entities_llm_names)} LLM name candidates ---")
        else:
            print("--- local_search: Skipping LLM name search (no precomputed entities) ---")

        combined_entities_map: Dict[str, Dict[str, Any]] = {}
        for entity in entities_vector:
            entity_id = entity['id']
            if entity_id not in combined_entities_map:
                 combined_entities_map[entity_id] = entity.copy() 
                 combined_entities_map[entity_id]['degree'] = entity.get('degree', 0)
                 combined_entities_map[entity_id]['source'] = 'vector' 
        for entity in entities_llm_names:
            entity_id = entity['id']
            if entity_id not in combined_entities_map:
                 combined_entities_map[entity_id] = entity.copy()
                 combined_entities_map[entity_id]['degree'] = entity.get('degree', 0)
                 combined_entities_map[entity_id]['source'] = 'llm_name'
            else:
                 combined_entities_map[entity_id]['source'] += '+llm_name'
                 combined_entities_map[entity_id]['degree'] = max(
                     combined_entities_map[entity_id].get('degree', 0),
                     entity.get('degree', 0)
                 )
        combined_entity_list = list(combined_entities_map.values())
        print(f"--- local_search: Combined unique candidate entities: {len(combined_entity_list)} ---")
        combined_entity_list.sort(key=lambda x: x.get('degree', 0), reverse=True)
        seed_entities = combined_entity_list[:max_total_seed_entities]
        seed_entity_ids = [entity['id'] for entity in seed_entities]
        print(f"--- local_search: Selected final {len(seed_entities)} seed entities (sorted by degree) ---")
        related_entities = self.find_related_entities(seed_entity_ids, graph_prefix, max_hops)
        all_entity_ids = seed_entity_ids + [e['id'] for e in related_entities if e['id'] not in seed_entity_ids]
        relationships = self.find_relationships(seed_entity_ids, all_entity_ids, graph_prefix)
        chunks_vector = self.find_chunks_by_vector(query_text, graph_prefix, max_chunks*2)
        print("local vector searching chunks length", len(chunks_vector))
        chunks_entities = []
        if seed_entity_ids:
             chunks_entities = self.find_chunks_by_entities(seed_entity_ids, graph_prefix, max_chunks*2) 
             print("local vector searching chunks length chunks_entities", len(chunks_entities))

        combined_chunks_map = {}
        for chunk in (chunks_vector or []):
            chunk_id = chunk.get('chunk_id')
            if chunk_id and chunk_id not in combined_chunks_map:
                combined_chunks_map[chunk_id] = chunk
        for chunk in (chunks_entities or []):
            chunk_id = chunk.get('chunk_id')
            if chunk_id and chunk_id not in combined_chunks_map:
                combined_chunks_map[chunk_id] = chunk

        final_chunks = list(combined_chunks_map.values())
        print("the length of final chunks", len(final_chunks))
        final_chunks_ = self.rerank_chunks_by_similarity(query_text, final_chunks, max_chunks)  
        context = self.build_local_context(
            seed_entities,
            related_entities,
            relationships,
            final_chunks_, 
            graph_prefix
        )

        result = {
            'seed_entities': seed_entities,
            'related_entities': related_entities,
            'relationships': relationships,
            'text_chunks': final_chunks,
            'context': context.get('text', ''),
            'approx_token_count': context.get('approx_token_count', 0),
            'query': query_text,
            'graph_prefix': graph_prefix,
        }
        return result
    
class GlobalSearch:   
    """
        Knowledge graph search class for retrieving information from Neo4j graph database.
        Supports multiple search strategies across different prefixed graphs.
    """
    
    def __init__(self, 
                 driver,
                 database=None, 
                 llm=None,
                 embedding_model=None):
        """
        Initialize the search class with Neo4j driver and configuration.
        
        Args:
            driver: Neo4j driver instance
            database: Name of Neo4j database to use (or None for default)
            embedding_model_name: Name of the embedding model to use
        """
        self.driver = driver
        self.database = database
        self.model = embedding_model
        self.llm = llm

    def _execute_query(self, query, params=None):
        """Execute a Cypher query and return results as a list of dictionaries"""
        if params is None:
            params = {}
            
        try:
            result = self.driver.execute_query(query, params, database_=self.database)
            return [record.data() for record in result.records]
        except Exception as e:
            print(f"Query execution error: {e}")
            print(f"Query: {query}")
            print(f"Parameters: {params}")
            return []
    
    def _rank_communities_by_embedding_similarity(self,
                                                  query_embedding: List[float],
                                                  communities_data: List[Dict[str, Any]],
                                                  graph_prefix: str 
                                                  ) -> List[Dict[str, Any]]:
        """
        Ranks communities based on the cosine similarity of their embeddings to the query embedding.
        Assumes communities_data contains dicts, each with an 'embedding' key.
        Adds a 'embedding_relevance_score' to each community.
        """
        if not query_embedding:
            print("Error (Rank Communities by Embedding): Query embedding not provided.")
            for comm in communities_data:
                comm['embedding_relevance_score'] = 0.0
            return communities_data
        
        if not communities_data:
            return []

        query_emb_np = np.array(query_embedding).reshape(1, -1)
        
        scored_communities = []
        for comm in communities_data:
            updated_comm = comm.copy()
            comm_emb_list = updated_comm.get('embedding')
            
            if comm_emb_list and isinstance(comm_emb_list, list) and len(comm_emb_list) > 0:
                try:
                    comm_emb_np = np.array(comm_emb_list).reshape(1, -1)
                    if query_emb_np.shape[1] != comm_emb_np.shape[1]:
                        print(f"Warning: Embedding dimension mismatch for community {updated_comm.get('id')}. Query: {query_emb_np.shape[1]}, Comm: {comm_emb_np.shape[1]}. Skipping.")
                        similarity = 0.0
                    else:
                        similarity = cosine_similarity(query_emb_np, comm_emb_np)[0][0]
                except ValueError as ve: 
                    print(f"Warning: ValueError converting embedding to NumPy array for community {updated_comm.get('id')}: {ve}. Skipping.")
                    similarity = 0.0
                except Exception as e:
                    print(f"Error calculating similarity for community {updated_comm.get('id')}: {e}")
                    similarity = 0.0
            else:
                print(f"Warning: Community {updated_comm.get('id')} has no valid embedding. Assigning 0 similarity.")
                similarity = 0.0
            
            updated_comm['embedding_relevance_score'] = float(similarity)
            scored_communities.append(updated_comm)
            
        scored_communities.sort(key=lambda x: x.get('embedding_relevance_score', 0.0), reverse=True)
        print(f"Ranked {len(scored_communities)} communities by embedding similarity.")
        return scored_communities
    
    
    def _fetch_all_community_content(self, graph_prefix: str, levels: Optional[List[int]] = None) -> List[Dict[str, Any]]:
        """Fetches ID, title, and full_content for all communities, optionally filtered by level."""
        community_label = f"{graph_prefix}__Community__"
        level_filter_str = ""
        if levels is not None and levels:
            level_filter_str = "WHERE c.level IN $levels"

        query = f"""
        MATCH (c:{community_label})
        {level_filter_str}
        WHERE c.full_content IS NOT NULL AND c.id IS NOT NULL AND c.title IS NOT NULL
        RETURN c.id AS id, c.community AS community_id, c.title AS title, c.full_content AS content, c.level as level, c.embedding AS embedding
        // Optionally add c.summary if full_content is too much for LLM
        """
        params = {}
        if levels:
            params['levels'] = levels
            
        results = self._execute_query(query, params)
        return results

    def build_global_context(self,
                             query_text: str, 
                             graph_prefix: str,
                             selected_communities: List[Dict[str, Any]], 
                             full_structural_hierarchy: Optional[Dict[str, Any]], 
                             top_reranked_chunks: List[Dict[str, Any]],   
                             cross_community_relationships: Optional[List[Dict[str, Any]]],
                             max_communities_to_display: int = 7,
                             max_cross_rels_to_display: int = 5
                            ) -> Dict[str, Any]:
        """
        Builds a global context string using LLM-ranked communities and LLM-reranked chunks.

        Args:
            query_text: The user's query.
            graph_prefix: The graph prefix string.
            selected_llm_ranked_communities: List of community dicts, already ranked by LLM.
            full_structural_hierarchy: The complete community hierarchy map.
            top_reranked_chunks: List of chunk dicts, already reranked by LLM and limited.
            cross_community_relationships: List of cross-community relationship dicts.
            max_communities_to_display: Max number of community summaries to include.
            max_cross_rels_to_display: Max number of cross-community relationships to show.

        Returns:
            A dictionary containing the "text" of the context and "approx_token_count".
        """
        context_lines = []
        
        context_lines.append(f"## Key Themes & Information Clusters (Identified by LLM, {graph_prefix} Graph)\n")
        communities_displayed_count = 0

        chunks_displayed_count = 0
        if top_reranked_chunks:
            context_lines.append("## Detailed Supporting Information (Snippets Ranked by LLM Relevance to Query)\n")
            seen_chunk_texts = set() 
            for chunk_data in top_reranked_chunks: 
                chunk_text_content = chunk_data.get('text') or chunk_data.get('chunk_text')
                if not chunk_text_content or chunk_text_content in seen_chunk_texts:
                    continue
                seen_chunk_texts.add(chunk_text_content)

                entity_name_origin = chunk_data.get('entity_name', 'Source Document') 
                chunk_llm_score = chunk_data.get('llm_relevance_score', 0.0)
                
                context_lines.append(f"### Snippet (LLM Relevance: {chunk_llm_score:.2f}, Related to: {entity_name_origin})\n")
                if chunk_data.get('llm_justification'):
                    context_lines.append(f"*LLM Rationale for this snippet:* {chunk_data['llm_justification']}\n")
                context_lines.append(f"{chunk_text_content}\n")
                chunks_displayed_count += 1
        
        cross_rels_displayed_count = 0
        if cross_community_relationships:
            context_lines.append("## Connections Between Identified Themes\n")
            for rel in cross_community_relationships[:max_cross_rels_to_display]:
                source = rel.get('source_name', 'Unknown Source')
                target = rel.get('target_name', 'Unknown Target')
                rel_type_str = str(rel.get('relationship_type', 'RELATED'))
                link_desc_raw = rel_type_str
                if graph_prefix and rel_type_str.startswith(graph_prefix + "_"):
                    link_desc_raw = rel_type_str[len(graph_prefix)+1:]
                link_desc_cleaned = link_desc_raw.replace("_", " ").lower()
                rel_desc_prop = rel.get('relationship_description', rel.get('description'))
                final_link_desc = rel_desc_prop if rel_desc_prop else link_desc_cleaned
                
                context_lines.append(f"- **{source}** --[{final_link_desc}]--> **{target}**")
                cross_rels_displayed_count += 1
            if cross_rels_displayed_count == 0:
                context_lines.append("(No significant direct connections found between the top themes for this query.)")
            context_lines.append("\n")

        if communities_displayed_count == 0 and chunks_displayed_count == 0 and cross_rels_displayed_count == 0:
             context_lines = ["No highly relevant themes, information snippets, or connections found for the query based on LLM assessment."]

        cleaned_context_text = "\n\n".join([line for line in "\n".join(context_lines).split('\n') if line.strip()])
        
        approx_token_count = len(cleaned_context_text.split()) 
        
        return {
            "text": cleaned_context_text,
            "approx_token_count": approx_token_count,
            "llm_ranked_communities_displayed": communities_displayed_count,
            "llm_reranked_chunks_displayed": chunks_displayed_count
        }
        
        
    def _find_communities_by_entity_names(self,
                                          graph_prefix: str,
                                          entity_names: List[str],
                                          level: int = 0,
                                          limit_per_entity: int = 2 
                                          ) -> List[Dict[str, Any]]:
        """
        Finds communities that contain entities matching the provided names at a specific level.
        Returns community details, assigning a simple presence score.
        """
        if not entity_names: return []

        community_label = f"{graph_prefix}__Community__"
        entity_label = f"{graph_prefix}__Entity__"
        in_community_rel = f"{graph_prefix}_IN_COMMUNITY"
        lowercase_names = [name.lower() for name in entity_names]

        query = f"""
        MATCH (e:{entity_label})
        WHERE toLower(e.name) IN $names
        WITH collect(e) AS matched_entities 

        UNWIND matched_entities AS entity_node

        MATCH (c:{community_label} {{level: $level}})<-[:{in_community_rel}]-(entity_node)
        WHERE c.summary IS NOT NULL

        RETURN DISTINCT c.id AS id,
            c.community AS community_id,
            c.title AS title,
            c.summary AS summary,
            c.rank AS rank,
            c.full_content AS full_content,
            c.period AS period,
            c.size AS size,
            1.0 AS entity_presence_score
        ORDER BY c.rank DESC 
        LIMIT $overall_limit 
        """
        overall_limit = len(lowercase_names) * limit_per_entity

        params = {'names': lowercase_names, 'level': level, 'overall_limit': overall_limit}
        results = self._execute_query(query, params)
        print(f"DEBUG: Found {len(results)} communities via entity names: {[r['title'][:30] for r in results]}")
        return results     
    
    def get_entity_text_chunks(self, 
                               entity_ids: List[str], 
                               graph_prefix: str, 
                               query_embedding: List[float], 
                               max_chunks_per_entity: int = 2
                               ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get source text chunks for entities, ranked by relevance to query embedding.
        Now also returns the embedding of each chunk.
        """
        if not entity_ids: 
            print("Warning (get_entity_text_chunks): No entity_ids provided.")
            return {}
        if not query_embedding: 
            print("Warning (get_entity_text_chunks): Query embedding not provided, cannot rank by similarity.")
            return {}


        entity_label = f"{graph_prefix}__Entity__"
        chunk_label = f"{graph_prefix}__Chunk__"
        has_entity_rel = f"{graph_prefix}_HAS_ENTITY"

        query = f"""
        MATCH (e:{entity_label})<-[:{has_entity_rel}]-(c:{chunk_label})
        WHERE e.id IN $entity_ids AND c.embedding IS NOT NULL
        WITH e, c, vector.similarity.cosine(c.embedding, $query_embedding) AS score
        ORDER BY score DESC 
        WITH e, collect({{
            chunk_node: c,  
            score: score
        }}) AS chunks_with_scores
        UNWIND chunks_with_scores[0..$max_per_entity] AS chunk_data_with_node

        WITH e, chunk_data_with_node.chunk_node AS chunk, chunk_data_with_node.score AS relevance_score
        RETURN e.id AS entity_id,
            e.name AS entity_name,
            chunk.id AS chunk_id,         
            chunk.text AS chunk_text,   
            chunk.n_tokens AS chunk_tokens,
            chunk.embedding AS chunk_embedding, 
            relevance_score AS relevance_score  
        """

        params = {
            'entity_ids': entity_ids,
            'max_per_entity': max_chunks_per_entity,
            'query_embedding': query_embedding
        }

        query_results = self._execute_query(query, params)

        entity_chunks_map = {} 
        for record in query_results:
            entity_id = record['entity_id']
            if entity_id not in entity_chunks_map:
                entity_chunks_map[entity_id] = []
            
            entity_chunks_map[entity_id].append({
                'chunk_id': record['chunk_id'],
                'text': record.get('chunk_text', ''), 
                'tokens': record.get('chunk_tokens', 0),
                'embedding': record.get('chunk_embedding', []), 
                'relevance': record.get('relevance_score', 0.0), 
                'entity_name': record.get('entity_name', 'Unknown Entity') 
            })
        
        return entity_chunks_map
    

    async def _process_single_llm_batch_async(self,
                                              chain: Any, # LangChain LCEL chain
                                              query_text: str,
                                              batch_chunks_for_prompt_payload: List[Dict[str, Any]],
                                              original_batch_data: List[Dict[str, Any]], # To map results back
                                              batch_number: int,
                                              max_retries: int
                                              ) -> List[Dict[str, Any]]:
        """
        Asynchronously processes a single batch of chunks with the LLM, including retries.
        Returns a list of chunks from this batch with LLM scores.
        """
        current_retries = 0
        processed_batch_chunks = [] # Chunks from this specific batch with LLM scores or fallback

        while current_retries < max_retries:
            try:
                print(f"LLM Chunk Reranker (Async): Processing Batch {batch_number}, Attempt {current_retries + 1}/{max_retries}.")
                # Ensure your LangChain chain supports ainvoke
                # If self.llm is a LangChain LLM, prompt | self.llm | StrOutputParser should support ainvoke
                response_str = await chain.ainvoke({ # Use await for asynchronous call
                    "query_input": query_text,
                    "chunks_json_input": json.dumps(batch_chunks_for_prompt_payload, indent=2)
                })
                
                response_str = response_str.strip().removeprefix("```json").removesuffix("```").strip()
                llm_evaluations = json.loads(response_str)

                if isinstance(llm_evaluations, list) and \
                   all(isinstance(item, dict) and "chunk_id" in item and "llm_relevance_score" in item for item in llm_evaluations):
                    for eval_item in llm_evaluations:
                        original_chunk_id_str = str(eval_item.get("chunk_id"))
                        original_chunk_data = next(
                            (c for c in original_batch_data if str(c.get('id') or c.get('chunk_id')) == original_chunk_id_str),
                            None
                        )
                        if original_chunk_data:
                            updated_chunk = original_chunk_data.copy()
                            try:
                                score_val = eval_item.get("llm_relevance_score", 0.0)
                                updated_chunk['llm_relevance_score'] = float(score_val)
                            except (ValueError, TypeError):
                                print(f"Warning (LLM Async Batch {batch_number}): Invalid LLM score '{score_val}' for chunk '{original_chunk_id_str}'. Defaulting to 0.0.")
                                updated_chunk['llm_relevance_score'] = 0.0
                            updated_chunk['llm_justification'] = eval_item.get("justification", "")
                            processed_batch_chunks.append(updated_chunk)
                        else:
                            print(f"Warning (LLM Async Batch {batch_number}): LLM returned ID '{original_chunk_id_str}' not found in original batch details.")
                    print(f"LLM Chunk Reranker (Async): Batch {batch_number} processed successfully.")
                    return processed_batch_chunks # Success for this batch
                else:
                    raise ValueError(f"LLM returned JSON in unexpected format: {str(llm_evaluations)[:200]}...")

            except (json.JSONDecodeError, ValueError) as e_parse:
                current_retries += 1
                response_for_log = locals().get('response_str', 'Response string not captured')
                print(f"Error (LLM Async Batch {batch_number}), Attempt {current_retries}/{max_retries}: Parse/Format Error: {e_parse}. LLM Output: '{response_for_log[:300]}...'")
                if current_retries >= max_retries:
                    print(f"LLM Chunk Reranker (Async): Batch {batch_number} max retries for Parse/Format. Applying fallback.")
                    break # Exit retry loop for this batch
                await asyncio.sleep(1 * (2**current_retries)) # Async sleep
            except Exception as e_chain:
                current_retries += 1
                print(f"Error (LLM Async Batch {batch_number}), Attempt {current_retries}/{max_retries}: LLM Chain Execution Error: {e_chain}")
                if current_retries >= max_retries:
                    print(f"LLM Chunk Reranker (Async): Batch {batch_number} max retries for Chain Error. Applying fallback.")
                    break # Exit retry loop for this batch
                await asyncio.sleep(1 * (2**current_retries))

        # Fallback for this batch if all retries failed
        print(f"LLM Chunk Reranker (Async): Batch {batch_number} failed all retries. Applying fallback scores.")
        fallback_batch_chunks = []
        for chunk_data in original_batch_data:
            updated_chunk = chunk_data.copy()
            updated_chunk['llm_relevance_score'] = updated_chunk.get('relevance', 0.0)
            updated_chunk['llm_justification'] = "Fallback - LLM processing failed for batch."
            fallback_batch_chunks.append(updated_chunk)
        return fallback_batch_chunks

    def _embedding_rerank_chunks(self,
                                 query_embedding_: List[float], 
                                 chunks_to_rerank: List[Dict[str, Any]], 
                                 target_final_chunk_count: int = 12,
                                 min_embedding_score_threshold: float = 0.5
                                 ) -> List[Dict[str, Any]]:
        """
        Reranks chunks based purely on the cosine similarity of their precomputed embeddings
        to the precomputed query embedding.

        Args:
            query_embedding_: The precomputed embedding vector for the query.
            chunks_to_rerank: List of chunk dicts. Expected keys: 'id' (or 'chunk_id'),
                               'text' (or 'chunk_text'), and a precomputed 'embedding' (list of floats).
            target_final_chunk_count: The desired number of top chunks to return.
            min_embedding_score_threshold: Minimum cosine similarity score to keep a chunk.

        Returns:
            A list of chunk dictionaries, sorted by 'embedding_relevance_score',
            filtered by threshold, and limited by target_final_chunk_count.
            Each chunk will have 'embedding_relevance_score' added.
        """
        if not query_embedding_:
            print("Error (_embedding_rerank_chunks): Query embedding not provided. Cannot rerank.")
            chunks_to_rerank.sort(key=lambda x: x.get('relevance', 0.0), reverse=True)
            return chunks_to_rerank[:target_final_chunk_count]

        if not chunks_to_rerank:
            print("Info (_embedding_rerank_chunks): No chunks provided to rerank.")
            return []

        query_emb_np = np.array(query_embedding_).reshape(1, -1)
        
        scored_chunks = []
        for i, chunk_data in enumerate(chunks_to_rerank):
            updated_chunk = chunk_data.copy()
            
            chunk_embedding_list = updated_chunk.get('embedding') 
            similarity_score = 0.0

            if chunk_embedding_list and isinstance(chunk_embedding_list, list) and len(chunk_embedding_list) > 0:
                try:
                    chunk_emb_np = np.array(chunk_embedding_list).reshape(1, -1)
                    if query_emb_np.shape[1] == chunk_emb_np.shape[1]:
                        similarity_score = cosine_similarity(query_emb_np, chunk_emb_np)[0][0]
                    else:
                        print(f"Warning (_embedding_rerank_chunks): Embedding dimension mismatch for chunk ID {updated_chunk.get('id') or updated_chunk.get('chunk_id', i)}. Query: {query_emb_np.shape[1]}, Chunk: {chunk_emb_np.shape[1]}. Score set to 0.")
                except ValueError as ve: 
                    print(f"Warning (_embedding_rerank_chunks): ValueError for chunk ID {updated_chunk.get('id') or updated_chunk.get('chunk_id', i)} embedding: {ve}. Score set to 0.")
                except Exception as e:
                    print(f"Error (_embedding_rerank_chunks): Calculating similarity for chunk ID {updated_chunk.get('id') or updated_chunk.get('chunk_id', i)}: {e}. Score set to 0.")
            else:
                print(f"Warning (_embedding_rerank_chunks): Chunk ID {updated_chunk.get('id') or updated_chunk.get('chunk_id', i)} has no valid precomputed 'embedding' attribute. Score set to 0.")
            
            updated_chunk['embedding_relevance_score'] = float(similarity_score)
            scored_chunks.append(updated_chunk)

        if min_embedding_score_threshold > -1.0: 
             relevant_by_embedding = [
                 chunk for chunk in scored_chunks
                 if chunk.get('embedding_relevance_score', -1.0) > min_embedding_score_threshold # Default to low if score missing
             ]
             print(f"Embedding Reranker: {len(relevant_by_embedding)} chunks after filtering by score > {min_embedding_score_threshold} (out of {len(scored_chunks)}).")
        else:
            relevant_by_embedding = scored_chunks

        relevant_by_embedding.sort(key=lambda x: x.get('embedding_relevance_score', 0.0), reverse=True)
        final_output_chunks = relevant_by_embedding[:target_final_chunk_count]
        print(f"Embedding Reranker: Finished. Returning {len(final_output_chunks)} chunks.")
        if final_output_chunks:
            top_c_id = final_output_chunks[0].get('id') or final_output_chunks[0].get('chunk_id')
            print(f"Top embedding-reranked chunk: ID: {top_c_id}, Embedding Score: {final_output_chunks[0].get('embedding_relevance_score')}")

        return final_output_chunks

    async def _llm_rerank_chunks_async(self, 
                                       query_text: str,
                                       chunks_to_rerank: List[Dict[str, Any]],
                                       llm_batch_size: int = 5,
                                       target_final_chunk_count: int = 10,
                                       max_retries_per_batch: int = 3,
                                       chunk_text_max_length: int = 1500,
                                       min_llm_score_threshold: float = 0.1 
                                       ) -> List[Dict[str, Any]]:
        if not self.llm:
            print("CRITICAL (LLM Chunk Reranker Async): LLM client (self.llm) is not initialized. Skipping.")
            chunks_to_rerank.sort(key=lambda x: x.get('relevance', x.get('similarity_score', 0.0)), reverse=True)
            
            return chunks_to_rerank[:target_final_chunk_count] 

        if not hasattr(self.llm, 'ainvoke') and not asyncio.iscoroutinefunction(self.llm.invoke):
             print("CRITICAL (LLM Chunk Reranker Async): LLM client does not support asynchronous 'ainvoke'. Falling back to synchronous (not implemented here, this function expects async).")
             chunks_to_rerank.sort(key=lambda x: x.get('relevance', 0.0), reverse=True)
             return chunks_to_rerank[:target_final_chunk_count]


        if not chunks_to_rerank:
            print("Info (LLM Chunk Reranker Async): No chunks provided.")
            return []

        sorted_candidate_chunks = sorted(
            chunks_to_rerank,
            key=lambda x: x.get('relevance', 0.0),
            reverse=True
        )
        chunks_for_llm_processing = sorted_candidate_chunks

        if not chunks_for_llm_processing:
            print("Info (LLM Chunk Reranker Async): No candidate chunks after initial sorting/limiting.")
            return []

        print(f"LLM Chunk Reranker (Async): Preparing to process {len(chunks_for_llm_processing)} chunks in batches of {llm_batch_size}.")

        prompt_template_str = """
    You are an expert relevance assessor. Given a User Query and a list of Text Chunks, your task is to evaluate how relevant each chunk is to *directly answering or contributing essential, specific information to answer* the User Query.
    Focus on identifying chunks that provide **specific facts, evidence, detailed descriptions, scientific findings, or arguments** that would be part of a comprehensive answer.
    **Critically, assign a very low score (0.0-0.1) to chunks that are primarily navigational website text (menus, lists of links, footers), boilerplate, or extremely generic summaries that lack actionable detail for the query, even if they contain keywords from the query.**
    A chunk is highly relevant (0.7-1.0) only if it contains substantive, detailed information directly pertinent to answering the query.

    User Query:
    "{query_input}"

    Text Chunks to Evaluate:
    ```json
    {chunks_json_input}
    ```

    Return ONLY a valid JSON list of objects, where each object has the following keys:
    - "chunk_id": The original ID of the chunk.
    - "llm_relevance_score": A float between 0.0 and 1.0.
    - "justification": A brief explanation for the score (1-2 sentences).

    Example for a single chunk object in the list:
    {{
        "chunk_id": "some_id_abc",
        "llm_relevance_score": 0.8,
        "justification": "This chunk directly mentions the key concept X and its impact on Y, which is central to the query."
    }}

    JSON List Output:
    """
        try:
            prompt = ChatPromptTemplate.from_template(prompt_template_str)
            chain = prompt | self.llm | StrOutputParser()
        except Exception as e_chain_setup:
            print(f"CRITICAL ERROR (LLM Chunk Reranker Async): Failed to set up LangChain chain: {e_chain_setup}. Aborting.")
            processed_chunks_with_llm_score = []
            for chunk_data in chunks_for_llm_processing:
                updated_chunk = chunk_data.copy()
                updated_chunk['llm_relevance_score'] = updated_chunk.get('relevance', 0.0)
                updated_chunk['llm_justification'] = "Fallback - LLM chain setup error."
                processed_chunks_with_llm_score.append(updated_chunk)
            processed_chunks_with_llm_score.sort(key=lambda x: x.get('llm_relevance_score', 0.0), reverse=True)
            filtered_chunks = [c for c in processed_chunks_with_llm_score if c.get('llm_relevance_score', 0.0) > min_llm_score_threshold]
            return filtered_chunks[:target_final_chunk_count]

        tasks = []
        batch_number_counter = 0
        for i in range(0, len(chunks_for_llm_processing), llm_batch_size):
            batch_data_orig_for_task = chunks_for_llm_processing[i:i+llm_batch_size]
            batch_number_counter += 1
            
            chunks_for_prompt_payload_for_task = []
            for chunk_data in batch_data_orig_for_task:
                cid = chunk_data.get('id') or chunk_data.get('chunk_id')
                ctext = chunk_data.get('text') or chunk_data.get('chunk_text', '')
                if cid and ctext:
                    chunks_for_prompt_payload_for_task.append({
                        "id": str(cid),
                        "text": ctext[:chunk_text_max_length]
                    })
            
            if not chunks_for_prompt_payload_for_task:
                print(f"LLM Chunk Reranker (Async): Batch {batch_number_counter} is empty for LLM. Adding original chunks with fallback score.")
                temp_fallback_chunks = []
                for cd in batch_data_orig_for_task:
                    uc = cd.copy()
                    uc['llm_relevance_score'] = uc.get('relevance', 0.0)
                    uc['llm_justification'] = "Fallback - Batch payload empty for LLM."
                    temp_fallback_chunks.append(uc)
                async def dummy_task(data): return data
                tasks.append(dummy_task(temp_fallback_chunks))
                continue

            tasks.append(self._process_single_llm_batch_async(
                chain,
                query_text,
                chunks_for_prompt_payload_for_task,
                batch_data_orig_for_task, 
                batch_number_counter,
                max_retries_per_batch
            ))

        all_batched_results = await asyncio.gather(*tasks)
        
        processed_chunks_with_llm_score = []
        for batch_result in all_batched_results:
            processed_chunks_with_llm_score.extend(batch_result) 
        highly_relevant_chunks = [
            chunk for chunk in processed_chunks_with_llm_score
            if chunk.get('llm_relevance_score', 0.0) > min_llm_score_threshold
        ]
        print(f"LLM Chunk Reranker (Async): {len(highly_relevant_chunks)} chunks after filtering by score > {min_llm_score_threshold} (out of {len(processed_chunks_with_llm_score)} processed).")

        highly_relevant_chunks.sort(key=lambda x: x.get('llm_relevance_score', 0.0), reverse=True)
        final_output_chunks = highly_relevant_chunks[:target_final_chunk_count]
        print(f"LLM Chunk Reranker (Async): Finished. Returning {len(final_output_chunks)} chunks.")
        if final_output_chunks:
            top_c_id = final_output_chunks[0].get('id') or final_output_chunks[0].get('chunk_id')
            print(f"Top LLM-reranked chunk (Async): ID: {top_c_id}, LLM Score: {final_output_chunks[0].get('llm_relevance_score')}")

        return final_output_chunks  
       
       
    def _rerank_communities_by_chunk_relevance(self,
                                               communities: List[Dict[str, Any]],
                                               query_embedding: List[float],
                                               graph_prefix: str,
                                               max_chunks_per_entity_for_reranking: int = 3,
                                               alpha: float = 0.6, 
                                               beta: float = 0.4   
                                               ) -> List[Dict[str, Any]]:
        """
        Re-ranks a list of communities based on the relevance of their constituent chunks to the query.
        """
        if not communities or not query_embedding:
            return communities

        print(f"Re-ranking {len(communities)} communities by chunk relevance...")
        reranked_communities = []

        all_entity_ids_in_candidates = list(set(
            eid for comm in communities for eid in comm.get('entity_ids', []) if eid
        ))

        if not all_entity_ids_in_candidates:
            print("No entities found in candidate communities for re-ranking. Returning original order.")
            for comm in communities:
                comm['final_rerank_score'] = comm.get('relevance_score', 0.0)
            return communities

        entity_chunks_map_for_reranking = self.get_entity_text_chunks(
            entity_ids=all_entity_ids_in_candidates,
            graph_prefix=graph_prefix,
            query_embedding=query_embedding,
            max_chunks_per_entity=max_chunks_per_entity_for_reranking
        )

        for comm in communities:
            original_community_score = comm.get('relevance_score', 0.0)
            chunk_relevance_for_community_score = 0.0
            
            community_entity_ids = comm.get('entity_ids', [])
            if community_entity_ids and entity_chunks_map_for_reranking:
                top_chunk_scores_for_this_community = []
                for entity_id in community_entity_ids:
                    if entity_id in entity_chunks_map_for_reranking:
        
                        for chunk_data in entity_chunks_map_for_reranking[entity_id]:
                            top_chunk_scores_for_this_community.append(chunk_data.get('relevance', 0.0))
                
                if top_chunk_scores_for_this_community:
                   
                    if top_chunk_scores_for_this_community:
                         chunk_relevance_for_community_score = sum(top_chunk_scores_for_this_community) / len(top_chunk_scores_for_this_community)


            new_community_score = (alpha * original_community_score) + (beta * chunk_relevance_for_community_score)
            comm_copy = comm.copy() 
            comm_copy['final_rerank_score'] = new_community_score
            comm_copy['score_components_rerank'] = {
                'original_relevance_score': original_community_score,
                'chunk_relevance_contribution': chunk_relevance_for_community_score,
                'alpha': alpha, 'beta': beta
            }
            reranked_communities.append(comm_copy)

        reranked_communities.sort(key=lambda x: x['final_rerank_score'], reverse=True)
        print("Re-ranking complete.")
        return reranked_communities
     
         
    def get_community_entities_map(self, community_ids, graph_prefix):
        """
        Get entity IDs for specific communities
        
        Returns:
            Dict mapping community IDs to lists of entity IDs
        """
        if not community_ids:
            return {}
            
        community_label = f"{graph_prefix}__Community__"
        entity_label = f"{graph_prefix}__Entity__"
        in_community_rel = f"{graph_prefix}_IN_COMMUNITY"
        
        query = f"""
        MATCH (c:{community_label})<-[:{in_community_rel}]-(e:{entity_label})
        WHERE c.id IN $community_ids
        RETURN c.id AS community_id, collect(e.id) AS entity_ids
        """
        
        result = self._execute_query(query, {'community_ids': community_ids})
        
        community_entity_map = {}
        for record in result:
            community_entity_map[record['community_id']] = record['entity_ids']

        return community_entity_map  
  
  
        
    def find_community_reports(self, graph_prefix, llm_entities, query_embedding, level=0, limit=30, min_similarity=0.2):
        """
        Finds relevant communities by combining results from vector similarity search
        and a search based on LLM-extracted entity names. Returns a de-duplicated list.

        Args:
            graph_prefix: Graph prefix.
            query_text: User query text (used if entities are not precomputed).
            query_embedding: Embedding of the user query.
            precomputed_query_entities: Optional list of lowercase entities from LLM.
            level: Community level.
            limit: Approximate final number of unique communities to return.
            min_vector_similarity: Min vector score for vector search candidates.
            vector_limit_factor: Fetch limit*factor candidates via vector search.
            max_llm_entities: Max entities to extract if not precomputed.
            entity_community_limit_per_entity: Max communities to fetch per matched entity name.

        Returns:
            List of unique community report dictionaries, potentially more than 'limit' initially.
            Further ranking might be needed downstream.
        """
        if not query_embedding:
            return []
        vector_search_limit = int(limit * limit)
        community_label = f"{graph_prefix}__Community__"
        vector_query = f"""
        MATCH (c:{community_label} {{level: $level}})
        WHERE c.summary IS NOT NULL AND c.embedding IS NOT NULL
        WITH c, vector.similarity.cosine(c.embedding, $query_embedding) AS vector_score
        WHERE vector_score >= $min_vector_similarity
        RETURN c.id AS id, c.community AS community_id, c.title AS title, c.summary AS summary,
               c.rank AS rank, c.full_content AS full_content, c.period AS period,
               c.size AS size, vector_score
        ORDER BY vector_score DESC LIMIT $vector_search_limit
        """
        vector_params = {
            'level': level, 'vector_search_limit': vector_search_limit,
            'query_embedding': query_embedding, 'min_vector_similarity': min_similarity
        }
        vector_results = self._execute_query(vector_query, vector_params)
        entity_results = self._find_communities_by_entity_names(
            graph_prefix=graph_prefix,
            entity_names=llm_entities,
            level=level,
            limit_per_entity=2
        )
        print(f"DEBUG: Entity search found {len(entity_results)} communities.")
        combined_results_map: Dict[str, Dict[str, Any]] = {}

        for comm in entity_results:
            comm_id = comm['id']
            if comm_id not in combined_results_map:
                 comm_data = comm.copy()
                 comm_data['retrieval_source'] = 'entity'
                 comm_data['relevance_score'] = comm.get('entity_presence_score', 0.5) 
                 combined_results_map[comm_id] = comm_data

        for comm in vector_results:
            comm_id = comm['id']
            if comm_id not in combined_results_map:
                comm_data = comm.copy()
                comm_data['retrieval_source'] = 'vector'
                comm_data['relevance_score'] = comm.get('vector_score', 0.0) 
                combined_results_map[comm_id] = comm_data
            else:
                 combined_results_map[comm_id]['vector_score_if_overlap'] = comm.get('vector_score')

        print(f"DEBUG: Entity search found vector_results {len(vector_results)} communities.")
        final_list = list(combined_results_map.values())
        final_list.sort(key=lambda x: x.get('relevance_score', 0.0), reverse=True)
        print(f"DEBUG: Entity search found final_list {len(final_list)} communities.")
        return final_list[:limit]
    
     
    
    def get_entities_for_query(self, query_embedding, graph_prefix, max_entities=5):
        """
        Find entities relevant to the query for weighting communities
        """
        entity_weights = {}
        
        if query_embedding:
            entity_label = f"{graph_prefix}__Entity__"
            
            query = f"""
            MATCH (e:{entity_label})
            WHERE e.embedding IS NOT NULL
            WITH e, vector.similarity.cosine(e.embedding, $query_embedding) AS score
            WHERE score > 0.7
            RETURN e.id AS id, e.name AS name, score AS relevance
            ORDER BY relevance DESC
            LIMIT $limit
            """
            
            results = self._execute_query(query, {
                'query_embedding': query_embedding,
                'limit': max_entities
            })
            
            for entity in results:
                entity_weights[entity['id']] = float(entity['relevance'])
                
        return entity_weights
  

    def _get_embedding(self, text):
        """Get embedding vector for a text string"""
        if self.model is None:
            self._load_embedding_model()
            if self.model is None:
                return None
                
        try:
            return self.model.encode(text).tolist()
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return None
             
    def find_community_hierarchy(self, graph_prefix):
        """
        Load community hierarchy relationships for dynamic community selection
        """
        community_label = f"{graph_prefix}__Community__"
        
        query = f"""
        MATCH (c:{community_label})
        WITH c
        OPTIONAL MATCH (parent:{community_label}) WHERE parent.community = c.parent
        RETURN c.id AS id, 
            c.community AS community_id,
            c.level AS level,
            c.parent AS parent_id,
            parent.community AS parent_community_id,
            c.children AS children_ids,
            c.title AS title
        ORDER BY c.level, c.rank DESC
        """
        
        result = self._execute_query(query, {})

        hierarchy = {
            'by_id': {},
            'by_level': {},
            'root_ids': []
        }
        
        for comm in result:
            comm_id = comm['community_id']
            level = comm.get('level', 0)
            
            hierarchy['by_id'][comm_id] = comm
            
            if level not in hierarchy['by_level']:
                hierarchy['by_level'][level] = []
            hierarchy['by_level'][level].append(comm)
            
            if level == 0:
                hierarchy['root_ids'].append(comm_id)
        return hierarchy


    def calculate_community_relevance(self, query_embedding, communities, entity_weights=None, search_type='drift'):
        for community in communities:
            base_rank = float(community.get('rank', 0.0)) 
            semantic_score = 0.0
            if query_embedding and community.get('embedding'):
                try:
                    comm_emb = np.array(community['embedding'])
                    query_emb_np = np.array(query_embedding)
                    norm_product = np.linalg.norm(comm_emb) * np.linalg.norm(query_emb_np)
                    if norm_product > 1e-9: 
                        semantic_score = float(np.dot(query_emb_np, comm_emb) / norm_product)
                    else:
                        semantic_score = 0.0
                except Exception as e:
                       print(e)
                       
            entity_score = 0.0
            if entity_weights and 'entity_ids' in community:
                community_entities = set(community.get('entity_ids', []))
                weighted_entities_in_community = community_entities.intersection(entity_weights.keys())
                if weighted_entities_in_community:
                    sum_weights = sum(entity_weights[e_id] for e_id in weighted_entities_in_community)

                    entity_score = sum_weights 

            if search_type=='drift':
                final_score = (0.1 * semantic_score) + (0.7 * entity_score) + (0.2 * base_rank)
            else:
                final_score = (0.5 * semantic_score) + (0.2 * entity_score) + (0.2 * base_rank)

            community['relevance_score'] = final_score

            community['score_components'] = {
                'semantic_score': semantic_score,
                'entity_score': entity_score,
                'base_rank': base_rank,
            }
        communities.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        return communities


    def find_cross_community_relationships(self,
                                        top_community_entity_map: Dict[str, List[str]],
                                        graph_prefix: str,
                                        max_relationships: int = 10) -> List[Dict[str, Any]]:
        """
        Finds relationships connecting entities that belong to *different* communities
        within the provided map. [Standard Cypher Version]

        Args:
            top_community_entity_map: A dictionary mapping {community_id: [list_of_entity_ids]}
                                    for the top relevant communities.
            graph_prefix: The graph prefix.
            max_relationships: Maximum number of relationships to return.

        Returns:
            A list of relationship dictionaries, each containing source/target entity names
            and relationship details. Returns empty list if fewer than 2 communities provided.
        """
        if len(top_community_entity_map) < 2:
            print('info', {
                'message': "Skipping cross-community search (less than 2 communities provided).",
                'community_count': len(top_community_entity_map)
            })
            return []

        all_entity_ids_in_map = [eid for entity_list in top_community_entity_map.values() for eid in entity_list]
        if not all_entity_ids_in_map:
            print('info', {
                'message': "Skipping cross-community search (no entities found in top communities map).",
                'community_count': len(top_community_entity_map)
            })
            return []

        entity_label = f"{graph_prefix}__Entity__"
        relationship_pattern = f":{graph_prefix}_RELATED" 

        query = f"""
        WITH $community_map AS comm_map, $all_entity_ids AS all_eids 
        MATCH (e1:{entity_label})
        WHERE e1.id IN all_eids
        MATCH (e1)-[r{relationship_pattern}]-(e2:{entity_label})
        WHERE e2.id IN all_eids 
        WITH e1, r, e2, comm_map, all_eids,
            [c_id IN keys(comm_map) WHERE e1.id IN comm_map[c_id] | c_id][0] AS comm1_id,
            [c_id IN keys(comm_map) WHERE e2.id IN comm_map[c_id] | c_id][0] AS comm2_id
        WHERE comm1_id <> comm2_id

        RETURN DISTINCT
            e1.id AS source_id, e1.name AS source_name, comm1_id AS source_community,
            e2.id AS target_id, e2.name AS target_name, comm2_id AS target_community,
            type(r) as relationship_type,
            r.description AS relationship_description,
            r.weight AS relationship_weight
        ORDER BY coalesce(r.weight, 0.0) DESC
        LIMIT $limit
        """
        params = {
            'community_map': top_community_entity_map,
            'all_entity_ids': all_entity_ids_in_map, 
            'limit': max_relationships
        }
        results = self._execute_query(query, params)
        print('cross_community_relationship_search', {
            'graph_prefix': graph_prefix,
            'community_count': len(top_community_entity_map),
            'entity_count': len(all_entity_ids_in_map),
            'found_relationships': len(results)
        })
        return results
        
    def global_search(self,
               query_text: str,
               graph_prefix: str,
               llm_entities: Optional[List[str]] = None, 
               top_k_llm_communities: int = 20,
               max_chunks_per_entity_from_top_comm: int = 3,
               enable_final_chunk_reranking: bool = True,
               chunk_reranker_llm_batch_size: int = 15,
               chunk_reranker_target_count: int = 15, 
               chunk_reranker_max_retries: int = 3,
               use_llm_reranker =True,
               max_communities_in_context: int = 7,
               max_cross_rels_in_context: int = 5,
               community_levels_to_consider: Optional[List[int]] = None 
               ) -> Dict[str, Any]:
        print(f"GlobalSearchLLMFocus: Starting search for '{query_text}'")
        query_embedding = self._get_embedding(query_text)
        if not query_embedding:
            return {'error': 'Failed to generate query embedding for chunk retrieval.', 'context': '', 'approx_token_count': 0}

        all_communities_data = self._fetch_all_community_content(graph_prefix, levels=community_levels_to_consider)
        if not all_communities_data:
            return {'error': 'No community data found.', 'context': '', 'approx_token_count': 0}
        embedding_ranked_communities  = self._rank_communities_by_embedding_similarity(
            query_embedding, all_communities_data, graph_prefix
        )
        selected_top_communities = embedding_ranked_communities [:top_k_llm_communities*60]    
        aggregated_candidate_chunks = []
        top_community_ids = [c['id'] for c in selected_top_communities]
        community_entities_map = self.get_community_entities_map(top_community_ids, graph_prefix)
        for comm in selected_top_communities:
            comm['entity_ids'] = community_entities_map.get(comm['id'], [])

        for community_data in selected_top_communities:
            entity_ids_in_comm = community_data.get('entity_ids', [])
            if entity_ids_in_comm:
                chunks_from_comm_entities = self.get_entity_text_chunks(
                    entity_ids_in_comm, graph_prefix, query_embedding, max_chunks_per_entity_from_top_comm
                )
                for entity_id, chunks_list in chunks_from_comm_entities.items():
                    aggregated_candidate_chunks.extend(chunks_list)
        
        deduped_chunks_map = {}
        for chunk in aggregated_candidate_chunks:
            key = chunk.get('chunk_id') or chunk.get('id')
            if key not in deduped_chunks_map:
                deduped_chunks_map[key] = chunk
            else: 
                if chunk.get('relevance',0) > deduped_chunks_map[key].get('relevance',0):
                    deduped_chunks_map[key] = chunk
        unique_candidate_chunks = list(deduped_chunks_map.values())
        unique_candidate_chunks.sort(key=lambda x: x.get('relevance', 0.0), reverse=True)
        print(f"Aggregated {len(unique_candidate_chunks)} unique candidate chunks from top communities.")
        final_top_chunks_for_context = []
        if enable_final_chunk_reranking and unique_candidate_chunks: 
            if use_llm_reranker:
                final_top_chunks_for_context = asyncio.run( self._llm_rerank_chunks_async(
                    query_text=query_text,
                    chunks_to_rerank = unique_candidate_chunks,
                    llm_batch_size=chunk_reranker_llm_batch_size,
                    target_final_chunk_count=chunk_reranker_target_count,
                    max_retries_per_batch=chunk_reranker_max_retries
                ))
            else:
                final_top_chunks_for_context = self._embedding_rerank_chunks(query_embedding_=query_embedding, chunks_to_rerank = unique_candidate_chunks)    
            
        elif unique_candidate_chunks: 
            for chunk in unique_candidate_chunks[:chunk_reranker_target_count]:
                chunk['llm_relevance_score'] = chunk.get('relevance', 0.0)
            final_top_chunks_for_context = unique_candidate_chunks[:chunk_reranker_target_count]

        print(f"Final {len(final_top_chunks_for_context)} chunks selected for context after reranking.")
        full_hierarchy_map = self.find_community_hierarchy(graph_prefix) 
        cross_comm_entity_map = {c['id']: c.get('entity_ids', []) for c in selected_top_communities}
        cross_community_rels = self.find_cross_community_relationships(
            cross_comm_entity_map, graph_prefix, max_cross_rels_in_context
        )
        context_data = self.build_global_context(
            query_text = query_text,
            graph_prefix=graph_prefix,
            selected_communities=selected_top_communities[:max_communities_in_context], 
            full_structural_hierarchy=full_hierarchy_map,
            top_reranked_chunks=final_top_chunks_for_context, 
            cross_community_relationships=cross_community_rels,
            max_communities_to_display=max_communities_in_context,
            max_cross_rels_to_display=max_cross_rels_in_context
        )
        return {
            'top_communities': selected_top_communities[:max_communities_in_context],
            'cross_community_relationships': cross_community_rels,
            'context': context_data.get('text', ''),
            'approx_token_count': context_data.get('approx_token_count', 0),
            'query': query_text,
            'graph_prefix': graph_prefix,
            'llm_reranked_chunks_count': len(final_top_chunks_for_context)
        }
        
    
def classical_and_llm_run_graph_search(
    driver,
    query: str,
    graph_prefix: str,
    embedding_model: SentenceTransformer,
    global_params: Optional[Dict[str, Any]] = None,
    local_params: Optional[Dict[str, Any]] = None,
    drift_params: Optional[Dict[str, Any]] = None
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Performs enhanced global, standard local, and enhanced DRIFT graph searches,
    measures execution time for each, adds the time to the respective result
    dictionary, and returns them. Allows passing specific parameters for each search type.

    Args:
        driver: Neo4j driver instance.
        query (str): The search query text.
        graph_prefix (str): A prefix used in graph operations (e.g., 'LLM', 'DL').
        global_params (Optional[Dict[str, Any]]): Dictionary of parameters to pass
            to the global_search method (e.g., {'max_cross_relationships': 5}).
        local_params (Optional[Dict[str, Any]]): Dictionary of parameters to pass
            to the local_search method (e.g., {'max_hops': 2}).
        drift_params (Optional[Dict[str, Any]]): Dictionary of parameters to pass
            to the drift_search method (e.g., {'max_depth': 2, 'max_context_chunks': 8}).

    Returns:
        Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
            A tuple containing the result dictionaries for global, local, and
            DRIFT searches. Each dictionary includes its original content
            plus a 'retrieval_time_taken' key. Returns dictionaries with an 'error'
            key if a search fails.
    """
    print(f"\nRunning graph searches for query: '{query}' with prefix: '{graph_prefix}'")
    measurer = TimeMeasurer()
    llm=gemini_llm(model_name="gemini-2.0-flash", temperature=0.1)

    global_search = GlobalSearch(driver, database=NEO4J_DATABASE, embedding_model= embedding_model, llm = llm)
    local_search = LocalSearch(driver, database=NEO4J_DATABASE, embedding_model= embedding_model, llm = llm)
    drift_search = DriftSearch(driver, database=NEO4J_DATABASE, embedding_model= embedding_model, llm = llm)
    global_params = global_params or {}
    local_params = local_params or {}
    drift_params = drift_params or {}

    global_measure_key = f'{graph_prefix}_global_time_measure'
    local_measure_key = f'{graph_prefix}_local_time_measure'
    drift_measure_key = f'{graph_prefix}_drift_time_measure'
    time_key_to_add = 'retrieval_time_taken'

    global_results = {}
    local_results = {}
    drift_results = {}
    
    llm_entities = _extract_entities_from_query_llm(llm, query, max_entities=6)
    
    try:
        print(f"--- Running Global Search (Params: {global_params}) ---")
        with measurer.measure(global_measure_key):
            global_results = global_search.global_search(
                query_text=query,
                llm_entities=llm_entities,
                graph_prefix=graph_prefix,
                **global_params  
            )
        if isinstance(global_results, dict):
             global_results[time_key_to_add] = measurer.get_timing(global_measure_key)
        else:
             global_results = {'error': 'Global search failed to return dictionary', time_key_to_add: measurer.get_timing(global_measure_key, 0.0)}
        print("===== GLOBAL SEARCH (ENHANCED) COMPLETED =====")

    except Exception as e:
        print(f"Error during Global search: {e}")
        global_results = {'error': str(e), time_key_to_add: measurer.get_timing(global_measure_key, 0.0)} 
  
    try:
        print(f"--- Running Local Search (Params: {local_params}) ---")
        with measurer.measure(local_measure_key):
            local_results = local_search.local_search(
                query_text=query,
                llm_entities=llm_entities,
                graph_prefix=graph_prefix,
                **local_params 
            )
        if isinstance(local_results, dict):
             local_results[time_key_to_add] = measurer.get_timing(local_measure_key)
        else:
             local_results = {'error': 'Local search failed to return dictionary', time_key_to_add: measurer.get_timing(local_measure_key, 0.0)}
        print("===== LOCAL SEARCH COMPLETED =====")

    except Exception as e:
        print(f"Error during Local search: {e}")
        local_results = {'error': str(e), time_key_to_add: measurer.get_timing(local_measure_key, 0.0)}

    try:
        print(f"--- Running DRIFT Search (Params: {drift_params}) ---")
        with measurer.measure(drift_measure_key):
            drift_results = drift_search.drift_search(
                query_text=query,
                llm_entities=llm_entities,
                graph_prefix=graph_prefix,
                **drift_params 
            )
        if isinstance(drift_results, dict):
             drift_results[time_key_to_add] = measurer.get_timing(drift_measure_key)
        else:
             drift_results = {'error': 'DRIFT search failed to return dictionary', time_key_to_add: measurer.get_timing(drift_measure_key, 0.0)}
        print("===== DRIFT SEARCH (ENHANCED) COMPLETED =====")

    except Exception as e:
        print(f"Error during DRIFT search: {e}")
        drift_results = {'error': str(e), time_key_to_add: measurer.get_timing(drift_measure_key, 0.0)}
    return global_results, local_results, drift_results