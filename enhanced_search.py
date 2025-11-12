"""
Enhanced Semantic Search Module
Integrates query understanding with multi-vector search strategies
"""

import numpy as np
import logging
from typing import List, Dict, Optional, Tuple
from query_understanding import QueryUnderstanding, ExpandedQuery, QueryIntent
from video_search_engine import VideoSearchEngine
from embedding_generator import TextEmbeddingGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedSearch:
    """Enhanced search with query understanding and multi-vector strategies"""
    
    def __init__(self, search_engine: VideoSearchEngine):
        """
        Initialize enhanced search
        
        Args:
            search_engine: Base VideoSearchEngine instance
        """
        self.search_engine = search_engine
        self.query_understanding = QueryUnderstanding()
        
        # Initialize embedding generator for query expansion
        if not self.search_engine.embedding_generator:
            self.search_engine._initialize_components()
        
        logger.info("Enhanced search initialized with query understanding")
    
    def search_with_understanding(self,
                                  query: str,
                                  top_k: int = 10,
                                  similarity_threshold: float = 0.6,
                                  use_query_expansion: bool = True,
                                  use_reranking: bool = True,
                                  explain: bool = False) -> Dict:
        """
        Perform enhanced search with query understanding
        
        Args:
            query: Natural language search query
            top_k: Number of results to return
            similarity_threshold: Minimum similarity score
            use_query_expansion: Whether to use query expansion
            use_reranking: Whether to rerank results
            explain: Whether to include explanation
            
        Returns:
            Dictionary with results and metadata
        """
        # Step 1: Understand the query
        expanded_query = self.query_understanding.understand_query(query)
        
        if explain:
            explanation = self.query_understanding.explain_query(expanded_query)
            logger.info(f"Query Understanding: {explanation}")
        
        # Step 2: Multi-query search
        if use_query_expansion and len(expanded_query.expanded_queries) > 1:
            results = self._multi_query_search(
                expanded_query=expanded_query,
                top_k=top_k,
                similarity_threshold=similarity_threshold
            )
        else:
            # Single query search
            results = self._single_query_search(
                query=query,
                expanded_query=expanded_query,
                top_k=top_k,
                similarity_threshold=similarity_threshold
            )
        
        # Step 3: Rerank results based on query understanding
        if use_reranking and results:
            results = self.query_understanding.rerank_results(
                results=results,
                intent=expanded_query.structured_intent
            )
        
        # Step 4: Prepare response
        response = {
            'query': query,
            'results': results[:top_k],  # Limit to top_k after reranking
            'total_results': len(results),
            'intent': expanded_query.structured_intent.primary_intent,
            'searched_namespaces': expanded_query.target_namespaces
        }
        
        if explain:
            response['explanation'] = self.query_understanding.explain_query(expanded_query)
            response['query_expansions'] = expanded_query.expanded_queries
        
        return response
    
    def _single_query_search(self,
                            query: str,
                            expanded_query: ExpandedQuery,
                            top_k: int,
                            similarity_threshold: float) -> List[Dict]:
        """
        Perform search with a single query using understood context
        """
        # Use understood namespaces and metadata filters
        results = []
        
        # Search across inferred namespaces
        for namespace in expanded_query.target_namespaces:
            try:
                namespace_results = self.search_engine.search(
                    query=query,
                    top_k=top_k * 2,  # Get more for filtering
                    similarity_threshold=similarity_threshold,
                    namespace_filter=namespace
                )
                results.extend(namespace_results)
            except Exception as e:
                logger.warning(f"Search failed for namespace {namespace}: {e}")
        
        # Remove duplicates
        results = self._deduplicate_results(results)
        
        return results
    
    def _multi_query_search(self,
                           expanded_query: ExpandedQuery,
                           top_k: int,
                           similarity_threshold: float) -> List[Dict]:
        """
        Perform multi-query search with query expansions
        Uses reciprocal rank fusion to combine results
        """
        all_results = {}  # frame_id -> result with metadata
        
        # Search with each expanded query
        for idx, query_variant in enumerate(expanded_query.expanded_queries):
            logger.info(f"Searching with query variant {idx+1}/{len(expanded_query.expanded_queries)}: '{query_variant}'")
            
            variant_results = self._single_query_search(
                query=query_variant,
                expanded_query=expanded_query,
                top_k=top_k * 2,
                similarity_threshold=similarity_threshold
            )
            
            # Add to combined results with rank information
            for rank, result in enumerate(variant_results):
                frame_id = result.get('frame_id')
                
                if frame_id not in all_results:
                    all_results[frame_id] = result
                    all_results[frame_id]['query_hits'] = []
                    all_results[frame_id]['rrf_score'] = 0
                
                # Track which query variants matched this result
                all_results[frame_id]['query_hits'].append({
                    'query': query_variant,
                    'rank': rank + 1,
                    'score': result.get('similarity_score', 0)
                })
                
                # Reciprocal Rank Fusion: 1 / (k + rank)
                # k=60 is a common constant
                all_results[frame_id]['rrf_score'] += 1 / (60 + rank + 1)
        
        # Convert to list and sort by RRF score
        results_list = list(all_results.values())
        
        # Update similarity_score to be the RRF score for sorting
        for result in results_list:
            result['original_similarity_score'] = result.get('similarity_score', 0)
            result['similarity_score'] = result['rrf_score']
            result['num_query_hits'] = len(result.get('query_hits', []))
        
        # Sort by RRF score
        results_list.sort(key=lambda x: x['rrf_score'], reverse=True)
        
        logger.info(f"Multi-query search combined {len(results_list)} unique results")
        
        return results_list
    
    def _deduplicate_results(self, results: List[Dict]) -> List[Dict]:
        """
        Remove duplicate results based on frame_id
        Keep the result with highest similarity score
        """
        unique_results = {}
        
        for result in results:
            frame_id = result.get('frame_id')
            current_score = result.get('similarity_score', 0)
            
            if frame_id not in unique_results:
                unique_results[frame_id] = result
            else:
                existing_score = unique_results[frame_id].get('similarity_score', 0)
                if current_score > existing_score:
                    unique_results[frame_id] = result
        
        return list(unique_results.values())
    
    def semantic_search_with_context(self,
                                     query: str,
                                     context: Optional[Dict] = None,
                                     top_k: int = 10) -> Dict:
        """
        Search with additional context (e.g., previous search, conversation history)
        
        Args:
            query: Search query
            context: Additional context (previous results, filters, etc.)
            top_k: Number of results
            
        Returns:
            Search results with context
        """
        # Incorporate context into query understanding
        if context:
            # Apply context filters
            if 'video_name' in context:
                logger.info(f"Filtering by video: {context['video_name']}")
            if 'date_range' in context:
                logger.info(f"Filtering by date range: {context['date_range']}")
        
        # Perform enhanced search
        results = self.search_with_understanding(
            query=query,
            top_k=top_k,
            use_query_expansion=True,
            use_reranking=True
        )
        
        # Add context information to results
        if context:
            results['context'] = context
        
        return results
    
    def hybrid_search(self,
                     query: str,
                     top_k: int = 10,
                     semantic_weight: float = 0.7,
                     keyword_weight: float = 0.3) -> Dict:
        """
        Hybrid search combining semantic and keyword-based approaches
        
        Args:
            query: Search query
            top_k: Number of results
            semantic_weight: Weight for semantic search
            keyword_weight: Weight for keyword matching
            
        Returns:
            Hybrid search results
        """
        # Semantic search
        semantic_results = self.search_with_understanding(
            query=query,
            top_k=top_k * 2,
            use_query_expansion=True,
            use_reranking=False  # We'll rerank after combining
        )
        
        # Keyword-based filtering
        query_terms = set(query.lower().split())
        
        # Score results based on keyword overlap
        for result in semantic_results['results']:
            caption = result.get('caption', '').lower()
            caption_terms = set(caption.split())
            
            # Calculate keyword overlap
            overlap = len(query_terms & caption_terms)
            keyword_score = overlap / max(len(query_terms), 1)
            
            # Combine scores
            semantic_score = result.get('similarity_score', 0)
            hybrid_score = (semantic_score * semantic_weight) + (keyword_score * keyword_weight)
            
            result['semantic_score'] = semantic_score
            result['keyword_score'] = keyword_score
            result['hybrid_score'] = hybrid_score
            result['similarity_score'] = hybrid_score
        
        # Sort by hybrid score
        semantic_results['results'].sort(key=lambda x: x['hybrid_score'], reverse=True)
        semantic_results['results'] = semantic_results['results'][:top_k]
        semantic_results['search_type'] = 'hybrid'
        
        return semantic_results
    
    def conversational_search(self, 
                            queries: List[str],
                            top_k: int = 10) -> Dict:
        """
        Handle multi-turn conversational search
        Later queries can refine earlier ones
        
        Args:
            queries: List of queries in conversation order
            top_k: Number of results
            
        Returns:
            Final search results with conversation history
        """
        conversation_history = []
        final_results = None
        
        for idx, query in enumerate(queries):
            logger.info(f"Processing query {idx+1}/{len(queries)}: '{query}'")
            
            # Build context from previous queries
            context = {
                'previous_queries': conversation_history,
                'query_number': idx + 1
            }
            
            # Search with context
            results = self.semantic_search_with_context(
                query=query,
                context=context,
                top_k=top_k
            )
            
            conversation_history.append({
                'query': query,
                'results_count': len(results['results'])
            })
            
            final_results = results
        
        # Add conversation history to final results
        final_results['conversation_history'] = conversation_history
        
        return final_results


# Example usage
if __name__ == "__main__":
    from video_search_config import Config
    
    # Initialize engines
    config = Config()
    base_engine = VideoSearchEngine(config)
    enhanced_search = EnhancedSearch(base_engine)
    
    # Test queries
    test_queries = [
        "person carrying a black backpack",
        "red duffel bag",
        "laptop on desk from yesterday",
        "someone without a bag"
    ]
    
    for query in test_queries:
        print(f"\n{'='*80}")
        print(f"Query: '{query}'")
        print('='*80)
        
        # Perform enhanced search
        results = enhanced_search.search_with_understanding(
            query=query,
            top_k=5,
            use_query_expansion=True,
            use_reranking=True,
            explain=True
        )
        
        print(f"\nExplanation: {results.get('explanation', 'N/A')}")
        print(f"Intent: {results['intent']}")
        print(f"Searched namespaces: {results['searched_namespaces']}")
        print(f"\nTop {len(results['results'])} results:")
        
        for idx, result in enumerate(results['results'][:3]):
            print(f"\n{idx+1}. Score: {result['similarity_score']:.3f}")
            print(f"   Caption: {result['caption']}")
            print(f"   Video: {result['video_name']} @ {result['time_formatted']}")
