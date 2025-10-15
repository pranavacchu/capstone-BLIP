"""
Pinecone Vector Database Integration Module
Handles Pinecone initialization, index management, and semantic search
"""

from pinecone import Pinecone, ServerlessSpec, PodSpec
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional, Any
from tqdm import tqdm
import time
from dataclasses import dataclass
from embedding_generator import EmbeddedFrame

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """Data structure for search results"""
    id: str
    score: float
    metadata: Dict[str, Any]
    timestamp: float
    caption: str
    frame_id: str
    video_name: str

class PineconeManager:
    """Manage Pinecone vector database operations"""
    
    def __init__(self,
                 api_key: str,
                 environment: str = "us-east-1",
                 index_name: str = "capstone",
                 dimension: int = 1024,
                 metric: str = "cosine",
                 use_serverless: bool = True,
                 host: Optional[str] = None):
        """
        Initialize Pinecone manager
        
        Args:
            api_key: Pinecone API key
            environment: Pinecone environment
            index_name: Name of the index
            dimension: Vector dimension
            metric: Distance metric (cosine, euclidean, dotproduct)
            use_serverless: Whether to use serverless or pod-based index
        """
        self.api_key = api_key
        self.environment = environment
        self.index_name = index_name
        self.dimension = dimension
        self.metric = metric
        self.use_serverless = use_serverless
        self.host = host
        
        # Initialize Pinecone
        self._initialize_pinecone()
        
        # Connect to existing index (don't create)
        self._setup_index()
    
    def _initialize_pinecone(self):
        """Initialize Pinecone client"""
        try:
            # Initialize Pinecone with API key
            self.pc = Pinecone(api_key=self.api_key)
            logger.info("Pinecone client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {e}")
            raise
    
    def _setup_index(self):
        """Connect to existing Pinecone index"""
        try:
            # Check if index exists
            existing_indexes = self.pc.list_indexes()
            index_exists = any(index.name == self.index_name for index in existing_indexes)
            
            if not index_exists:
                logger.warning(f"Index '{self.index_name}' does not exist. It should already be created in Pinecone.")
                logger.info(f"Attempting to connect anyway...")
            else:
                logger.info(f"Connecting to existing index: {self.index_name}")
            
            # Connect to index (use host if provided)
            if self.host:
                self.index = self.pc.Index(self.index_name, host=self.host)
            else:
                self.index = self.pc.Index(self.index_name)
            
            # Get index stats
            stats = self.index.describe_index_stats()
            logger.info(f"Index stats - Total vectors: {stats.get('total_vector_count', 0)}")
            logger.info(f"Index dimension: {stats.get('dimension', 'unknown')}")
            
        except Exception as e:
            logger.error(f"Failed to setup index: {e}")
            raise
    
    def _create_index(self):
        """Create a new Pinecone index"""
        try:
            if self.use_serverless:
                # Create serverless index
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric=self.metric,
                    spec=ServerlessSpec(
                        cloud='aws',
                        region=self.environment
                    )
                )
            else:
                # Create pod-based index
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric=self.metric,
                    spec=PodSpec(
                        environment=self.environment,
                        pod_type='p1.x1',
                        pods=1
                    )
                )
            
            # Wait for index to be ready
            logger.info("Waiting for index to be ready...")
            time.sleep(10)  # Give it some time to initialize
            
            logger.info(f"Index '{self.index_name}' created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create index: {e}")
            raise
    
    def upload_embeddings(self,
                         data: List[Tuple[str, List[float], Dict]],
                         batch_size: int = 100,
                         namespace: str = "") -> bool:
        """
        Upload embeddings to Pinecone
        
        Args:
            data: List of (id, vector, metadata) tuples
            batch_size: Batch size for uploads
            namespace: Namespace for organization (optional)
            
        Returns:
            Success status
        """
        logger.info(f"Uploading {len(data)} vectors to Pinecone")
        
        try:
            # Upload in batches
            for i in tqdm(range(0, len(data), batch_size), desc="Uploading to Pinecone"):
                batch = data[i:i + batch_size]
                
                # Prepare vectors for upsert
                vectors = []
                for item_id, vector, metadata in batch:
                    vectors.append({
                        'id': item_id,
                        'values': vector,
                        'metadata': metadata
                    })
                
                # Upsert batch
                self.index.upsert(vectors=vectors, namespace=namespace)
            
            logger.info(f"Successfully uploaded {len(data)} vectors")
            return True
            
        except Exception as e:
            logger.error(f"Failed to upload embeddings: {e}")
            return False
    
    def query(self,
             query_vector: np.ndarray,
             top_k: int = 10,
             filter: Optional[Dict] = None,
             namespace: str = "",
             include_metadata: bool = True) -> List[SearchResult]:
        """
        Query the Pinecone index
        
        Args:
            query_vector: Query embedding vector
            top_k: Number of results to return
            filter: Metadata filter (optional)
            namespace: Namespace to query
            include_metadata: Whether to include metadata in results
            
        Returns:
            List of SearchResult objects
        """
        try:
            # Convert numpy array to list
            query_vector_list = query_vector.tolist()
            
            # Query index
            results = self.index.query(
                vector=query_vector_list,
                top_k=top_k,
                filter=filter,
                namespace=namespace,
                include_metadata=include_metadata
            )
            
            # Parse results
            search_results = []
            for match in results['matches']:
                metadata = match.get('metadata', {})
                
                result = SearchResult(
                    id=match['id'],
                    score=match['score'],
                    metadata=metadata,
                    timestamp=metadata.get('timestamp', 0.0),
                    caption=metadata.get('caption', ''),
                    frame_id=metadata.get('frame_id', ''),
                    video_name=metadata.get('video_name', '')
                )
                search_results.append(result)
            
            return search_results
            
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return []
    
    def semantic_search(self,
                       query_embedding: np.ndarray,
                       top_k: int = 10,
                       similarity_threshold: float = 0.6,
                       video_filter: Optional[str] = None,
                       time_window: Optional[Tuple[float, float]] = None) -> List[SearchResult]:
        """
        Perform semantic search with filtering
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            similarity_threshold: Minimum similarity score
            video_filter: Filter by video name
            time_window: Filter by time range (start, end) in seconds
            
        Returns:
            Filtered list of SearchResult objects
        """
        # Build metadata filter
        metadata_filter = {}
        
        if video_filter:
            metadata_filter['video_name'] = video_filter
        
        if time_window:
            metadata_filter['timestamp'] = {
                '$gte': time_window[0],
                '$lte': time_window[1]
            }
        
        # Query index
        results = self.query(
            query_vector=query_embedding,
            top_k=top_k * 2,  # Get more results for filtering
            filter=metadata_filter if metadata_filter else None
        )
        
        # Filter by similarity threshold
        filtered_results = [r for r in results if r.score >= similarity_threshold]
        
        # Remove near-duplicate results
        filtered_results = self._remove_duplicate_results(filtered_results)
        
        # Return top-k results
        return filtered_results[:top_k]
    
    def _remove_duplicate_results(self,
                                 results: List[SearchResult],
                                 time_window: float = 2.0) -> List[SearchResult]:
        """
        Remove near-duplicate results within a time window
        
        Args:
            results: List of search results
            time_window: Time window in seconds
            
        Returns:
            Filtered list without duplicates
        """
        if not results:
            return []
        
        filtered = []
        seen_timestamps = {}
        
        for result in results:
            video_name = result.video_name
            timestamp = result.timestamp
            
            # Check if we've seen a result from this video near this timestamp
            key = video_name
            if key in seen_timestamps:
                # Check if timestamp is too close to any seen timestamp
                is_duplicate = False
                for seen_ts in seen_timestamps[key]:
                    if abs(timestamp - seen_ts) < time_window:
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    filtered.append(result)
                    seen_timestamps[key].append(timestamp)
            else:
                filtered.append(result)
                seen_timestamps[key] = [timestamp]
        
        return filtered
    
    def batch_query(self,
                   query_vectors: List[np.ndarray],
                   top_k: int = 10,
                   namespace: str = "") -> List[List[SearchResult]]:
        """
        Perform batch queries
        
        Args:
            query_vectors: List of query embedding vectors
            top_k: Number of results per query
            namespace: Namespace to query
            
        Returns:
            List of search results for each query
        """
        all_results = []
        
        for query_vector in tqdm(query_vectors, desc="Batch querying"):
            results = self.query(
                query_vector=query_vector,
                top_k=top_k,
                namespace=namespace
            )
            all_results.append(results)
        
        return all_results
    
    def delete_vectors(self,
                      ids: List[str],
                      namespace: str = "") -> bool:
        """
        Delete vectors from index
        
        Args:
            ids: List of vector IDs to delete
            namespace: Namespace
            
        Returns:
            Success status
        """
        try:
            self.index.delete(ids=ids, namespace=namespace)
            logger.info(f"Deleted {len(ids)} vectors")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete vectors: {e}")
            return False
    
    def update_metadata(self,
                       id: str,
                       metadata: Dict,
                       namespace: str = "") -> bool:
        """
        Update metadata for a vector
        
        Args:
            id: Vector ID
            metadata: New metadata
            namespace: Namespace
            
        Returns:
            Success status
        """
        try:
            # Fetch existing vector
            fetch_response = self.index.fetch(ids=[id], namespace=namespace)
            
            if id in fetch_response['vectors']:
                vector = fetch_response['vectors'][id]['values']
                
                # Update with new metadata
                self.index.upsert(
                    vectors=[{
                        'id': id,
                        'values': vector,
                        'metadata': metadata
                    }],
                    namespace=namespace
                )
                
                logger.info(f"Updated metadata for vector {id}")
                return True
            else:
                logger.warning(f"Vector {id} not found")
                return False
                
        except Exception as e:
            logger.error(f"Failed to update metadata: {e}")
            return False
    
    def get_index_stats(self) -> Dict:
        """Get index statistics"""
        try:
            stats = self.index.describe_index_stats()
            return {
                'total_vectors': stats.get('total_vector_count', 0),
                'dimension': stats.get('dimension', self.dimension),
                'index_fullness': stats.get('index_fullness', 0),
                'namespaces': stats.get('namespaces', {})
            }
            
        except Exception as e:
            logger.error(f"Failed to get index stats: {e}")
            return {}
    
    def clear_index(self, namespace: str = "") -> bool:
        """
        Clear all vectors from index or namespace
        
        Args:
            namespace: Namespace to clear (empty for all)
            
        Returns:
            Success status
        """
        try:
            self.index.delete(delete_all=True, namespace=namespace)
            logger.info(f"Cleared index/namespace: {namespace or 'default'}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear index: {e}")
            return False
    
    def delete_index(self) -> bool:
        """Delete the entire index"""
        try:
            self.pc.delete_index(self.index_name)
            logger.info(f"Deleted index: {self.index_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete index: {e}")
            return False


# Example usage and testing
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv()
    
    # Test Pinecone manager
    # api_key = os.getenv('PINECONE_API_KEY')
    # 
    # if api_key:
    #     manager = PineconeManager(
    #         api_key=api_key,
    #         index_name="video-frame-search-test",
    #         dimension=384,
    #         use_serverless=True
    #     )
    #     
    #     # Get index stats
    #     stats = manager.get_index_stats()
    #     print(f"Index stats: {stats}")
    #     
    #     # Example upload
    #     # test_data = [
    #     #     ("test_1", [0.1] * 384, {"caption": "Test caption 1", "timestamp": 1.0}),
    #     #     ("test_2", [0.2] * 384, {"caption": "Test caption 2", "timestamp": 2.0}),
    #     # ]
    #     # manager.upload_embeddings(test_data)
    #     
    #     # Example query
    #     # query_vector = np.random.randn(384)
    #     # results = manager.semantic_search(query_vector, top_k=5)
    #     # for result in results:
    #     #     print(f"Score: {result.score:.3f} - Caption: {result.caption}")