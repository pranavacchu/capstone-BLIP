"""
Video Frame Search Engine - Main Application
Production-ready video semantic search system with BLIP and Pinecone
"""

import os
import logging
from typing import List, Optional, Dict, Tuple, Any
from pathlib import Path
import json
import time
from datetime import datetime
import pandas as pd
import numpy as np

# Import all modules
from video_search_config import Config
from frame_extractor import VideoFrameExtractor, FrameData
from caption_generator import BlipCaptionGenerator, CaptionedFrame
from embedding_generator import TextEmbeddingGenerator, EmbeddedFrame
from pinecone_manager import PineconeManager, SearchResult
from object_caption_pipeline import ObjectCaptionPipeline, ObjectCaption

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('video_search_engine.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class VideoSearchEngine:
    """
    Complete video search engine integrating all components
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the video search engine
        
        Args:
            config: Configuration object (uses default if None)
        """
        # Use provided config or default
        self.config = config or Config()
        
        # Validate configuration
        try:
            self.config.validate()
        except ValueError as e:
            logger.error(f"Configuration validation failed: {e}")
            raise
        
        # Initialize components (lazy loading)
        self.frame_extractor = None
        self.caption_generator = None
        self.embedding_generator = None
        self.pinecone_manager = None
        self.object_pipeline = None  # Object-focused captioning pipeline
        
        # Processing state
        self.current_video = None
        self.processed_frames = []
        self.processing_stats = {}
        
        logger.info("Video Search Engine initialized")
    
    def _initialize_components(self):
        """Initialize all components if not already initialized"""
        if not self.frame_extractor:
            self.frame_extractor = VideoFrameExtractor(
                similarity_threshold=self.config.FRAME_SIMILARITY_THRESHOLD,
                max_frames=self.config.MAX_FRAMES_PER_VIDEO,
                resize_width=self.config.FRAME_RESIZE_WIDTH
            )
            logger.info("Frame extractor initialized")
        
        if not self.caption_generator:
            self.caption_generator = BlipCaptionGenerator(
                model_name=self.config.BLIP_MODEL,
                batch_size=self.config.BLIP_BATCH_SIZE,
                use_gpu=self.config.USE_GPU,
                generate_multiple_captions=getattr(self.config, 'GENERATE_MULTIPLE_CAPTIONS', False),
                captions_per_frame=getattr(self.config, 'CAPTIONS_PER_FRAME', 3)
            )
            logger.info("Caption generator initialized with multi-caption support")
        
        if not self.embedding_generator:
            self.embedding_generator = TextEmbeddingGenerator(
                model_name=self.config.EMBEDDING_MODEL,
                batch_size=self.config.EMBEDDING_BATCH_SIZE,
                use_gpu=self.config.USE_GPU,
                normalize=True
            )
            logger.info("Embedding generator initialized")
        
        if not self.pinecone_manager:
            self.pinecone_manager = PineconeManager(
                api_key=self.config.PINECONE_API_KEY,
                environment=self.config.PINECONE_ENVIRONMENT,
                index_name=self.config.PINECONE_INDEX_NAME,
                dimension=self.config.PINECONE_DIMENSION,
                metric=self.config.PINECONE_METRIC,
                host=getattr(self.config, 'PINECONE_HOST', None)
            )
            logger.info("Pinecone manager initialized")
    
    def process_video(self, 
                     video_path: str,
                     video_name: Optional[str] = None,
                     video_date: Optional[str] = None,
                     save_frames: bool = False,
                     upload_to_pinecone: bool = True,
                     use_object_detection: bool = False) -> Dict[str, Any]:
        """
        Process a video file end-to-end
        
        Args:
            video_path: Path to video file
            video_name: Name for the video (uses filename if None)
            video_date: Date when video was recorded (YYYY-MM-DD format, uses today if None)
            save_frames: Whether to save extracted frames to disk
            upload_to_pinecone: Whether to upload embeddings to Pinecone
            use_object_detection: Whether to use object detection + captioning pipeline
            
        Returns:
            Processing statistics and results
        """
        start_time = time.time()
        
        # Validate video file
        if not os.path.exists(video_path):
            raise ValueError(f"Video file not found: {video_path}")
        
        # Set video name
        if not video_name:
            video_name = Path(video_path).stem
        
        # Set video date (use today's date if not provided)
        if not video_date:
            from datetime import date
            video_date = date.today().strftime("%Y-%m-%d")
            logger.info(f"No video_date provided, using today's date: {video_date}")
        
        self.current_video = video_name
        logger.info(f"Processing video: {video_name} ({video_path})")
        logger.info(f"Video date: {video_date}")
        
        # Initialize components
        self._initialize_components()
        
        try:
            # Step 1: Extract frames
            logger.info("Step 1/4: Extracting frames...")
            frames = self.frame_extractor.extract_frames(
                video_path=video_path,
                use_similarity_filter=True,
                video_date=video_date  # Pass video date to frame extractor
            )
            
            if save_frames:
                output_dir = os.path.join(self.config.OUTPUT_DIR, video_name, "frames")
                self.frame_extractor.save_frames_to_disk(output_dir)
            
            # Step 2: Generate captions
            logger.info("Step 2/4: Generating captions...")
            
            if use_object_detection:
                # Use object-focused captioning pipeline
                logger.info("Using object detection + captioning pipeline")
                print("\n" + "="*80)
                print("🎯 OBJECT DETECTION + CAPTIONING MODE")
                print("="*80)
                
                if not self.object_pipeline:
                    from object_caption_pipeline import ObjectCaptionPipeline
                    self.object_pipeline = ObjectCaptionPipeline(
                        use_gpu=self.config.USE_GPU,
                        min_object_size=30,
                        max_objects_per_frame=10,
                        include_scene_caption=False,  # Don't generate scene captions
                        caption_similarity_threshold=0.85  # Filter similar captions
                    )
                
                # Reset caption history for new video
                self.object_pipeline.reset_caption_history()
                
                # Process frames with object detection (with verbose logging)
                object_captions = self.object_pipeline.process_frames(
                    frames=frames,
                    show_progress=True
                )
                
                # Convert ObjectCaption to CaptionedFrame format
                captioned_frames = []
                for oc in object_captions:
                    cf = CaptionedFrame(
                        frame_data=oc.frame_data,
                        caption=oc.attribute_caption,
                        confidence=oc.confidence
                    )
                    # Store namespace info in frame_data for later use
                    cf.frame_data.namespace = oc.namespace
                    captioned_frames.append(cf)
                
                logger.info(f"Object detection pipeline generated {len(captioned_frames)} captions")
                
            else:
                # Use standard BLIP captioning
                captioned_frames = self.caption_generator.generate_captions(
                    frames=frames,
                    filter_empty=True
                )
            
            # Filter duplicate captions
            captioned_frames = self.caption_generator.filter_duplicate_captions(
                captioned_frames=captioned_frames,
                time_window=self.config.DUPLICATE_TIME_WINDOW
            )
            
            # Step 3: Generate embeddings
            logger.info("Step 3/4: Generating embeddings...")
            embedded_frames = self.embedding_generator.generate_embeddings(
                captioned_frames=captioned_frames
            )
            
            # Step 3.5: Deduplicate embeddings before upload
            logger.info("Deduplicating embeddings...")
            captions_before_dedupe = len(embedded_frames)
            # Fallback if embedding generator lacks deduplication API
            if hasattr(self.embedding_generator, 'deduplicate_embeddings'):
                embedded_frames = self.embedding_generator.deduplicate_embeddings(
                    embedded_frames=embedded_frames,
                    similarity_threshold=0.95  # Remove very similar embeddings
                )
            else:
                embedded_frames = self._deduplicate_embeddings(
                    embedded_frames=embedded_frames,
                    similarity_threshold=0.95
                )
            logger.info(f"After deduplication: {len(embedded_frames)} unique embeddings")
            
            # Step 4: Upload to Pinecone
            actual_uploaded = 0
            if upload_to_pinecone:
                logger.info("Step 4/4: Uploading to Pinecone...")
                print("\n" + "="*80)
                print("☁️  UPLOADING TO PINECONE VECTOR DATABASE")
                print("="*80)
                
                pinecone_data = self.embedding_generator.prepare_for_pinecone(
                    embedded_frames=embedded_frames,
                    video_name=video_name,
                    source_file_path=video_path
                )
                
                # Group by namespace if using object detection
                if use_object_detection:
                    namespace_groups = {}
                    for i, (vec_id, vector, metadata) in enumerate(pinecone_data):
                        # Get namespace from frame_data
                        ef = embedded_frames[i]
                        object_category = getattr(ef.captioned_frame.frame_data, 'namespace', '')
                        
                        # Create date-based namespace: videos:YYYY-MM-DD:category
                        if object_category:
                            namespace = f"videos:{video_date}:{object_category}"
                        else:
                            namespace = f"videos:{video_date}:general"
                        
                        if namespace not in namespace_groups:
                            namespace_groups[namespace] = []
                        namespace_groups[namespace].append((vec_id, vector, metadata))
                    
                    # Upload each namespace separately with logging
                    for namespace, data in namespace_groups.items():
                        print(f"\n📁 Namespace: {namespace}")
                        print(f"   Uploading {len(data)} vectors...")
                        
                        uploaded = self.pinecone_manager.upload_embeddings(
                            data=data,
                            batch_size=self.config.PINECONE_BATCH_SIZE,
                            namespace=namespace
                        )
                        actual_uploaded += uploaded
                        
                        # Show sample
                        if data:
                            sample_caption = data[0][2].get('caption', 'N/A')
                            print(f"   ✓ Uploaded {uploaded} vectors")
                            print(f"   Sample caption: {sample_caption[:70]}...")
                else:
                    # Upload to default namespace
                    actual_uploaded = self.pinecone_manager.upload_embeddings(
                        data=pinecone_data,
                        batch_size=self.config.PINECONE_BATCH_SIZE
                    )
                
                # Print verification
                print(f"\n{'='*80}")
                if actual_uploaded > 0:
                    print(f"✅ UPLOAD COMPLETE: {actual_uploaded} vectors uploaded for '{video_name}'")
                    sample_ids = [pinecone_data[i][0] for i in range(min(3, len(pinecone_data)))]
                    print(f"   Sample vector IDs: {', '.join(sample_ids[:3])}...")
                else:
                    print("❌ WARNING: No vectors were successfully uploaded to Pinecone")
                print(f"{'='*80}\n")
            
            # Store processed frames
            self.processed_frames = embedded_frames
            
            # Calculate statistics
            processing_time = time.time() - start_time
            
            # Calculate frame reduction correctly
            total_video_frames = len(frames)  # Frames before similarity filtering
            frames_after_caption = len(captioned_frames)  # Frames that got captions
            frame_reduction_pct = ((total_video_frames - frames_after_caption) / total_video_frames * 100) if total_video_frames > 0 else 0
            
            stats = {
                "video_name": video_name,
                "video_path": video_path,
                "total_frames_extracted": total_video_frames,
                "frames_with_captions": frames_after_caption,
                "captions_before_dedupe": captions_before_dedupe,
                "embeddings_generated": len(embedded_frames),  # After dedupe
                "embeddings_uploaded": actual_uploaded if upload_to_pinecone else 0,
                "processing_time_seconds": processing_time,
                "frame_reduction_percent": frame_reduction_pct,
                "caption_stats": self.caption_generator.get_caption_statistics(captioned_frames),
                "embedding_stats": self.embedding_generator.get_embedding_statistics(embedded_frames)
            }
            
            self.processing_stats = stats
            
            logger.info(f"Video processing completed in {processing_time:.2f} seconds")
            logger.info(f"Statistics: {json.dumps(stats, indent=2)}")
            
            # Save processing report
            self._save_processing_report(stats, video_name)
            
            return stats
            
        except Exception as e:
            logger.error(f"Error processing video: {e}")
            raise
        
        finally:
            # Clear GPU cache
            if self.caption_generator:
                self.caption_generator.clear_gpu_cache()
            if self.embedding_generator:
                self.embedding_generator.clear_cache()
            if self.object_pipeline:
                self.object_pipeline.clear_cache()
    
    def _deduplicate_embeddings(self,
                               embedded_frames: List[EmbeddedFrame],
                               similarity_threshold: float = 0.95) -> List[EmbeddedFrame]:
        """Deduplicate embeddings locally if generator lacks the method."""
        if not embedded_frames:
            return []
        if len(embedded_frames) == 1:
            return embedded_frames
        # Stack embeddings
        embeddings = np.array([ef.embedding for ef in embedded_frames])
        # If embeddings are normalized, cosine similarity is dot product
        normalized = True
        # Heuristic: check mean norm ~1.0
        norms = np.linalg.norm(embeddings, axis=1)
        if not np.allclose(norms.mean(), 1.0, atol=1e-2):
            normalized = False
        keep = np.ones(len(embeddings), dtype=bool)
        for i in range(len(embeddings)):
            if not keep[i]:
                continue
            vec_i = embeddings[i]
            for j in range(i + 1, len(embeddings)):
                if not keep[j]:
                    continue
                vec_j = embeddings[j]
                if normalized:
                    sim = float(np.dot(vec_i, vec_j))
                else:
                    denom = (np.linalg.norm(vec_i) * np.linalg.norm(vec_j)) or 1e-8
                    sim = float(np.dot(vec_i, vec_j) / denom)
                if sim >= similarity_threshold:
                    keep[j] = False
        return [ef for ef, k in zip(embedded_frames, keep) if k]

    def search(self,
              query: str,
              top_k: int = None,
              similarity_threshold: float = None,
              video_filter: Optional[str] = None,
              time_window: Optional[Tuple[float, float]] = None,
              date_filter: Optional[str] = None,
              namespace_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search for video frames using natural language query
        
        Args:
            query: Natural language search query
            top_k: Number of results to return
            similarity_threshold: Minimum similarity score
            video_filter: Filter by video name
            time_window: Filter by time range (start, end) in seconds
            date_filter: Filter by video date (YYYY-MM-DD format)
            namespace_filter: Filter by specific namespace/category
            
        Returns:
            List of search results with timestamps and metadata
        """
        # Use config defaults if not specified
        top_k = top_k or self.config.QUERY_TOP_K
        similarity_threshold = similarity_threshold or self.config.QUERY_SIMILARITY_THRESHOLD
        
        # Initialize components if needed
        if not self.embedding_generator:
            self.embedding_generator = TextEmbeddingGenerator(
                model_name=self.config.EMBEDDING_MODEL,
                batch_size=self.config.EMBEDDING_BATCH_SIZE,
                use_gpu=self.config.USE_GPU,
                normalize=True
            )
        
        if not self.pinecone_manager:
            self.pinecone_manager = PineconeManager(
                api_key=self.config.PINECONE_API_KEY,
                environment=self.config.PINECONE_ENVIRONMENT,
                index_name=self.config.PINECONE_INDEX_NAME,
                dimension=self.config.PINECONE_DIMENSION,
                metric=self.config.PINECONE_METRIC,
                host=getattr(self.config, 'PINECONE_HOST', None)
            )
        
        logger.info(f"Searching for: '{query}'")
        if date_filter:
            logger.info(f"Date filter: {date_filter}")
        if namespace_filter:
            logger.info(f"Namespace filter: {namespace_filter}")
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_generator.encode_query(query)
            
            # If date_filter is provided, search date-specific namespaces
            if date_filter and namespace_filter:
                # Search specific date + category namespace
                target_namespace = f"videos:{date_filter}:{namespace_filter}"
                search_results = self.pinecone_manager.query(
                    query_vector=query_embedding,
                    top_k=top_k,
                    namespace=target_namespace,
                    include_metadata=True
                )
            elif date_filter:
                # Search all categories for this date
                # Get all namespaces and filter by date prefix
                stats = self.pinecone_manager.get_index_stats()
                namespaces = stats.get('namespaces', {})
                
                date_namespaces = [ns for ns in namespaces.keys() if ns.startswith(f"videos:{date_filter}:")]
                
                if not date_namespaces:
                    logger.warning(f"No namespaces found for date: {date_filter}")
                    return []
                
                # Query each namespace and combine results
                all_results = []
                for ns in date_namespaces:
                    ns_results = self.pinecone_manager.query(
                        query_vector=query_embedding,
                        top_k=top_k,
                        namespace=ns,
                        include_metadata=True
                    )
                    all_results.extend(ns_results)
                
                # Sort by score and take top_k
                all_results.sort(key=lambda x: x.score, reverse=True)
                search_results = all_results[:top_k]
            elif namespace_filter:
                # Search specific category across all dates
                # This requires querying multiple date namespaces
                stats = self.pinecone_manager.get_index_stats()
                namespaces = stats.get('namespaces', {})
                
                category_namespaces = [ns for ns in namespaces.keys() if ns.endswith(f":{namespace_filter}")]
                
                if not category_namespaces:
                    logger.warning(f"No namespaces found for category: {namespace_filter}")
                    return []
                
                # Query each namespace and combine results
                all_results = []
                for ns in category_namespaces:
                    ns_results = self.pinecone_manager.query(
                        query_vector=query_embedding,
                        top_k=top_k,
                        namespace=ns,
                        include_metadata=True
                    )
                    all_results.extend(ns_results)
                
                # Sort by score and take top_k
                all_results.sort(key=lambda x: x.score, reverse=True)
                search_results = all_results[:top_k]
            else:
                # No filters - search across ALL namespaces
                stats = self.pinecone_manager.get_index_stats()
                namespaces = stats.get('namespaces', {})
                
                if not namespaces:
                    # No namespaces exist, try default namespace
                    logger.warning("No namespaces found in index, searching default namespace")
                    search_results = self.pinecone_manager.semantic_search(
                        query_embedding=query_embedding,
                        top_k=top_k,
                        similarity_threshold=similarity_threshold,
                        video_filter=video_filter,
                        time_window=time_window
                    )
                else:
                    # Query each namespace and combine results
                    logger.info(f"Searching across {len(namespaces)} namespaces")
                    all_results = []
                    for ns in namespaces.keys():
                        ns_results = self.pinecone_manager.query(
                            query_vector=query_embedding,
                            top_k=top_k,
                            namespace=ns,
                            include_metadata=True
                        )
                        all_results.extend(ns_results)
                    
                    # Filter by similarity threshold
                    filtered_results = [r for r in all_results if r.score >= similarity_threshold]
                    
                    # Filter by video if specified
                    if video_filter:
                        filtered_results = [r for r in filtered_results if r.video_name == video_filter]
                    
                    # Filter by time window if specified
                    if time_window:
                        start_time, end_time = time_window
                        filtered_results = [r for r in filtered_results if start_time <= r.timestamp <= end_time]
                    
                    # Sort by score and take top_k
                    filtered_results.sort(key=lambda x: x.score, reverse=True)
                    search_results = filtered_results[:top_k]
            
            # Format results
            formatted_results = []
            for result in search_results:
                formatted_result = {
                    "timestamp": result.timestamp,
                    "caption": result.caption,
                    "similarity_score": result.score,
                    "frame_id": result.frame_id,
                    "video_name": result.video_name,
                    "time_formatted": self._format_timestamp(result.timestamp)
                }
                formatted_results.append(formatted_result)
            
            logger.info(f"Found {len(formatted_results)} results for query: '{query}'")
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise
    
    def batch_search(self,
                    queries: List[str],
                    top_k: int = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Perform batch search for multiple queries
        
        Args:
            queries: List of search queries
            top_k: Number of results per query
            
        Returns:
            Dictionary mapping queries to results
        """
        results = {}
        
        for query in queries:
            try:
                results[query] = self.search(query, top_k=top_k)
            except Exception as e:
                logger.error(f"Failed to search for '{query}': {e}")
                results[query] = []
        
        return results
    
    def _format_timestamp(self, seconds: float) -> str:
        """Format timestamp in readable format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:05.2f}"
        else:
            return f"{minutes:02d}:{secs:05.2f}"
    
    def _save_processing_report(self, stats: Dict, video_name: str):
        """Save processing report to file"""
        report_dir = os.path.join(self.config.OUTPUT_DIR, video_name)
        os.makedirs(report_dir, exist_ok=True)
        
        report_path = os.path.join(report_dir, "processing_report.json")
        
        with open(report_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Processing report saved to: {report_path}")
    
    def get_index_stats(self) -> Dict:
        """Get Pinecone index statistics"""
        if not self.pinecone_manager:
            self.pinecone_manager = PineconeManager(
                api_key=self.config.PINECONE_API_KEY,
                environment=self.config.PINECONE_ENVIRONMENT,
                index_name=self.config.PINECONE_INDEX_NAME,
                dimension=self.config.PINECONE_DIMENSION,
                metric=self.config.PINECONE_METRIC,
                host=getattr(self.config, 'PINECONE_HOST', None)
            )
        
        return self.pinecone_manager.get_index_stats()
    
    def get_available_dates(self) -> List[str]:
        """
        Get list of all dates that have videos in the index
        
        Returns:
            List of dates in YYYY-MM-DD format, sorted
        """
        if not self.pinecone_manager:
            self.pinecone_manager = PineconeManager(
                api_key=self.config.PINECONE_API_KEY,
                environment=self.config.PINECONE_ENVIRONMENT,
                index_name=self.config.PINECONE_INDEX_NAME,
                dimension=self.config.PINECONE_DIMENSION,
                metric=self.config.PINECONE_METRIC,
                host=getattr(self.config, 'PINECONE_HOST', None)
            )
        
        stats = self.pinecone_manager.get_index_stats()
        namespaces = stats.get('namespaces', {})
        
        # Extract unique dates from namespace names
        dates = set()
        for ns in namespaces.keys():
            if ns.startswith('videos:'):
                parts = ns.split(':')
                if len(parts) >= 2:
                    date_part = parts[1]
                    # Validate date format (YYYY-MM-DD)
                    if len(date_part) == 10 and date_part[4] == '-' and date_part[7] == '-':
                        dates.add(date_part)
        
        return sorted(list(dates))
    
    def search_by_date_range(self,
                           query: str,
                           start_date: str,
                           end_date: str,
                           top_k: int = None,
                           similarity_threshold: float = None,
                           namespace_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search across a date range
        
        Args:
            query: Natural language search query
            start_date: Start date (YYYY-MM-DD format)
            end_date: End date (YYYY-MM-DD format)
            top_k: Number of results to return
            similarity_threshold: Minimum similarity score
            namespace_filter: Filter by specific category
            
        Returns:
            List of search results sorted by score
        """
        from datetime import datetime, timedelta
        
        top_k = top_k or self.config.QUERY_TOP_K
        similarity_threshold = similarity_threshold or self.config.QUERY_SIMILARITY_THRESHOLD
        
        # Generate date range
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        
        date_range = []
        current_dt = start_dt
        while current_dt <= end_dt:
            date_range.append(current_dt.strftime("%Y-%m-%d"))
            current_dt += timedelta(days=1)
        
        logger.info(f"Searching across date range: {start_date} to {end_date} ({len(date_range)} days)")
        
        # Initialize components
        if not self.embedding_generator:
            self.embedding_generator = TextEmbeddingGenerator(
                model_name=self.config.EMBEDDING_MODEL,
                batch_size=self.config.EMBEDDING_BATCH_SIZE,
                use_gpu=self.config.USE_GPU,
                normalize=True
            )
        
        if not self.pinecone_manager:
            self.pinecone_manager = PineconeManager(
                api_key=self.config.PINECONE_API_KEY,
                environment=self.config.PINECONE_ENVIRONMENT,
                index_name=self.config.PINECONE_INDEX_NAME,
                dimension=self.config.PINECONE_DIMENSION,
                metric=self.config.PINECONE_METRIC,
                host=getattr(self.config, 'PINECONE_HOST', None)
            )
        
        # Generate query embedding once
        query_embedding = self.embedding_generator.encode_query(query)
        
        # Get all namespaces
        stats = self.pinecone_manager.get_index_stats()
        all_namespaces = stats.get('namespaces', {}).keys()
        
        # Find relevant namespaces for date range
        target_namespaces = []
        for date in date_range:
            if namespace_filter:
                # Specific category
                ns = f"videos:{date}:{namespace_filter}"
                if ns in all_namespaces:
                    target_namespaces.append(ns)
            else:
                # All categories for this date
                date_ns = [ns for ns in all_namespaces if ns.startswith(f"videos:{date}:")]
                target_namespaces.extend(date_ns)
        
        if not target_namespaces:
            logger.warning(f"No namespaces found for date range {start_date} to {end_date}")
            return []
        
        logger.info(f"Searching {len(target_namespaces)} namespaces")
        
        # Query each namespace and collect results
        all_results = []
        for ns in target_namespaces:
            ns_results = self.pinecone_manager.query(
                query_vector=query_embedding,
                top_k=top_k,
                namespace=ns,
                include_metadata=True
            )
            all_results.extend(ns_results)
        
        # Filter by similarity threshold
        filtered_results = [r for r in all_results if r.score >= similarity_threshold]
        
        # Sort by score and take top_k
        filtered_results.sort(key=lambda x: x.score, reverse=True)
        top_results = filtered_results[:top_k]
        
        # Format results
        formatted_results = []
        for result in top_results:
            formatted_result = {
                "timestamp": result.timestamp,
                "caption": result.caption,
                "similarity_score": result.score,
                "frame_id": result.frame_id,
                "video_name": result.video_name,
                "video_date": result.metadata.get('video_date', 'unknown'),
                "time_formatted": self._format_timestamp(result.timestamp)
            }
            formatted_results.append(formatted_result)
        
        logger.info(f"Found {len(formatted_results)} results across date range")
        
        return formatted_results
    
    def clear_index(self) -> bool:
        """Clear all vectors from Pinecone index"""
        if not self.pinecone_manager:
            self.pinecone_manager = PineconeManager(
                api_key=self.config.PINECONE_API_KEY,
                environment=self.config.PINECONE_ENVIRONMENT,
                index_name=self.config.PINECONE_INDEX_NAME,
                dimension=self.config.PINECONE_DIMENSION,
                metric=self.config.PINECONE_METRIC,
                host=getattr(self.config, 'PINECONE_HOST', None)
            )
        
        return self.pinecone_manager.clear_index()
    
    def cleanup(self):
        """Clean up resources"""
        if self.caption_generator:
            self.caption_generator.unload_model()
        if self.embedding_generator:
            self.embedding_generator.unload_model()
        if self.object_pipeline:
            self.object_pipeline.unload_models()
        
        logger.info("Resources cleaned up")


def demo_usage():
    """Demonstrate usage of the Video Search Engine"""
    
    # Initialize engine
    engine = VideoSearchEngine()
    
    # Example: Process a video
    # stats = engine.process_video(
    #     video_path="sample_video.mp4",
    #     video_name="sample_demo",
    #     save_frames=False,
    #     upload_to_pinecone=True
    # )
    
    # Example: Search for content
    # results = engine.search(
    #     query="person walking with a black bag",
    #     top_k=5
    # )
    # 
    # for result in results:
    #     print(f"Time: {result['time_formatted']} - Score: {result['similarity_score']:.3f}")
    #     print(f"  Caption: {result['caption']}")
    #     print(f"  Video: {result['video_name']}")
    #     print()
    
    # Example: Batch search
    # queries = [
    #     "black bag",
    #     "yellow bottle",
    #     "person walking",
    #     "car driving"
    # ]
    # 
    # batch_results = engine.batch_search(queries, top_k=3)
    # 
    # for query, results in batch_results.items():
    #     print(f"\nQuery: '{query}' - Found {len(results)} results")
    #     for result in results[:2]:  # Show top 2
    #         print(f"  {result['time_formatted']} (score: {result['similarity_score']:.3f})")
    
    # Get index stats
    stats = engine.get_index_stats()
    print(f"Index statistics: {json.dumps(stats, indent=2)}")
    
    # Cleanup
    engine.cleanup()


if __name__ == "__main__":
    demo_usage()