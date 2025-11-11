"""
Configuration file for Video Frame Search System
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for the video search system"""
    
    # API Keys (load from environment variables for security)
    PINECONE_API_KEY = os.getenv('PINECONE_API_KEY', '')
    PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT', 'us-east-1')
    PINECONE_HOST = os.getenv('PINECONE_HOST', 'https://test-b5a0x4x.svc.aped-4627-b74a.pinecone.io')
    
    # Pinecone Index Configuration
    PINECONE_INDEX_NAME = 'test'
    PINECONE_DIMENSION = 1024  # For multilingual-e5-large
    PINECONE_METRIC = 'cosine'
    PINECONE_CLOUD = 'aws'
    PINECONE_REGION = 'us-east-1'
    # Optional separate indices for multi-modal storage
    PINECONE_TEXT_INDEX_NAME = 'test-text'
    PINECONE_IMAGE_INDEX_NAME = 'test-image'
    PINECONE_IMAGE_DIMENSION = 512  # Typical CLIP image embedding dim
    
    # Model Configuration
    BLIP_MODEL = 'Salesforce/blip-image-captioning-base'
    # Using multilingual-e5-large for better semantic understanding
    EMBEDDING_MODEL = 'intfloat/multilingual-e5-large'  # 1024 dimensions
    # Alternative: 'sentence-transformers/all-MiniLM-L6-v2' (384 dimensions)
    
    # Frame Extraction Configuration
    FRAME_SIMILARITY_THRESHOLD = 0.90  # Higher threshold to capture more frames (only skip very similar frames)
    FRAME_EXTRACTION_INTERVAL = 2.0  # Extract frame every N seconds (if not using similarity)
    MAX_FRAMES_PER_VIDEO = 1000  # Maximum frames to extract per video
    FRAME_RESIZE_WIDTH = 640  # Resize frames for memory efficiency (None for original size)
    MIN_FRAMES_PER_VIDEO = 10  # Minimum frames to extract regardless of similarity
    
    # Enhanced Caption Configuration
    GENERATE_MULTIPLE_CAPTIONS = True  # Generate multiple object-focused captions per frame
    CAPTIONS_PER_FRAME = 3  # Number of different captions to generate per frame
    USE_OBJECT_FOCUSED_PROMPTS = True  # Use object-focused prompts for more detailed descriptions
    
    # Processing Configuration
    BLIP_BATCH_SIZE = 8  # Batch size for BLIP caption generation
    EMBEDDING_BATCH_SIZE = 32  # Batch size for embedding generation
    PINECONE_BATCH_SIZE = 100  # Batch size for Pinecone uploads
    # Enhanced options
    ENABLE_DUAL_EMBEDDINGS = True  # Upload caption/image vectors to separate indices
    ENABLE_TEMPORAL_BOOTSTRAPPING = True  # Compute temporal confidence and smoothing
    CONFIDENCE_THRESHOLD = 0.5  # Default threshold for filtering search results
    # Fusion / multi-index search options
    FUSION_TEXT_WEIGHT = 0.6  # Weight for text/caption index during fusion
    FUSION_IMAGE_WEIGHT = 0.4  # Weight for image index during fusion
    CLIP_MODEL_NAME = 'sentence-transformers/clip-vit-base-patch32'  # CLIP model for image embeddings and text->image queries
    ENABLE_CLIP_DEDUPE = False  # If True, use CLIP-based semantic dedupe instead of histogram-based
    CLIP_DEDUPE_THRESHOLD = 0.88  # Similarity threshold for CLIP semantic dedupe (0-1)
    # Thumbnails
    SAVE_THUMBNAILS = True  # Save small thumbnails for UI and metadata
    THUMBNAIL_SIZE = (256, 256)  # Size of generated thumbnails (width, height)
    
    # Query Configuration
    QUERY_TOP_K = 10  # Number of results to return
    QUERY_SIMILARITY_THRESHOLD = 0.6  # Minimum similarity score for results
    DUPLICATE_TIME_WINDOW = 5.0  # Seconds within which to consider frames as duplicates (increased for multi-captions)
    
    # Logging Configuration
    LOG_LEVEL = 'INFO'
    LOG_FILE = 'video_search.log'
    
    # Performance Configuration
    USE_GPU = True  # Use GPU if available
    NUM_WORKERS = 4  # Number of workers for data loading
    
    # File paths
    TEMP_DIR = './temp'
    OUTPUT_DIR = './output'
    
    @classmethod
    def validate(cls):
        """Validate configuration settings"""
        if not cls.PINECONE_API_KEY:
            raise ValueError("PINECONE_API_KEY not set. Please set it in .env file or environment variables")
        
        if cls.PINECONE_DIMENSION not in [384, 768, 1024, 1536]:
            print(f"Warning: Unusual embedding dimension {cls.PINECONE_DIMENSION}. Common values are 384, 768, 1024, or 1536")
        
        if cls.FRAME_SIMILARITY_THRESHOLD < 0 or cls.FRAME_SIMILARITY_THRESHOLD > 1:
            raise ValueError("FRAME_SIMILARITY_THRESHOLD must be between 0 and 1")
        
        # Create directories if they don't exist
        os.makedirs(cls.TEMP_DIR, exist_ok=True)
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        
        return True