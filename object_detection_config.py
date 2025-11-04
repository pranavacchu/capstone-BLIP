"""
Configuration for Object Detection + Captioning Pipeline
Optimized for Google Colab T4 GPU
"""

import os
from typing import Optional

class ObjectDetectionConfig:
    """Configuration for object-focused video search pipeline"""
    
    # Model Configuration
    GROUNDING_DINO_MODEL = "IDEA-Research/grounding-dino-tiny"  # Smaller, faster model for T4
    BLIP_MODEL = "Salesforce/blip-image-captioning-base"
    EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
    
    # GPU and Memory Settings
    USE_GPU = True
    USE_HALF_PRECISION = True  # FP16 for faster inference on T4
    MAX_BATCH_SIZE_DETECTION = 1  # Process one frame at a time for detection
    MAX_BATCH_SIZE_CAPTION = 4  # Batch cropped objects for captioning
    MAX_BATCH_SIZE_EMBEDDING = 32
    
    # Object Detection Settings
    OBJECT_CONFIDENCE_THRESHOLD = 0.25  # Lower threshold to catch more objects
    BOX_THRESHOLD = 0.25
    TEXT_THRESHOLD = 0.25
    MIN_OBJECT_SIZE = 30  # Minimum object size in pixels
    MAX_OBJECTS_PER_FRAME = 10  # Limit objects to caption per frame
    
    # Target Objects Only - Specific items to detect and caption
    SURVEILLANCE_OBJECTS = [
        # Bags and carrying items
        "duffel bag", "duffel", "backpack", "bag",
        
        # Electronics
        "laptop", "computer", "tablet",
        
        # Safety equipment
        "helmet",
        
        # Containers
        "bottle", "water bottle",
        
        # Personal items
        "file folder", "folder", "document folder",
        "umbrella", "coat", "jacket",
        
        # Travel items
        "suitcase", "luggage"
    ]
    
    # Frame Extraction Settings
    FRAME_SIMILARITY_THRESHOLD = 0.85
    MAX_FRAMES_PER_VIDEO = 500  # Limit for memory management
    FRAME_RESIZE_WIDTH = 640  # Smaller for faster processing
    
    # Caption Settings
    CAPTION_MAX_LENGTH = 30  # Shorter for attribute-focused captions
    CAPTION_NUM_BEAMS = 3
    INCLUDE_SCENE_CAPTION = True  # Fallback if no objects detected
    
    # Deduplication Settings
    DUPLICATE_TIME_WINDOW = 2.0  # Seconds
    EMBEDDING_SIMILARITY_THRESHOLD = 0.95
    
    # Pinecone Settings (from existing config)
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
    PINECONE_HOST = os.getenv("PINECONE_HOST", "")
    PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
    PINECONE_INDEX_NAME = "capstone"
    PINECONE_DIMENSION = 384  # For BGE-small
    PINECONE_METRIC = "cosine"
    PINECONE_BATCH_SIZE = 100
    
    # Query Settings
    QUERY_TOP_K = 5
    QUERY_SIMILARITY_THRESHOLD = 0.5
    
    # Output Settings
    OUTPUT_DIR = "./output"
    SAVE_FRAMES = False
    SAVE_DETECTIONS = False  # Save bounding box visualizations
    
    @classmethod
    def for_colab(cls, api_key: str, host: str):
        """
        Create configuration optimized for Google Colab
        
        Args:
            api_key: Pinecone API key
            host: Pinecone host URL
        
        Returns:
            ObjectDetectionConfig instance
        """
        config = cls()
        config.PINECONE_API_KEY = api_key
        config.PINECONE_HOST = host
        config.USE_GPU = True
        config.USE_HALF_PRECISION = True
        return config
    
    @classmethod
    def for_local(cls, api_key: str, host: str, use_gpu: bool = False):
        """
        Create configuration for local testing (CPU)
        
        Args:
            api_key: Pinecone API key
            host: Pinecone host URL
            use_gpu: Whether GPU is available locally
        
        Returns:
            ObjectDetectionConfig instance
        """
        config = cls()
        config.PINECONE_API_KEY = api_key
        config.PINECONE_HOST = host
        config.USE_GPU = use_gpu
        config.USE_HALF_PRECISION = False  # FP32 for CPU
        config.MAX_BATCH_SIZE_DETECTION = 1
        config.MAX_BATCH_SIZE_CAPTION = 2
        return config
    
    def validate(self):
        """Validate configuration"""
        if not self.PINECONE_API_KEY:
            raise ValueError("PINECONE_API_KEY is required")
        if not self.PINECONE_HOST:
            raise ValueError("PINECONE_HOST is required")
        
        return True
    
    def to_dict(self):
        """Convert configuration to dictionary"""
        return {k: v for k, v in vars(self).items() if not k.startswith('_')}
