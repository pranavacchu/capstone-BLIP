"""
BLIP Caption Generation Module
Generates semantic captions for video frames using the BLIP vision-language model
"""

import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import logging
from typing import List, Tuple, Optional, Dict
from tqdm import tqdm
import gc
from dataclasses import dataclass
from frame_extractor import FrameData

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CaptionedFrame:
    """Data structure for frame with caption"""
    frame_data: FrameData
    caption: str
    confidence: Optional[float] = None

class BlipCaptionGenerator:
    """Generate captions for frames using BLIP model"""
    
    def __init__(self, 
                 model_name: str = "Salesforce/blip-image-captioning-base",
                 batch_size: int = 8,
                 use_gpu: bool = True,
                 max_length: int = 50,
                 num_beams: int = 4,
                 generate_multiple_captions: bool = False,
                 captions_per_frame: int = 3):
        """
        Initialize the BLIP caption generator
        
        Args:
            model_name: Hugging Face model identifier
            batch_size: Batch size for processing
            use_gpu: Whether to use GPU if available
            max_length: Maximum caption length
            num_beams: Number of beams for beam search
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.num_beams = num_beams
        self.generate_multiple_captions = generate_multiple_captions
        self.captions_per_frame = captions_per_frame
        
        # Object-focused prompts for diverse, detailed captions
        self.object_prompts = [
            "a photo of",  # General caption
            "this image shows",  # Descriptive focus
            "visible in this scene are"  # Object enumeration focus
        ]
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Load model and processor
        self._load_model()
        
    def _load_model(self):
        """Load BLIP model and processor"""
        logger.info(f"Loading BLIP model: {self.model_name}")
        
        try:
            # Load processor for image preprocessing
            self.processor = BlipProcessor.from_pretrained(self.model_name)
            
            # Load model
            self.model = BlipForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32
            )
            
            # Move model to device
            self.model = self.model.to(self.device)
            
            # Set model to evaluation mode
            self.model.eval()
            
            logger.info("BLIP model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load BLIP model: {e}")
            raise
    
    def generate_captions(self, frames: List[FrameData], 
                         filter_empty: bool = True,
                         min_caption_length: int = 3) -> List[CaptionedFrame]:
        """
        Generate captions for a list of frames
        
        Args:
            frames: List of FrameData objects
            filter_empty: Whether to filter out empty/short captions
            min_caption_length: Minimum caption length (in words)
            
        Returns:
            List of CaptionedFrame objects
        """
        logger.info(f"Generating captions for {len(frames)} frames")
        
        captioned_frames = []
        
        # If multiple captions per frame is enabled, generate variants
        if self.generate_multiple_captions:
            logger.info(f"Generating {self.captions_per_frame} caption variants per frame")
            for frame_data in tqdm(frames, desc="Generating multi-captions"):
                variants = self.generate_object_focused_captions(frame_data, 
                                                                 num_variants=self.captions_per_frame)
                
                for caption in variants:
                    # Filter empty or short captions if requested
                    if filter_empty and len(caption.split()) < min_caption_length:
                        continue
                    
                    captioned_frame = CaptionedFrame(
                        frame_data=frame_data,
                        caption=caption
                    )
                    captioned_frames.append(captioned_frame)
        else:
            # Process frames in batches (original behavior)
            for batch_start in tqdm(range(0, len(frames), self.batch_size), 
                                    desc="Generating captions"):
                batch_end = min(batch_start + self.batch_size, len(frames))
                batch_frames = frames[batch_start:batch_end]
                
                # Generate captions for batch
                batch_captions = self._generate_batch_captions(batch_frames)
                
                # Create CaptionedFrame objects
                for frame_data, caption in zip(batch_frames, batch_captions):
                    # Filter empty or short captions if requested
                    if filter_empty and len(caption.split()) < min_caption_length:
                        logger.debug(f"Skipping short caption: '{caption}' for frame {frame_data.frame_id}")
                        continue
                    
                    captioned_frame = CaptionedFrame(
                        frame_data=frame_data,
                        caption=caption
                    )
                    captioned_frames.append(captioned_frame)
        
        logger.info(f"Generated {len(captioned_frames)} valid captions")
        return captioned_frames
    
    def _generate_batch_captions(self, batch_frames: List[FrameData], 
                                  text_prompt: Optional[str] = None) -> List[str]:
        """Generate captions for a batch of frames with optional text prompting"""
        try:
            # Extract PIL images from frame data
            images = [frame.image for frame in batch_frames]
            
            # Preprocess images with optional text prompt
            if text_prompt:
                inputs = self.processor(images=images, 
                                       text=[text_prompt] * len(images),
                                       return_tensors="pt", 
                                       padding=True)
            else:
                inputs = self.processor(images=images, return_tensors="pt", padding=True)
            
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate captions
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=self.max_length,
                    num_beams=self.num_beams,
                    do_sample=False,  # Deterministic generation
                    early_stopping=True
                )
            
            # Decode captions
            captions = self.processor.batch_decode(outputs, skip_special_tokens=True)
            
            # Clean up captions
            captions = [self._clean_caption(caption) for caption in captions]
            
            return captions
            
        except Exception as e:
            logger.error(f"Error generating batch captions: {e}")
            # Return empty captions for failed batch
            return ["" for _ in batch_frames]
    
    def _clean_caption(self, caption: str) -> str:
        """Clean and normalize generated caption"""
        # Remove extra whitespace
        caption = " ".join(caption.split())
        
        # Capitalize first letter
        if caption:
            caption = caption[0].upper() + caption[1:]
        
        # Ensure caption ends with period if it doesn't have punctuation
        if caption and caption[-1] not in '.!?':
            caption += '.'
        
        return caption
    
    def generate_object_focused_captions(self, frame: FrameData, 
                                         num_variants: int = 3) -> List[str]:
        """
        Generate multiple object-focused caption variants for a single frame
        Uses different prompting strategies to focus on objects, attributes, and scene details
        
        Args:
            frame: Single FrameData object
            num_variants: Number of caption variants to generate
            
        Returns:
            List of diverse, object-focused caption variants
        """
        image = frame.image
        captions = []
        
        with torch.no_grad():
            # Strategy 1: Standard unconditional caption (most detailed)
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            outputs = self.model.generate(
                **inputs,
                max_length=self.max_length,
                num_beams=self.num_beams,
                do_sample=False,
                early_stopping=True
            )
            caption = self.processor.decode(outputs[0], skip_special_tokens=True)
            captions.append(self._clean_caption(caption))
            
            # Strategy 2: Sample with higher diversity for alternative descriptions
            for i in range(1, min(num_variants, 3)):
                outputs = self.model.generate(
                    **inputs,
                    max_length=self.max_length,
                    num_beams=self.num_beams,
                    do_sample=True,  # Enable sampling for diversity
                    top_p=0.92,
                    temperature=0.7 + (i * 0.15),  # Vary temperature for diversity
                    early_stopping=True
                )
                caption = self.processor.decode(outputs[0], skip_special_tokens=True)
                captions.append(self._clean_caption(caption))
        
        # Remove duplicates while preserving order
        seen = set()
        unique_captions = []
        for caption in captions:
            caption_lower = caption.lower().strip()
            if caption_lower not in seen and len(caption.split()) >= 3:
                seen.add(caption_lower)
                unique_captions.append(caption)
        
        # If we don't have enough unique captions, keep at least one
        if not unique_captions and captions:
            unique_captions = [captions[0]]
        
        return unique_captions
    
    def generate_caption_variants(self, frame: FrameData, 
                                 num_variants: int = 3) -> List[str]:
        """
        Generate multiple caption variants for a single frame
        Useful for improving search diversity
        
        Args:
            frame: Single FrameData object
            num_variants: Number of caption variants to generate
            
        Returns:
            List of caption variants
        """
        # Delegate to the new object-focused method
        return self.generate_object_focused_captions(frame, num_variants)
    
    def filter_duplicate_captions(self, 
                                 captioned_frames: List[CaptionedFrame],
                                 time_window: float = 2.0) -> List[CaptionedFrame]:
        """
        Filter duplicate captions within a time window
        Note: When generating multiple captions per frame, this allows different captions
        for the same frame but prevents exact duplicate captions nearby in time
        
        Args:
            captioned_frames: List of CaptionedFrame objects
            time_window: Time window in seconds to check for duplicates
            
        Returns:
            Filtered list of CaptionedFrame objects
        """
        filtered = []
        caption_timestamps = {}  # Track last timestamp for each caption
        
        for cf in captioned_frames:
            caption_lower = cf.caption.lower().strip()
            timestamp = cf.frame_data.timestamp
            
            # Check if we've seen this EXACT caption text recently
            if caption_lower in caption_timestamps:
                last_timestamp = caption_timestamps[caption_lower]
                # Only filter if it's the exact same caption within time window
                # Different captions for same frame are allowed
                if abs(timestamp - last_timestamp) < time_window:
                    # Skip exact duplicate within time window
                    logger.debug(f"Skipping duplicate caption at {timestamp:.2f}s: '{cf.caption}'")
                    continue
            
            # Keep this caption
            filtered.append(cf)
            caption_timestamps[caption_lower] = timestamp
        
        filtered_count = len(captioned_frames) - len(filtered)
        if filtered_count > 0:
            logger.info(f"Filtered {filtered_count} duplicate captions")
        return filtered
    
    def enhance_captions_with_context(self, 
                                     captioned_frames: List[CaptionedFrame],
                                     video_name: str = "video") -> List[CaptionedFrame]:
        """
        Enhance captions with contextual information
        
        Args:
            captioned_frames: List of CaptionedFrame objects
            video_name: Name of the video for context
            
        Returns:
            Enhanced CaptionedFrame objects
        """
        for cf in captioned_frames:
            # Add timestamp context to caption
            timestamp_str = f"[{cf.frame_data.timestamp:.1f}s]"
            
            # You could add more context here based on your needs
            # For now, we'll keep the original caption but store metadata
            cf.frame_data.video_name = video_name
        
        return captioned_frames
    
    def get_caption_statistics(self, captioned_frames: List[CaptionedFrame]) -> Dict:
        """Get statistics about generated captions"""
        if not captioned_frames:
            return {"total": 0}
        
        captions = [cf.caption for cf in captioned_frames]
        caption_lengths = [len(c.split()) for c in captions]
        
        stats = {
            "total": len(captions),
            "unique": len(set(captions)),
            "avg_length": sum(caption_lengths) / len(caption_lengths),
            "min_length": min(caption_lengths),
            "max_length": max(caption_lengths),
            "duplicate_rate": 1 - (len(set(captions)) / len(captions))
        }
        
        return stats
    
    def clear_gpu_cache(self):
        """Clear GPU cache to free memory"""
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            gc.collect()
            logger.info("GPU cache cleared")
    
    def unload_model(self):
        """Unload model from memory"""
        del self.model
        del self.processor
        self.clear_gpu_cache()
        logger.info("Model unloaded from memory")


# Example usage and testing
if __name__ == "__main__":
    # Test the caption generator
    generator = BlipCaptionGenerator(
        batch_size=8,
        use_gpu=True,
        max_length=50,
        num_beams=4
    )
    
    # Example: Generate captions for frames
    # from frame_extractor import VideoFrameExtractor
    # 
    # extractor = VideoFrameExtractor()
    # frames = extractor.extract_frames("sample_video.mp4")
    # 
    # captioned_frames = generator.generate_captions(frames)
    # 
    # # Get statistics
    # stats = generator.get_caption_statistics(captioned_frames)
    # print(f"Caption statistics: {stats}")
    # 
    # # Print some examples
    # for cf in captioned_frames[:5]:
    #     print(f"Time: {cf.frame_data.timestamp:.2f}s - Caption: {cf.caption}")