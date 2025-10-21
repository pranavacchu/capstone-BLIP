"""
Object Detection Colab Code Cells
Python code only - ready to copy into Colab notebook
Each section is a separate cell
"""

# =============================================================================
# CELL 1: Install Grounding DINO Dependencies (after Step 1)
# =============================================================================

# Install Grounding DINO dependencies
print("Installing Grounding DINO and dependencies...")
print("This may take 2-3 minutes...")

# Install timm for vision models
import subprocess
subprocess.run(['pip', 'install', '-q', 'timm'], check=False)

# Install supervision for visualization (optional)
subprocess.run(['pip', 'install', '-q', 'supervision'], check=False)

print("\nGrounding DINO dependencies installed!")
print("Models will be downloaded automatically from Hugging Face on first use")


# =============================================================================
# CELL 2: Choose Captioning Method (before Step 5)
# =============================================================================

print("Choose your captioning method:\n")
print("1. Standard BLIP (faster, general scene captions)")
print("2. Object Detection + BLIP (slower, object-focused)")
print()

method_choice = input("Enter choice (1/2, default=1): ").strip() or "1"

use_object_detection = (method_choice == "2")

if use_object_detection:
    print("\nUsing Object Detection + BLIP pipeline")
    print("   This will detect specific objects and caption them individually")
    print("   Focuses on: bags, laptops, helmets, phones, bottles, vehicles, etc.")
else:
    print("\nUsing Standard BLIP captioning")
    print("   This will generate general scene captions")


# =============================================================================
# CELL 3: Modified Step 5 - Process Video with Chosen Method
# =============================================================================

import time
from datetime import datetime

if 'video_path' not in locals() or not video_path:
    print("Please upload a video first (run the previous cell)")
else:
    # Set video name
    video_name = input("Enter a name for this video (or press Enter for auto-name): ").strip()
    if not video_name:
        video_name = f"video_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print(f"\nProcessing video: {video_name}")
    
    if use_object_detection:
        print("Using Object Detection + BLIP (may take longer)")
    else:
        print("Using Standard BLIP captioning")
    
    print("This will take a few minutes... Please wait.\n")
    print("=" * 60)

    start_time = time.time()

    try:
        # Process the video with chosen method
        stats = engine.process_video(
            video_path=video_path,
            video_name=video_name,
            save_frames=False,
            upload_to_pinecone=True,
            use_object_detection=use_object_detection  # NEW PARAMETER
        )

        processing_time = time.time() - start_time

        print("\n" + "=" * 60)
        print("\nVIDEO PROCESSING COMPLETE!\n")
        print(f"Processing Statistics:")
        print(f"   Video name: {video_name}")
        print(f"   Method: {'Object Detection + BLIP' if use_object_detection else 'Standard BLIP'}")
        print(f"   Frames extracted: {stats['total_frames_extracted']:,}")
        print(f"   Captions generated: {stats['frames_with_captions']:,}")
        print(f"   Unique embeddings: {stats.get('embeddings_generated', 0):,}")
        print(f"   Uploaded to Pinecone: {stats['embeddings_uploaded']:,}")
        print(f"   Processing time: {processing_time/60:.1f} minutes")
        
        if use_object_detection:
            print(f"\nObject Detection Stats:")
            print(f"   Object-focused captions generated")
            print(f"   Focus: bags, laptops, helmets, phones, etc.")

        # Save video_name for next steps
        processed_video_name = video_name

    except Exception as e:
        print(f"\nError processing video: {e}")
        print("\nTroubleshooting tips:")
        print("- If GPU memory error: Restart runtime and try again")
        print("- If video format error: Convert video to MP4 format")
        print("- If object detection fails: Try standard BLIP method instead")


# =============================================================================
# CELL 4: Demo Object Detection on Single Frame (Optional)
# =============================================================================

# Demo object detection on a single frame
print("Object Detection Demo\n")

if 'video_path' not in locals() or not video_path:
    print("Please upload a video first")
else:
    # Extract one frame for demo
    import cv2
    from PIL import Image
    
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        # Convert to PIL Image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        # Initialize object detector
        from object_detector import GroundingDINODetector
        
        print("Loading Grounding DINO model...")
        detector = GroundingDINODetector(
            confidence_threshold=0.3,
            use_gpu=True
        )
        
        # Detect objects
        print("Detecting objects...")
        detections = detector.detect_objects(pil_image)
        
        print(f"\nDetected {len(detections)} objects:\n")
        
        for i, det in enumerate(detections[:10], 1):  # Show top 10
            print(f"{i}. {det.label.upper()}")
            print(f"   Confidence: {det.confidence:.2%}")
            print(f"   Bounding box: {det.bbox}")
            print()
        
        if len(detections) > 10:
            print(f"... and {len(detections) - 10} more objects")
        
        # Display image with detections (optional, requires matplotlib)
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
            
            fig, ax = plt.subplots(1, figsize=(12, 8))
            ax.imshow(pil_image)
            
            # Draw bounding boxes
            for det in detections[:15]:  # Show top 15
                x1, y1, x2, y2 = det.bbox
                width = x2 - x1
                height = y2 - y1
                
                rect = patches.Rectangle(
                    (x1, y1), width, height,
                    linewidth=2, edgecolor='red', facecolor='none'
                )
                ax.add_patch(rect)
                
                # Add label
                ax.text(x1, y1 - 5, f"{det.label} ({det.confidence:.2f})",
                       color='red', fontsize=10, weight='bold',
                       bbox=dict(facecolor='white', alpha=0.7))
            
            ax.axis('off')
            plt.title("Object Detection Results")
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("Install matplotlib to visualize detections")
        
        # Clean up
        detector.clear_gpu_cache()
    else:
        print("Failed to read frame from video")


# =============================================================================
# CELL 5: Compare Standard vs Object-Focused Captions (Optional)
# =============================================================================

# Compare standard BLIP vs object-focused captions
print("Comparing Caption Methods\n")
print("=" * 60)

if 'video_path' in locals() and video_path:
    # Extract a few frames
    from frame_extractor import VideoFrameExtractor
    
    print("Extracting sample frames...")
    extractor = VideoFrameExtractor(
        similarity_threshold=0.85,
        max_frames=3  # Just 3 frames for demo
    )
    frames = extractor.extract_frames(video_path, use_similarity_filter=True)
    
    if len(frames) > 0:
        # Standard BLIP
        print("\n1. STANDARD BLIP CAPTIONS:\n")
        from caption_generator import BlipCaptionGenerator
        
        blip_gen = BlipCaptionGenerator(batch_size=4, use_gpu=True)
        standard_captions = blip_gen.generate_captions(frames, filter_empty=True)
        
        for cf in standard_captions[:3]:
            print(f"Time {cf.frame_data.timestamp:.1f}s:")
            print(f"   {cf.caption}")
            print()
        
        # Object-focused captions
        print("\n2. OBJECT-FOCUSED CAPTIONS:\n")
        from object_caption_pipeline import ObjectCaptionPipeline
        
        obj_pipeline = ObjectCaptionPipeline(use_gpu=True)
        object_captions = obj_pipeline.process_frames(frames, show_progress=False)
        
        for oc in object_captions[:5]:
            print(f"Time {oc.frame_data.timestamp:.1f}s:")
            print(f"   Object: {oc.object_label} (confidence: {oc.confidence:.2f})")
            print(f"   Caption: {oc.attribute_caption}")
            print()
        
        print("\n" + "=" * 60)
        print("\nComparison:")
        print(f"   Standard BLIP: {len(standard_captions)} scene captions")
        print(f"   Object-focused: {len(object_captions)} object captions")
        print()
        print("Notice:")
        print("   - Object-focused captions identify specific items")
        print("   - Include attributes like color, type, appearance")
        print("   - Better for search: 'black backpack', 'red helmet', etc.")
        
        # Clean up
        blip_gen.clear_gpu_cache()
        obj_pipeline.clear_cache()
        
    else:
        print("No frames extracted")
else:
    print("Please upload a video first")


# =============================================================================
# CELL 6: Quick Test - Search for Objects (Optional)
# =============================================================================

# Test searching for specific objects
if 'processed_video_name' in locals():
    print("Testing Object Search\n")
    print("=" * 60)
    
    # Test queries
    test_queries = [
        "backpack",
        "bag",
        "laptop",
        "phone",
        "helmet"
    ]
    
    for query in test_queries:
        print(f"\nSearching for: '{query}'")
        results = engine.search(query, top_k=3)
        
        if results:
            print(f"Found {len(results)} results:")
            for i, result in enumerate(results, 1):
                print(f"  {i}. Time {result['time_formatted']}: {result['caption']}")
        else:
            print(f"  No results found")
    
    print("\n" + "=" * 60)
else:
    print("Process a video first before searching")
