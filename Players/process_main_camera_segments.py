import os
from tqdm import tqdm
from player_detection_lib import process_video_with_player_detection

# Configuration
MAIN_CAMERA_SEGMENTS_DIR = "../Camera/camera_outputs/main_camera_segments"
OUTPUT_DIR = "annotated_main_camera_segments"
YOLO_MODEL_PATH = "yolov8n.pt"
CROP_MARGIN_PERCENT = 0.10

def get_video_files(directory):
    """Get all mp4 files from directory."""
    video_files = []
    for file in os.listdir(directory):
        if file.lower().endswith('.mp4'):
            video_files.append(file)
    return sorted(video_files)

def main():
    """Process all main camera segments with player detection."""
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Get all video files
    if not os.path.exists(MAIN_CAMERA_SEGMENTS_DIR):
        print(f"Error: Main camera segments directory not found: {MAIN_CAMERA_SEGMENTS_DIR}")
        return
    
    video_files = get_video_files(MAIN_CAMERA_SEGMENTS_DIR)
    
    if not video_files:
        print(f"No MP4 files found in {MAIN_CAMERA_SEGMENTS_DIR}")
        return
    
    print(f"Found {len(video_files)} videos to process")
    
    # Process each video
    successful = 0
    failed = 0
    
    with tqdm(total=len(video_files), desc="Processing videos", position=0, unit='videos') as overall_pbar:
        for i, video_file in enumerate(video_files):
            input_path = os.path.join(MAIN_CAMERA_SEGMENTS_DIR, video_file)
            output_path = os.path.join(OUTPUT_DIR, video_file)
            
            # Update overall progress description with current video
            short_name = video_file[:30] + "..." if len(video_file) > 30 else video_file
            overall_pbar.set_description(f"[{i+1}/{len(video_files)}] {short_name}")
            
            # Skip if already processed
            if os.path.exists(output_path):
                tqdm.write(f"⏭️  Skipping {video_file} - already processed")
                overall_pbar.update(1)
                continue
            
            tqdm.write(f"\n🎬 Processing: {video_file}")
            
            # Process video with nested progress bar
            success = process_video_with_player_detection(
                input_path=input_path,
                output_path=output_path,
                model_path=YOLO_MODEL_PATH,
                crop_margin_percent=CROP_MARGIN_PERCENT,
                show_progress=True,  # Enable individual progress bars
                inference_stride=5,  # Run inference every 5th frame for speed
                progress_position=1  # Nested progress bar position
            )
            
            if success:
                successful += 1
                tqdm.write(f"✅ Successfully processed: {video_file}")
            else:
                failed += 1
                tqdm.write(f"❌ Failed to process: {video_file}")
            
            overall_pbar.update(1)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"🎉 PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"📊 Total videos: {len(video_files)}")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"📁 Output directory: {OUTPUT_DIR}")
    
    if failed > 0:
        print(f"\n⚠️  Check the error messages above for failed videos.")
    else:
        print(f"\n🎯 All videos processed successfully!")

if __name__ == "__main__":
    main()