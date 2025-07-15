# 🏸 Squash Video Analysis Pipeline

A clean, modular pipeline for analyzing squash videos with player detection, pose estimation, and court mapping.

## 🚀 Features

- **📥 Video Download** - Download squash videos from YouTube  
- **📹 Camera Extraction** - Extract main camera angles automatically
- **🤸 Player Detection** - Detect players with YOLO + MediaPipe pose estimation
- **🏟️ Court Mapping** - Map player positions to court coordinates
- **📊 Data Export** - Export frame-by-frame analysis to CSV
- **🎬 Video Overlay** - Create annotated videos with court overlays

## 🔧 Installation

1. **Install dependencies:**
```bash
pip install opencv-python mediapipe ultralytics yt-dlp pandas tqdm numpy
```

2. **Clone or download the pipeline**

## 🎯 Usage

### Basic Usage
```bash
# Analyze full video
python3 squash_pipeline.py "https://www.youtube.com/watch?v=VIDEO_ID"

# Analyze first 30 seconds (for testing)
python3 squash_pipeline.py "https://www.youtube.com/watch?v=VIDEO_ID" --max-duration 30

# CSV-only mode (faster, no video output)
python3 squash_pipeline.py "https://www.youtube.com/watch?v=VIDEO_ID" --csv-only
```

### Command Line Options
- `--max-duration SECONDS` - Process only first N seconds
- `--csv-only` - Extract data without creating annotated videos
- `--output-dir PATH` - Specify output directory (default: Pipeline_Output)

### Example
```bash
# Analyze first 30 seconds of a match
python3 squash_pipeline.py "https://www.youtube.com/watch?v=yMKDa4aHMsk" --max-duration 30
```

## 📁 Project Structure

```
Pipeline/
├── squash_pipeline.py     # Main pipeline (single entry point)
├── lib/                   # Modular libraries
│   ├── utils.py           # All utilities (downloader, file manager, data exporter)
│   ├── camera_processor.py       # Camera extraction
│   ├── player_detection.py       # Complete player detection system
│   └── yolov8n.pt              # YOLO model weights
└── Pipeline_Output/        # Generated outputs
    ├── 01_Downloads/       # Downloaded videos
    ├── 02_Camera_Segments/ # Main camera extracts
    ├── 03_Player_Annotations/ # Player detection results
    └── 04_Final_Outputs/   # Final annotated videos
```

## 🎬 Processing Pipeline

The pipeline follows this workflow:

1. **📋 Playlist Discovery**: Fetches recent playlists from the channel
2. **📥 Parallel Downloads**: Downloads videos using multiple workers
3. **🎬 Progressive Processing**: Processes videos as they download
4. **📹 Camera Extraction**: Identifies and extracts main camera angles
5. **🤸 Player Detection**: Adds player bounding boxes and pose estimation
6. **🏟️ Court Mapping**: Creates court overlay showing player positions
7. **📊 Final Assembly**: Combines all annotations into final videos

## 🔧 Advanced Configuration

### Performance Tuning

- **Download Workers**: More workers = faster downloads but higher bandwidth usage
- **Process Workers**: More workers = faster processing but higher CPU/GPU usage
- **Memory Usage**: Each worker uses ~1-2GB RAM during processing

### Quality Settings

The pipeline uses optimized settings for speed vs quality:
- **Camera Detection**: Template matching with 3-frame stride
- **Player Detection**: YOLO + MediaPipe with 5-frame inference stride
- **Video Quality**: Maintains original resolution with mp4v codec

## 📊 Monitoring Progress

The pipeline provides comprehensive progress tracking:

- **📋 Playlist Analysis**: Shows playlists being discovered
- **📥 Download Progress**: Real-time download status for all videos
- **🎬 Processing Progress**: Camera extraction and player detection progress
- **📈 Overall Pipeline**: Combined progress across all stages

## 🔍 Troubleshooting

### Common Issues

1. **Download Failures**
   - Check internet connection
   - Verify channel URL is correct
   - Some videos may be region-locked or private

2. **Processing Failures**
   - Ensure sufficient disk space (each video ~50-200MB processed)
   - Check that input videos are valid MP4 files
   - GPU/CPU intensive - reduce workers if system struggles

3. **Memory Issues**
   - Reduce `--process-workers` to 1
   - Close other applications
   - Each worker needs ~1-2GB RAM

### Logs

Check the logs folder for detailed information:
- `download_summary.json`: Download statistics and file paths
- `processing_summary.json`: Processing results and error messages

## 🎯 Output Quality

The final videos include:

- **📹 Clean Camera Angles**: Only main court camera, no crowd shots
- **🤸 Pose Detection**: Stick figure overlays showing player postures
- **🏟️ Court Mapping**: Mini court map in top-right corner
- **🎨 Professional Annotations**: Color-coded player tracking

## 🔄 Resuming Interrupted Processing

The pipeline is designed to be resume-friendly:
- Downloaded videos are preserved
- Partial processing can be continued
- Use the same command to resume from where it left off

## 📈 Performance Expectations

Typical processing times (per video):
- **Download**: 30-120 seconds (depends on video length and connection)
- **Camera Extraction**: 60-180 seconds
- **Player Detection**: 120-300 seconds
- **Total**: 3-10 minutes per video

## 🎉 Example Output

After running the pipeline, you'll have:
- Original downloaded videos organized by playlist
- Extracted main camera segments
- Fully annotated videos with player poses and court mapping
- Comprehensive logs and statistics

The final videos in `04_Final_Outputs/` are ready for analysis, coaching, or sharing!