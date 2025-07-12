# 🏸 Squash Video Analysis Pipeline

A complete automated pipeline that downloads YouTube playlists and processes squash videos with camera angle detection and player pose analysis.

## 🌟 Features

- **📥 Parallel Downloads**: Downloads multiple videos simultaneously from YouTube playlists
- **🎬 Progressive Processing**: Starts processing videos as soon as they're downloaded
- **📹 Camera Analysis**: Extracts main camera angles using template matching
- **🤸 Player Detection**: Detects players with full pose estimation (stick figures)
- **🏟️ Court Mapping**: Shows player positions on a top-down court overlay
- **📊 Progress Tracking**: Real-time progress bars for all operations
- **📁 Organized Output**: Well-structured folder hierarchy for all outputs

## 🚀 Quick Start

```bash
# Navigate to the pipeline directory
cd /Users/soumitraagarwal/Squash/Pipeline

# Run the complete pipeline
python main_pipeline.py "https://www.youtube.com/@PSAWorldTour"
```

## 📋 Requirements

The pipeline automatically installs dependencies, but you can install them manually:

```bash
pip install yt-dlp tqdm opencv-python ultralytics mediapipe numpy
```

## 🎯 Usage

### Basic Usage
```bash
python main_pipeline.py <channel_url>
```

### Advanced Usage
```bash
python main_pipeline.py "https://www.youtube.com/@PSAWorldTour" \\
    --max-playlists 5 \\
    --download-workers 2 \\
    --process-workers 1 \\
    --output-dir "MySquashAnalysis"
```

### Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `channel_url` | Required | YouTube channel URL |
| `--max-playlists` | 12 | Number of recent playlists to process |
| `--download-workers` | 3 | Parallel download threads |
| `--process-workers` | 2 | Parallel processing threads |
| `--output-dir` | Pipeline_Output | Output directory |
| `--quiet` | False | Reduce output verbosity |

## 📁 Output Structure

The pipeline creates a well-organized folder structure:

```
Pipeline_Output/
├── 01_Downloads/                    # Original downloaded videos
│   ├── Playlist_1/
│   └── Playlist_2/
├── 02_Camera_Segments/              # Main camera extracted segments
│   ├── Playlist_1/
│   │   ├── main_camera_segments/
│   │   └── templates_jpeg/
│   └── Playlist_2/
├── 03_Player_Annotations/           # Videos with player detection
│   ├── Playlist_1/
│   └── Playlist_2/
├── 04_Final_Outputs/               # Final processed videos
│   ├── Playlist_1/
│   └── Playlist_2/
└── logs/                           # Processing logs and summaries
    ├── download_summary.json
    └── processing_summary.json
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