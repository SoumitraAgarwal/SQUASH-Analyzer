# 🏸 Squash Video Analysis Suite

A comprehensive toolkit for automated squash video analysis, featuring camera angle detection, player pose estimation, court mapping, and complete YouTube-to-analysis pipelines.

## 🌟 Features

### 📥 Automated Video Pipeline
- **Parallel Downloads**: Downloads multiple YouTube playlists simultaneously
- **Progressive Processing**: Starts analyzing videos as soon as they're downloaded
- **Smart Folder Organization**: Automatically organizes outputs by playlist and processing stage

### 🎬 Advanced Video Analysis
- **Camera Angle Detection**: Automatically identifies and extracts main court camera views
- **Template Matching**: Uses optimized template matching to filter out crowd shots and close-ups
- **Player Detection**: YOLO-based player detection with bounding boxes
- **Pose Estimation**: Full body pose detection using MediaPipe with stick figure visualization

### 🏟️ Court Mapping & Visualization
- **Court Boundary Filtering**: Removes false detections (audience, cameramen) using court constraints
- **Real-time Position Mapping**: Shows player positions on a top-down court overlay
- **Color-coded Tracking**: Distinguishes between players with consistent color coding

### 📊 Comprehensive Progress Tracking
- **Real-time Progress Bars**: Nested progress tracking for all pipeline stages
- **Detailed Logging**: Complete processing logs and error reporting
- **Performance Metrics**: Processing statistics and success rates

## 🚀 Quick Start

### Prerequisites
```bash
pip install yt-dlp tqdm opencv-python ultralytics mediapipe numpy
```

### Basic Usage
```bash
# Navigate to the pipeline directory
cd Pipeline

# Run the complete pipeline
python main_pipeline.py "https://www.youtube.com/@PSAWorldTour"
```

### Advanced Usage
```bash
# Customize processing parameters
python main_pipeline.py "https://www.youtube.com/@PSAWorldTour" \
    --max-playlists 5 \
    --download-workers 2 \
    --process-workers 1 \
    --output-dir "MySquashAnalysis"
```

## 📁 Repository Structure

```
Squash/
├── Pipeline/                           # 🎯 Complete Automation Pipeline
│   ├── main_pipeline.py               # Main entry point - run this for full automation!
│   ├── enhanced_downloader.py          # Parallel YouTube playlist downloader
│   ├── video_processor.py             # Video processing orchestrator
│   └── README.md                      # Pipeline-specific documentation
│
├── Camera/                            # 📹 Camera Angle Detection System
│   ├── extract_main_camera.py         # Main camera extraction logic
│   ├── camera_templates.py            # Template creation and matching algorithms
│   ├── court_detection.py            # Court boundary detection
│   └── README.md                      # Camera system documentation
│
├── Players/                           # 🤸 Player Detection & Analysis
│   ├── enhanced_player_detection.py   # YOLO + MediaPipe integration
│   ├── player_detection_lib.py        # Core detection utilities and tracking
│   ├── court_mapping.py              # Court position mapping and visualization
│   ├── video_with_court_overlay.py   # Video annotation and overlay system
│   └── README.md                      # Player detection documentation
│
├── video_downloader.py               # 📥 Simple playlist downloader (legacy)
├── README.md                         # This comprehensive guide
└── requirements.txt                  # Python dependencies
```

## 🎯 Output Structure

The pipeline creates a well-organized folder hierarchy:

```
Pipeline_Output/
├── 01_Downloads/                    # Original downloaded videos
│   ├── PSA_Squash_Tour_Finals/
│   └── World_Championships_2024/
├── 02_Camera_Segments/              # Main camera extracted segments
│   ├── PSA_Squash_Tour_Finals/
│   │   ├── main_camera_segments/
│   │   └── templates_jpeg/
│   └── World_Championships_2024/
├── 03_Player_Annotations/           # Videos with player detection
│   ├── PSA_Squash_Tour_Finals/
│   └── World_Championships_2024/
├── 04_Final_Outputs/               # Final processed videos ⭐
│   ├── PSA_Squash_Tour_Finals/
│   └── World_Championships_2024/
└── logs/                           # Processing logs and summaries
    ├── download_summary.json
    └── processing_summary.json
```

**🎬 Your final annotated videos are in `04_Final_Outputs/`!**

## 🛠️ Component Usage

### 🎯 Full Automation Pipeline (Recommended)
The complete end-to-end solution:
```bash
cd Pipeline
python main_pipeline.py "https://www.youtube.com/@PSAWorldTour"
```

### 📹 Camera Angle Detection System
Extract main court camera views and filter out crowd shots:
```bash
cd Camera
python extract_main_camera.py
```
**Features:**
- Template-based camera angle classification
- Automatic filtering of crowd shots and close-ups
- Optimized frame processing with early exit
- JPEG template visualization

### 🤸 Player Detection & Analysis
Add comprehensive player analysis to videos:
```bash
cd Players
python enhanced_player_detection.py
```
**Features:**
- YOLO v8 player detection with bounding boxes
- MediaPipe 33-point pose estimation
- Stick figure pose visualization
- Court boundary constraint filtering
- Player tracking and labeling

### 🏟️ Court Mapping & Visualization
Create court overlay with player positions:
```bash
cd Players
python video_with_court_overlay.py
```
**Features:**
- Top-down court perspective mapping
- Real-time player position tracking
- Color-coded player identification
- Court line and boundary visualization

### 📥 Simple Video Downloader (Legacy)
Basic playlist downloader for manual workflows:
```bash
python video_downloader.py
```
**Note:** Use the Pipeline system for automated workflows

## ⚙️ Configuration Options

### Pipeline Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--max-playlists` | 12 | Number of recent playlists to process |
| `--download-workers` | 3 | Parallel download threads |
| `--process-workers` | 2 | Parallel processing threads |
| `--output-dir` | Pipeline_Output | Output directory |
| `--quiet` | False | Reduce output verbosity |

### Performance Tuning
- **More download workers** = faster downloads (bandwidth limited)
- **More process workers** = faster processing (CPU/GPU limited)
- **Memory usage**: ~1-2GB RAM per processing worker

### Quality Settings
The pipeline uses optimized settings for speed vs accuracy:
- **Camera Detection**: 3-frame stride, 5-second no-match threshold
- **Player Detection**: 5-frame inference stride with pose caching
- **Video Quality**: Maintains original resolution

## 📊 Processing Workflow

1. **📋 Playlist Discovery**: Fetches recent playlists from YouTube channel
2. **📥 Parallel Downloads**: Downloads videos using multiple workers
3. **🎬 Progressive Processing**: Processes videos as they complete downloading
4. **📹 Camera Extraction**: Identifies and extracts main camera angles
5. **🤸 Player Detection**: Adds player bounding boxes and pose estimation
6. **🏟️ Court Mapping**: Creates court overlay showing player positions
7. **📊 Final Assembly**: Combines all annotations into final videos

## 🔍 Troubleshooting

### Common Issues

**Download Failures**
- Check internet connection and channel URL
- Some videos may be region-locked or private
- Reduce `--download-workers` if connection is unstable

**Processing Failures**
- Ensure sufficient disk space (50-200MB per processed video)
- Reduce `--process-workers` if running out of memory
- Check that downloaded videos are valid

**Performance Issues**
- Close other applications to free up RAM
- Use SSD storage for better I/O performance
- GPU acceleration automatically used when available

### Error Logs
Check detailed logs in the `logs/` folder:
- `download_summary.json`: Download statistics and file paths
- `processing_summary.json`: Processing results and error messages

## 📈 Performance Expectations

Typical processing times per video:
- **Download**: 30-120 seconds (depends on video length and connection)
- **Camera Extraction**: 60-180 seconds
- **Player Detection**: 120-300 seconds
- **Total**: 3-10 minutes per video

**Example**: Processing 20 videos with 2 workers = ~30-60 minutes total

## 🎉 Example Results

After running the pipeline, you'll have:
- ✅ Original videos organized by playlist
- ✅ Extracted main camera segments (no crowd shots)
- ✅ Fully annotated videos with:
  - Player bounding boxes with labels
  - Stick figure pose overlays
  - Real-time court position mapping
  - Professional color-coded tracking

## 🎮 Usage Scenarios

### 🚀 Complete Automation (Recommended)
For full end-to-end processing from YouTube to analyzed videos:
```bash
cd Pipeline
python main_pipeline.py "https://www.youtube.com/@PSAWorldTour"
```

### 🔧 Custom Workflows
For specific analysis needs, use individual components:

**1. Extract camera angles only:**
```bash
cd Camera
python extract_main_camera.py
```

**2. Add player detection to existing videos:**
```bash
cd Players
python enhanced_player_detection.py
```

**3. Create court overlays:**
```bash
cd Players
python video_with_court_overlay.py
```

## 🚀 Getting Started Examples

### Beginner: Quick Test
```bash
cd Pipeline
python main_pipeline.py "https://www.youtube.com/@PSAWorldTour" --max-playlists 3
```

### Standard: Full Analysis
```bash
cd Pipeline
python main_pipeline.py "https://www.youtube.com/@PSAWorldTour"
```

### Advanced: High-Performance Setup
```bash
cd Pipeline
python main_pipeline.py "https://www.youtube.com/@PSAWorldTour" \
    --download-workers 5 \
    --process-workers 3 \
    --output-dir "HighSpeed_Analysis"
```

### Resource-Constrained: Low-Resource Setup
```bash
cd Pipeline
python main_pipeline.py "https://www.youtube.com/@PSAWorldTour" \
    --download-workers 1 \
    --process-workers 1
```

## 🔬 Technical Details

### Computer Vision Pipeline
- **YOLO v8**: Real-time object detection for player identification
- **MediaPipe**: 33-point pose estimation with body landmarks
- **OpenCV**: Video processing and template matching
- **Template Matching**: Optimized camera angle classification

### Performance Optimizations
- **Frame Skipping**: Intelligent stride patterns for processing efficiency
- **Caching**: Pose data caching between inference frames
- **Parallel Processing**: Concurrent downloads and processing
- **Memory Management**: Efficient video stream handling

### Court Analysis
- **Perspective Transformation**: Maps pixel coordinates to court positions
- **Boundary Constraints**: Filters detections outside court boundaries
- **Position Tracking**: Maintains player identity across frames

## 📄 License

This project is for educational and research purposes. Please respect YouTube's terms of service when downloading videos.

## 🤝 Contributing

Feel free to submit issues and pull requests to improve the pipeline!

---

**🎬 Ready to analyze some squash videos? Run the pipeline and watch the magic happen!**