#!/usr/bin/env python3
"""
Complete YouTube to Squash Analysis Pipeline

This pipeline:
1. Downloads recent playlists from a YouTube channel
2. Processes videos in parallel as they download
3. Extracts main camera angles
4. Adds player detection with pose estimation
5. Creates annotated videos with court overlays

Usage:
    python main_pipeline.py <channel_url> [options]
"""

import os
import sys
import threading
import signal
import time
import argparse
from tqdm import tqdm

from enhanced_downloader import EnhancedDownloader
from video_processor import VideoProcessor

class SquashAnalysisPipeline:
    """
    Main orchestrator for the complete squash video analysis pipeline.
    Coordinates downloading and processing with real-time progress tracking.
    """
    
    def __init__(self, output_dir="Pipeline_Output", max_download_workers=3, max_process_workers=2):
        self.output_dir = output_dir
        self.max_download_workers = max_download_workers
        self.max_process_workers = max_process_workers
        
        # Initialize components
        self.downloader = EnhancedDownloader(output_dir, max_download_workers)
        self.processor = VideoProcessor(output_dir, max_process_workers)
        
        # Setup callback for progressive processing
        self.downloader.set_video_completed_callback(self.on_video_downloaded)
        
        # Progress tracking
        self.total_playlists = 0
        self.total_videos = 0
        self.downloaded_videos = 0
        self.processed_videos = 0
        self.failed_downloads = 0
        self.failed_processing = 0
        
        # Threading control
        self.stop_event = threading.Event()
        self.processing_thread = None
        
        # Progress bars
        self.main_pbar = None
        
        print(f"🚀 Squash Analysis Pipeline initialized")
        print(f"📁 Output directory: {output_dir}")
        print(f"⚡ Download workers: {max_download_workers}")
        print(f"🎬 Processing workers: {max_process_workers}")
    
    def on_video_downloaded(self, video_info):
        """Callback when a video is downloaded - immediately queue for processing."""
        self.downloaded_videos += 1
        self.processor.add_video_for_processing(video_info)
        
        # Update main progress bar
        if self.main_pbar:
            self.main_pbar.set_postfix_str(f"📥 Downloaded: {self.downloaded_videos}, 🎬 Processing...")
            self.main_pbar.refresh()
    
    def on_video_processed(self, task):
        """Callback when a video is processed."""
        if task.stage == "completed":
            self.processed_videos += 1
        else:
            self.failed_processing += 1
        
        # Update main progress bar
        if self.main_pbar:
            self.main_pbar.set_postfix_str(
                f"📥 Downloaded: {self.downloaded_videos}, "
                f"✅ Processed: {self.processed_videos}, "
                f"❌ Failed: {self.failed_processing}"
            )
            self.main_pbar.refresh()
    
    def run_pipeline(self, channel_url: str, max_playlists: int = 12):
        """Run the complete pipeline."""
        try:
            print(f"\\n{'='*80}")
            print(f"🎯 STARTING SQUASH ANALYSIS PIPELINE")
            print(f"{'='*80}")
            print(f"📺 Channel: {channel_url}")
            print(f"📋 Max playlists: {max_playlists}")
            
            # Phase 1: Get playlists
            print(f"\\n🔍 Phase 1: Discovering playlists...")
            playlists = self.downloader.get_channel_playlists(channel_url, max_playlists)
            
            if not playlists:
                print("❌ No playlists found. Exiting.")
                return
            
            self.total_playlists = len(playlists)
            
            # Calculate total videos
            total_videos = sum(playlist.get('video_count', 0) for playlist in playlists)
            self.total_videos = total_videos
            
            print(f"\\n📊 Pipeline Overview:")
            print(f"   📋 Playlists found: {self.total_playlists}")
            print(f"   🎥 Total videos: {self.total_videos}")
            print(f"   📁 Output folder: {self.output_dir}")
            
            # Show playlist details
            print(f"\\n📋 Playlists to process:")
            for i, playlist in enumerate(playlists, 1):
                print(f"   {i:2d}. {playlist['title']} ({playlist['video_count']} videos)")
            
            # Phase 2: Start processing thread
            print(f"\\n🎬 Phase 2: Starting processing pipeline...")
            self.processing_thread = threading.Thread(target=self._run_processing_loop)
            self.processing_thread.daemon = True
            self.processing_thread.start()
            
            # Phase 3: Download and process concurrently
            print(f"\\n📥 Phase 3: Downloading and processing...")
            
            # Setup main progress tracking
            # We track downloads + processing (roughly 2x total videos for progress)
            with tqdm(total=self.total_videos * 2, desc="Overall Pipeline Progress") as self.main_pbar:
                
                # Start downloads (this will trigger processing via callbacks)
                downloaded_videos = self.downloader.download_playlists_parallel(playlists)
                
                # Wait for all processing to complete
                while (self.processed_videos + self.failed_processing) < len(downloaded_videos):
                    if self.stop_event.is_set():
                        break
                    time.sleep(1)
                    
                    # Update progress bar to show processing completion
                    total_completed = self.downloaded_videos + self.processed_videos
                    self.main_pbar.n = total_completed
                    self.main_pbar.refresh()
            
            # Wait for processing thread to finish
            if self.processing_thread and self.processing_thread.is_alive():
                print("⏳ Waiting for processing to complete...")
                self.processing_thread.join(timeout=30)
            
            # Phase 4: Final summary
            self._print_final_summary()
            
        except KeyboardInterrupt:
            print("\\n🛑 Pipeline interrupted by user")
            self.stop_pipeline()
        except Exception as e:
            print(f"\\n❌ Pipeline error: {e}")
            self.stop_pipeline()
    
    def _run_processing_loop(self):
        """Run the video processing in a separate thread."""
        try:
            self.processor.process_videos_parallel(progress_callback=self.on_video_processed)
        except Exception as e:
            print(f"❌ Processing thread error: {e}")
    
    def _print_final_summary(self):
        """Print comprehensive final summary."""
        print(f"\\n{'='*80}")
        print(f"🎉 PIPELINE COMPLETED")
        print(f"{'='*80}")
        
        print(f"\\n📊 Summary Statistics:")
        print(f"   📋 Playlists processed: {self.total_playlists}")
        print(f"   📥 Videos downloaded: {self.downloaded_videos}/{self.total_videos}")
        print(f"   🎬 Videos processed: {self.processed_videos}/{self.downloaded_videos}")
        print(f"   ❌ Download failures: {self.failed_downloads}")
        print(f"   ❌ Processing failures: {self.failed_processing}")
        
        # Calculate success rate
        if self.total_videos > 0:
            success_rate = (self.processed_videos / self.total_videos) * 100
            print(f"   📈 Overall success rate: {success_rate:.1f}%")
        
        print(f"\\n📁 Output Structure:")
        print(f"   📥 Downloads: {self.downloader.folders['downloads']}")
        print(f"   📹 Camera segments: {self.processor.folders['camera_segments']}")
        print(f"   🤸 Player annotations: {self.processor.folders['player_annotations']}")
        print(f"   🎯 Final outputs: {self.processor.folders['final_outputs']}")
        print(f"   📄 Logs: {self.processor.folders['logs']}")
        
        # Show folder sizes
        self._print_folder_sizes()
        
        print(f"\\n✨ Pipeline completed successfully!")
        print(f"🎬 Your annotated squash videos are ready in: {self.processor.folders['final_outputs']}")
    
    def _print_folder_sizes(self):
        """Print sizes of output folders."""
        def get_folder_size(folder_path):
            total_size = 0
            try:
                for dirpath, _, filenames in os.walk(folder_path):
                    for filename in filenames:
                        filepath = os.path.join(dirpath, filename)
                        if os.path.exists(filepath):
                            total_size += os.path.getsize(filepath)
            except:
                pass
            return total_size
        
        def format_size(size_bytes):
            if size_bytes == 0:
                return "0 B"
            for unit in ['B', 'KB', 'MB', 'GB']:
                if size_bytes < 1024.0:
                    return f"{size_bytes:.1f} {unit}"
                size_bytes /= 1024.0
            return f"{size_bytes:.1f} TB"
        
        print(f"\\n💾 Storage Usage:")
        total_size = 0
        for name, folder in self.processor.folders.items():
            if os.path.exists(folder):
                size = get_folder_size(folder)
                total_size += size
                print(f"   📁 {name.replace('_', ' ').title()}: {format_size(size)}")
        
        print(f"   📊 Total pipeline output: {format_size(total_size)}")
    
    def stop_pipeline(self):
        """Stop the entire pipeline gracefully."""
        print("🛑 Stopping pipeline...")
        self.stop_event.set()
        if hasattr(self.downloader, 'stop_downloads'):
            self.downloader.stop_downloads()
        self.processor.stop_processing()
        
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=10)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Complete YouTube to Squash Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main_pipeline.py "https://www.youtube.com/@PSAWorldTour"
    python main_pipeline.py "https://www.youtube.com/@PSAWorldTour" --max-playlists 5 --output-dir "MyAnalysis"
    python main_pipeline.py "https://www.youtube.com/@PSAWorldTour" --download-workers 2 --process-workers 1
        """
    )
    
    parser.add_argument("channel_url", help="YouTube channel URL")
    parser.add_argument("--max-playlists", type=int, default=12, 
                       help="Maximum number of playlists to process (default: 12)")
    parser.add_argument("--output-dir", default="Pipeline_Output",
                       help="Output directory (default: Pipeline_Output)")
    parser.add_argument("--download-workers", type=int, default=3,
                       help="Number of parallel download workers (default: 3)")
    parser.add_argument("--process-workers", type=int, default=2,
                       help="Number of parallel processing workers (default: 2)")
    parser.add_argument("--quiet", action="store_true",
                       help="Reduce output verbosity")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.channel_url.startswith(('http://', 'https://')):
        print("❌ Error: Please provide a valid YouTube channel URL")
        sys.exit(1)
    
    if args.max_playlists <= 0:
        print("❌ Error: max-playlists must be positive")
        sys.exit(1)
    
    # Create pipeline
    pipeline = SquashAnalysisPipeline(
        output_dir=args.output_dir,
        max_download_workers=args.download_workers,
        max_process_workers=args.process_workers
    )
    
    # Setup signal handler for graceful shutdown
    def signal_handler(signum, _):
        print("\\n🛑 Received interrupt signal")
        pipeline.stop_pipeline()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Run the complete pipeline
        pipeline.run_pipeline(args.channel_url, args.max_playlists)
        
    except Exception as e:
        print(f"\\n❌ Pipeline failed: {e}")
        pipeline.stop_pipeline()
        sys.exit(1)


if __name__ == "__main__":
    main()