import os
import subprocess
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from dataclasses import dataclass
from typing import List, Dict, Optional, Callable
import queue
import signal
import sys

@dataclass
class PlaylistInfo:
    """Information about a playlist."""
    id: str
    title: str
    video_count: int
    url: str

@dataclass
class VideoInfo:
    """Information about a video."""
    id: str
    title: str
    playlist_id: str
    playlist_title: str
    duration: Optional[int] = None
    download_path: Optional[str] = None
    status: str = "pending"  # pending, downloading, downloaded, failed

class PlaylistDownloader:
    """
    Downloads YouTube playlists with parallel processing and progress tracking.
    """
    
    def __init__(self, output_dir="downloads", max_workers=4):
        self.output_dir = output_dir
        self.max_workers = max_workers
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Progress tracking
        self.download_queue = queue.Queue()
        self.completed_videos = queue.Queue()
        self.failed_videos = queue.Queue()
        
        # Threading control
        self.stop_event = threading.Event()
        
        # Progress bars
        self.playlist_pbar = None
        self.download_pbar = None
        
        # Callbacks for pipeline integration
        self.video_completed_callback = None
        
    def set_video_completed_callback(self, callback: Callable[[VideoInfo], None]):
        """Set callback to be called when a video is downloaded."""
        self.video_completed_callback = callback
    
    def check_dependencies(self):
        """Check if required dependencies are installed."""
        try:
            # Check yt-dlp
            result = subprocess.run(['yt-dlp', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                print("❌ yt-dlp not found. Installing...")\n                self._install_yt_dlp()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print("❌ yt-dlp not found. Installing...")
            self._install_yt_dlp()
        
        print("✅ Dependencies verified")
    
    def _install_yt_dlp(self):
        """Install yt-dlp."""
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'yt-dlp'])
            print("✅ yt-dlp installed successfully")
        except subprocess.CalledProcessError:
            print("❌ Failed to install yt-dlp")
            sys.exit(1)
    
    def get_channel_playlists(self, channel_url: str, max_playlists: int = 12) -> List[PlaylistInfo]:
        """Get recent playlists from a YouTube channel."""
        print(f"🔍 Fetching playlists from channel...")
        
        try:
            # Get channel playlists info
            cmd = [
                'yt-dlp',
                '--flat-playlist',
                '--print', '%(playlist_id)s|||%(playlist_title)s|||%(playlist_count)s',
                f'{channel_url}/playlists'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode != 0:
                print(f"❌ Error fetching playlists: {result.stderr}")
                return []
            
            playlists = []
            lines = result.stdout.strip().split('\n')
            
            for line in lines[:max_playlists]:
                if '|||' in line:
                    parts = line.split('|||')
                    if len(parts) >= 3:
                        playlist_id = parts[0].strip()
                        title = parts[1].strip()
                        try:
                            video_count = int(parts[2].strip())
                        except ValueError:
                            video_count = 0
                        
                        playlist_url = f"https://www.youtube.com/playlist?list={playlist_id}"
                        
                        playlists.append(PlaylistInfo(
                            id=playlist_id,
                            title=title,
                            video_count=video_count,
                            url=playlist_url
                        ))
            
            print(f"✅ Found {len(playlists)} playlists")
            return playlists
            
        except subprocess.TimeoutExpired:
            print("❌ Timeout fetching playlists")
            return []
        except Exception as e:
            print(f"❌ Error fetching playlists: {e}")
            return []
    
    def get_playlist_videos(self, playlist: PlaylistInfo) -> List[VideoInfo]:
        """Get video information from a playlist."""
        try:
            cmd = [
                'yt-dlp',
                '--flat-playlist',
                '--print', '%(id)s|||%(title)s|||%(duration)s',
                playlist.url
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            
            if result.returncode != 0:
                print(f"❌ Error fetching videos from {playlist.title}: {result.stderr}")
                return []
            
            videos = []
            lines = result.stdout.strip().split('\n')
            
            for line in lines:
                if '|||' in line:
                    parts = line.split('|||')
                    if len(parts) >= 2:
                        video_id = parts[0].strip()
                        title = parts[1].strip()
                        
                        try:
                            duration = int(parts[2].strip()) if len(parts) > 2 and parts[2].strip() != 'NA' else None
                        except (ValueError, IndexError):
                            duration = None
                        
                        videos.append(VideoInfo(
                            id=video_id,
                            title=title,
                            playlist_id=playlist.id,
                            playlist_title=playlist.title,
                            duration=duration
                        ))
            
            return videos
            
        except Exception as e:
            print(f"❌ Error fetching videos from {playlist.title}: {e}")
            return []
    
    def download_video(self, video: VideoInfo) -> VideoInfo:
        """Download a single video."""
        if self.stop_event.is_set():
            video.status = "cancelled"
            return video
        
        video.status = "downloading"
        
        # Create playlist subdirectory
        playlist_dir = os.path.join(self.output_dir, self._sanitize_filename(video.playlist_title))
        os.makedirs(playlist_dir, exist_ok=True)
        
        try:
            # Download with yt-dlp
            cmd = [
                'yt-dlp',
                '--format', 'best[ext=mp4]/best',
                '--output', os.path.join(playlist_dir, '%(title)s.%(ext)s'),
                '--no-playlist',
                f'https://www.youtube.com/watch?v={video.id}'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            if result.returncode == 0:
                # Find the downloaded file
                expected_filename = self._sanitize_filename(video.title) + '.mp4'
                video.download_path = os.path.join(playlist_dir, expected_filename)
                video.status = "downloaded"
                
                # Call completion callback if set
                if self.video_completed_callback:
                    self.video_completed_callback(video)
                
            else:
                video.status = "failed"
                print(f"❌ Failed to download {video.title}: {result.stderr}")
            
        except subprocess.TimeoutExpired:
            video.status = "failed"
            print(f"❌ Download timeout for {video.title}")
        except Exception as e:
            video.status = "failed"
            print(f"❌ Error downloading {video.title}: {e}")
        
        return video
    
    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for filesystem."""
        import re
        # Remove invalid characters
        filename = re.sub(r'[<>:"/\\|?*]', '', filename)
        # Replace multiple spaces with single space
        filename = re.sub(r'\\s+', ' ', filename)
        # Trim and limit length
        return filename.strip()[:100]
    
    def download_playlists(self, playlists: List[PlaylistInfo]) -> Dict[str, List[VideoInfo]]:
        """Download videos from multiple playlists with parallel processing."""
        if not playlists:
            print("❌ No playlists to download")
            return {}
        
        # Collect all videos
        all_videos = []
        playlist_videos = {}
        
        print("📋 Collecting video information...")
        
        with tqdm(playlists, desc="Analyzing playlists") as pbar:
            for playlist in pbar:
                pbar.set_postfix_str(f"Processing {playlist.title[:30]}")
                videos = self.get_playlist_videos(playlist)
                all_videos.extend(videos)
                playlist_videos[playlist.id] = videos
        
        if not all_videos:
            print("❌ No videos found in playlists")
            return playlist_videos
        
        print(f"📥 Starting download of {len(all_videos)} videos...")
        
        # Setup progress bars
        download_pbar = tqdm(total=len(all_videos), desc="Downloading videos", position=0)
        
        # Download videos in parallel
        completed_videos = []
        failed_videos = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all download tasks
            future_to_video = {executor.submit(self.download_video, video): video 
                             for video in all_videos}
            
            # Process completed downloads
            for future in as_completed(future_to_video):
                if self.stop_event.is_set():
                    break
                
                video = future.result()
                
                if video.status == "downloaded":
                    completed_videos.append(video)
                    download_pbar.set_postfix_str(f"✅ {video.title[:30]}")
                else:
                    failed_videos.append(video)
                    download_pbar.set_postfix_str(f"❌ {video.title[:30]}")
                
                download_pbar.update(1)
        
        download_pbar.close()
        
        # Print summary
        print(f"\\n{'='*60}")
        print(f"📊 DOWNLOAD SUMMARY")
        print(f"{'='*60}")
        print(f"✅ Successfully downloaded: {len(completed_videos)}")
        print(f"❌ Failed downloads: {len(failed_videos)}")
        print(f"📁 Output directory: {self.output_dir}")
        
        if failed_videos:
            print(f"\\n❌ Failed videos:")
            for video in failed_videos[:5]:  # Show first 5 failures
                print(f"   - {video.title}")
            if len(failed_videos) > 5:
                print(f"   ... and {len(failed_videos) - 5} more")
        
        return playlist_videos
    
    def stop_downloads(self):
        """Stop all downloads gracefully."""
        print("🛑 Stopping downloads...")
        self.stop_event.set()


def main():
    """Test the playlist downloader."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download YouTube playlists")
    parser.add_argument("channel_url", help="YouTube channel URL")
    parser.add_argument("--max-playlists", type=int, default=12, help="Maximum playlists to download")
    parser.add_argument("--max-workers", type=int, default=4, help="Maximum parallel downloads")
    parser.add_argument("--output-dir", default="downloads", help="Output directory")
    
    args = parser.parse_args()
    
    # Setup signal handler for graceful shutdown
    downloader = PlaylistDownloader(output_dir=args.output_dir, max_workers=args.max_workers)
    
    def signal_handler(sig, frame):
        downloader.stop_downloads()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        # Check dependencies
        downloader.check_dependencies()
        
        # Get playlists
        playlists = downloader.get_channel_playlists(args.channel_url, args.max_playlists)
        
        if not playlists:
            print("❌ No playlists found")
            return
        
        print(f"\\n📋 Found playlists:")
        for i, playlist in enumerate(playlists, 1):
            print(f"   {i}. {playlist.title} ({playlist.video_count} videos)")
        
        # Download playlists
        print(f"\\n🚀 Starting downloads...")
        playlist_videos = downloader.download_playlists(playlists)
        
        print(f"\\n🎉 Download process completed!")
        
    except KeyboardInterrupt:
        print("\\n🛑 Download cancelled by user")
    except Exception as e:
        print(f"\\n❌ Error: {e}")


if __name__ == "__main__":
    main()