import os
import yt_dlp
import threading
import queue
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from dataclasses import dataclass
from typing import List, Dict, Optional, Callable
import json

@dataclass
class VideoInfo:
    """Information about a downloaded video."""
    id: str
    title: str
    playlist_title: str
    filepath: str
    status: str = "pending"  # pending, downloading, completed, failed
    file_size: Optional[int] = None

class EnhancedDownloader:
    """
    Enhanced video downloader that builds on the existing yt-dlp approach
    with parallel processing and callback support for pipeline integration.
    """
    
    def __init__(self, base_output_dir="Pipeline_Output", max_workers=3):
        self.base_output_dir = base_output_dir
        self.max_workers = max_workers
        
        # Create organized folder structure
        self.folders = {
            'downloads': os.path.join(base_output_dir, "01_Downloads"),
            'camera_segments': os.path.join(base_output_dir, "02_Camera_Segments"),
            'player_annotations': os.path.join(base_output_dir, "03_Player_Annotations"),
            'final_outputs': os.path.join(base_output_dir, "04_Final_Outputs"),
            'logs': os.path.join(base_output_dir, "logs")
        }
        
        # Create all directories
        for folder in self.folders.values():
            os.makedirs(folder, exist_ok=True)
        
        # Callbacks for pipeline integration
        self.video_completed_callback = None
        
        # Progress tracking
        self.completed_videos = queue.Queue()
        self.failed_videos = queue.Queue()
        
        print(f"📁 Created pipeline folder structure in: {base_output_dir}")
    
    def set_video_completed_callback(self, callback: Callable[[VideoInfo], None]):
        """Set callback to be called when a video is downloaded."""
        self.video_completed_callback = callback
    
    def get_channel_playlists(self, channel_url: str, max_playlists: int = 12) -> List[Dict]:
        """Get recent playlists from a YouTube channel."""
        print(f"🔍 Fetching recent {max_playlists} playlists from channel...")
        
        ydl_opts = {
            'quiet': True,
            'extract_flat': True,
            'playlistend': max_playlists
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # Extract channel playlists
                channel_info = ydl.extract_info(f"{channel_url}/playlists", download=False)
                
                playlists = []
                if 'entries' in channel_info:
                    for entry in channel_info['entries'][:max_playlists]:
                        if entry and 'id' in entry:
                            playlists.append({
                                'id': entry['id'],
                                'title': entry.get('title', 'Unknown'),
                                'url': f"https://www.youtube.com/playlist?list={entry['id']}",
                                'video_count': entry.get('playlist_count', 0)
                            })
                
                print(f"✅ Found {len(playlists)} playlists")
                return playlists
                
        except Exception as e:
            print(f"❌ Error fetching playlists: {e}")
            return []
    
    def get_playlist_videos(self, playlist_url: str) -> List[Dict]:
        """Get video information from a playlist."""
        ydl_opts = {
            'quiet': True,
            'extract_flat': True
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                playlist_info = ydl.extract_info(playlist_url, download=False)
                
                videos = []
                if 'entries' in playlist_info:
                    for entry in playlist_info['entries']:
                        if entry and 'id' in entry:
                            videos.append({
                                'id': entry['id'],
                                'title': entry.get('title', 'Unknown'),
                                'url': f"https://www.youtube.com/watch?v={entry['id']}",
                                'duration': entry.get('duration', 0),
                                'playlist_title': playlist_info.get('title', 'Unknown')
                            })
                
                return videos
                
        except Exception as e:
            print(f"❌ Error fetching playlist videos: {e}")
            return []
    
    def download_video(self, video_info: Dict) -> VideoInfo:
        """Download a single video with progress tracking."""
        video_obj = VideoInfo(
            id=video_info['id'],
            title=video_info['title'],
            playlist_title=video_info['playlist_title'],
            filepath=""
        )
        
        video_obj.status = "downloading"
        
        # Create playlist subfolder
        playlist_folder = os.path.join(
            self.folders['downloads'], 
            self._sanitize_filename(video_info['playlist_title'])
        )
        os.makedirs(playlist_folder, exist_ok=True)
        
        # Set up yt-dlp options - single format to avoid ffmpeg dependency
        ydl_opts = {
            'outtmpl': os.path.join(playlist_folder, '%(title)s.%(ext)s'),
            'format': 'best[ext=mp4]/best',  # Prefer mp4 but don't require merging
            'quiet': True,
            'no_warnings': True,
            'ignoreerrors': False,
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # Extract info to get final filename
                info = ydl.extract_info(video_info['url'], download=False)
                if info:
                    filename = ydl.prepare_filename(info)
                    
                    # Download the video
                    ydl.download([video_info['url']])
                    
                    # Check if file exists and get its path
                    if os.path.exists(filename):
                        video_obj.filepath = filename
                        video_obj.file_size = os.path.getsize(filename)
                        video_obj.status = "completed"
                        
                        # Call completion callback for pipeline integration
                        if self.video_completed_callback:
                            self.video_completed_callback(video_obj)
                    else:
                        video_obj.status = "failed"
                else:
                    video_obj.status = "failed"
                    
        except Exception as e:
            print(f"❌ Error downloading {video_info['title']}: {e}")
            video_obj.status = "failed"
        
        return video_obj
    
    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for filesystem."""
        import re
        # Remove invalid characters
        filename = re.sub(r'[<>:"/\\\\|?*]', '', filename)
        # Replace multiple spaces with single space
        filename = re.sub(r'\\s+', ' ', filename)
        # Trim and limit length
        return filename.strip()[:100]
    
    def download_playlists_parallel(self, playlists: List[Dict]) -> List[VideoInfo]:
        """Download videos from playlists with parallel processing."""
        if not playlists:
            print("❌ No playlists to download")
            return []
        
        # Collect all videos from all playlists
        all_videos = []
        print("📋 Collecting videos from playlists...")
        
        with tqdm(playlists, desc="Analyzing playlists") as pbar:
            for playlist in pbar:
                pbar.set_postfix_str(f"Processing {playlist['title'][:30]}")
                videos = self.get_playlist_videos(playlist['url'])
                all_videos.extend(videos)
        
        if not all_videos:
            print("❌ No videos found in playlists")
            return []
        
        print(f"📥 Starting parallel download of {len(all_videos)} videos...")
        
        # Download videos in parallel
        completed_videos = []
        failed_count = 0
        
        with tqdm(total=len(all_videos), desc="Downloading videos") as pbar:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all download tasks
                future_to_video = {
                    executor.submit(self.download_video, video): video 
                    for video in all_videos
                }
                
                # Process completed downloads
                for future in as_completed(future_to_video):
                    video_result = future.result()
                    
                    if video_result.status == "completed":
                        completed_videos.append(video_result)
                        pbar.set_postfix_str(f"✅ {video_result.title[:30]}")
                    else:
                        failed_count += 1
                        pbar.set_postfix_str(f"❌ {video_result.title[:30]}")
                    
                    pbar.update(1)
        
        # Save download summary
        self._save_download_summary(completed_videos, failed_count)
        
        print(f"\\n{'='*60}")
        print(f"📊 DOWNLOAD SUMMARY")
        print(f"{'='*60}")
        print(f"✅ Successfully downloaded: {len(completed_videos)}")
        print(f"❌ Failed downloads: {failed_count}")
        print(f"📁 Downloads saved to: {self.folders['downloads']}")
        
        return completed_videos
    
    def _save_download_summary(self, completed_videos: List[VideoInfo], failed_count: int):
        """Save download summary to log file."""
        summary = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_downloaded': len(completed_videos),
            'total_failed': failed_count,
            'videos': [
                {
                    'title': video.title,
                    'playlist': video.playlist_title,
                    'filepath': video.filepath,
                    'file_size': video.file_size
                }
                for video in completed_videos
            ]
        }
        
        log_file = os.path.join(self.folders['logs'], 'download_summary.json')
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"📄 Download summary saved to: {log_file}")


def test_downloader():
    """Test the enhanced downloader."""
    # Example usage
    channel_url = "https://www.youtube.com/@PSAWorldTour"  # Example channel
    
    downloader = EnhancedDownloader()
    
    # Get recent playlists
    playlists = downloader.get_channel_playlists(channel_url, max_playlists=3)  # Small test
    
    if playlists:
        print(f"\\n📋 Found playlists:")
        for i, playlist in enumerate(playlists, 1):
            print(f"   {i}. {playlist['title']} ({playlist['video_count']} videos)")
        
        # Download playlists
        completed_videos = downloader.download_playlists_parallel(playlists)
        print(f"\\n🎉 Downloaded {len(completed_videos)} videos!")
    else:
        print("❌ No playlists found")


if __name__ == "__main__":
    test_downloader()