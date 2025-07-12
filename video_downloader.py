import os
import yt_dlp

# Paste your playlist URL here
PLAYLIST_URL = "https://www.youtube.com/watch?v=5X32HYQ5zBc&list=PLxmhcE3iz1lAgkaW9ijsMJDBBZVoObQk3"

def main():
    # Set up yt-dlp options
    ydl_opts = {
        'outtmpl': os.path.join('%(playlist_title)s', '%(title)s.%(ext)s'),
        'format': 'bestvideo+bestaudio/best',
        'merge_output_format': 'mp4',
        'quiet': False,
        'ignoreerrors': True,  # Skip videos with errors
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([PLAYLIST_URL])

if __name__ == "__main__":
    main()