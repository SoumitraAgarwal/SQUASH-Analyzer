import yt_dlp

url = 'https://www.youtube.com/watch?v=yMKDa4aHMsk'
with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
    info = ydl.extract_info(url, download=False)
    
    # Check video metadata
    print('Video Info:')
    print(f"Title: {info.get('title')}")
    print(f"Duration: {info.get('duration')} seconds ({info.get('duration', 0)/60:.1f} minutes)")
    print(f"View count: {info.get('view_count')}")
    print(f"Upload date: {info.get('upload_date')}")
    print(f"Is live: {info.get('is_live')}")
    print(f"Was live: {info.get('was_live')}")
    
    # Check if it's a livestream or has chapters
    chapters = info.get('chapters', [])
    if chapters:
        print(f"\nChapters ({len(chapters)}):")
        for i, chapter in enumerate(chapters):
            start_time = chapter.get('start_time', 0)
            end_time = chapter.get('end_time', 0)
            title = chapter.get('title', 'Unknown')
            print(f"  {i+1}. {title}: {start_time}s - {end_time}s ({end_time-start_time}s)")
    
    formats = info.get('formats', [])
    
    print(f'\nAvailable formats ({len(formats)} total):')
    for fmt in formats:
        if fmt.get('vcodec') != 'none':  # Only video formats
            height = fmt.get('height', '?')
            fps = fmt.get('fps', '?')
            ext = fmt.get('ext', '?')
            filesize = fmt.get('filesize', 0)
            filesize_mb = filesize / (1024*1024) if filesize else 0
            format_id = fmt.get('format_id', '?')
            print(f'{format_id}: {height}p @ {fps}fps ({ext}) - {filesize_mb:.1f}MB')