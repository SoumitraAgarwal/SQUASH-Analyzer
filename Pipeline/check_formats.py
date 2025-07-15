import yt_dlp

url = 'https://www.youtube.com/watch?v=yMKDa4aHMsk'
with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
    info = ydl.extract_info(url, download=False)
    formats = info.get('formats', [])
    
    print('Available formats:')
    for fmt in formats:
        if fmt.get('vcodec') != 'none':  # Only video formats
            height = fmt.get('height', '?')
            fps = fmt.get('fps', '?')
            ext = fmt.get('ext', '?')
            filesize = fmt.get('filesize', 0)
            filesize_mb = filesize / (1024*1024) if filesize else 0
            format_id = fmt.get('format_id', '?')
            print(f'{format_id}: {height}p @ {fps}fps ({ext}) - {filesize_mb:.1f}MB')