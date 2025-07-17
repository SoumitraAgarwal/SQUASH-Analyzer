import yt_dlp

url = 'https://www.youtube.com/watch?v=yMKDa4aHMsk'
with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
    info = ydl.extract_info(url, download=False)
    
    formats = info.get('formats', [])
    
    print('Available formats with audio:')
    audio_formats = []
    for fmt in formats:
        acodec = fmt.get('acodec', 'unknown')
        if acodec != 'none':
            format_id = fmt.get('format_id', 'unknown')
            ext = fmt.get('ext', 'unknown')
            vcodec = fmt.get('vcodec', 'unknown')
            resolution = fmt.get('resolution', 'unknown')
            height = fmt.get('height', 'unknown')
            fps = fmt.get('fps', 'unknown')
            
            audio_formats.append(fmt)
            print(f'Format {format_id}: {ext} - {height}p@{fps}fps - Video: {vcodec}, Audio: {acodec}')
    
    print(f'\nFound {len(audio_formats)} formats with audio')
    
    # Show some video-only formats too
    print('\nSome video-only formats:')
    video_only = []
    for fmt in formats:
        acodec = fmt.get('acodec', 'unknown')
        vcodec = fmt.get('vcodec', 'unknown')
        if acodec == 'none' and vcodec != 'none':
            format_id = fmt.get('format_id', 'unknown')
            ext = fmt.get('ext', 'unknown')
            height = fmt.get('height', 'unknown')
            fps = fmt.get('fps', 'unknown')
            
            video_only.append(fmt)
            print(f'Format {format_id}: {ext} - {height}p@{fps}fps - Video: {vcodec}, Audio: none')
            
            if len(video_only) >= 5:  # Show first 5
                break