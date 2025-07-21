import os
import librosa
import numpy as np
from utils import extract_audio_from_video
import cv2
import ffmpeg
import glob
from tqdm import tqdm

def has_audio_stream(video_path):
    """Return True if the video file has an audio stream."""
    import subprocess
    cmd = [
        'ffprobe', '-v', 'error', '-select_streams', 'a', '-show_entries', 'stream=index', '-of', 'csv=p=0', video_path
    ]
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return bool(result.stdout.strip())
    except Exception:
        return False

def detect_hits_and_walls(video_path, sr=44100, hop_length=512, min_separation=0.05):
    """
    Given a video file, extract audio and detect hit and wall events.
    Returns a list of (timestamp, event_type) where event_type is 'hit' or 'wall'.
    """
    # 1. Check if video has audio
    use_path = video_path
    if not has_audio_stream(video_path):
        # Try to find a matching audio file in the downloads folder
        base = os.path.splitext(os.path.basename(video_path))[0]
        # Remove .f399 suffix if present (from camera processing)
        if base.endswith('.f399'):
            base = base[:-5]
        # Remove enhanced_ prefix if present
        if base.startswith('enhanced_'):
            base = base[9:]
        # Remove shell escaping from filename
        base = base.replace('\\!', '!')
        
        # Try both OLD Outputs and Pipeline_Output paths
        # Go up from 03_Player_Annotations to Pipeline_Output
        base_output_dir = os.path.dirname(os.path.dirname(os.path.dirname(video_path)))
        possible_download_dirs = [
            os.path.join(base_output_dir, 'OLD Outputs', '01_Downloads'),
            os.path.join(base_output_dir, '01_Downloads')
        ]
        
        matches = []
        for downloads_dir in possible_download_dirs:
            if os.path.exists(downloads_dir):
                # Look for various audio formats
                for ext in ['mp4a', 'm4a', 'aac', 'wav', 'mp3']:
                    pattern = os.path.join(downloads_dir, '**', f'{base}.{ext}')
                    matches.extend(glob.glob(pattern, recursive=True))
                    # Also try with _audio suffix
                    pattern = os.path.join(downloads_dir, '**', f'{base}_audio.{ext}')
                    matches.extend(glob.glob(pattern, recursive=True))
        if matches:
            use_path = matches[0]
            print(f"   🔊 No audio in video, using audio file: {use_path}")
        else:
            print(f"   ❌ No audio found in video or matching .mp4a file.")
            return []
    else:
        print(f"   🔊 Using audio from: {use_path}")
    # 2. Extract audio
    audio_path = extract_audio_from_video(use_path)
    if not audio_path or not os.path.exists(audio_path):
        print(f"❌ Could not extract audio from {use_path}")
        return []
    # 2. Load audio
    print(f"   📊 Loading audio file...")
    y, sr = librosa.load(audio_path, sr=sr, mono=True)
    # 3. Onset detection (find percussive events)
    print(f"   🔍 Detecting audio events...")
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop_length, backtrack=True, units='frames')
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)
    # 4. Heuristic: hits and walls are two close onsets (within 0.2s)
    print(f"   🎯 Processing {len(onset_times)} audio events...")
    events = []
    i = 0
    
    with tqdm(total=len(onset_times), desc="   Processing audio events", unit="event") as pbar:
        while i < len(onset_times) - 1:
            t1 = onset_times[i]
            t2 = onset_times[i+1]
            if 0.03 < (t2 - t1) < 0.25:  # likely a hit followed by wall
                events.append((t1, 'hit'))
                events.append((t2, 'wall'))
                i += 2
                pbar.update(2)
            else:
                # Could be a single event (ignore or mark as hit only)
                events.append((t1, 'hit'))
                i += 1
                pbar.update(1)
        # Handle last event if not paired
        if i == len(onset_times) - 1:
            events.append((onset_times[-1], 'hit'))
            pbar.update(1)
    
    return events

def annotate_video_with_events(video_path, events, output_path=None, text_pos=(30, 60)):
    """Overlay 'hit' and 'wall' text on video frames at detected event times."""
    if output_path is None:
        base, ext = os.path.splitext(video_path)
        output_path = base + '_with_audio_events' + ext
    # Open video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Build a frame-to-event mapping
    event_frames = {}
    for t, label in events:
        frame_idx = int(round(t * fps))
        event_frames[frame_idx] = label
    
    frame_idx = 0
    with tqdm(total=total_frames, desc="   Annotating video with audio events", unit="frame") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # If this frame has an event, overlay text
            if frame_idx in event_frames:
                label = event_frames[frame_idx]
                color = (0, 255, 255) if label == 'hit' else (255, 0, 255)
                cv2.putText(frame, label.upper(), text_pos, cv2.FONT_HERSHEY_SIMPLEX, 2.0, color, 4, cv2.LINE_AA)
            out.write(frame)
            frame_idx += 1
            pbar.update(1)
    
    cap.release()
    out.release()
    print(f"✅ Annotated video saved to {output_path}")
    return output_path 