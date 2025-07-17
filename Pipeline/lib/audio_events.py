import os
import librosa
import numpy as np
from utils import extract_audio_from_video
import cv2

def detect_hits_and_walls(video_path, sr=44100, hop_length=512, min_separation=0.05):
    """
    Given a video file, extract audio and detect hit and wall events.
    Returns a list of (timestamp, event_type) where event_type is 'hit' or 'wall'.
    """
    # 1. Extract audio
    audio_path = extract_audio_from_video(video_path)
    if not audio_path or not os.path.exists(audio_path):
        print(f"❌ Could not extract audio from {video_path}")
        return []
    # 2. Load audio
    y, sr = librosa.load(audio_path, sr=sr, mono=True)
    # 3. Onset detection (find percussive events)
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop_length, backtrack=True, units='frames')
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)
    # 4. Heuristic: hits and walls are two close onsets (within 0.2s)
    events = []
    i = 0
    while i < len(onset_times) - 1:
        t1 = onset_times[i]
        t2 = onset_times[i+1]
        if 0.03 < (t2 - t1) < 0.25:  # likely a hit followed by wall
            events.append((t1, 'hit'))
            events.append((t2, 'wall'))
            i += 2
        else:
            # Could be a single event (ignore or mark as hit only)
            events.append((t1, 'hit'))
            i += 1
    # Handle last event if not paired
    if i == len(onset_times) - 1:
        events.append((onset_times[-1], 'hit'))
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
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    # Build a frame-to-event mapping
    event_frames = {}
    for t, label in events:
        frame_idx = int(round(t * fps))
        event_frames[frame_idx] = label
    frame_idx = 0
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
    cap.release()
    out.release()
    print(f"✅ Annotated video saved to {output_path}")
    return output_path 