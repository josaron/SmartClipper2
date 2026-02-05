"""
Video cropping with MediaPipe face/pose detection for intelligent subject centering.
"""

import subprocess
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

import mediapipe as mp

from src.clip_extractor import get_ffmpeg_path, get_video_duration


# Target output dimensions (9:16 vertical)
TARGET_WIDTH = 720
TARGET_HEIGHT = 1280
TARGET_ASPECT = TARGET_WIDTH / TARGET_HEIGHT  # 0.5625

# Cached MediaPipe detectors (Fix 4: avoid recreating models for each clip)
_face_detector = None
_pose_detector = None


def _get_face_detector():
    """Get or create cached face detector."""
    global _face_detector
    if _face_detector is None:
        _face_detector = mp.solutions.face_detection.FaceDetection(
            model_selection=1,  # Full range model
            min_detection_confidence=0.5
        )
    return _face_detector


def _get_pose_detector():
    """Get or create cached pose detector."""
    global _pose_detector
    if _pose_detector is None:
        _pose_detector = mp.solutions.pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            min_detection_confidence=0.5
        )
    return _pose_detector


def detect_subject_center(frame: np.ndarray) -> Optional[Tuple[int, int]]:
    """
    Detect the center of the main subject (face or pose) in a frame.
    
    Uses MediaPipe Face Detection first, falls back to Pose Detection.
    Uses cached detectors to avoid model reloading overhead.
    
    Args:
        frame: BGR image (numpy array from OpenCV)
        
    Returns:
        (x, y) center coordinates, or None if no subject detected
    """
    # Convert BGR to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = frame.shape[:2]
    
    # Try face detection first (using cached detector)
    face_detection = _get_face_detector()
    results = face_detection.process(rgb_frame)
    
    if results.detections:
        # Use the first (most confident) detection
        detection = results.detections[0]
        bbox = detection.location_data.relative_bounding_box
        
        # Calculate center of face bounding box
        center_x = int((bbox.xmin + bbox.width / 2) * w)
        center_y = int((bbox.ymin + bbox.height / 2) * h)
        
        return (center_x, center_y)
    
    # Fall back to pose detection (using cached detector)
    pose = _get_pose_detector()
    results = pose.process(rgb_frame)
    
    if results.pose_landmarks:
        # Use nose or center of shoulders as reference
        landmarks = results.pose_landmarks.landmark
        
        # Try nose first (landmark 0)
        nose = landmarks[0]
        if nose.visibility > 0.5:
            return (int(nose.x * w), int(nose.y * h))
        
        # Fall back to midpoint between shoulders
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        
        if left_shoulder.visibility > 0.3 and right_shoulder.visibility > 0.3:
            center_x = int((left_shoulder.x + right_shoulder.x) / 2 * w)
            center_y = int((left_shoulder.y + right_shoulder.y) / 2 * h)
            return (center_x, center_y)
    
    # No subject detected
    return None


def calculate_crop_region(
    frame_width: int,
    frame_height: int,
    subject_center: Optional[Tuple[int, int]] = None,
    target_width: int = TARGET_WIDTH,
    target_height: int = TARGET_HEIGHT,
) -> Tuple[int, int, int, int]:
    """
    Calculate the crop region for 9:16 output centered on subject.
    
    Args:
        frame_width: Original frame width
        frame_height: Original frame height
        subject_center: (x, y) center of detected subject, or None for center crop
        target_width: Output width
        target_height: Output height
        
    Returns:
        (x, y, width, height) crop region
    """
    target_aspect = target_width / target_height
    frame_aspect = frame_width / frame_height
    
    if frame_aspect > target_aspect:
        # Frame is wider than target - crop width
        crop_height = frame_height
        crop_width = int(frame_height * target_aspect)
    else:
        # Frame is taller than target - crop height
        crop_width = frame_width
        crop_height = int(frame_width / target_aspect)
    
    # Default to center if no subject detected
    if subject_center is None:
        center_x = frame_width // 2
        center_y = frame_height // 2
    else:
        center_x, center_y = subject_center
    
    # Calculate crop position centered on subject
    x = center_x - crop_width // 2
    y = center_y - crop_height // 2
    
    # Clamp to frame boundaries
    x = max(0, min(x, frame_width - crop_width))
    y = max(0, min(y, frame_height - crop_height))
    
    return (x, y, crop_width, crop_height)


def crop_to_vertical(
    input_path: str,
    output_path: str,
    target_width: int = TARGET_WIDTH,
    target_height: int = TARGET_HEIGHT,
    subject_center: Optional[Tuple[int, int]] = None,
) -> str:
    """
    Crop a video to 9:16 vertical format, centered on detected subject.
    
    If subject_center is not provided, detects subject from first frame.
    
    Args:
        input_path: Path to input video
        output_path: Path for output video
        target_width: Output width (default 720)
        target_height: Output height (default 1280)
        subject_center: Optional pre-computed subject center
        
    Returns:
        Path to cropped video
    """
    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Open video to get dimensions and first frame
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Detect subject if not provided
    if subject_center is None:
        ret, first_frame = cap.read()
        if ret:
            subject_center = detect_subject_center(first_frame)
            if subject_center:
                print(f"Detected subject at: {subject_center}")
            else:
                print("No subject detected, using center crop")
    
    cap.release()
    
    # Calculate crop region
    x, y, crop_w, crop_h = calculate_crop_region(
        frame_width, frame_height, subject_center, target_width, target_height
    )
    
    # Build FFmpeg command for cropping and scaling
    crop_filter = f"crop={crop_w}:{crop_h}:{x}:{y},scale={target_width}:{target_height}"
    ffmpeg = get_ffmpeg_path()
    
    cmd = [
        ffmpeg,
        '-y',
        '-i', input_path,
        '-vf', crop_filter,
        '-c:v', 'libx264',
        '-preset', 'ultrafast',  # Fast encoding for intermediate files
        '-crf', '23',
        '-an',  # No audio (we'll add TTS later)
        output_path,
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg crop failed: {result.stderr}")
    
    return output_path


def extract_crop_and_speed(
    source_path: str,
    start_time: float,
    extract_duration: float,
    target_duration: float,
    output_path: str,
    target_width: int = TARGET_WIDTH,
    target_height: int = TARGET_HEIGHT,
) -> str:
    """
    Extract, crop, and speed-adjust a clip in a single FFmpeg pass.
    
    Combines what was previously 3 separate encoding steps into 1,
    significantly reducing processing time.
    
    Args:
        source_path: Path to source video
        start_time: Start time in seconds
        extract_duration: Duration to extract from source
        target_duration: Desired output duration (for speed adjustment)
        output_path: Path for output video
        target_width: Output width (default 720)
        target_height: Output height (default 1280)
        
    Returns:
        Path to processed video
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # First, extract a frame to detect subject center
    # Use OpenCV to seek to the start position and grab a frame
    cap = cv2.VideoCapture(source_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {source_path}")
    
    # Seek to start time (in milliseconds)
    cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)
    actual_pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Read frame and detect subject
    ret, frame = cap.read()
    subject_center = None
    if ret:
        subject_center = detect_subject_center(frame)
        if subject_center:
            print(f"Detected subject at: {subject_center}")
        else:
            print("No subject detected, using center crop")
    
    cap.release()
    
    # Calculate crop region
    x, y, crop_w, crop_h = calculate_crop_region(
        frame_width, frame_height, subject_center, target_width, target_height
    )
    
    # Calculate speed adjustment factor
    # setpts: PTS * factor, where factor < 1 speeds up, > 1 slows down
    pts_multiplier = target_duration / extract_duration
    
    # Build combined filter chain: crop -> scale -> speed adjust -> trim to target duration
    # The trim filter is essential: setpts changes timestamps but doesn't drop frames,
    # so we must explicitly trim to the target duration after adjusting speed
    filter_str = f"crop={crop_w}:{crop_h}:{x}:{y},scale={target_width}:{target_height},setpts={pts_multiplier}*PTS,trim=duration={target_duration}"
    
    ffmpeg = get_ffmpeg_path()
    
    cmd = [
        ffmpeg,
        '-y',
        '-ss', str(start_time),  # Input seeking (fast)
        '-i', source_path,
        '-t', str(extract_duration),  # Duration to extract
        '-vf', filter_str,
        '-c:v', 'libx264',
        '-preset', 'ultrafast',  # Fast encoding for intermediate files
        '-crf', '23',
        '-an',  # No audio
        output_path,
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg extract+crop+speed failed: {result.stderr}")
    
    return output_path


def extract_still_with_ken_burns(
    source_path: str,
    timestamp: float,
    target_duration: float,
    output_path: str,
    target_width: int = TARGET_WIDTH,
    target_height: int = TARGET_HEIGHT,
    zoom_factor: float = 1.08,
) -> str:
    """
    Extract a still frame and apply Ken Burns effect (subtle zoom/pan).
    
    Creates a video from a single frame with smooth zoom animation
    to add visual interest to still images.
    
    Args:
        source_path: Path to source video
        timestamp: Time in seconds to extract frame from
        target_duration: Duration of output video
        output_path: Path for output video
        target_width: Output width (default 720)
        target_height: Output height (default 1280)
        zoom_factor: How much to zoom in (1.08 = 8% zoom over duration)
        
    Returns:
        Path to processed video
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Extract frame and detect subject center
    cap = cv2.VideoCapture(source_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {source_path}")
    
    # Seek to timestamp
    cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    
    # Read frame and detect subject
    ret, frame = cap.read()
    if not ret:
        # Try reading from the beginning if seek failed
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
        if not ret:
            cap.release()
            raise RuntimeError(f"Could not read frame from: {source_path}")
    
    subject_center = detect_subject_center(frame)
    if subject_center:
        print(f"Ken Burns: Detected subject at {subject_center}")
    else:
        print("Ken Burns: No subject detected, using center")
    
    cap.release()
    
    # Calculate crop region centered on subject
    x, y, crop_w, crop_h = calculate_crop_region(
        frame_width, frame_height, subject_center, target_width, target_height
    )
    
    # Extract the cropped region from the frame
    cropped_frame = frame[y:y+crop_h, x:x+crop_w]
    resized_frame = cv2.resize(cropped_frame, (target_width, target_height))
    
    # Save the still frame temporarily
    temp_frame_path = output_path + ".frame.png"
    cv2.imwrite(temp_frame_path, resized_frame)
    
    # Use FFmpeg to create Ken Burns effect
    # The zoompan filter creates smooth zoom animation from a still image
    # zoom: starts at 1, ends at zoom_factor
    # d: duration in frames
    # fps: output framerate
    # s: output size
    
    total_frames = int(target_duration * fps)
    
    # zoompan parameters:
    # z: zoom level expression (linear interpolation from 1 to zoom_factor)
    # d: total duration in frames
    # s: output size
    # x,y: pan position (center the crop, accounting for zoom)
    # The 'on' variable gives current frame number, 'zoom' gives current zoom
    
    # Center-focused zoom: zoom in toward center of frame
    zoom_expr = f"min(zoom+{(zoom_factor-1)/total_frames},{{zoom_factor}})"
    zoom_expr = f"1+({zoom_factor}-1)*on/{total_frames}"
    
    # Calculate center offset as zoom increases to keep subject centered
    # As we zoom in, we need to offset x,y to keep the center stable
    pan_x = f"(iw-iw/zoom)/2"
    pan_y = f"(ih-ih/zoom)/2"
    
    ffmpeg = get_ffmpeg_path()
    
    # Build the zoompan filter
    zoompan_filter = f"zoompan=z='{zoom_expr}':x='{pan_x}':y='{pan_y}':d={total_frames}:s={target_width}x{target_height}:fps={fps}"
    
    cmd = [
        ffmpeg,
        '-y',
        '-loop', '1',
        '-i', temp_frame_path,
        '-vf', zoompan_filter,
        '-t', str(target_duration),
        '-c:v', 'libx264',
        '-preset', 'ultrafast',
        '-crf', '23',
        '-pix_fmt', 'yuv420p',
        output_path,
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Clean up temp frame
    try:
        Path(temp_frame_path).unlink()
    except Exception:
        pass
    
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg Ken Burns failed: {result.stderr}")
    
    return output_path


def crop_video_with_tracking(
    input_path: str,
    output_path: str,
    target_width: int = TARGET_WIDTH,
    target_height: int = TARGET_HEIGHT,
    sample_interval: int = 30,  # Detect subject every N frames
) -> str:
    """
    Crop video with subject tracking (detects subject periodically).
    
    More expensive but better for videos where subject moves significantly.
    
    Args:
        input_path: Path to input video
        output_path: Path for output video
        target_width: Output width
        target_height: Output height
        sample_interval: Frames between subject detection
        
    Returns:
        Path to cropped video
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate target dimensions maintaining aspect
    target_aspect = target_width / target_height
    frame_aspect = frame_width / frame_height
    
    if frame_aspect > target_aspect:
        crop_height = frame_height
        crop_width = int(frame_height * target_aspect)
    else:
        crop_width = frame_width
        crop_height = int(frame_width / target_aspect)
    
    # Create video writer
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (target_width, target_height))
    
    current_center = (frame_width // 2, frame_height // 2)
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Update subject detection periodically
        if frame_idx % sample_interval == 0:
            detected = detect_subject_center(frame)
            if detected:
                # Smooth transition to new center
                current_center = (
                    int(0.7 * current_center[0] + 0.3 * detected[0]),
                    int(0.7 * current_center[1] + 0.3 * detected[1]),
                )
        
        # Calculate crop region
        x, y, _, _ = calculate_crop_region(
            frame_width, frame_height, current_center, target_width, target_height
        )
        
        # Crop and resize
        cropped = frame[y:y+crop_height, x:x+crop_width]
        resized = cv2.resize(cropped, (target_width, target_height))
        
        out.write(resized)
        frame_idx += 1
    
    cap.release()
    out.release()
    
    # Re-encode with proper codec for compatibility
    temp_path = output_path + ".temp.mp4"
    Path(output_path).rename(temp_path)
    ffmpeg = get_ffmpeg_path()
    
    cmd = [
        ffmpeg, '-y',
        '-i', temp_path,
        '-c:v', 'libx264',
        '-preset', 'ultrafast',  # Fast encoding for intermediate files
        '-crf', '23',
        '-an',
        output_path,
    ]
    subprocess.run(cmd, capture_output=True)
    Path(temp_path).unlink()
    
    return output_path
