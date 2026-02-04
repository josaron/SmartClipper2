"""
Video clip extraction using FFmpeg with fast seeking.
"""

import subprocess
import shutil
from pathlib import Path
from typing import Optional, Tuple

# Cache for FFmpeg paths
_ffmpeg_path: Optional[str] = None
_ffprobe_path: Optional[str] = None


def get_ffmpeg_path() -> str:
    """Get the path to FFmpeg, preferring system install, falling back to imageio-ffmpeg."""
    global _ffmpeg_path
    
    if _ffmpeg_path is not None:
        return _ffmpeg_path
    
    # Try system FFmpeg first
    system_ffmpeg = shutil.which('ffmpeg')
    if system_ffmpeg:
        _ffmpeg_path = system_ffmpeg
        return _ffmpeg_path
    
    # Fall back to imageio-ffmpeg bundled binary
    try:
        import imageio_ffmpeg
        _ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
        return _ffmpeg_path
    except ImportError:
        pass
    
    raise RuntimeError(
        "FFmpeg is not available. Install it via your system package manager "
        "or install imageio-ffmpeg: pip install imageio-ffmpeg"
    )


def get_ffprobe_path() -> str:
    """Get the path to FFprobe."""
    global _ffprobe_path
    
    if _ffprobe_path is not None:
        return _ffprobe_path
    
    # Try system FFprobe first
    system_ffprobe = shutil.which('ffprobe')
    if system_ffprobe:
        _ffprobe_path = system_ffprobe
        return _ffprobe_path
    
    # imageio-ffmpeg doesn't include ffprobe, so we need to find it
    # Try common locations relative to ffmpeg
    ffmpeg_path = get_ffmpeg_path()
    ffmpeg_dir = Path(ffmpeg_path).parent
    
    # Check for ffprobe in same directory
    possible_ffprobe = ffmpeg_dir / "ffprobe"
    if possible_ffprobe.exists():
        _ffprobe_path = str(possible_ffprobe)
        return _ffprobe_path
    
    # Use ffmpeg with -i flag as fallback for duration detection
    _ffprobe_path = ffmpeg_path  # Will use ffmpeg -i as fallback
    return _ffprobe_path


def check_ffmpeg() -> bool:
    """Check if FFmpeg is installed and accessible."""
    try:
        get_ffmpeg_path()
        return True
    except RuntimeError:
        return False


def extract_clip(
    source_path: str,
    start_time: float,
    duration: float,
    output_path: str,
    strip_audio: bool = True,
) -> str:
    """
    Extract a clip from a video using FFmpeg fast seeking.
    
    Uses input seeking (-ss before -i) for fast seeking without decoding
    the entire video up to the seek point.
    
    Args:
        source_path: Path to the source video
        start_time: Start time in seconds
        duration: Duration to extract in seconds (max 15s recommended)
        output_path: Path for the output clip
        strip_audio: If True, remove audio track from output
        
    Returns:
        Path to the extracted clip
    """
    ffmpeg = get_ffmpeg_path()
    
    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Build FFmpeg command with fast seeking
    # -ss before -i enables input seeking (fast)
    cmd = [
        ffmpeg,
        '-y',  # Overwrite output
        '-ss', str(start_time),  # Seek to start time (input seeking = fast)
        '-i', source_path,
        '-t', str(duration),  # Duration to extract
        '-c:v', 'libx264',  # Re-encode video for clean cuts
        '-preset', 'ultrafast',  # Fast encoding for intermediate files
        '-crf', '23',  # Good quality
    ]
    
    if strip_audio:
        cmd.extend(['-an'])  # No audio
    else:
        cmd.extend(['-c:a', 'aac'])
    
    # Add output path
    cmd.append(output_path)
    
    # Run FFmpeg
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )
    
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg failed: {result.stderr}")
    
    if not Path(output_path).exists():
        raise FileNotFoundError(f"FFmpeg completed but output not found: {output_path}")
    
    return output_path


def get_video_duration(video_path: str) -> float:
    """
    Get the duration of a video file.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        Duration in seconds
    """
    # Try using ffprobe if available
    ffprobe = shutil.which('ffprobe')
    if ffprobe:
        cmd = [
            ffprobe,
            '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            video_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            return float(result.stdout.strip())
    
    # Fallback: use ffmpeg -i to get duration from stderr
    ffmpeg = get_ffmpeg_path()
    cmd = [ffmpeg, '-i', video_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Parse duration from stderr (format: Duration: HH:MM:SS.ms)
    import re
    match = re.search(r'Duration: (\d+):(\d+):(\d+)\.(\d+)', result.stderr)
    if match:
        hours, minutes, seconds, ms = match.groups()
        return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(ms) / 100
    
    raise RuntimeError(f"Could not determine video duration: {video_path}")


def get_video_info(video_path: str) -> dict:
    """
    Get video information (resolution, fps, duration).
    
    Args:
        video_path: Path to the video file
        
    Returns:
        Dictionary with video info
    """
    import cv2
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        
        return {
            'width': width,
            'height': height,
            'fps': fps,
            'duration': duration,
        }
    finally:
        cap.release()


def extract_clips_batch(
    source_path: str,
    segments: list[tuple[float, float]],  # List of (start_time, duration)
    output_dir: str,
    prefix: str = "clip",
) -> list[str]:
    """
    Extract multiple clips from a video.
    
    Args:
        source_path: Path to the source video
        segments: List of (start_time, duration) tuples
        output_dir: Directory for output clips
        prefix: Filename prefix for clips
        
    Returns:
        List of paths to extracted clips
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    output_paths = []
    for i, (start_time, duration) in enumerate(segments):
        output_path = str(Path(output_dir) / f"{prefix}_{i}.mp4")
        extract_clip(source_path, start_time, duration, output_path)
        output_paths.append(output_path)
    
    return output_paths
