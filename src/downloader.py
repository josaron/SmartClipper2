"""
YouTube video downloader using yt-dlp.
"""

import yt_dlp
import subprocess
from pathlib import Path
import re

from src.clip_extractor import get_ffmpeg_path


def _fix_video_timestamps(input_path: str) -> str:
    """
    Re-encode a video to fix timestamp issues.
    
    Some YouTube downloads have broken presentation timestamps (PTS)
    that cause seeking to fail. Re-encoding with vsync creates proper timestamps.
    
    Args:
        input_path: Path to input video
        
    Returns:
        Path to fixed video (same as input, file is replaced)
    """
    ffmpeg = get_ffmpeg_path()
    temp_path = input_path + ".fixed.mp4"
    
    # Re-encode with constant frame rate and proper timestamps
    cmd = [
        ffmpeg,
        '-y',
        '-i', input_path,
        '-c:v', 'libx264',       # Re-encode video
        '-preset', 'fast',       # Balance speed/quality  
        '-crf', '18',            # High quality
        '-vsync', 'cfr',         # Constant frame rate
        '-r', '30',              # Force 30fps
        '-c:a', 'aac',           # Re-encode audio
        '-b:a', '192k',
        '-movflags', '+faststart',
        temp_path,
    ]
    
    print(f"Re-encoding video to fix timestamps (this may take a while)...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0 and Path(temp_path).exists():
        # Replace original with fixed version
        Path(input_path).unlink()
        Path(temp_path).rename(input_path)
        print(f"Fixed video timestamps: {input_path}")
    else:
        # Cleanup on failure
        if Path(temp_path).exists():
            Path(temp_path).unlink()
        print(f"Warning: Re-encode failed: {result.stderr[:200] if result.stderr else 'unknown error'}")
    
    return input_path


def extract_video_id(url: str) -> str:
    """Extract the video ID from a YouTube URL."""
    patterns = [
        r'(?:v=|/v/|youtu\.be/)([a-zA-Z0-9_-]{11})',
        r'(?:embed/)([a-zA-Z0-9_-]{11})',
        r'(?:shorts/)([a-zA-Z0-9_-]{11})',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    
    raise ValueError(f"Could not extract video ID from URL: {url}")


def download_video(url: str, output_dir: str = "temp") -> str:
    """
    Download a YouTube video using yt-dlp.
    
    Args:
        url: YouTube video URL
        output_dir: Directory to save the video
        
    Returns:
        Path to the downloaded video file
    """
    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Extract video ID for consistent naming
    video_id = extract_video_id(url)
    output_template = str(Path(output_dir) / f"{video_id}.%(ext)s")
    
    # Check if already downloaded (look for remuxed version first)
    remuxed_marker = Path(output_dir) / f"{video_id}.remuxed"
    existing_files = list(Path(output_dir).glob(f"{video_id}.*"))
    video_extensions = ['.mp4', '.webm', '.mkv', '.avi', '.mov']
    for f in existing_files:
        if f.suffix.lower() in video_extensions:
            # Check if this video has been remuxed already
            if remuxed_marker.exists():
                print(f"Video already downloaded and remuxed: {f}")
                return str(f)
            else:
                # Re-encode to fix potential timestamp issues
                print(f"Found cached video, re-encoding to fix timestamps: {f}")
                _fix_video_timestamps(str(f))
                remuxed_marker.touch()  # Mark as fixed
                return str(f)
    
    # yt-dlp options - configured to bypass YouTube SABR streaming restrictions
    ydl_opts = {
        'format': 'best[ext=mp4]/best',
        'outtmpl': output_template,
        'merge_output_format': 'mp4',
        'quiet': False,
        'no_warnings': False,
        'extract_flat': False,
        # Use Android client which is less likely to hit SABR restrictions
        'extractor_args': {
            'youtube': {
                'player_client': ['android', 'web'],
            }
        },
        # Add HTTP headers for better compatibility
        'http_headers': {
            'User-Agent': 'Mozilla/5.0 (Linux; Android 12; Pixel 6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36',
        },
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        
        # Get the actual filename
        if 'requested_downloads' in info and info['requested_downloads']:
            video_path = info['requested_downloads'][0]['filepath']
        else:
            # Fallback: look for the file
            video_path = str(Path(output_dir) / f"{video_id}.mp4")
    
    if not Path(video_path).exists():
        raise FileNotFoundError(f"Download completed but file not found: {video_path}")
    
    # Re-encode to fix potential timestamp issues
    _fix_video_timestamps(video_path)
    remuxed_marker.touch()  # Mark as fixed
    
    return video_path


def get_video_info(url: str) -> dict:
    """
    Get video information without downloading.
    
    Args:
        url: YouTube video URL
        
    Returns:
        Dictionary with video metadata (title, duration, etc.)
    """
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'extract_flat': False,
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        
    return {
        'id': info.get('id'),
        'title': info.get('title'),
        'duration': info.get('duration'),  # in seconds
        'uploader': info.get('uploader'),
        'thumbnail': info.get('thumbnail'),
    }
