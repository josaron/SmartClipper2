"""
Video compositor for assembling final video with audio.
"""

import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional

from src.clip_extractor import get_ffmpeg_path, get_video_duration
from src.captions import (
    WordTiming, 
    CaptionStyle, 
    render_caption_frame,
    render_caption_frame_fast,
    merge_word_timings,
    get_font,
    group_words_into_lines,
    generate_ass_subtitles,
)


def speed_adjust_clip(
    input_path: str,
    output_path: str,
    target_duration: float,
) -> str:
    """
    Adjust video speed to match target duration.
    
    Args:
        input_path: Path to input video
        output_path: Path for output video
        target_duration: Desired duration in seconds
        
    Returns:
        Path to speed-adjusted video
    """
    current_duration = get_video_duration(input_path)
    
    ffmpeg = get_ffmpeg_path()
    
    if abs(current_duration - target_duration) < 0.1:
        # Durations are close enough, just copy
        cmd = [ffmpeg, '-y', '-i', input_path, '-c', 'copy', output_path]
        subprocess.run(cmd, capture_output=True)
        return output_path
    
    # Calculate speed factor
    # setpts: lower = faster, higher = slower
    # speed = current / target
    # PTS multiplier = target / current (inverse of speed)
    pts_multiplier = target_duration / current_duration
    
    # FFmpeg setpts filter: PTS*factor
    # factor < 1 = speed up, factor > 1 = slow down
    filter_str = f"setpts={pts_multiplier}*PTS"
    
    cmd = [
        ffmpeg, '-y',
        '-i', input_path,
        '-filter:v', filter_str,
        '-c:v', 'libx264',
        '-preset', 'ultrafast',  # Fast encoding for intermediate files
        '-crf', '23',
        '-an',  # No audio
        output_path,
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Speed adjustment failed: {result.stderr}")
    
    return output_path


def concatenate_videos(
    video_paths: List[str],
    output_path: str,
) -> str:
    """
    Concatenate multiple video clips into one.
    
    Args:
        video_paths: List of video file paths
        output_path: Path for concatenated output
        
    Returns:
        Path to concatenated video
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Create concat file list with absolute paths to avoid path resolution issues
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        for path in video_paths:
            # Use absolute path to avoid working directory issues
            abs_path = str(Path(path).resolve())
            # Escape special characters in path
            escaped_path = abs_path.replace("'", "'\\''")
            f.write(f"file '{escaped_path}'\n")
        concat_file = f.name
    
    ffmpeg = get_ffmpeg_path()
    
    try:
        cmd = [
            ffmpeg, '-y',
            '-f', 'concat',
            '-safe', '0',
            '-i', concat_file,
            '-c:v', 'libx264',
            '-preset', 'ultrafast',  # Fast encoding for intermediate files
            '-crf', '23',
            '-an',
            output_path,
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Concatenation failed: {result.stderr}")
    finally:
        Path(concat_file).unlink()
    
    return output_path


def concatenate_audio(
    audio_paths: List[str],
    output_path: str,
) -> str:
    """
    Concatenate multiple audio files into one.
    
    Args:
        audio_paths: List of audio file paths
        output_path: Path for concatenated output
        
    Returns:
        Path to concatenated audio
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Create concat file list
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        for path in audio_paths:
            escaped_path = path.replace("'", "'\\''")
            f.write(f"file '{escaped_path}'\n")
        concat_file = f.name
    
    ffmpeg = get_ffmpeg_path()
    
    try:
        cmd = [
            ffmpeg, '-y',
            '-f', 'concat',
            '-safe', '0',
            '-i', concat_file,
            '-c:a', 'aac',
            '-b:a', '128k',
            output_path,
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Audio concatenation failed: {result.stderr}")
    finally:
        Path(concat_file).unlink()
    
    return output_path


def merge_video_audio(
    video_path: str,
    audio_path: str,
    output_path: str,
) -> str:
    """
    Merge video and audio tracks into final output.
    
    Args:
        video_path: Path to video file (no audio)
        audio_path: Path to audio file
        output_path: Path for final output
        
    Returns:
        Path to merged video
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = get_ffmpeg_path()
    
    cmd = [
        ffmpeg, '-y',
        '-i', video_path,
        '-i', audio_path,
        '-c:v', 'copy',  # Copy video stream
        '-c:a', 'aac',   # Encode audio as AAC
        '-b:a', '128k',
        '-shortest',     # Cut to shortest stream
        output_path,
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Merge failed: {result.stderr}")
    
    return output_path


def compose_final_video_fast(
    clips: List[str],
    audio_segments: List[str],
    output_path: str,
) -> str:
    """
    Compose the final video from pre-processed clips and audio.
    
    This is a faster version that assumes clips are already speed-adjusted.
    Steps:
    1. Concatenate all video clips
    2. Concatenate all audio segments
    3. Merge video and audio
    
    Args:
        clips: List of pre-processed video clip paths (already speed-adjusted)
        audio_segments: List of TTS audio paths
        output_path: Path for final output
        
    Returns:
        Path to final video
    """
    if len(clips) != len(audio_segments):
        raise ValueError("Clips and audio segments must have same length")
    
    # Use absolute paths to avoid working directory issues
    output_path_abs = str(Path(output_path).resolve())
    Path(output_path_abs).parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(output_path_abs).parent / "temp_compose"
    temp_dir.mkdir(exist_ok=True)
    
    try:
        # Resolve clip paths to absolute
        clips_abs = [str(Path(c).resolve()) for c in clips]
        
        # Step 1: Concatenate video clips
        concat_video_path = str(temp_dir / "concat_video.mp4")
        concatenate_videos(clips_abs, concat_video_path)
        
        # Step 2: Concatenate audio segments
        concat_audio_path = str(temp_dir / "concat_audio.aac")
        audio_segments_abs = [str(Path(a).resolve()) for a in audio_segments]
        concatenate_audio(audio_segments_abs, concat_audio_path)
        
        # Step 3: Merge video and audio
        merge_video_audio(concat_video_path, concat_audio_path, output_path_abs)
        
        return output_path_abs
        
    finally:
        # Clean up temp files
        for f in temp_dir.glob("*"):
            try:
                f.unlink()
            except:
                pass
        try:
            temp_dir.rmdir()
        except:
            pass


def compose_final_video_fast_with_captions(
    clips: List[str],
    audio_segments: List[str],
    audio_durations: List[float],
    word_timings_list: List[List[WordTiming]],
    output_path: str,
    caption_style: Optional[CaptionStyle] = None,
) -> str:
    """
    Compose the final video from pre-processed clips, audio, and captions.
    
    This is a faster version that assumes clips are already speed-adjusted.
    Steps:
    1. Concatenate all video clips
    2. Concatenate all audio segments
    3. Merge video and audio
    4. Add caption overlay
    
    Args:
        clips: List of pre-processed video clip paths (already speed-adjusted)
        audio_segments: List of TTS audio paths
        audio_durations: List of audio durations in seconds (for caption timing)
        word_timings_list: List of word timing lists (one per segment)
        output_path: Path for final output
        caption_style: Optional caption style configuration
        
    Returns:
        Path to final video with captions
    """
    if len(clips) != len(audio_segments):
        raise ValueError("Clips and audio segments must have same length")
    if len(word_timings_list) != len(clips):
        raise ValueError("Word timings list must match number of clips")
    
    # Use absolute paths to avoid working directory issues
    output_path_abs = str(Path(output_path).resolve())
    Path(output_path_abs).parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(output_path_abs).parent / "temp_compose"
    temp_dir.mkdir(exist_ok=True)
    
    try:
        # Resolve clip paths to absolute
        clips_abs = [str(Path(c).resolve()) for c in clips]
        
        # Step 1: Concatenate video clips
        concat_video_path = str(temp_dir / "concat_video.mp4")
        concatenate_videos(clips_abs, concat_video_path)
        
        # Step 2: Concatenate audio segments
        concat_audio_path = str(temp_dir / "concat_audio.aac")
        audio_segments_abs = [str(Path(a).resolve()) for a in audio_segments]
        concatenate_audio(audio_segments_abs, concat_audio_path)
        
        # Step 3: Merge video and audio (to temp file)
        merged_path = str(temp_dir / "merged.mp4")
        merge_video_audio(concat_video_path, concat_audio_path, merged_path)
        
        # Step 4: Merge word timings and add captions
        all_words = merge_word_timings(word_timings_list, audio_durations)
        
        add_captions_to_video(merged_path, all_words, output_path_abs, caption_style)
        
        return output_path_abs
        
    finally:
        # Clean up temp files
        for f in temp_dir.glob("*"):
            try:
                f.unlink()
            except:
                pass
        try:
            temp_dir.rmdir()
        except:
            pass


def compose_final_video(
    clips: List[str],
    audio_segments: List[str],
    audio_durations: List[float],
    output_path: str,
) -> str:
    """
    Compose the final video from clips and audio.
    
    Steps:
    1. Speed-adjust each clip to match corresponding audio duration
    2. Concatenate all video clips
    3. Concatenate all audio segments
    4. Merge video and audio
    
    Args:
        clips: List of cropped video clip paths
        audio_segments: List of TTS audio paths
        audio_durations: List of audio durations in seconds
        output_path: Path for final output
        
    Returns:
        Path to final video
    """
    if len(clips) != len(audio_segments) or len(clips) != len(audio_durations):
        raise ValueError("Clips, audio segments, and durations must have same length")
    
    # Use absolute paths to avoid working directory issues
    output_path_abs = str(Path(output_path).resolve())
    Path(output_path_abs).parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(output_path_abs).parent / "temp_compose"
    temp_dir.mkdir(exist_ok=True)
    
    try:
        # Step 1: Speed-adjust each clip to match audio duration
        adjusted_clips = []
        for i, (clip, duration) in enumerate(zip(clips, audio_durations)):
            # Resolve input clip path to absolute
            clip_abs = str(Path(clip).resolve())
            adjusted_path = str(temp_dir / f"adjusted_{i}.mp4")
            speed_adjust_clip(clip_abs, adjusted_path, duration)
            adjusted_clips.append(adjusted_path)
        
        # Step 2: Concatenate video clips
        concat_video_path = str(temp_dir / "concat_video.mp4")
        concatenate_videos(adjusted_clips, concat_video_path)
        
        # Step 3: Concatenate audio segments
        concat_audio_path = str(temp_dir / "concat_audio.aac")
        # Resolve audio segment paths to absolute
        audio_segments_abs = [str(Path(a).resolve()) for a in audio_segments]
        concatenate_audio(audio_segments_abs, concat_audio_path)
        
        # Step 4: Merge video and audio
        merge_video_audio(concat_video_path, concat_audio_path, output_path_abs)
        
        return output_path_abs
        
    finally:
        # Clean up temp files
        for f in temp_dir.glob("*"):
            try:
                f.unlink()
            except:
                pass
        try:
            temp_dir.rmdir()
        except:
            pass


def _get_video_dimensions(video_path: str) -> tuple:
    """
    Get video dimensions using ffprobe or ffmpeg fallback.
    
    Args:
        video_path: Path to video file
        
    Returns:
        Tuple of (width, height)
    """
    import os
    import re
    from src.clip_extractor import get_ffprobe_path, get_ffmpeg_path
    
    probe_path = get_ffprobe_path()
    is_ffprobe = 'ffprobe' in os.path.basename(probe_path)
    
    if is_ffprobe:
        # Use ffprobe with proper options
        cmd = [
            probe_path,
            '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height',
            '-of', 'csv=p=0',
            video_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to get video dimensions: {result.stderr}")
        dimensions = result.stdout.strip().split(',')
        return int(dimensions[0]), int(dimensions[1])
    else:
        # Fallback: use ffmpeg -i and parse the output
        ffmpeg_path = get_ffmpeg_path()
        cmd = [ffmpeg_path, '-i', video_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        # ffmpeg -i returns error code 1 but prints info to stderr
        output = result.stderr
        
        # Parse dimensions from output like "Stream #0:0: Video: h264, 1920x1080"
        match = re.search(r'Stream.*Video:.*, (\d{2,5})x(\d{2,5})', output)
        if match:
            return int(match.group(1)), int(match.group(2))
        
        # Alternative pattern: look for explicit resolution
        match = re.search(r'(\d{2,5})x(\d{2,5})', output)
        if match:
            return int(match.group(1)), int(match.group(2))
        
        raise RuntimeError(f"Failed to parse video dimensions from ffmpeg output")


def add_captions_to_video(
    video_path: str,
    word_timings: List[WordTiming],
    output_path: str,
    style: Optional[CaptionStyle] = None,
) -> str:
    """
    Add word-by-word highlighting captions to a video using FFmpeg.
    
    This uses FFmpeg's native ASS subtitle filter which is MUCH faster than
    Python-based frame-by-frame rendering (typically 10-20x faster).
    
    The captions use karaoke-style highlighting where the current word
    is shown in yellow while other words are white.
    
    Args:
        video_path: Path to input video (with audio)
        word_timings: List of WordTiming objects for the entire video
        output_path: Path for output video with captions
        style: Caption style configuration (optional)
        
    Returns:
        Path to the captioned video
    """
    if style is None:
        style = CaptionStyle()
    
    # Get video dimensions for proper caption positioning
    width, height = _get_video_dimensions(video_path)
    
    # Generate ASS subtitle file
    temp_dir = Path(output_path).parent / "temp_captions"
    temp_dir.mkdir(parents=True, exist_ok=True)
    ass_path = str(temp_dir / "captions.ass")
    
    try:
        generate_ass_subtitles(
            word_timings=word_timings,
            output_path=ass_path,
            style=style,
            video_width=width,
            video_height=height,
        )
        
        # Use FFmpeg to burn in subtitles
        ffmpeg = get_ffmpeg_path()
        
        # Escape the ASS path for FFmpeg filter syntax
        # FFmpeg filter paths need special escaping: \ -> \\, : -> \:, ' -> \'
        escaped_ass_path = ass_path.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
        
        cmd = [
            ffmpeg, '-y',
            '-i', video_path,
            '-vf', f"ass={escaped_ass_path}",
            '-c:v', 'libx264',
            '-preset', 'ultrafast',
            '-crf', '23',
            '-c:a', 'copy',  # Copy audio without re-encoding
            output_path,
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg caption overlay failed: {result.stderr}")
        
        return output_path
        
    finally:
        # Clean up temp files
        try:
            if Path(ass_path).exists():
                Path(ass_path).unlink()
            temp_dir.rmdir()
        except Exception:
            pass


def add_captions_to_video_moviepy(
    video_path: str,
    word_timings: List[WordTiming],
    output_path: str,
    style: Optional[CaptionStyle] = None,
) -> str:
    """
    Add word-by-word highlighting captions to a video using MoviePy.
    
    This is the legacy/fallback method that uses Python-based frame-by-frame
    rendering. It's slower but doesn't require FFmpeg ASS filter support.
    
    For better performance, use add_captions_to_video() which uses FFmpeg.
    
    Args:
        video_path: Path to input video (with audio)
        word_timings: List of WordTiming objects for the entire video
        output_path: Path for output video with captions
        style: Caption style configuration (optional)
        
    Returns:
        Path to the captioned video
    """
    from moviepy import VideoFileClip, VideoClip, CompositeVideoClip
    
    if style is None:
        style = CaptionStyle()
    
    # Load the video
    video = VideoFileClip(video_path)
    width, height = video.size
    
    # Pre-compute font and word groupings ONCE
    font = get_font(style.font_path, style.font_size)
    lines = group_words_into_lines(word_timings, style.max_words_per_line)
    
    # Create a function that generates caption frames using cached values
    def make_caption_frame(t):
        """Generate caption overlay frame for time t."""
        frame = render_caption_frame_fast(
            width=width,
            height=height,
            lines=lines,
            current_time=t,
            style=style,
            font=font,
        )
        return frame
    
    # Create caption overlay clip
    caption_overlay = VideoClip(make_caption_frame, duration=video.duration)
    caption_overlay = caption_overlay.with_position((0, 0))
    
    # Composite video with caption overlay
    final = CompositeVideoClip([video, caption_overlay])
    
    # Write output with optimized settings
    final.write_videofile(
        output_path,
        codec='libx264',
        audio_codec='aac',
        fps=video.fps or 30,
        preset='ultrafast',
        threads=4,
    )
    
    # Clean up
    video.close()
    final.close()
    
    return output_path


def compose_final_video_with_captions(
    clips: List[str],
    audio_segments: List[str],
    audio_durations: List[float],
    word_timings_list: List[List[WordTiming]],
    output_path: str,
    caption_style: Optional[CaptionStyle] = None,
) -> str:
    """
    Compose the final video from clips, audio, and captions.
    
    Steps:
    1. Speed-adjust each clip to match corresponding audio duration
    2. Concatenate all video clips
    3. Concatenate all audio segments
    4. Merge video and audio
    5. Add caption overlay
    
    Args:
        clips: List of cropped video clip paths
        audio_segments: List of TTS audio paths
        audio_durations: List of audio durations in seconds
        word_timings_list: List of word timing lists (one per segment)
        output_path: Path for final output
        caption_style: Optional caption style configuration
        
    Returns:
        Path to final video with captions
    """
    if len(clips) != len(audio_segments) or len(clips) != len(audio_durations):
        raise ValueError("Clips, audio segments, and durations must have same length")
    if len(word_timings_list) != len(clips):
        raise ValueError("Word timings list must match number of clips")
    
    # Use absolute paths to avoid working directory issues
    output_path_abs = str(Path(output_path).resolve())
    Path(output_path_abs).parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(output_path_abs).parent / "temp_compose"
    temp_dir.mkdir(exist_ok=True)
    
    try:
        # Step 1: Speed-adjust each clip to match audio duration
        adjusted_clips = []
        for i, (clip, duration) in enumerate(zip(clips, audio_durations)):
            clip_abs = str(Path(clip).resolve())
            adjusted_path = str(temp_dir / f"adjusted_{i}.mp4")
            speed_adjust_clip(clip_abs, adjusted_path, duration)
            adjusted_clips.append(adjusted_path)
        
        # Step 2: Concatenate video clips
        concat_video_path = str(temp_dir / "concat_video.mp4")
        concatenate_videos(adjusted_clips, concat_video_path)
        
        # Step 3: Concatenate audio segments
        concat_audio_path = str(temp_dir / "concat_audio.aac")
        audio_segments_abs = [str(Path(a).resolve()) for a in audio_segments]
        concatenate_audio(audio_segments_abs, concat_audio_path)
        
        # Step 4: Merge video and audio (to temp file)
        merged_path = str(temp_dir / "merged.mp4")
        merge_video_audio(concat_video_path, concat_audio_path, merged_path)
        
        # Step 5: Merge word timings and add captions
        all_words = merge_word_timings(word_timings_list, audio_durations)
        add_captions_to_video(merged_path, all_words, output_path_abs, caption_style)
        
        return output_path_abs
        
    finally:
        # Clean up temp files
        for f in temp_dir.glob("*"):
            try:
                f.unlink()
            except:
                pass
        try:
            temp_dir.rmdir()
        except:
            pass


def compose_with_moviepy(
    clips: List[str],
    audio_segments: List[str],
    audio_durations: List[float],
    output_path: str,
) -> str:
    """
    Alternative compositor using MoviePy (for comparison/fallback).
    
    Args:
        clips: List of cropped video clip paths
        audio_segments: List of TTS audio paths
        audio_durations: List of audio durations in seconds
        output_path: Path for final output
        
    Returns:
        Path to final video
    """
    from moviepy import VideoFileClip, AudioFileClip, concatenate_videoclips
    from moviepy.video.fx import MultiplySpeed
    
    video_clips = []
    
    for clip_path, audio_path, target_duration in zip(clips, audio_segments, audio_durations):
        # Load video clip
        video = VideoFileClip(clip_path)
        
        # Speed adjust to match audio duration
        if abs(video.duration - target_duration) > 0.1:
            speed_factor = video.duration / target_duration
            video = video.with_effects([MultiplySpeed(factor=speed_factor)])
        
        # Load and attach audio
        audio = AudioFileClip(audio_path)
        video = video.with_audio(audio)
        
        video_clips.append(video)
    
    # Concatenate all clips
    final = concatenate_videoclips(video_clips, method="compose")
    
    # Export
    final.write_videofile(
        output_path,
        codec='libx264',
        audio_codec='aac',
        fps=30,
    )
    
    # Clean up
    for clip in video_clips:
        clip.close()
    final.close()
    
    return output_path
