"""
Utility functions for parsing and time conversion.
"""

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class Segment:
    """Represents a single segment of the video script."""
    script_time: float      # When this segment starts in final video (seconds)
    text: str               # Voiceover text
    footage_start: float    # Where to pull footage from source video (seconds)
    is_still: bool = False  # True = still image with Ken Burns, False = video clip
    layout: str = "crop"    # Layout mode: "crop" (face-centered) or "letterbox" (blur background)


def parse_timestamp(timestamp: str) -> float:
    """
    Convert a timestamp string to seconds.
    
    Supports formats:
    - MM:SS (e.g., "23:45")
    - HH:MM:SS (e.g., "1:23:45")
    - [MM:SS] (bracketed, e.g., "[23:45]")
    - Seconds only (e.g., "145.5")
    
    Args:
        timestamp: Timestamp string
        
    Returns:
        Time in seconds (float)
    """
    # Remove brackets if present
    original = timestamp
    timestamp = timestamp.strip().strip('[]')
    
    # Try to parse as float (seconds only)
    try:
        return float(timestamp)
    except ValueError:
        pass
    
    # Parse MM:SS or HH:MM:SS
    parts = timestamp.split(':')
    
    if len(parts) == 2:
        # MM:SS
        minutes, seconds = parts
        return int(minutes) * 60 + float(seconds)
    elif len(parts) == 3:
        # HH:MM:SS
        hours, minutes, seconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    else:
        raise ValueError(f"Invalid timestamp format: {timestamp}")


def format_timestamp(seconds: float) -> str:
    """
    Convert seconds to MM:SS format.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted timestamp string
    """
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes:02d}:{secs:05.2f}"


def parse_script_table(text: str) -> list[Segment]:
    """
    Parse a 2-column script table into Segment objects.
    
    Expected format (tab or pipe separated):
    Script | Footage Timestamp
    Did you know... | [23:23]
    This is Murphy... | [24:02]
    
    Args:
        text: Raw table text (can be pasted from spreadsheet)
        
    Returns:
        List of Segment objects
    """
    segments = []
    lines = text.strip().split('\n')
    
    for line_num, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        
        # Skip header rows
        if _is_header_row(line):
            continue
        
        # Try different delimiters: tab, pipe, multiple spaces
        parts = _split_row(line)
        
        if len(parts) < 2:
            print(f"Warning: Skipping line {line_num + 1}, not enough columns: {line}")
            continue
        
        try:
            text_content = parts[0].strip().strip('"\'')
            footage_start = parse_timestamp(parts[1])
            
            # Parse 3rd column for still/video indicator (default to video)
            is_still = False
            if len(parts) >= 3:
                media_type = parts[2].strip().lower()
                is_still = media_type == 'still'
            
            # Parse 4th column for layout mode (default to crop)
            # Supports: "crop" (face-centered), "letterbox" (blur background)
            layout = "crop"
            if len(parts) >= 4:
                layout_value = parts[3].strip().lower()
                if layout_value in ('letterbox', 'letter', 'blur', 'full'):
                    layout = "letterbox"
            
            if text_content:  # Only add if there's actual text
                segments.append(Segment(
                    script_time=0,  # Not used, segments are concatenated sequentially
                    text=text_content,
                    footage_start=footage_start,
                    is_still=is_still,
                    layout=layout,
                ))
        except ValueError as e:
            print(f"Warning: Could not parse line {line_num + 1}: {line} ({e})")
            continue
    
    return segments


def _is_header_row(line: str) -> bool:
    """Check if a line appears to be a table header."""
    lower = line.lower()
    header_keywords = ['time', 'script', 'voiceover', 'timestamp', 'footage', 'still/video', 'still', 'video', '---', '===']
    # Only consider it a header if it looks like a header row (multiple keywords or separator)
    keyword_count = sum(1 for kw in header_keywords if kw in lower)
    return keyword_count >= 2 or '---' in lower or '===' in lower


def _split_row(line: str) -> list[str]:
    """
    Split a table row by various delimiters.
    
    Tries in order: tab, pipe, bracketed timestamp at end, multiple spaces
    """
    # Try tab first (most common from spreadsheet paste)
    if '\t' in line:
        return [p.strip() for p in line.split('\t')]
    
    # Try pipe (markdown table format)
    if '|' in line:
        parts = [p.strip() for p in line.split('|')]
        # Remove empty parts from leading/trailing pipes
        return [p for p in parts if p]
    
    # Try to detect bracketed timestamp at end of line
    # Matches patterns like [12:34] or [1:23:45] at the end
    match = re.match(r'^(.+?)\s*(\[\d{1,2}:\d{2}(?::\d{2})?\])\s*$', line)
    if match:
        return [match.group(1).strip(), match.group(2)]
    
    # Try multiple spaces (at least 2)
    parts = re.split(r'\s{2,}', line)
    if len(parts) >= 2:
        return [p.strip() for p in parts]
    
    # Fallback: try comma
    if ',' in line:
        return [p.strip() for p in line.split(',')]
    
    # Last resort: return as-is (single column)
    return [line]


def clean_temp_files(temp_dir: str = "temp", keep_source: bool = True) -> int:
    """
    Clean up temporary files.
    
    Args:
        temp_dir: Directory containing temp files
        keep_source: If True, keep downloaded source videos
        
    Returns:
        Number of files deleted
    """
    from pathlib import Path
    
    temp_path = Path(temp_dir)
    if not temp_path.exists():
        return 0
    
    deleted = 0
    for file in temp_path.iterdir():
        if file.is_file():
            # Keep source videos if requested
            if keep_source and file.stem in ['source', 'video'] or len(file.stem) == 11:
                continue
            
            # Delete temp clips and audio
            if 'clip' in file.name or 'audio' in file.name:
                file.unlink()
                deleted += 1
    
    return deleted


def clean_old_outputs(output_dir: str = "output", keep_count: int = 10) -> int:
    """
    Keep only the most recent output files, deleting older ones.
    
    Args:
        output_dir: Directory containing output videos
        keep_count: Number of most recent files to keep
        
    Returns:
        Number of files deleted
    """
    from pathlib import Path
    
    output_path = Path(output_dir)
    if not output_path.exists():
        return 0
    
    # Get all mp4 files sorted by modification time (newest first)
    files = sorted(
        output_path.glob("*.mp4"),
        key=lambda f: f.stat().st_mtime,
        reverse=True
    )
    
    deleted = 0
    for file in files[keep_count:]:
        try:
            file.unlink()
            deleted += 1
        except Exception as e:
            print(f"Warning: Could not delete {file}: {e}")
    
    if deleted > 0:
        print(f"Cleaned up {deleted} old output file(s)")
    
    return deleted
