"""
Caption rendering module for word-by-word highlighting captions.

This module renders karaoke-style captions where the current word
is highlighted as it's being spoken.
"""

import math
from dataclasses import dataclass
from typing import List, Tuple, Optional
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


@dataclass
class WordTiming:
    """Represents a word with its timing information."""
    text: str
    start: float  # Start time in seconds
    end: float    # End time in seconds


@dataclass 
class CaptionStyle:
    """Configuration for caption appearance.
    
    Safe zone positioning is optimized for major shorts platforms:
    - YouTube Shorts: Right-side buttons, bottom description/subscribe
    - TikTok: Right-side buttons, bottom username/description/hashtags
    - Instagram Reels: Right-side buttons, bottom username/audio info
    """
    font_path: Optional[str] = None  # Path to TTF font, None = default
    font_size: int = 58
    normal_color: Tuple[int, int, int] = (255, 255, 255)  # White
    highlight_color: Tuple[int, int, int] = (255, 255, 0)  # Yellow
    outline_color: Tuple[int, int, int] = (0, 0, 0)  # Black
    outline_width: int = 3
    max_words_per_line: int = 5
    y_position_ratio: float = 0.70  # Position from top (0-1), moved up to avoid bottom UI
    line_spacing: int = 10
    # Safe zone margins to avoid platform UI elements (as ratio of video dimensions)
    # These margins ensure captions don't overlap with like/comment/share buttons
    # on the right side or username/description at the bottom
    horizontal_margin_ratio: float = 0.12  # 12% margin on each side for right-side buttons
    safe_zone_enabled: bool = True  # Enable platform-aware safe zones


# Default system fonts to try (cross-platform)
FALLBACK_FONTS = [
    # macOS
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/Library/Fonts/Arial Bold.ttf",
    # Linux
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    # Windows
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/arial.ttf",
]


def get_font(font_path: Optional[str], size: int) -> ImageFont.FreeTypeFont:
    """Load a font, falling back to system fonts if needed."""
    if font_path and Path(font_path).exists():
        try:
            return ImageFont.truetype(font_path, size)
        except Exception:
            pass
    
    # Try fallback fonts
    for fallback in FALLBACK_FONTS:
        if Path(fallback).exists():
            try:
                return ImageFont.truetype(fallback, size)
            except Exception:
                continue
    
    # Last resort: PIL default font (not ideal but works)
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        # Older PIL versions don't support size parameter
        return ImageFont.load_default()


def group_words_into_lines(
    words: List[WordTiming], 
    max_words: int = 5
) -> List[List[WordTiming]]:
    """
    Group words into display lines.
    
    Each line contains up to max_words words and is displayed
    while any of its words are being spoken.
    """
    lines = []
    current_line = []
    
    for word in words:
        current_line.append(word)
        if len(current_line) >= max_words:
            lines.append(current_line)
            current_line = []
    
    # Add remaining words
    if current_line:
        lines.append(current_line)
    
    return lines


def get_line_timing(line: List[WordTiming]) -> Tuple[float, float]:
    """Get the start and end time for a line of words."""
    if not line:
        return (0, 0)
    return (line[0].start, line[-1].end)


def find_active_line(
    lines: List[List[WordTiming]], 
    current_time: float
) -> Optional[int]:
    """Find which line should be displayed at the current time."""
    for i, line in enumerate(lines):
        start, end = get_line_timing(line)
        if start <= current_time <= end:
            return i
    return None


def draw_text_with_outline(
    draw: ImageDraw.ImageDraw,
    position: Tuple[int, int],
    text: str,
    font: ImageFont.FreeTypeFont,
    fill_color: Tuple[int, int, int],
    outline_color: Tuple[int, int, int],
    outline_width: int = 2,
) -> None:
    """Draw text with an outline for readability on any background.
    
    Uses PIL's native stroke support (Pillow 8.0+) for much better performance.
    """
    x, y = position
    
    # Use PIL's native stroke support - much faster than manual outline loop
    # This replaces the previous approach of drawing text (2*outline_width+1)^2 - 1 times
    draw.text(
        (x, y), 
        text, 
        font=font, 
        fill=fill_color,
        stroke_width=outline_width,
        stroke_fill=outline_color,
    )


def render_caption_frame_fast(
    width: int,
    height: int,
    lines: List[List[WordTiming]],
    current_time: float,
    style: CaptionStyle,
    font: ImageFont.FreeTypeFont,
) -> np.ndarray:
    """
    Render a single caption frame with word highlighting (optimized version).
    
    This version accepts pre-computed font and lines to avoid redundant
    computation when rendering many frames.
    
    Args:
        width: Frame width in pixels
        height: Frame height in pixels
        lines: Pre-computed list of word groups (from group_words_into_lines)
        current_time: Current video time in seconds
        style: Caption style configuration
        font: Pre-loaded font object
        
    Returns:
        RGBA numpy array of the caption overlay
    """
    # Create transparent image
    img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Find which line to display
    active_line_idx = find_active_line(lines, current_time)
    
    if active_line_idx is None:
        # No caption to show at this time
        return np.array(img)
    
    active_line = lines[active_line_idx]
    
    # Calculate line text and width
    line_text = " ".join(word.text for word in active_line)
    
    # Get text bounding box
    bbox = draw.textbbox((0, 0), line_text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # Calculate safe zone margins for shorts platforms (YouTube, TikTok, Instagram)
    if style.safe_zone_enabled:
        horizontal_margin = int(width * style.horizontal_margin_ratio)
    else:
        horizontal_margin = 0
    
    # Calculate available width within safe zone
    safe_width = width - (2 * horizontal_margin)
    
    # Center horizontally within safe zone, position vertically according to style
    # If text is wider than safe zone, still center but it may extend slightly
    x_start = horizontal_margin + (safe_width - text_width) // 2
    x_start = max(horizontal_margin, x_start)  # Ensure we don't go past left margin
    
    y_pos = int(height * style.y_position_ratio) - text_height // 2
    
    # Draw each word with appropriate color
    x_offset = x_start
    
    for word in active_line:
        # Check if this word is currently being spoken
        is_highlighted = word.start <= current_time < word.end
        
        color = style.highlight_color if is_highlighted else style.normal_color
        
        # Draw word with outline
        draw_text_with_outline(
            draw,
            (x_offset, y_pos),
            word.text,
            font,
            color,
            style.outline_color,
            style.outline_width,
        )
        
        # Move to next word position (add space)
        word_bbox = draw.textbbox((0, 0), word.text + " ", font=font)
        x_offset += word_bbox[2] - word_bbox[0]
    
    return np.array(img)


def render_caption_frame(
    width: int,
    height: int,
    words: List[WordTiming],
    current_time: float,
    style: Optional[CaptionStyle] = None,
) -> np.ndarray:
    """
    Render a single caption frame with word highlighting.
    
    Note: For rendering many frames, use render_caption_frame_fast() with
    pre-computed font and lines for better performance.
    
    Args:
        width: Frame width in pixels
        height: Frame height in pixels
        words: List of WordTiming objects
        current_time: Current video time in seconds
        style: Caption style configuration
        
    Returns:
        RGBA numpy array of the caption overlay
    """
    if style is None:
        style = CaptionStyle()
    
    # Load font and group words (these are cached in the fast version)
    font = get_font(style.font_path, style.font_size)
    lines = group_words_into_lines(words, style.max_words_per_line)
    
    return render_caption_frame_fast(width, height, lines, current_time, style, font)


def create_caption_clip(
    width: int,
    height: int,
    words: List[WordTiming],
    duration: float,
    style: Optional[CaptionStyle] = None,
    fps: int = 30,
) -> List[np.ndarray]:
    """
    Create all caption frames for a video segment.
    
    Args:
        width: Frame width in pixels
        height: Frame height in pixels
        words: List of WordTiming objects
        duration: Total duration in seconds
        style: Caption style configuration
        fps: Frames per second
        
    Returns:
        List of RGBA numpy arrays
    """
    if style is None:
        style = CaptionStyle()
    
    # Pre-compute font and lines once for all frames (performance optimization)
    font = get_font(style.font_path, style.font_size)
    lines = group_words_into_lines(words, style.max_words_per_line)
    
    frames = []
    num_frames = int(duration * fps)
    
    for frame_idx in range(num_frames):
        current_time = frame_idx / fps
        frame = render_caption_frame_fast(width, height, lines, current_time, style, font)
        frames.append(frame)
    
    return frames


def offset_word_timings(
    words: List[WordTiming], 
    offset: float
) -> List[WordTiming]:
    """Offset all word timings by a given amount."""
    return [
        WordTiming(
            text=w.text,
            start=w.start + offset,
            end=w.end + offset,
        )
        for w in words
    ]


def merge_word_timings(
    word_lists: List[List[WordTiming]],
    durations: List[float],
) -> List[WordTiming]:
    """
    Merge multiple word timing lists with proper offsets.
    
    Args:
        word_lists: List of word timing lists (one per segment)
        durations: Duration of each segment
        
    Returns:
        Single merged list with adjusted timings
    """
    merged = []
    offset = 0.0
    
    for words, duration in zip(word_lists, durations):
        offset_words = offset_word_timings(words, offset)
        merged.extend(offset_words)
        offset += duration
    
    return merged


# =============================================================================
# ASS Subtitle Generation (for FFmpeg-based caption rendering)
# =============================================================================

def _format_ass_time(seconds: float) -> str:
    """
    Format time in ASS subtitle format: H:MM:SS.cs (centiseconds).
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string like "0:01:23.45"
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours}:{minutes:02d}:{secs:05.2f}"


def _rgb_to_ass_color(rgb: Tuple[int, int, int], alpha: int = 0) -> str:
    """
    Convert RGB color to ASS color format (&HAABBGGRR).
    
    ASS uses BGR order with alpha, where alpha 0 = fully opaque.
    
    Args:
        rgb: Tuple of (R, G, B) values 0-255
        alpha: Alpha value 0-255 (0 = opaque, 255 = transparent)
        
    Returns:
        ASS color string like "&H00FFFFFF"
    """
    r, g, b = rgb
    return f"&H{alpha:02X}{b:02X}{g:02X}{r:02X}"


def _escape_ass_text(text: str) -> str:
    """
    Escape special characters in ASS subtitle text.
    
    Args:
        text: Raw text to escape
        
    Returns:
        Escaped text safe for ASS format
    """
    # Escape backslashes first, then other special chars
    text = text.replace("\\", "\\\\")
    text = text.replace("{", "\\{")
    text = text.replace("}", "\\}")
    # Newlines in ASS use \N
    text = text.replace("\n", "\\N")
    return text


def _get_ass_font_name(font_path: Optional[str]) -> str:
    """
    Get font name for ASS from font path.
    
    Args:
        font_path: Path to font file or None
        
    Returns:
        Font family name for ASS
    """
    if font_path:
        # Extract font name from path (e.g., "Arial Bold.ttf" -> "Arial Bold")
        name = Path(font_path).stem
        # Remove common suffixes
        for suffix in ["-Regular", "-Bold", "-Italic", "-BoldItalic"]:
            name = name.replace(suffix, "")
        return name
    return "Arial"


def generate_ass_subtitles(
    word_timings: List[WordTiming],
    output_path: str,
    style: Optional[CaptionStyle] = None,
    video_width: int = 1080,
    video_height: int = 1920,
) -> str:
    """
    Generate ASS subtitle file with word-by-word karaoke highlighting.
    
    This creates an ASS (Advanced SubStation Alpha) subtitle file that can be
    burned into video using FFmpeg's 'ass' filter. This is MUCH faster than
    Python-based frame-by-frame rendering.
    
    The subtitles use ASS karaoke tags (\\kf) to highlight each word as it's spoken,
    transitioning from white to yellow.
    
    Args:
        word_timings: List of WordTiming objects for the entire video
        output_path: Path to write the .ass file
        style: Caption style configuration (optional)
        video_width: Video width in pixels (for positioning)
        video_height: Video height in pixels (for positioning)
        
    Returns:
        Path to the generated ASS file
    """
    if style is None:
        style = CaptionStyle()
    
    # Group words into display lines
    lines = group_words_into_lines(word_timings, style.max_words_per_line)
    
    # Convert colors to ASS format
    normal_color = _rgb_to_ass_color(style.normal_color)
    highlight_color = _rgb_to_ass_color(style.highlight_color)
    outline_color = _rgb_to_ass_color(style.outline_color)
    
    # Get font name
    font_name = _get_ass_font_name(style.font_path)
    
    # Calculate vertical margin from bottom (ASS uses margin from edges)
    # y_position_ratio is from top, so margin from bottom = height * (1 - ratio)
    margin_v = int(video_height * (1 - style.y_position_ratio))
    
    # Calculate horizontal margins for safe zones (shorts platform UI avoidance)
    # These margins keep captions away from right-side buttons (like, comment, share)
    if style.safe_zone_enabled:
        margin_h = int(video_width * style.horizontal_margin_ratio)
    else:
        margin_h = 10  # Minimal default margin
    
    # ASS header
    ass_content = f"""[Script Info]
Title: Auto-generated captions
ScriptType: v4.00+
WrapStyle: 0
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.709
PlayResX: {video_width}
PlayResY: {video_height}

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font_name},{style.font_size},{normal_color},{highlight_color},{outline_color},&H00000000,1,0,0,0,100,100,0,0,1,{style.outline_width},0,2,{margin_h},{margin_h},{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    
    # Generate dialogue events for each line
    for line in lines:
        if not line:
            continue
            
        start_time = line[0].start
        end_time = line[-1].end
        
        # Build karaoke text with {\kf} tags for word-by-word highlighting
        # \kf = smooth fill from secondary (highlight) to primary (normal)
        # Duration is in centiseconds
        text_parts = []
        
        for i, word in enumerate(line):
            # Calculate duration in centiseconds for this word
            duration_cs = int((word.end - word.start) * 100)
            
            # Escape the word text
            escaped_word = _escape_ass_text(word.text)
            
            # Add karaoke tag with fill effect
            # {\kf<duration>} highlights the text over the duration
            text_parts.append(f"{{\\kf{duration_cs}}}{escaped_word}")
            
            # Add space between words (except after last word)
            if i < len(line) - 1:
                text_parts.append(" ")
        
        text = "".join(text_parts)
        
        # Format times
        start_str = _format_ass_time(start_time)
        end_str = _format_ass_time(end_time)
        
        # Add dialogue line
        ass_content += f"Dialogue: 0,{start_str},{end_str},Default,,0,0,0,,{text}\n"
    
    # Write the ASS file
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(ass_content)
    
    return output_path
