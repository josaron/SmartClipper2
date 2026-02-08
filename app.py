"""
SmartClipper: YouTube Shorts Generator

A local app that creates 45-75 second vertical shorts from long-form YouTube videos
using TTS voiceover and intelligent face-tracking crop.
"""

import gradio as gr
from pathlib import Path
from datetime import datetime
import traceback
import time
from typing import Optional, Tuple

from src.downloader import download_video
from src.tts import generate_audio, generate_audio_with_words, list_voices, DEFAULT_VOICE
from src.utils import (
    parse_script_table,
    parse_script_table_with_errors,
    segments_to_rows,
    segments_to_text,
    rows_to_text,
    format_timestamps_in_text,
    clean_temp_files,
    clean_old_outputs,
)
from src.clip_extractor import check_ffmpeg
from src.cropper import (
    extract_crop_and_speed, 
    extract_still_with_ken_burns, 
    extract_with_letterbox,
    extract_still_with_letterbox,
    KEN_BURNS_STYLES,
)
from src.compositor import compose_final_video_fast, compose_final_video_fast_with_captions

# Resolution presets (width x height) for 9:16 vertical video
RESOLUTION_PRESETS = {
    "720p (720x1280)": (720, 1280),
    "1080p (1080x1920)": (1080, 1920),
    "1440p (1440x2560)": (1440, 2560),
}
DEFAULT_RESOLUTION = "1080p (1080x1920)"

# Ensure temp and output directories exist
Path("temp").mkdir(exist_ok=True)
Path("output").mkdir(exist_ok=True)

# Clean old output files on startup (keep only the 10 most recent)
clean_old_outputs(keep_count=10)

# Sample script for demo (4-column format: Script | Timestamp | Still/Video | Layout)
# Layout column is optional: "crop" (default, face-centered) or "letterbox" (blur background, full width)
SAMPLE_SCRIPT = """Did you know a 51-foot fire-breathing dragon used to live on the Las Vegas Strip?\t[23:23]\tVideo\tcrop
This is Murphy. He was the star of the Excalibur Hotel, emerging from a cave every hour to battle Merlin.\t[24:02]\tStill\tcrop
Built by Disney veterans, Murphy was a hydraulic beast who spent his days submerged in the castle moat.\t[23:37]\tVideo\tletterbox
But the desert wasn't kind. Between constant breakdowns and the family-friendly era ending, the show was axed in 2003.\t[27:44]\tStill\tletterbox
For 20 years, Murphy didn't leave. He was simply locked behind a wall in his dark, underwater cave.\t[28:01]\tVideo\tcrop
In 2024, his lair was finally sealed for good. Most think he was scrapped, but some say he's still in there...\t[28:22]\tStill\tcrop
...waiting for the day the magic returns to Vegas. Subscribe for more lost history!\t[31:01]\tVideo\tletterbox"""


def validate_inputs(youtube_url: str, script_table: str, voice: str) -> Tuple[bool, str]:
    """Validate user inputs before processing."""
    if not youtube_url or not youtube_url.strip():
        return False, "Please enter a YouTube URL"
    
    if 'youtube.com' not in youtube_url and 'youtu.be' not in youtube_url:
        return False, "Invalid YouTube URL. Please use a youtube.com or youtu.be link"
    
    if not script_table or not script_table.strip():
        return False, "Please enter a script table"
    
    if not voice:
        return False, "Please select a voice"
    
    if not check_ffmpeg():
        return False, "FFmpeg is not installed. Please install FFmpeg first."
    
    return True, ""


def process_video(youtube_url: str, script_table: str, voice: str, resolution: str, add_captions: bool = True, progress=gr.Progress()):
    """
    Main processing pipeline:
    1. Download YouTube video
    2. Parse script/timestamps
    3. Generate TTS audio for each segment
    4. Extract and crop video clips
    5. Compose final video (with or without captions)
    """
    # Validate inputs
    valid, error_msg = validate_inputs(youtube_url, script_table, voice)
    if not valid:
        return None, error_msg
    
    # Parse resolution
    target_width, target_height = RESOLUTION_PRESETS.get(resolution, RESOLUTION_PRESETS[DEFAULT_RESOLUTION])
    
    # Track generation time
    start_time = time.time()
    
    try:
        # Step 1: Download video
        progress(0.05, desc="Downloading video...")
        video_path = download_video(youtube_url.strip())
        progress(0.15, desc="Video downloaded!")
        
        # Step 2: Parse script table
        progress(0.2, desc="Parsing script...")
        segments = parse_script_table(script_table)
        
        if not segments:
            return None, "Error: No valid segments found in script table. Check the format."
        
        progress(0.25, desc=f"Found {len(segments)} segments")
        
        # Step 3: Generate TTS audio for each segment (with word timings for captions)
        audio_paths = []
        audio_durations = []
        word_timings_list = []
        
        for i, segment in enumerate(segments):
            progress(0.25 + (0.2 * i / len(segments)), desc=f"Generating audio {i+1}/{len(segments)}...")
            audio_path = f"temp/audio_{i}.mp3"
            duration, word_timings = generate_audio_with_words(segment.text, voice, audio_path)
            audio_paths.append(audio_path)
            audio_durations.append(duration)
            word_timings_list.append(word_timings)
        
        # Calculate total duration
        total_duration = sum(audio_durations)
        if total_duration < 30:
            progress(0.45, desc=f"Warning: Short video ({total_duration:.1f}s)")
        elif total_duration > 90:
            progress(0.45, desc=f"Warning: Long video ({total_duration:.1f}s)")
        else:
            progress(0.45, desc=f"Audio ready ({total_duration:.1f}s total)")
        
        # Step 4: Extract, crop, and speed-adjust clips (or Ken Burns stills)
        processed_clips = []
        
        for i, (segment, duration) in enumerate(zip(segments, audio_durations)):
            clip_type = "still" if segment.is_still else "clip"
            layout_desc = f" ({segment.layout})" if segment.layout != "crop" else ""
            progress(0.45 + (0.35 * i / len(segments)), desc=f"Processing {clip_type}{layout_desc} {i+1}/{len(segments)}...")
            
            processed_path = f"temp/clip_processed_{i}.mp4"
            
            if segment.is_still:
                if segment.layout == "letterbox":
                    # Extract still with blur background letterbox + subtle zoom
                    extract_still_with_letterbox(
                        video_path,
                        segment.footage_start,
                        duration,  # Target duration matches audio
                        processed_path,
                        target_width=target_width,
                        target_height=target_height,
                    )
                else:
                    # Extract still frame with Ken Burns effect (face-centered crop)
                    # Cycle through different styles for variety
                    kb_style = KEN_BURNS_STYLES[i % len(KEN_BURNS_STYLES)]
                    extract_still_with_ken_burns(
                        video_path,
                        segment.footage_start,
                        duration,  # Target duration matches audio
                        processed_path,
                        target_width=target_width,
                        target_height=target_height,
                        zoom_factor=1.08,  # Subtle 8% zoom/pan amount
                        style=kb_style,
                    )
            else:
                # Extract video clip with speed adjustment
                # IMPORTANT: extract_duration must be >= target_duration to avoid sync issues
                extract_duration = max(duration + 2.0, duration * 1.2)  # At least 20% more or +2 seconds
                
                if segment.layout == "letterbox":
                    # Use blur background letterbox (preserves full horizontal view)
                    extract_with_letterbox(
                        video_path,
                        segment.footage_start,
                        extract_duration,
                        duration,  # Target duration matches audio
                        processed_path,
                        target_width=target_width,
                        target_height=target_height,
                    )
                else:
                    # Use face-centered crop (default)
                    extract_crop_and_speed(
                        video_path,
                        segment.footage_start,
                        extract_duration,
                        duration,  # Target duration matches audio
                        processed_path,
                        target_width=target_width,
                        target_height=target_height,
                    )
            
            processed_clips.append(processed_path)
        
        # Step 5: Compose final video (with or without captions; clips are already speed-adjusted)
        progress(0.85, desc="Composing final video (with captions)..." if add_captions else "Composing final video...")
        
        # Generate unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"output/short_{timestamp}.mp4"
        
        if add_captions:
            compose_final_video_fast_with_captions(
                processed_clips,
                audio_paths,
                audio_durations,
                word_timings_list,
                output_path,
            )
        else:
            compose_final_video_fast(
                processed_clips,
                audio_paths,
                output_path,
            )
        
        progress(1.0, desc="Done!")
        
        # Clean up temp files (keep source video)
        clean_temp_files(keep_source=True)
        
        # Calculate generation time
        elapsed_time = time.time() - start_time
        elapsed_min = int(elapsed_time // 60)
        elapsed_sec = int(elapsed_time % 60)
        if elapsed_min > 0:
            time_str = f"{elapsed_min}m {elapsed_sec}s"
        else:
            time_str = f"{elapsed_sec}s"
        
        return output_path, f"Video created successfully! Duration: {total_duration:.1f}s | Generated in {time_str}"
        
    except Exception as e:
        error_detail = traceback.format_exc()
        print(f"Error: {error_detail}")
        return None, f"Error: {str(e)}"


def load_sample_script():
    """Load the sample script into the textbox."""
    return SAMPLE_SCRIPT


def load_paste_into_table(script_table: str):
    """Parse paste text and return rows for the Dataframe."""
    if not (script_table or script_table.strip()):
        return []
    segments, _ = parse_script_table_with_errors(script_table)
    rows = segments_to_rows(segments)
    return rows


def _dataframe_to_rows(script_dataframe):
    """Normalize Gradio Dataframe value to list of rows (list of lists)."""
    if script_dataframe is None:
        return []
    if isinstance(script_dataframe, dict):
        # Gradio may pass {data: [...], headers: [...]}
        return script_dataframe.get("data", []) or []
    if isinstance(script_dataframe, list):
        return script_dataframe if script_dataframe else []
    if hasattr(script_dataframe, "values") and hasattr(script_dataframe.values, "tolist"):
        return script_dataframe.values.tolist()
    return list(script_dataframe) if script_dataframe else []


def copy_table_to_paste(script_dataframe) -> str:
    """Convert Dataframe value to tab-separated text for the paste textbox."""
    rows = _dataframe_to_rows(script_dataframe)
    if not rows:
        return ""
    return rows_to_text(rows)


def validate_script_display(script_table: str, script_dataframe) -> str:
    """Validate the effective script (table if has rows, else paste) and return a message."""
    script_text = get_effective_script(script_table, script_dataframe)
    if not (script_text or script_text.strip()):
        return "No script to validate. Paste a table or add segments in the Edit table tab."
    segments, errors = parse_script_table_with_errors(script_text)
    if errors:
        return "Validation found issues:\n" + "\n".join(errors) + (f"\n\nParsed {len(segments)} segment(s) from valid lines." if segments else "")
    return f"Valid: {len(segments)} segment(s) ready."


def format_timestamps_click(script_table: str) -> str:
    """Normalize timestamps in the paste text and return updated text."""
    if not (script_table or script_table.strip()):
        return script_table or ""
    return format_timestamps_in_text(script_table)


def _default_row():
    """Default new segment row."""
    return ["", "[0:00]", "Video", "crop"]


def table_add_row_at_end(script_dataframe):
    """Append a new row to the script table."""
    rows = _dataframe_to_rows(script_dataframe)
    rows.append(_default_row())
    return rows


def table_insert_row_above(script_dataframe, selected_row) -> Tuple[list, Optional[int]]:
    """Insert a new row above the selected row. If no selection, insert at start."""
    rows = _dataframe_to_rows(script_dataframe)
    idx = selected_row if isinstance(selected_row, int) and 0 <= selected_row < len(rows) else 0
    rows.insert(idx, _default_row())
    return rows, idx


def table_insert_row_below(script_dataframe, selected_row) -> Tuple[list, Optional[int]]:
    """Insert a new row below the selected row. If no selection, append at end."""
    rows = _dataframe_to_rows(script_dataframe)
    if isinstance(selected_row, int) and 0 <= selected_row < len(rows):
        idx = selected_row + 1
    else:
        idx = len(rows)
    rows.insert(idx, _default_row())
    return rows, idx


def table_delete_row(script_dataframe, selected_row) -> Tuple[list, Optional[int]]:
    """Delete the selected row. If no selection, delete last row. If one row, leave empty."""
    rows = _dataframe_to_rows(script_dataframe)
    if not rows:
        return [], None
    if isinstance(selected_row, int) and 0 <= selected_row < len(rows):
        idx = selected_row
    else:
        idx = len(rows) - 1
    rows.pop(idx)
    # Keep selection on same position or previous row
    new_sel = idx if idx < len(rows) else (len(rows) - 1 if len(rows) > 0 else None)
    return rows, new_sel


def get_effective_script(script_table: str, script_dataframe) -> str:
    """Script text to use for generation: from Dataframe if it has rows, else from paste textbox."""
    rows = _dataframe_to_rows(script_dataframe)
    if rows:
        return rows_to_text(rows)
    return script_table or ""


def process_video_with_source(youtube_url: str, script_table: str, script_dataframe, voice: str, resolution: str, add_captions: bool, progress=gr.Progress()):
    """Wrapper that picks effective script from table or paste, then runs process_video."""
    script_text = get_effective_script(script_table, script_dataframe)
    return process_video(youtube_url, script_text, voice, resolution, add_captions, progress)


def preview_voice(voice_name: str) -> Optional[str]:
    """Generate a short audio preview for the selected voice."""
    if not voice_name:
        return None
    
    sample_text = "Hello! This is a preview of how I sound."
    output_path = "temp/voice_preview.mp3"
    
    try:
        generate_audio(sample_text, voice_name, output_path)
        return output_path
    except Exception as e:
        print(f"Voice preview error: {e}")
        return None


def create_ui():
    """Create the Gradio interface."""
    
    # Get available voices (with fallback)
    try:
        voices = list_voices()
        default_voice = DEFAULT_VOICE if DEFAULT_VOICE in voices else (voices[0] if voices else None)
    except Exception:
        voices = ["Ryan (UK English)"]
        default_voice = voices[0]
    
    with gr.Blocks(title="SmartClipper", theme=gr.themes.Glass()) as app:
        gr.Markdown("# SmartClipper: YouTube Shorts Generator")
        gr.Markdown("Create 45-75 second vertical shorts from long-form YouTube videos with AI voiceover.")
        
        # Script section: full width
        with gr.Tabs():
            with gr.Tab("Paste table"):
                gr.Markdown("Paste a tab-separated table from a spreadsheet or AI output.")
                paste_script = gr.Textbox(
                    label="Script / Timestamps",
                    placeholder="Script\tFootage Timestamp\tStill/Video\tLayout\nYour text here...\t[0:30]\tVideo\tcrop",
                    lines=12,
                    info="Tab-separated: Script | [Timestamp] | Still/Video | Layout (crop/letterbox)",
                )
                with gr.Row():
                    sample_btn = gr.Button("Load Sample", variant="secondary", size="sm")
                    validate_btn = gr.Button("Validate script", variant="secondary", size="sm")
                    load_into_table_btn = gr.Button("Load into table", variant="secondary", size="sm")
            
            with gr.Tab("Edit table"):
                gr.Markdown("Edit segments in the table. **Click a row** to select it, then use the buttons below to add or delete rows.")
                edit_dataframe = gr.Dataframe(
                    headers=["Script", "Timestamp", "Still/Video", "Layout"],
                    datatype=["str", "str", "str", "str"],
                    label="Script table",
                    value=[],
                    col_count=(4, "fixed"),
                    type="array",
                    interactive=True,
                )
                selected_row_state = gr.State(value=None)  # int | None: row index for insert/delete
                with gr.Row():
                    add_row_end_btn = gr.Button("Add row at end", variant="secondary", size="sm")
                    insert_above_btn = gr.Button("Insert row above selected", variant="secondary", size="sm")
                    insert_below_btn = gr.Button("Insert row below selected", variant="secondary", size="sm")
                    delete_row_btn = gr.Button("Delete selected row", variant="secondary", size="sm")
                with gr.Row():
                    copy_to_paste_btn = gr.Button("Copy as text (to Paste tab)", variant="secondary", size="sm")
                    validate_btn_edit = gr.Button("Validate script", variant="secondary", size="sm")
        
        script_validation_msg = gr.Textbox(
            label="Validation",
            interactive=False,
            lines=2,
            placeholder="Click \"Validate script\" to check format.",
        )
        
        with gr.Row():
            # Left column: URL, voice, settings, generate
            with gr.Column(scale=1):
                youtube_url = gr.Textbox(
                    label="YouTube URL",
                    placeholder="https://www.youtube.com/watch?v=...",
                    info="Paste the full YouTube video URL"
                )
                
                voice_dropdown = gr.Dropdown(
                    label="Voice",
                    choices=voices,
                    value=default_voice,
                    info="Select TTS voice for voiceover"
                )
                with gr.Accordion("Preview voice", open=False):
                    voice_preview = gr.Audio(
                        label="Voice Preview",
                        show_label=False,
                        autoplay=True,
                    )
                
                with gr.Accordion("Settings", open=False):
                    resolution_dropdown = gr.Dropdown(
                        label="Output Resolution",
                        choices=list(RESOLUTION_PRESETS.keys()),
                        value=DEFAULT_RESOLUTION,
                        info="Higher resolution = better quality but larger file size"
                    )
                    add_captions_checkbox = gr.Checkbox(
                        label="Add captions",
                        value=True,
                        info="Word-by-word captions on the video",
                    )
                    format_ts_btn = gr.Button("Format timestamps in script", variant="secondary", size="sm")
                
                with gr.Row():
                    process_btn = gr.Button("Generate Short", variant="primary", size="lg")
            
            # Right column: Output (slightly larger for preview)
            with gr.Column(scale=1.2):
                status_text = gr.Textbox(
                    label="Status",
                    interactive=False,
                    placeholder="Ready to process..."
                )
                
                video_output = gr.Video(
                    label="Preview",
                    height=400,
                )
                
                download_btn = gr.DownloadButton(
                    "Download Video",
                    visible=False,
                    variant="primary",
                    size="lg",
                )
        
        # Instructions accordion
        with gr.Accordion("How to Use", open=False):
            gr.Markdown("""
### Step 1: Prepare Your Script
Use Gemini or another AI to analyze a YouTube video and create a script with timestamps.

### Step 2: Format Your Script
Paste a tab-separated table with 3-4 columns:
- **Script**: The text to be spoken
- **Footage Timestamp**: Where to pull footage from the source video (e.g., `[23:45]`)
- **Still/Video**: Use `Still` for a freeze frame with Ken Burns effect, or `Video` for motion footage
- **Layout** (optional): Choose how wide shots are handled:
  - `crop` (default): Face-centered intelligent crop - best for talking heads and close-ups
  - `letterbox`: Blur background with full-width video - best for wide shots, landscapes, groups, or screen recordings

### Step 3: Generate
1. Paste the YouTube URL
2. Paste your script table
3. Select a voice
4. Choose output resolution (1080p recommended)
5. Click "Generate Short"

### Output
- Format: MP4 (9:16 vertical, resolution based on selection)
- Duration: Based on your script length
- Video clips are cropped to focus on detected faces/subjects (or letterboxed if specified)

### Layout Tips
- Use **crop** when there's a clear subject (person, face) you want to focus on
- Use **letterbox** when the full scene matters (wide establishing shots, text on screen, multiple subjects)
""")
        
        # Wire up events
        sample_btn.click(
            fn=load_sample_script,
            inputs=[],
            outputs=[paste_script],
        )
        def load_into_table_and_clear_selection(paste_text):
            rows = load_paste_into_table(paste_text)
            return rows, None  # clear selected row so insert/delete use sensible defaults

        load_into_table_btn.click(
            fn=load_into_table_and_clear_selection,
            inputs=[paste_script],
            outputs=[edit_dataframe, selected_row_state],
        )
        copy_to_paste_btn.click(
            fn=copy_table_to_paste,
            inputs=[edit_dataframe],
            outputs=[paste_script],
        )
        validate_btn.click(
            fn=validate_script_display,
            inputs=[paste_script, edit_dataframe],
            outputs=[script_validation_msg],
        )
        validate_btn_edit.click(
            fn=validate_script_display,
            inputs=[paste_script, edit_dataframe],
            outputs=[script_validation_msg],
        )
        format_ts_btn.click(
            fn=format_timestamps_click,
            inputs=[paste_script],
            outputs=[paste_script],
        )
        
        # Script table: track selected row when user clicks a cell
        def on_table_select(evt: gr.SelectData):
            idx = evt.index
            # Gradio sends index as list [row, col], not tuple
            if isinstance(idx, (tuple, list)) and len(idx) >= 1:
                return int(idx[0])
            if isinstance(idx, int):
                return idx
            return 0
        
        edit_dataframe.select(fn=on_table_select, outputs=[selected_row_state])
        
        add_row_end_btn.click(
            fn=table_add_row_at_end,
            inputs=[edit_dataframe],
            outputs=[edit_dataframe],
        )
        insert_above_btn.click(
            fn=table_insert_row_above,
            inputs=[edit_dataframe, selected_row_state],
            outputs=[edit_dataframe, selected_row_state],
        )
        insert_below_btn.click(
            fn=table_insert_row_below,
            inputs=[edit_dataframe, selected_row_state],
            outputs=[edit_dataframe, selected_row_state],
        )
        delete_row_btn.click(
            fn=table_delete_row,
            inputs=[edit_dataframe, selected_row_state],
            outputs=[edit_dataframe, selected_row_state],
        )
        
        # Auto-preview voice when selection changes
        voice_dropdown.change(
            fn=preview_voice,
            inputs=[voice_dropdown],
            outputs=[voice_preview]
        )
        
        process_btn.click(
            fn=process_video_with_source,
            inputs=[youtube_url, paste_script, edit_dataframe, voice_dropdown, resolution_dropdown, add_captions_checkbox],
            outputs=[video_output, status_text]
        )
        
        # Update download button when video is ready
        def update_download_btn(video_path):
            if video_path:
                return gr.update(value=video_path, visible=True)
            return gr.update(value=None, visible=False)
        
        video_output.change(
            fn=update_download_btn,
            inputs=[video_output],
            outputs=[download_btn]
        )
    
    return app


if __name__ == "__main__":
    print("Starting SmartClipper...")
    print("Open http://127.0.0.1:7860 in your browser")
    app = create_ui()
    app.launch(server_name="127.0.0.1", server_port=7860)
