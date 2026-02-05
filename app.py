"""
SmartClipper: YouTube Shorts Generator

A local app that creates 45-75 second vertical shorts from long-form YouTube videos
using TTS voiceover and intelligent face-tracking crop.
"""

import gradio as gr
from pathlib import Path
from datetime import datetime
import traceback
from typing import Optional, Tuple

from src.downloader import download_video
from src.tts import generate_audio, generate_audio_with_words, list_voices, DEFAULT_VOICE
from src.utils import parse_script_table, clean_temp_files, clean_old_outputs
from src.clip_extractor import check_ffmpeg
from src.cropper import extract_crop_and_speed, extract_still_with_ken_burns, KEN_BURNS_STYLES
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

# Sample script for demo (3-column format: Script | Timestamp | Still/Video)
SAMPLE_SCRIPT = """Did you know a 51-foot fire-breathing dragon used to live on the Las Vegas Strip?\t[23:23]\tVideo
This is Murphy. He was the star of the Excalibur Hotel, emerging from a cave every hour to battle Merlin.\t[24:02]\tStill
Built by Disney veterans, Murphy was a hydraulic beast who spent his days submerged in the castle moat.\t[23:37]\tVideo
But the desert wasn't kind. Between constant breakdowns and the family-friendly era ending, the show was axed in 2003.\t[27:44]\tStill
For 20 years, Murphy didn't leave. He was simply locked behind a wall in his dark, underwater cave.\t[28:01]\tVideo
In 2024, his lair was finally sealed for good. Most think he was scrapped, but some say he's still in there...\t[28:22]\tStill
...waiting for the day the magic returns to Vegas. Subscribe for more lost history!\t[31:01]\tVideo"""


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


def process_video(youtube_url: str, script_table: str, voice: str, resolution: str, progress=gr.Progress()):
    """
    Main processing pipeline:
    1. Download YouTube video
    2. Parse script/timestamps
    3. Generate TTS audio for each segment
    4. Extract and crop video clips
    5. Compose final video
    """
    # Validate inputs
    valid, error_msg = validate_inputs(youtube_url, script_table, voice)
    if not valid:
        return None, error_msg
    
    # Parse resolution
    target_width, target_height = RESOLUTION_PRESETS.get(resolution, RESOLUTION_PRESETS[DEFAULT_RESOLUTION])
    
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
            progress(0.45 + (0.35 * i / len(segments)), desc=f"Processing {clip_type} {i+1}/{len(segments)}...")
            
            processed_path = f"temp/clip_processed_{i}.mp4"
            
            if segment.is_still:
                # Extract still frame with Ken Burns effect
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
        
        # Step 5: Compose final video with captions (clips are already speed-adjusted)
        progress(0.85, desc="Composing final video with captions...")
        
        # Generate unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"output/short_{timestamp}.mp4"
        
        compose_final_video_fast_with_captions(
            processed_clips, 
            audio_paths, 
            audio_durations,
            word_timings_list,
            output_path
        )
        
        progress(1.0, desc="Done!")
        
        # Clean up temp files (keep source video)
        clean_temp_files(keep_source=True)
        
        return output_path, f"Video created successfully! Duration: {total_duration:.1f}s"
        
    except Exception as e:
        error_detail = traceback.format_exc()
        print(f"Error: {error_detail}")
        return None, f"Error: {str(e)}"


def load_sample_script():
    """Load the sample script into the textbox."""
    return SAMPLE_SCRIPT


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
    
    with gr.Blocks(title="SmartClipper", theme=gr.themes.Soft()) as app:
        gr.Markdown("# SmartClipper: YouTube Shorts Generator")
        gr.Markdown("Create 45-75 second vertical shorts from long-form YouTube videos with AI voiceover.")
        
        with gr.Row():
            # Left column: Inputs
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
                    info="Select TTS voice for voiceover - changes auto-preview"
                )
                
                resolution_dropdown = gr.Dropdown(
                    label="Output Resolution",
                    choices=list(RESOLUTION_PRESETS.keys()),
                    value=DEFAULT_RESOLUTION,
                    info="Higher resolution = better quality but larger file size"
                )
                
                voice_preview = gr.Audio(
                    label="Voice Preview",
                    show_label=False,
                    autoplay=True,
                )
                
                script_table = gr.Textbox(
                    label="Script / Timestamps",
                    placeholder="Script\tFootage Timestamp\tStill/Video\nYour text here...\t[0:30]\tVideo",
                    lines=12,
                    info="Tab-separated: Script | [Footage Timestamp] | Still/Video"
                )
                
                with gr.Row():
                    sample_btn = gr.Button("Load Sample", variant="secondary", size="sm")
                    process_btn = gr.Button("Generate Short", variant="primary", size="lg")
            
            # Right column: Output
            with gr.Column(scale=1):
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
Paste a tab-separated table with 3 columns:
- **Script**: The text to be spoken
- **Footage Timestamp**: Where to pull footage from the source video (e.g., `[23:45]`)
- **Still/Video**: Use `Still` for a freeze frame with Ken Burns effect, or `Video` for motion footage

### Step 3: Generate
1. Paste the YouTube URL
2. Paste your script table
3. Select a voice
4. Choose output resolution (1080p recommended)
5. Click "Generate Short"

### Output
- Format: MP4 (9:16 vertical, resolution based on selection)
- Duration: Based on your script length
- Video clips are cropped to focus on detected faces/subjects
""")
        
        # Wire up events
        sample_btn.click(
            fn=load_sample_script,
            inputs=[],
            outputs=[script_table]
        )
        
        # Auto-preview voice when selection changes
        voice_dropdown.change(
            fn=preview_voice,
            inputs=[voice_dropdown],
            outputs=[voice_preview]
        )
        
        process_btn.click(
            fn=process_video,
            inputs=[youtube_url, script_table, voice_dropdown, resolution_dropdown],
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
