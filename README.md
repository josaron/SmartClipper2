# SmartClipper: YouTube Shorts Generator

A local Python app that creates 45-75 second vertical shorts from long-form YouTube videos using TTS voiceover and intelligent face-tracking crop.

## Features

- Download YouTube videos via yt-dlp
- Generate voiceover using Piper TTS (offline, high quality)
- Intelligent face/pose detection for 9:16 cropping (MediaPipe)
- Speed-adjust video clips to match voiceover duration
- Simple Gradio web interface

## Installation

1. **Install FFmpeg** (required for video processing):
   ```bash
   # macOS
   brew install ffmpeg
   
   # Ubuntu/Debian
   sudo apt install ffmpeg
   
   # Windows (via chocolatey)
   choco install ffmpeg
   ```

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download a Piper voice model** (first run will auto-download):
   The app uses `en_US-lessac-medium` by default. Voice models are cached in `~/.local/share/piper/`.

## Usage

1. **Start the app**:
   ```bash
   python app.py
   ```

2. **Open the web interface** at `http://localhost:7860`

3. **Provide inputs**:
   - YouTube URL of the source video
   - Script/timestamp table in 3-column format (paste from spreadsheet)
   - Select a voice for the voiceover

4. **Click "Generate Short"** and wait for processing

5. **Preview and download** the final video

## Script Table Format

The script table should be tab-separated with 3 columns:

| Time | Script/Voiceover | Footage Timestamp |
|------|------------------|-------------------|
| 00:00 | Your voiceover text here... | [23:23] |
| 00:06 | Next segment of voiceover... | [24:02] |

- **Time**: When this segment starts in the final video (used for reference)
- **Script/Voiceover**: The text to be spoken
- **Footage Timestamp**: `[MM:SS]` timestamp to pull from source video

## Output

- Format: MP4 (H.264 + AAC)
- Dimensions: 720x1280 (9:16 vertical)
- Duration: 45-75 seconds (based on your script)
- No original audio from source video

## Project Structure

```
SmartClipper2/
├── app.py                 # Main Gradio app
├── requirements.txt       # Dependencies
├── src/
│   ├── downloader.py     # YouTube download
│   ├── tts.py            # Text-to-speech
│   ├── clip_extractor.py # Video clip extraction
│   ├── cropper.py        # Face detection + crop
│   ├── compositor.py     # Final video assembly
│   └── utils.py          # Parsing utilities
├── temp/                 # Temporary files (auto-cleaned)
└── output/               # Final exported videos
```
