"""
Text-to-Speech engine using edge-tts (Microsoft Edge TTS).

This module uses the free Microsoft Edge TTS API via edge-tts library.
No API key required, works offline after initial voice list fetch.
"""

import asyncio
import wave
from pathlib import Path
from typing import Optional, List, Tuple

import edge_tts

from src.captions import WordTiming


# Available voices (subset of edge-tts voices)
# Full list: edge-tts --list-voices
EDGE_VOICES = {
    "Guy (US English)": "en-US-GuyNeural",
    "Jenny (US English)": "en-US-JennyNeural",
    "Aria (US English)": "en-US-AriaNeural",
    "Davis (US English)": "en-US-DavisNeural",
    "Tony (US English)": "en-US-TonyNeural",
    "Sara (US English)": "en-US-SaraNeural",
    "Ryan (UK English)": "en-GB-RyanNeural",
    "Sonia (UK English)": "en-GB-SoniaNeural",
    "Libby (UK English)": "en-GB-LibbyNeural",
    "Natasha (AU English)": "en-AU-NatashaNeural",
    "William (AU English)": "en-AU-WilliamNeural",
}

# Default voice
DEFAULT_VOICE = "Ryan (UK English)"


def list_voices() -> list[str]:
    """Return list of available voice names for UI dropdown."""
    return list(EDGE_VOICES.keys())


def get_voice_id(voice_name: str) -> str:
    """Get the edge-tts voice ID for a voice name."""
    if voice_name not in EDGE_VOICES:
        voice_name = DEFAULT_VOICE
    return EDGE_VOICES[voice_name]


async def _generate_audio_async(text: str, voice_id: str, output_path: str) -> None:
    """
    Async function to generate TTS audio using edge-tts.
    """
    communicate = edge_tts.Communicate(text, voice_id)
    await communicate.save(output_path)


async def _generate_audio_with_words_async(
    text: str, 
    voice_id: str, 
    output_path: str
) -> List[WordTiming]:
    """
    Generate TTS audio and capture word-level timestamps.
    
    Args:
        text: Text to convert to speech
        voice_id: Edge TTS voice ID
        output_path: Path to save the audio file
        
    Returns:
        List of WordTiming objects with timing info for each word
    """
    # Enable receive_timeout to ensure we get all metadata including word boundaries
    communicate = edge_tts.Communicate(text, voice_id)
    word_timings = []
    
    # Use SubMaker to capture word-level timing data
    submaker = edge_tts.SubMaker()
    
    with open(output_path, "wb") as f:
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                f.write(chunk["data"])
            elif chunk["type"] == "WordBoundary":
                # edge-tts provides offset and duration in 100-nanosecond units
                offset_ns = chunk["offset"]
                duration_ns = chunk["duration"]
                
                # Convert to seconds
                start_sec = offset_ns / 10_000_000
                end_sec = start_sec + (duration_ns / 10_000_000)
                
                word_timings.append(WordTiming(
                    text=chunk["text"],
                    start=start_sec,
                    end=end_sec,
                ))
                
                # Also feed to submaker for backup
                submaker.feed(chunk)
    
    # If no WordBoundary events were received, try to extract from submaker or generate estimates
    if not word_timings:
        # Fallback: generate estimated word timings based on text and audio duration
        # We'll need to get audio duration first, then distribute words evenly
        pass  # Will be handled by caller with actual audio duration
    
    return word_timings


def generate_audio(text: str, voice_name: str, output_path: str) -> float:
    """
    Generate TTS audio from text using edge-tts.
    
    Args:
        text: Text to convert to speech
        voice_name: Voice name from list_voices()
        output_path: Path to save the audio file (MP3 format)
        
    Returns:
        Duration of the generated audio in seconds
    """
    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Get voice ID
    voice_id = get_voice_id(voice_name)
    
    # edge-tts outputs MP3, so adjust path if needed
    if output_path.endswith('.wav'):
        output_path = output_path[:-4] + '.mp3'
    
    # Run async function
    asyncio.run(_generate_audio_async(text, voice_id, output_path))
    
    # Get duration
    return get_audio_duration(output_path)


def _generate_estimated_word_timings(text: str, duration: float) -> List[WordTiming]:
    """
    Generate estimated word timings by distributing words evenly across the audio duration.
    
    This is a fallback when edge-tts doesn't provide WordBoundary events.
    
    Args:
        text: The spoken text
        duration: Total audio duration in seconds
        
    Returns:
        List of WordTiming objects with estimated timing
    """
    import re
    # Split text into words, keeping punctuation attached
    words = text.split()
    if not words:
        return []
    
    # Calculate time per word (with small gaps between words)
    word_count = len(words)
    # Reserve 5% for gaps, distribute rest evenly
    active_time = duration * 0.95
    time_per_word = active_time / word_count
    gap_time = (duration * 0.05) / max(1, word_count - 1)
    
    timings = []
    current_time = 0.0
    
    for i, word in enumerate(words):
        start = current_time
        end = start + time_per_word
        
        timings.append(WordTiming(
            text=word,
            start=start,
            end=end,
        ))
        
        current_time = end + (gap_time if i < word_count - 1 else 0)
    
    return timings


def generate_audio_with_words(
    text: str, 
    voice_name: str, 
    output_path: str
) -> Tuple[float, List[WordTiming]]:
    """
    Generate TTS audio from text and return word-level timestamps.
    
    Args:
        text: Text to convert to speech
        voice_name: Voice name from list_voices()
        output_path: Path to save the audio file (MP3 format)
        
    Returns:
        Tuple of (duration in seconds, list of WordTiming objects)
    """
    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Get voice ID
    voice_id = get_voice_id(voice_name)
    
    # edge-tts outputs MP3, so adjust path if needed
    if output_path.endswith('.wav'):
        output_path = output_path[:-4] + '.mp3'
    
    # Run async function to get audio and word timings
    word_timings = asyncio.run(_generate_audio_with_words_async(text, voice_id, output_path))
    
    # Get duration
    duration = get_audio_duration(output_path)
    
    # Fallback: if no word timings from edge-tts, generate estimates
    if not word_timings:
        word_timings = _generate_estimated_word_timings(text, duration)
    
    return duration, word_timings


def get_audio_duration(audio_path: str) -> float:
    """
    Get the duration of an audio file in seconds.
    
    Args:
        audio_path: Path to the audio file
        
    Returns:
        Duration in seconds
    """
    # Try using mutagen for MP3
    try:
        from mutagen.mp3 import MP3
        audio = MP3(audio_path)
        return audio.info.length
    except Exception:
        pass
    
    # Try WAV
    try:
        with wave.open(audio_path, 'rb') as wav_file:
            frames = wav_file.getnframes()
            rate = wav_file.getframerate()
            return frames / rate
    except Exception:
        pass
    
    # Fallback to mutagen generic
    try:
        from mutagen import File
        audio = File(audio_path)
        if audio is not None:
            return audio.info.length
    except Exception:
        pass
    
    raise ValueError(f"Could not determine duration of {audio_path}")


async def list_all_voices_async() -> list[dict]:
    """
    Get full list of available edge-tts voices.
    
    Returns:
        List of voice dictionaries with Name, ShortName, Gender, Locale
    """
    voices = await edge_tts.list_voices()
    return voices


def list_all_voices() -> list[dict]:
    """
    Get full list of available edge-tts voices (sync wrapper).
    """
    return asyncio.run(list_all_voices_async())
