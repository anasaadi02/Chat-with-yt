# YouTube Video Script Extractor

A Python application that extracts the complete script (transcript), title, and description from YouTube videos.

## Features

- ✅ Extract video title, author, length, and view count
- ✅ Extract video description
- ✅ Extract complete video transcript/script
- ✅ **NEW:** Audio transcription using OpenAI Whisper (works even when YouTube transcripts aren't available)
- ✅ Display transcript with timestamps
- ✅ Save extracted data to a text file
- ✅ Support for various YouTube URL formats
- ✅ Multiple transcription methods with automatic fallback

## Requirements

- Python 3.12
- Dependencies listed in `requirements.txt`
- **FFmpeg** (required for audio processing with Whisper)
  - Windows: Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH
  - macOS: `brew install ffmpeg`
  - Linux: `sudo apt install ffmpeg` (Ubuntu/Debian) or `sudo yum install ffmpeg` (CentOS/RHEL)

## Installation

1. **Clone or download this project**

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Basic Usage

Run the script and enter a YouTube URL when prompted:

```bash
python youtube_extractor.py
```

### Example

```
Enter YouTube video URL: https://www.youtube.com/watch?v=dQw4w9WgXcQ
```

The script will:
1. Extract the video ID from the URL
2. Fetch video metadata (title, description, author, etc.)
3. Ask you to choose a transcription method:
   - **Method 1 (default):** Try YouTube API first (fast, but may not be available)
   - **Method 2:** Use audio transcription with Whisper (slower, but works for all videos)
4. Fetch the complete transcript using your chosen method
5. If Method 1 fails, automatically offer audio transcription as fallback
6. Display all information in a formatted output
7. Optionally save the data to a text file

### Transcription Methods

**YouTube API (Method 1):**
- Fast and accurate
- Only works if the video has captions/subtitles enabled
- Uses YouTube's official transcript API

**Audio Transcription with Whisper (Method 2):**
- Works for ANY video (even without captions)
- More accurate for natural speech
- Slower (requires downloading audio and processing)
- Choose from 5 model sizes: tiny, base, small, medium, large
  - **tiny**: Fastest, least accurate (~1GB RAM)
  - **base**: Balanced, recommended (~1GB RAM)
  - **small**: Better accuracy (~2GB RAM)
  - **medium**: High accuracy (~5GB RAM)
  - **large**: Best accuracy, slowest (~10GB RAM)

### Supported URL Formats

- `https://www.youtube.com/watch?v=VIDEO_ID`
- `https://youtu.be/VIDEO_ID`
- `https://www.youtube.com/embed/VIDEO_ID`
- `https://www.youtube.com/v/VIDEO_ID`

## Output

The script provides:

1. **Console Output:**
   - Video title, author, length, and views
   - Complete description
   - Full transcript text
   - Transcript statistics (segments, words, characters)

2. **File Output (optional):**
   - All the above information
   - Transcript with timestamps for each segment

## Limitations

- **YouTube API method:** Only works with videos that have transcripts/captions available
- **Audio transcription method:** Works for all videos, but requires more time and processing power
- Requires an active internet connection
- Audio transcription requires FFmpeg to be installed
- Some videos may have restricted access (age-restricted, region-locked, etc.)

## Dependencies

- **youtube-transcript-api**: For extracting video transcripts via YouTube API
- **yt-dlp**: For extracting video metadata and downloading audio
- **openai-whisper**: For transcribing audio using AI models

## Troubleshooting

### "No transcript available"
- If using YouTube API method: The video doesn't have captions/subtitles enabled
- **Solution:** Choose Method 2 (audio transcription) or use the automatic fallback option
- Audio transcription works for all videos, even without captions

### "Error downloading audio" or "Error transcribing audio"
- Make sure FFmpeg is installed and accessible in your PATH
- Check your internet connection
- For long videos, audio transcription may take significant time
- Try using a smaller Whisper model (tiny or base) if you're running out of memory

### "Error fetching video metadata"
- Check your internet connection
- Verify the URL is correct and the video is publicly accessible
- The video may be age-restricted or region-locked

## License

This project is open source and available for educational purposes.
