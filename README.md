# YouTube Video Script Extractor & Chat System

A Python application that extracts the complete script (transcript), title, and description from YouTube videos, and provides an AI-powered chatbot to ask questions about the video content using local LLM (Ollama).

## Features

- ✅ Extract video title, author, length, and view count
- ✅ Extract video description
- ✅ Extract complete video transcript/script
- ✅ **Audio transcription** using OpenAI Whisper (works even when YouTube transcripts aren't available)
- ✅ Display transcript with timestamps
- ✅ Save extracted data to a text file
- ✅ Support for various YouTube URL formats
- ✅ Multiple transcription methods with automatic fallback
- ✅ **AI Chatbot** - Ask questions about video content using local LLM (Ollama)
- ✅ **Semantic Search** - Intelligent chunk retrieval for accurate answers
- ✅ **Streamlit Web Interface** - Beautiful, interactive web app for video extraction and chat

## Requirements

- Python 3.12
- Dependencies listed in `requirements.txt`
- **FFmpeg** (required for audio processing with Whisper)
  - Windows: Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH
  - macOS: `brew install ffmpeg`
  - Linux: `sudo apt install ffmpeg` (Ubuntu/Debian) or `sudo yum install ffmpeg` (CentOS/RHEL)
- **Ollama** (required for chatbot functionality)
  - Download from [ollama.ai](https://ollama.ai)
  - Install a model: `ollama pull llama3.2` (or mistral, phi3, etc.)

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
- **ollama**: Python client for Ollama LLM
- **sentence-transformers**: For semantic search and embeddings
- **streamlit**: For web interface
- **numpy**: For numerical operations

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

## Streamlit Web Interface

The project includes a beautiful Streamlit web app that combines video extraction and chat functionality in one interface.

### Running the Streamlit App

```bash
streamlit run app.py
```

The app will open automatically in your browser (usually at `http://localhost:8501`).

### Features

1. **Video URL Input**: Paste a YouTube URL to extract video data
2. **Metadata Display**: View video title, author, description, length, and views
3. **Transcript Viewer**: Browse the full transcript with timestamps
4. **AI Chatbot**: Ask questions about the video content
5. **Model Selection**: Choose from different Ollama models
6. **Chat History**: View conversation history with retrieval information

### Usage Flow

1. Open the Streamlit app in your browser
2. Paste a YouTube video URL
3. Click "Extract Video Data" to process the video
4. View metadata and transcript
5. Click "Open Chatbot" to start asking questions
6. Select an Ollama model and initialize the chat system
7. Start chatting about the video!

## Video Chat System

The chatbot uses advanced techniques to provide accurate answers about video content.

### How It Works

#### 1. **Data Processing**
- Extracts metadata (title, author, description, etc.) and transcript from the video
- Splits transcript into chunks (500 words each, 50-word overlap)
- Generates embeddings for each chunk using sentence-transformers

#### 2. **Question Classification**
The system automatically detects question types:

**General Questions** (e.g., "What is this video about?"):
- Uses metadata + diverse chunks from different parts of the video
- Provides overview and context
- Retrieves chunks from beginning, middle, and end

**Specific Questions** (e.g., "What did they say about X?"):
- Uses semantic search to find most relevant chunks
- Focuses on specific information
- Still includes metadata for context

#### 3. **Semantic Search**
- Uses cosine similarity to find most relevant transcript chunks
- Embeds both questions and chunks using `all-MiniLM-L6-v2` model
- Retrieves top-k most similar chunks (5 for specific, 7 for general questions)

#### 4. **LLM Processing**
- Sends relevant context (metadata + chunks) to Ollama LLM
- LLM synthesizes answer from the provided context
- Returns accurate, context-aware responses

### Chatbot Logic

```python
# Question Classification
if is_general_question(question):
    # General: metadata + diverse chunks
    context = metadata + chunks_from_beginning + chunks_from_middle + chunks_from_end + relevant_chunks
else:
    # Specific: metadata + semantically similar chunks
    context = metadata + semantic_search(question, top_k=5)

# LLM Processing
answer = ollama.chat(model, context + question)
```

### Example Questions

**General Questions:**
- "What is this video about?"
- "Summarize the main points"
- "What happens in this video?"
- "What is the purpose of this video?"

**Specific Questions:**
- "What did they say about X?"
- "Who won the game?"
- "What was the result?"
- "When did they mention Y?"

### Command-Line Chat Interface

You can also use the chatbot from the command line:

```bash
python video_chat.py
```

This will:
1. Load video data from `output.txt`
2. Initialize the chat system with embeddings
3. Allow you to ask questions interactively

### Chat System Architecture

```
Video Data
    ↓
[Metadata] + [Transcript]
    ↓
[Chunking] → [Embeddings] → [Vector Store]
    ↓
[Question] → [Classification] → [Retrieval]
    ↓
[Context] + [Question] → [Ollama LLM] → [Answer]
```

### Configuration

**Ollama Models:**
- `llama3.2` (recommended, ~2GB) - Good balance of speed and quality
- `mistral` (~4GB) - High quality
- `phi3` (~2GB) - Fast and efficient
- `llama3.1` (~4GB) - Latest Llama model

**Chunking Parameters:**
- Chunk size: 500 words
- Overlap: 50 words
- Adjustable in `video_chat.py`

**Retrieval Parameters:**
- General questions: 7 chunks (diverse selection)
- Specific questions: 5 chunks (semantic search)

## License

This project is open source and available for educational purposes.
