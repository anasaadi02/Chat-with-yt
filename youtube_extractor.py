"""
YouTube Video Script Extractor
Extracts title, description, and transcript from YouTube videos.
"""

import re
import os
import tempfile
import urllib.request
import yt_dlp
import whisper
from youtube_transcript_api import YouTubeTranscriptApi


def extract_video_id(url):
    """
    Extract video ID from various YouTube URL formats.
    
    Args:
        url (str): YouTube video URL
        
    Returns:
        str: Video ID or None if invalid
    """
    patterns = [
        r'(?:youtube\.com\/watch\?v=|youtu\.be\/)([^&\n?#]+)',
        r'youtube\.com\/embed\/([^&\n?#]+)',
        r'youtube\.com\/v\/([^&\n?#]+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    
    return None


def get_video_metadata(url):
    """
    Extract video title and description using yt-dlp.
    
    Args:
        url (str): YouTube video URL
        
    Returns:
        dict: Dictionary containing title and description
    """
    try:
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            
            return {
                'title': info.get('title', 'N/A'),
                'description': info.get('description', 'No description available'),
                'author': info.get('uploader', 'N/A'),
                'length': info.get('duration', 0),
                'views': info.get('view_count', 0)
            }
    except Exception as e:
        raise Exception(f"Error fetching video metadata: {str(e)}")


def get_video_transcript(video_id):
    """
    Extract video transcript using youtube-transcript-api.
    Tries multiple languages and methods to get the transcript.
    
    Args:
        video_id (str): YouTube video ID
        
    Returns:
        tuple: (full_transcript, transcript_segments)
    """
    # Try to get transcript using youtube-transcript-api
    try:
        # First, list available transcripts
        transcript_list_obj = YouTubeTranscriptApi.list_transcripts(video_id)
        
        # Try to get manually created transcript first (more accurate)
        try:
            transcript = transcript_list_obj.find_manually_created_transcript(['en', 'en-US', 'en-GB'])
            transcript_list = transcript.fetch()
        except:
            # If no manual transcript, try auto-generated
            try:
                transcript = transcript_list_obj.find_generated_transcript(['en', 'en-US', 'en-GB'])
                transcript_list = transcript.fetch()
            except:
                # Try any available language
                try:
                    transcript = transcript_list_obj.find_transcript(['en'])
                    transcript_list = transcript.fetch()
                except:
                    # List all available transcripts for better error message
                    available = []
                    for t in transcript_list_obj:
                        available.append(f"{t.language} ({'manual' if t.is_generated == False else 'auto-generated'})")
                    raise Exception(f"No English transcript found. Available transcripts: {', '.join(available) if available else 'None'}")
        
        # Combine all transcript segments into a single text
        full_transcript = ' '.join([segment['text'] for segment in transcript_list])
        
        return full_transcript, transcript_list
        
    except Exception as api_error:
        # Fallback: Try using yt-dlp to get subtitles
        try:
            return get_transcript_via_ytdlp(video_id)
        except Exception as ytdlp_error:
            error_msg = str(api_error)
            if "no element found" in error_msg.lower() or "xml" in error_msg.lower():
                error_msg = "YouTube API parsing error (possibly due to API changes or network issues)"
            raise Exception(f"Error fetching transcript: {error_msg}. Fallback method also failed: {str(ytdlp_error)}")


def get_transcript_via_ytdlp(video_id):
    """
    Fallback method to extract transcript using yt-dlp.
    
    Args:
        video_id (str): YouTube video ID
        
    Returns:
        tuple: (full_transcript, transcript_segments)
    """
    url = f"https://www.youtube.com/watch?v={video_id}"
    
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'skip_download': True,
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        
        # Try to get subtitles from info
        subtitles = info.get('subtitles', {})
        auto_captions = info.get('automatic_captions', {})
        
        # Prefer manual subtitles over auto-generated
        all_subs = {**subtitles, **auto_captions}
        
        if all_subs:
            # Try different language codes
            lang_codes = ['en', 'en-US', 'en-GB', 'en-CA', 'en-AU']
            
            # Also try any available language if English not found
            if not any(lang in all_subs for lang in lang_codes):
                lang_codes = list(all_subs.keys())
            
            for lang in lang_codes:
                if lang in all_subs and all_subs[lang]:
                    try:
                        sub_url = all_subs[lang][0]['url']
                        with urllib.request.urlopen(sub_url) as response:
                            sub_data = response.read().decode('utf-8')
                            
                        # Parse VTT or SRT format
                        transcript_segments = parse_subtitle_data(sub_data)
                        if transcript_segments:
                            full_transcript = ' '.join([seg['text'] for seg in transcript_segments])
                            return full_transcript, transcript_segments
                    except Exception as e:
                        continue  # Try next language
    
    raise Exception("No subtitles found via yt-dlp")


def parse_subtitle_data(sub_data):
    """
    Parse subtitle data (VTT or SRT format) into segments.
    
    Args:
        sub_data (str): Raw subtitle data
        
    Returns:
        list: List of transcript segments with 'text' and 'start' keys
    """
    segments = []
    
    # Try VTT format first
    vtt_pattern = r'(\d{2}:\d{2}:\d{2}\.\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}\.\d{3})\s*\n(.*?)(?=\n\n|\n\d{2}:|\Z)'
    matches = re.findall(vtt_pattern, sub_data, re.DOTALL | re.MULTILINE)
    
    if matches:
        for match in matches:
            start_time = match[0]
            text = match[2].strip().replace('\n', ' ')
            # Remove VTT tags
            text = re.sub(r'<[^>]+>', '', text)
            if text:
                # Convert time to seconds
                time_parts = start_time.split(':')
                seconds = float(time_parts[0]) * 3600 + float(time_parts[1]) * 60 + float(time_parts[2])
                segments.append({'text': text, 'start': seconds})
    else:
        # Try SRT format
        srt_pattern = r'(\d+)\s*\n(\d{2}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,.]\d{3})\s*\n(.*?)(?=\n\n|\n\d+\s*\n|\Z)'
        matches = re.findall(srt_pattern, sub_data, re.DOTALL | re.MULTILINE)
        
        for match in matches:
            start_time = match[1].replace(',', '.')
            text = match[3].strip().replace('\n', ' ')
            if text:
                # Convert time to seconds
                time_parts = start_time.split(':')
                seconds = float(time_parts[0]) * 3600 + float(time_parts[1]) * 60 + float(time_parts[2])
                segments.append({'text': text, 'start': seconds})
    
    return segments


def download_audio(url, output_path=None):
    """
    Download audio from YouTube video using yt-dlp.
    
    Args:
        url (str): YouTube video URL
        output_path (str): Optional path to save audio file. If None, uses temp file.
        
    Returns:
        str: Path to downloaded audio file
    """
    if output_path is None:
        # Create a temporary file
        temp_dir = tempfile.gettempdir()
        video_id = extract_video_id(url)
        output_path = os.path.join(temp_dir, f"youtube_audio_{video_id}")
    
    # Ensure output_path doesn't have extension (yt-dlp will add it)
    base_path = output_path
    if base_path.endswith('.mp3'):
        base_path = base_path[:-4]
    
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': base_path + '.%(ext)s',
        'quiet': True,
        'no_warnings': True,
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        
        # yt-dlp should create the file with .mp3 extension
        output_path = base_path + '.mp3'
        
        # Check if file exists
        if not os.path.exists(output_path):
            # Try to find the file with different extensions
            for ext in ['.mp3', '.m4a', '.webm', '.opus']:
                test_path = base_path + ext
                if os.path.exists(test_path):
                    output_path = test_path
                    break
        
        if not os.path.exists(output_path):
            raise Exception("Audio file was not created successfully")
        
        return output_path
    except Exception as e:
        raise Exception(f"Error downloading audio: {str(e)}")


def transcribe_audio_with_whisper(audio_path, model_size='base', language='en'):
    """
    Transcribe audio file using OpenAI Whisper.
    
    Args:
        audio_path (str): Path to audio file
        model_size (str): Whisper model size ('tiny', 'base', 'small', 'medium', 'large')
                         Larger models are more accurate but slower
        language (str): Language code (e.g., 'en' for English)
        
    Returns:
        tuple: (full_transcript, transcript_segments)
    """
    try:
        print(f"Loading Whisper model '{model_size}'...")
        model = whisper.load_model(model_size)
        
        print("Transcribing audio (this may take a while)...")
        result = model.transcribe(audio_path, language=language)
        
        # Extract segments
        transcript_segments = []
        for segment in result['segments']:
            transcript_segments.append({
                'text': segment['text'].strip(),
                'start': segment['start']
            })
        
        # Combine into full transcript
        full_transcript = ' '.join([seg['text'] for seg in transcript_segments])
        
        return full_transcript, transcript_segments
        
    except Exception as e:
        raise Exception(f"Error transcribing audio: {str(e)}")


def get_transcript_via_audio_transcription(video_id, url, model_size='base', language='en', keep_audio=False):
    """
    Download audio and transcribe it using Whisper as a fallback method.
    
    Args:
        video_id (str): YouTube video ID
        url (str): YouTube video URL
        model_size (str): Whisper model size
        language (str): Language code
        keep_audio (bool): Whether to keep the audio file after transcription
        
    Returns:
        tuple: (full_transcript, transcript_segments)
    """
    audio_path = None
    try:
        print("Downloading audio from video...")
        audio_path = download_audio(url)
        print(f"Audio downloaded to: {audio_path}")
        
        # Transcribe the audio
        full_transcript, transcript_segments = transcribe_audio_with_whisper(
            audio_path, model_size=model_size, language=language
        )
        
        return full_transcript, transcript_segments
        
    finally:
        # Clean up audio file if requested
        if audio_path and os.path.exists(audio_path) and not keep_audio:
            try:
                os.remove(audio_path)
                print(f"Cleaned up temporary audio file: {audio_path}")
            except:
                pass


def format_time(seconds):
    """Convert seconds to HH:MM:SS format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes:02d}:{secs:02d}"


def extract_youtube_data(url, use_audio_transcription=False, whisper_model='base', language='en'):
    """
    Main function to extract all data from a YouTube video.
    
    Args:
        url (str): YouTube video URL
        use_audio_transcription (bool): If True, skip API transcript and use audio transcription
        whisper_model (str): Whisper model size if using audio transcription
        language (str): Language code for transcription
        
    Returns:
        dict: Dictionary containing all extracted data
    """
    # Validate URL and extract video ID
    video_id = extract_video_id(url)
    if not video_id:
        raise ValueError("Invalid YouTube URL. Please provide a valid YouTube video link.")
    
    print(f"Extracting data for video ID: {video_id}")
    print("-" * 80)
    
    # Get metadata
    print("Fetching video metadata...")
    metadata = get_video_metadata(url)
    
    # Get transcript
    full_transcript = None
    transcript_segments = None
    
    if use_audio_transcription:
        print("Using audio transcription method...")
        try:
            full_transcript, transcript_segments = get_transcript_via_audio_transcription(
                video_id, url, model_size=whisper_model, language=language
            )
            print("✅ Audio transcription completed successfully!")
        except Exception as e:
            print(f"Error: {str(e)}")
            raise
    else:
        print("Fetching video transcript from YouTube API...")
        try:
            full_transcript, transcript_segments = get_video_transcript(video_id)
        except Exception as e:
            print(f"Warning: {str(e)}")
            full_transcript = None
            transcript_segments = None
    
    return {
        'video_id': video_id,
        'url': url,
        'metadata': metadata,
        'transcript': full_transcript,
        'transcript_segments': transcript_segments,
        'transcription_method': 'audio_whisper' if use_audio_transcription else 'youtube_api'
    }


def display_results(data):
    """
    Display extracted data in a formatted way.
    
    Args:
        data (dict): Extracted video data
    """
    print("\n" + "=" * 80)
    print("YOUTUBE VIDEO INFORMATION")
    print("=" * 80)
    
    metadata = data['metadata']
    
    print(f"\n📹 TITLE: {metadata['title']}")
    print(f"👤 AUTHOR: {metadata['author']}")
    print(f"⏱️  LENGTH: {format_time(metadata['length'])}")
    print(f"👁️  VIEWS: {metadata['views']:,}")
    
    print(f"\n📝 DESCRIPTION:")
    print("-" * 80)
    print(metadata['description'] if metadata['description'] else "No description available")
    
    if data['transcript']:
        # Show transcription method
        method = data.get('transcription_method', 'unknown')
        method_names = {
            'youtube_api': 'YouTube API',
            'audio_whisper': 'Audio Transcription (Whisper)',
            'audio_whisper_fallback': 'Audio Transcription (Whisper - Fallback)'
        }
        method_display = method_names.get(method, method)
        
        print(f"\n📜 TRANSCRIPT (via {method_display}):")
        print("-" * 80)
        print(data['transcript'])
        
        print(f"\n📊 TRANSCRIPT STATISTICS:")
        print(f"   - Total segments: {len(data['transcript_segments'])}")
        print(f"   - Total words: {len(data['transcript'].split())}")
        print(f"   - Total characters: {len(data['transcript'])}")
    else:
        print("\n⚠️  No transcript available for this video.")
    
    print("\n" + "=" * 80)


def save_to_file(data, filename='youtube_extract.txt'):
    """
    Save extracted data to a text file.
    
    Args:
        data (dict): Extracted video data
        filename (str): Output filename
    """
    with open(filename, 'w', encoding='utf-8') as f:
        metadata = data['metadata']
        
        f.write("=" * 80 + "\n")
        f.write("YOUTUBE VIDEO INFORMATION\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"TITLE: {metadata['title']}\n")
        f.write(f"AUTHOR: {metadata['author']}\n")
        f.write(f"LENGTH: {format_time(metadata['length'])}\n")
        f.write(f"VIEWS: {metadata['views']:,}\n")
        f.write(f"URL: {data['url']}\n")
        
        f.write(f"\nDESCRIPTION:\n")
        f.write("-" * 80 + "\n")
        f.write(metadata['description'] if metadata['description'] else "No description available")
        f.write("\n\n")
        
        if data['transcript']:
            f.write(f"TRANSCRIPT:\n")
            f.write("-" * 80 + "\n")
            f.write(data['transcript'])
            f.write("\n\n")
            
            f.write(f"TRANSCRIPT WITH TIMESTAMPS:\n")
            f.write("-" * 80 + "\n")
            for segment in data['transcript_segments']:
                timestamp = format_time(segment['start'])
                f.write(f"[{timestamp}] {segment['text']}\n")
        else:
            f.write("\nNo transcript available for this video.\n")
    
    print(f"\n✅ Data saved to: {filename}")


def main():
    """Main function to run the YouTube extractor."""
    print("=" * 80)
    print("YOUTUBE VIDEO SCRIPT EXTRACTOR")
    print("=" * 80)
    
    # Get YouTube URL from user
    url = input("\nEnter YouTube video URL: ").strip()
    
    if not url:
        print("❌ Error: No URL provided.")
        return
    
    # Always use audio transcription with Whisper base model
    use_audio = True
    whisper_model = 'base'
    language = 'en'
    
    print("\nUsing audio transcription with Whisper (base model)...")
    
    try:
        # Extract data
        data = extract_youtube_data(url, use_audio_transcription=use_audio, 
                                   whisper_model=whisper_model, language=language)
        
        # Display results
        display_results(data)
        
        # Always save to output.txt automatically
        filename = 'output.txt'
        save_to_file(data, filename)
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")


if __name__ == "__main__":
    main()
