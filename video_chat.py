"""
Video Chat System using Ollama
Allows asking questions about YouTube video transcripts using local LLM
"""

import re
import json
import os
from typing import Dict, List, Tuple, Optional
import ollama
from sentence_transformers import SentenceTransformer
import numpy as np


class VideoChat:
    def __init__(self, output_file: str = None, model_name: str = 'llama3.2', video_data: dict = None):
        """
        Initialize the video chat system.
        
        Args:
            output_file: Path to the output.txt file from youtube_extractor (optional)
            model_name: Ollama model name to use (e.g., 'llama3.2', 'mistral', 'phi3')
            video_data: Dictionary with video data (metadata, transcript, transcript_segments) - alternative to output_file
        """
        self.output_file = output_file
        self.model_name = model_name
        self.metadata = {}
        self.transcript = ""
        self.transcript_segments = []
        self.chunks = []
        self.chunk_embeddings = None
        self.embedding_model = None
        
        # Load video data from file or direct data
        if video_data:
            self._load_from_data(video_data)
        elif output_file:
            self._load_video_data()
        else:
            raise ValueError("Either output_file or video_data must be provided")
        
        # Initialize embedding model for semantic search
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Chunk the transcript
        self._chunk_transcript()
        
        # Generate embeddings for chunks
        self._generate_embeddings()
    
    def _load_from_data(self, video_data: dict):
        """Load video data directly from a dictionary."""
        self.metadata = video_data.get('metadata', {})
        self.transcript = video_data.get('transcript', '')
        
        # Convert transcript_segments if available
        segments = video_data.get('transcript_segments', [])
        if segments:
            # Normalize segments to have both 'timestamp' and 'seconds'
            normalized_segments = []
            for segment in segments:
                normalized_seg = segment.copy()
                
                # If segment has 'start' (from Whisper) but no 'timestamp', convert it
                if 'start' in normalized_seg and 'timestamp' not in normalized_seg:
                    seconds = normalized_seg['start']
                    normalized_seg['seconds'] = seconds
                    normalized_seg['timestamp'] = self._seconds_to_timestamp(seconds)
                # If segment has 'timestamp' but no 'seconds', convert it
                elif 'timestamp' in normalized_seg and 'seconds' not in normalized_seg:
                    timestamp = normalized_seg['timestamp']
                    normalized_seg['seconds'] = self._timestamp_to_seconds(timestamp)
                # If segment has 'seconds' but no 'timestamp', convert it
                elif 'seconds' in normalized_seg and 'timestamp' not in normalized_seg:
                    seconds = normalized_seg['seconds']
                    normalized_seg['timestamp'] = self._seconds_to_timestamp(seconds)
                
                normalized_segments.append(normalized_seg)
            
            self.transcript_segments = normalized_segments
        else:
            # If no segments, create empty list
            self.transcript_segments = []
    
    def _seconds_to_timestamp(self, seconds: float) -> str:
        """Convert seconds to HH:MM:SS or MM:SS format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes:02d}:{secs:02d}"
    
    def _timestamp_to_seconds(self, timestamp: str) -> float:
        """Convert timestamp string to seconds."""
        time_parts = timestamp.split(':')
        if len(time_parts) == 3:
            return int(time_parts[0]) * 3600 + int(time_parts[1]) * 60 + int(time_parts[2])
        elif len(time_parts) == 2:
            return int(time_parts[0]) * 60 + int(time_parts[1])
        else:
            return 0.0
    
    def _load_video_data(self):
        """Load and parse the output.txt file to extract metadata and transcript."""
        if not os.path.exists(self.output_file):
            raise FileNotFoundError(f"Output file not found: {self.output_file}")
        
        with open(self.output_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract metadata
        title_match = re.search(r'TITLE:\s*(.+?)(?:\n|$)', content)
        author_match = re.search(r'AUTHOR:\s*(.+?)(?:\n|$)', content)
        length_match = re.search(r'LENGTH:\s*(.+?)(?:\n|$)', content)
        views_match = re.search(r'VIEWS:\s*([\d,]+)', content)
        url_match = re.search(r'URL:\s*(.+?)(?:\n|$)', content)
        
        # Extract description (between DESCRIPTION: and TRANSCRIPT:)
        desc_match = re.search(r'DESCRIPTION:\s*\n-+\n(.*?)(?=\n\nTRANSCRIPT:)', content, re.DOTALL)
        
        self.metadata = {
            'title': title_match.group(1).strip() if title_match else 'N/A',
            'author': author_match.group(1).strip() if author_match else 'N/A',
            'length': length_match.group(1).strip() if length_match else 'N/A',
            'views': views_match.group(1).strip() if views_match else 'N/A',
            'url': url_match.group(1).strip() if url_match else 'N/A',
            'description': desc_match.group(1).strip() if desc_match else 'No description available'
        }
        
        # Extract transcript (between TRANSCRIPT: and TRANSCRIPT WITH TIMESTAMPS:)
        transcript_match = re.search(r'TRANSCRIPT:\s*\n-+\n(.*?)(?=\n\nTRANSCRIPT WITH TIMESTAMPS:)', content, re.DOTALL)
        if transcript_match:
            self.transcript = transcript_match.group(1).strip()
        
        # Extract timestamped segments
        timestamp_section = re.search(r'TRANSCRIPT WITH TIMESTAMPS:\s*\n-+\n(.*?)$', content, re.DOTALL)
        if timestamp_section:
            timestamp_text = timestamp_section.group(1)
            # Parse timestamped segments
            pattern = r'\[(\d{2}:\d{2}:\d{2}|\d{2}:\d{2})\]\s*(.+?)(?=\n\[|\Z)'
            matches = re.findall(pattern, timestamp_text, re.DOTALL)
            for timestamp, text in matches:
                # Convert timestamp to seconds
                time_parts = timestamp.split(':')
                if len(time_parts) == 3:
                    seconds = int(time_parts[0]) * 3600 + int(time_parts[1]) * 60 + int(time_parts[2])
                else:
                    seconds = int(time_parts[0]) * 60 + int(time_parts[1])
                
                self.transcript_segments.append({
                    'timestamp': timestamp,
                    'seconds': seconds,
                    'text': text.strip()
                })
    
    def _chunk_transcript(self, chunk_size: int = 500, overlap: int = 50):
        """
        Split transcript into chunks for better processing.
        
        Args:
            chunk_size: Target number of words per chunk
            overlap: Number of words to overlap between chunks
        """
        words = self.transcript.split()
        chunks = []
        
        i = 0
        while i < len(words):
            chunk_words = words[i:i + chunk_size]
            chunk_text = ' '.join(chunk_words)
            
            # Find the timestamp for this chunk (approximate)
            chunk_start_idx = len(' '.join(words[:i]).split())
            timestamp = self._find_timestamp_for_position(chunk_start_idx)
            
            chunks.append({
                'text': chunk_text,
                'index': len(chunks),
                'timestamp': timestamp,
                'word_start': i,
                'word_end': min(i + chunk_size, len(words))
            })
            
            i += chunk_size - overlap
        
        self.chunks = chunks
    
    def _find_timestamp_for_position(self, word_position: int) -> str:
        """Find approximate timestamp for a word position in the transcript."""
        if not self.transcript_segments:
            return "00:00"
        
        # Count words up to the position
        word_count = 0
        for segment in self.transcript_segments:
            segment_words = len(segment.get('text', '').split())
            if word_count + segment_words >= word_position:
                # Get timestamp, converting from seconds if needed
                if 'timestamp' in segment:
                    return segment['timestamp']
                elif 'seconds' in segment:
                    return self._seconds_to_timestamp(segment['seconds'])
                elif 'start' in segment:
                    return self._seconds_to_timestamp(segment['start'])
                else:
                    return "00:00"
            word_count += segment_words
        
        # Return last timestamp if position is beyond transcript
        if self.transcript_segments:
            last_seg = self.transcript_segments[-1]
            if 'timestamp' in last_seg:
                return last_seg['timestamp']
            elif 'seconds' in last_seg:
                return self._seconds_to_timestamp(last_seg['seconds'])
            elif 'start' in last_seg:
                return self._seconds_to_timestamp(last_seg['start'])
        
        return "00:00"
    
    def _generate_embeddings(self):
        """Generate embeddings for all transcript chunks."""
        if not self.chunks:
            return
        
        chunk_texts = [chunk['text'] for chunk in self.chunks]
        self.chunk_embeddings = self.embedding_model.encode(chunk_texts, show_progress_bar=False)
    
    def is_general_question(self, question: str) -> bool:
        """
        Determine if a question is general (needs overview) or specific.
        
        Args:
            question: The user's question
            
        Returns:
            True if general question, False if specific
        """
        question_lower = question.lower()
        
        # General question indicators
        general_patterns = [
            r'what (is|was|are|were) (this|the) (video|video about|about)',
            r'what (is|was|are|were) (it|they) (about|talking about|discussing)',
            r'summarize',
            r'summary',
            r'overview',
            r'main (points|topics|ideas|themes)',
            r'what (happens|happened)',
            r'tell me about',
            r'explain (this|the) video',
            r'what (is|was) the (purpose|goal|objective)',
        ]
        
        for pattern in general_patterns:
            if re.search(pattern, question_lower):
                return True
        
        return False
    
    def _retrieve_relevant_chunks(self, question: str, top_k: int = 5) -> List[Dict]:
        """
        Retrieve most relevant chunks using semantic search.
        
        Args:
            question: The user's question
            top_k: Number of chunks to retrieve
            
        Returns:
            List of relevant chunks with similarity scores
        """
        if self.chunk_embeddings is None or not self.chunks:
            return []
        
        # Generate embedding for the question
        question_embedding = self.embedding_model.encode([question])
        
        # Calculate cosine similarity
        similarities = np.dot(self.chunk_embeddings, question_embedding.T).flatten()
        
        # Get top-k most similar chunks
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        relevant_chunks = []
        for idx in top_indices:
            chunk = self.chunks[idx].copy()
            chunk['similarity'] = float(similarities[idx])
            relevant_chunks.append(chunk)
        
        return relevant_chunks
    
    def _format_metadata_context(self) -> str:
        """Format metadata as context string."""
        return f"""Video Information:
- Title: {self.metadata['title']}
- Author: {self.metadata['author']}
- Length: {self.metadata['length']}
- Views: {self.metadata['views']}
- Description: {self.metadata['description']}
- URL: {self.metadata['url']}
"""
    
    def _format_chunks_context(self, chunks: List[Dict]) -> str:
        """Format chunks as context string."""
        context_parts = []
        for chunk in chunks:
            timestamp = chunk.get('timestamp', 'N/A')
            text = chunk['text']
            context_parts.append(f"[{timestamp}] {text}")
        
        return "\n\n".join(context_parts)
    
    def ask(self, question: str) -> str:
        """
        Ask a question about the video.
        
        Args:
            question: The user's question
            
        Returns:
            The LLM's answer
        """
        is_general = self.is_general_question(question)
        
        # Build context based on question type
        if is_general:
            # For general questions: use metadata + multiple diverse chunks
            
            # Get diverse chunks (from beginning, middle, end, and relevant ones)
            diverse_chunks = []
            if len(self.chunks) > 0:
                # Beginning chunk
                diverse_chunks.append(self.chunks[0])
                # Middle chunk
                if len(self.chunks) > 2:
                    diverse_chunks.append(self.chunks[len(self.chunks) // 2])
                # End chunk
                diverse_chunks.append(self.chunks[-1])
            
            # Also get some semantically relevant chunks
            relevant_chunks = self._retrieve_relevant_chunks(question, top_k=3)
            for chunk in relevant_chunks:
                if chunk not in diverse_chunks:
                    diverse_chunks.append(chunk)
            
            # Limit to top 5-7 chunks
            diverse_chunks = diverse_chunks[:7]
            
            context = f"""{self._format_metadata_context()}

Relevant transcript segments:
{self._format_chunks_context(diverse_chunks)}
"""
        else:
            # For specific questions: use semantic search to find relevant chunks
            
            relevant_chunks = self._retrieve_relevant_chunks(question, top_k=5)
            
            if not relevant_chunks:
                context = self._format_metadata_context()
            else:
                context = f"""{self._format_metadata_context()}

Relevant transcript segments:
{self._format_chunks_context(relevant_chunks)}
"""
        
        # Build the prompt
        prompt = f"""You are a helpful assistant that answers questions about YouTube videos based on their transcripts.

{context}

Question: {question}

Answer the question based on the video information and transcript segments provided above. Be specific and accurate. If the information is not available in the provided context, say so.
"""
        
        # Query Ollama
        try:
            # Check if model is available
            try:
                models = ollama.list()
                available_models = [m['name'] for m in models.get('models', [])]
                if self.model_name not in available_models:
                    return f"❌ Error: Model '{self.model_name}' is not installed.\n\nPlease install it first:\n  ollama pull {self.model_name}\n\nAvailable models: {', '.join(available_models) if available_models else 'None'}"
            except:
                pass  # Continue anyway, let Ollama handle the error
            
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {
                        'role': 'system',
                        'content': 'You are a helpful assistant that answers questions about YouTube video transcripts accurately and concisely.'
                    },
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ]
            )
            
            answer = response['message']['content']
            return answer.strip()
            
        except Exception as e:
            error_msg = str(e)
            if "connection" in error_msg.lower() or "refused" in error_msg.lower():
                return f"❌ Error: Cannot connect to Ollama.\n\nMake sure Ollama is running:\n  - Install: https://ollama.ai\n  - Start: ollama serve\n  - Or just run: ollama pull {self.model_name}"
            return f"❌ Error querying Ollama: {error_msg}\n\nMake sure Ollama is running and the model '{self.model_name}' is installed.\nRun: ollama pull {self.model_name}"


def main():
    """Main function to run the video chat interface."""
    print("=" * 80)
    print("VIDEO CHAT SYSTEM - Ask Questions About YouTube Videos")
    print("=" * 80)
    
    # Check if output.txt exists
    if not os.path.exists('output.txt'):
        print("❌ Error: output.txt not found!")
        print("   Please run youtube_extractor.py first to generate output.txt")
        return
    
    # Ask for Ollama model
    print("\nAvailable Ollama models (common ones):")
    print("  - llama3.2 (recommended, ~2GB)")
    print("  - mistral (~4GB)")
    print("  - phi3 (~2GB)")
    print("  - llama3.1 (~4GB)")
    
    model_name = input("\nEnter Ollama model name (default: llama3.2): ").strip() or "llama3.2"
    
    try:
        # Initialize the chat system
        chat = VideoChat(output_file='output.txt', model_name=model_name)
        
        print("\n" + "=" * 80)
        print("Chat ready! Ask questions about the video.")
        print("Type 'quit' or 'exit' to stop.")
        print("=" * 80)
        
        # Chat loop
        while True:
            question = input("\n💬 Your question: ").strip()
            
            if not question:
                continue
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            # Get answer
            answer = chat.ask(question)
            print(f"\n🤖 Answer:\n{answer}")
            
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

