"""
Unified Streamlit App for YouTube Video Extraction and Chat
"""

import streamlit as st
import os
from youtube_extractor import extract_youtube_data, format_time
from video_chat import VideoChat

# Page configuration
st.set_page_config(
    page_title="YouTube Video Chat",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .video-info-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .stButton>button {
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'video_data' not in st.session_state:
    st.session_state.video_data = None
if 'chat_system' not in st.session_state:
    st.session_state.chat_system = None
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'chat_initialized' not in st.session_state:
    st.session_state.chat_initialized = False
if 'current_url' not in st.session_state:
    st.session_state.current_url = None


def main():
    # Header
    st.markdown('<h1 class="main-header">🎥 YouTube Video Chat System</h1>', unsafe_allow_html=True)
    st.markdown("### Extract video transcripts and chat with them using AI")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Ollama model selection
        model_name = st.selectbox(
            "Select Ollama Model",
            options=['llama3.2', 'mistral', 'phi3', 'llama3.1', 'llama2', 'codellama'],
            index=0,
            help="Make sure the model is installed: ollama pull <model_name>"
        )
        
        st.markdown("---")
        st.markdown("### 📋 Instructions")
        st.markdown("""
        1. Paste a YouTube URL below
        2. Click "Extract Video Data"
        3. View metadata and transcript
        4. Click "Start Chat" to ask questions
        """)
        
        if st.button("🗑️ Clear All Data", type="secondary"):
            st.session_state.video_data = None
            st.session_state.chat_system = None
            st.session_state.messages = []
            st.session_state.chat_initialized = False
            st.session_state.current_url = None
            st.rerun()
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["📥 Extract Video", "📄 View Data", "💬 Chat"])
    
    # Tab 1: Extract Video
    with tab1:
        st.header("Extract Video Data")
        st.markdown("Paste a YouTube video URL to extract its metadata and transcript")
        
        url = st.text_input(
            "YouTube Video URL",
            value=st.session_state.current_url or "",
            placeholder="https://www.youtube.com/watch?v=...",
            help="Supports various YouTube URL formats"
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            extract_button = st.button("🔍 Extract Video Data", type="primary", use_container_width=True)
        
        if extract_button:
            if not url:
                st.error("❌ Please enter a YouTube URL")
            else:
                with st.spinner("Extracting video data... This may take a few minutes for transcription."):
                    try:
                        # Extract video data (always use audio transcription)
                        video_data = extract_youtube_data(
                            url, 
                            use_audio_transcription=True, 
                            whisper_model='base', 
                            language='en'
                        )
                        st.session_state.video_data = video_data
                        st.session_state.current_url = url
                        st.session_state.chat_initialized = False  # Reset chat when new video is loaded
                        st.session_state.messages = []  # Clear chat history
                        st.session_state.chat_system = None
                        
                        st.success("✅ Video data extracted successfully!")
                        st.info("👉 Go to the 'View Data' tab to see the metadata and transcript, then use 'Chat' tab to ask questions.")
                        
                    except Exception as e:
                        st.error(f"❌ Error extracting video: {str(e)}")
                        st.info("💡 Make sure the video is publicly accessible and has audio for transcription. Also ensure FFmpeg is installed.")
        
        # Show status
        if st.session_state.video_data:
            st.success(f"✅ Video loaded: {st.session_state.video_data['metadata']['title']}")
    
    # Tab 2: View Data
    with tab2:
        st.header("Video Information & Transcript")
        
        if not st.session_state.video_data:
            st.info("👈 Please extract a video first using the 'Extract Video' tab.")
        else:
            video_data = st.session_state.video_data
            metadata = video_data['metadata']
            
            # Video Metadata
            st.subheader("📹 Video Metadata")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**Title:** {metadata.get('title', 'N/A')}")
                st.markdown(f"**Author:** {metadata.get('author', 'N/A')}")
                st.markdown(f"**Length:** {metadata.get('length', 'N/A')}")
            
            with col2:
                views = metadata.get('views', 0)
                if isinstance(views, (int, float)) and views > 0:
                    st.markdown(f"**Views:** {views:,}")
                else:
                    st.markdown(f"**Views:** N/A")
                st.markdown(f"**URL:** {video_data.get('url', 'N/A')}")
            
            st.markdown("**Description:**")
            description = metadata.get('description', 'No description available')
            st.text_area("", value=description, height=100, disabled=True, label_visibility="collapsed")
            
            st.markdown("---")
            
            # Transcript
            st.subheader("📜 Transcript")
            
            if video_data.get('transcript'):
                # Show transcript statistics
                transcript = video_data['transcript']
                segments = video_data.get('transcript_segments', [])
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Words", len(transcript.split()))
                with col2:
                    st.metric("Total Characters", len(transcript))
                with col3:
                    st.metric("Segments", len(segments) if segments else "N/A")
                
                # Full transcript
                with st.expander("📄 View Full Transcript", expanded=False):
                    st.text_area("", value=transcript, height=400, disabled=True, label_visibility="collapsed")
                
                # Timestamped transcript
                if segments:
                    with st.expander("⏱️ View Transcript with Timestamps", expanded=False):
                        timestamped_text = ""
                        for segment in segments[:100]:  # Show first 100 segments
                            timestamp = segment.get('timestamp', 'N/A')
                            text = segment.get('text', '')
                            timestamped_text += f"[{timestamp}] {text}\n\n"
                        
                        if len(segments) > 100:
                            timestamped_text += f"\n... and {len(segments) - 100} more segments"
                        
                        st.text_area("", value=timestamped_text, height=400, disabled=True, label_visibility="collapsed")
            else:
                st.warning("⚠️ No transcript available for this video.")
            
            # Chat button
            st.markdown("---")
            if st.button("💬 Start Chat with This Video", type="primary", use_container_width=True):
                st.session_state.chat_initialized = False  # Force re-initialization
                st.info("👉 Go to the 'Chat' tab to start asking questions!")
    
    # Tab 3: Chat
    with tab3:
        st.header("💬 Chat with Video")
        
        if not st.session_state.video_data:
            st.info("👈 Please extract a video first using the 'Extract Video' tab.")
        else:
            # Initialize chat system if not already done
            if not st.session_state.chat_initialized or st.session_state.chat_system is None:
                with st.spinner("Initializing chat system... This may take a moment to load the embedding model."):
                    try:
                        chat_system = VideoChat(
                            video_data=st.session_state.video_data,
                            model_name=model_name
                        )
                        st.session_state.chat_system = chat_system
                        st.session_state.chat_initialized = True
                        st.success("✅ Chat system ready!")
                    except Exception as e:
                        st.error(f"❌ Error initializing chat: {str(e)}")
                        st.info("💡 Make sure Ollama is running and the model is installed.")
                        st.stop()
            
            # Show video info
            if st.session_state.chat_system:
                metadata = st.session_state.video_data['metadata']
                with st.expander("📹 Current Video", expanded=False):
                    title = metadata.get('title', 'N/A')
                    author = metadata.get('author', 'N/A')
                    length = metadata.get('length', 'N/A')
                    views = metadata.get('views', 0)
                    views_str = f"{views:,}" if isinstance(views, (int, float)) and views > 0 else "N/A"
                    st.write(f"**{title}** by {author}")
                    st.write(f"Length: {length} | Views: {views_str}")
                
                # Display chat messages
                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])
                        if "metadata" in message:
                            with st.expander("🔍 Retrieval Info"):
                                st.write(f"**Question Type:** {message['metadata'].get('question_type', 'N/A')}")
                                st.write(f"**Chunks Used:** {message['metadata'].get('chunks_used', 0)}")
                
                # Chat input
                if prompt := st.chat_input("Ask a question about the video..."):
                    # Add user message
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    
                    # Display user message
                    with st.chat_message("user"):
                        st.markdown(prompt)
                    
                    # Get answer
                    with st.chat_message("assistant"):
                        with st.spinner("Thinking..."):
                            try:
                                # Determine question type
                                is_general = st.session_state.chat_system.is_general_question(prompt)
                                question_type = "General" if is_general else "Specific"
                                
                                # Get answer
                                answer = st.session_state.chat_system.ask(prompt)
                                
                                # Count chunks used
                                if is_general:
                                    chunks_used = min(7, len(st.session_state.chat_system.chunks))
                                else:
                                    chunks_used = min(5, len(st.session_state.chat_system.chunks))
                                
                                # Display answer
                                st.markdown(answer)
                                
                                # Add to history
                                st.session_state.messages.append({
                                    "role": "assistant",
                                    "content": answer,
                                    "metadata": {
                                        "question_type": question_type,
                                        "chunks_used": chunks_used
                                    }
                                })
                                
                            except Exception as e:
                                error_msg = f"❌ Error: {str(e)}"
                                st.error(error_msg)
                                st.session_state.messages.append({
                                    "role": "assistant",
                                    "content": error_msg
                                })
                
                # Example questions
                with st.expander("💡 Example Questions"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**General Questions:**")
                        st.write("- What is this video about?")
                        st.write("- Summarize the main points")
                        st.write("- What happens in this video?")
                        st.write("- What is the purpose of this video?")
                    with col2:
                        st.write("**Specific Questions:**")
                        st.write("- What did they say about X?")
                        st.write("- Who won the game?")
                        st.write("- What was the result?")
                        st.write("- When did they mention Y?")


if __name__ == "__main__":
    main()
