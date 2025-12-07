"""
Streamlit Web App for Video Chat System
"""

import streamlit as st
import os
from video_chat import VideoChat

# Page configuration
st.set_page_config(
    page_title="Video Chat - Ask Questions About YouTube Videos",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .video-info {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #e3f2fd;
        margin-left: 20%;
    }
    .assistant-message {
        background-color: #f5f5f5;
        margin-right: 20%;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chat_system' not in st.session_state:
    st.session_state.chat_system = None
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'initialized' not in st.session_state:
    st.session_state.initialized = False


def initialize_chat_system(model_name: str):
    """Initialize the video chat system."""
    try:
        with st.spinner("Loading video data and initializing chat system..."):
            chat = VideoChat(output_file='output.txt', model_name=model_name)
            st.session_state.chat_system = chat
            st.session_state.initialized = True
            return True, None
    except FileNotFoundError:
        return False, "output.txt not found! Please run youtube_extractor.py first."
    except Exception as e:
        return False, f"Error initializing: {str(e)}"


def main():
    # Header
    st.markdown('<h1 class="main-header">🎥 Video Chat System</h1>', unsafe_allow_html=True)
    st.markdown("### Ask questions about your YouTube video transcript")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Check if output.txt exists
        if not os.path.exists('output.txt'):
            st.error("❌ output.txt not found!")
            st.info("Please run `python youtube_extractor.py` first to generate the transcript.")
            st.stop()
        
        # Model selection
        model_name = st.selectbox(
            "Select Ollama Model",
            options=['llama3.2', 'mistral', 'phi3', 'llama3.1', 'llama2', 'codellama'],
            index=0,
            help="Make sure the model is installed: ollama pull <model_name>"
        )
        
        # Initialize button
        if st.button("🔄 Initialize Chat System", type="primary"):
            success, error = initialize_chat_system(model_name)
            if success:
                st.success("✅ Chat system initialized!")
                st.session_state.messages = []  # Clear previous messages
            else:
                st.error(f"❌ {error}")
        
        # Show initialization status
        if st.session_state.initialized:
            st.success("✅ System Ready")
            if st.session_state.chat_system:
                st.info(f"**Model:** {st.session_state.chat_system.model_name}")
                st.info(f"**Chunks:** {len(st.session_state.chat_system.chunks)}")
        else:
            st.warning("⚠️ Click 'Initialize Chat System' to start")
        
        st.markdown("---")
        
        # Video metadata display
        if st.session_state.chat_system:
            st.header("📹 Video Info")
            metadata = st.session_state.chat_system.metadata
            st.write(f"**Title:** {metadata['title']}")
            st.write(f"**Author:** {metadata['author']}")
            st.write(f"**Length:** {metadata['length']}")
            st.write(f"**Views:** {metadata['views']}")
            
            if st.button("🗑️ Clear Chat History"):
                st.session_state.messages = []
                st.rerun()
    
    # Main content area
    if not st.session_state.initialized or st.session_state.chat_system is None:
        st.info("👈 Please initialize the chat system using the sidebar configuration.")
        st.markdown("""
        ### Getting Started:
        1. Make sure you have `output.txt` (run `youtube_extractor.py` first)
        2. Install Ollama: https://ollama.ai
        3. Pull a model: `ollama pull llama3.2`
        4. Select model and click "Initialize Chat System" in the sidebar
        """)
    else:
        # Display chat messages
        chat_container = st.container()
        
        with chat_container:
            # Show video metadata at the top
            metadata = st.session_state.chat_system.metadata
            with st.expander("📹 Video Information", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Title:** {metadata['title']}")
                    st.write(f"**Author:** {metadata['author']}")
                    st.write(f"**Length:** {metadata['length']}")
                with col2:
                    st.write(f"**Views:** {metadata['views']}")
                    st.write(f"**URL:** {metadata['url']}")
                st.write(f"**Description:** {metadata['description']}")
            
            # Display chat history
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    if "metadata" in message:
                        with st.expander("🔍 Retrieval Info"):
                            st.write(f"**Question Type:** {message['metadata'].get('question_type', 'N/A')}")
                            st.write(f"**Chunks Used:** {message['metadata'].get('chunks_used', 0)}")
            
            # Chat input
            if prompt := st.chat_input("Ask a question about the video..."):
                # Add user message to chat history
                st.session_state.messages.append({"role": "user", "content": prompt})
                
                # Display user message
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                # Get answer from chat system
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        try:
                            # Determine question type for metadata
                            is_general = st.session_state.chat_system.is_general_question(prompt)
                            question_type = "General" if is_general else "Specific"
                            
                            # Get answer
                            answer = st.session_state.chat_system.ask(prompt)
                            
                            # Count chunks that would be used (for display)
                            if is_general:
                                chunks_used = min(7, len(st.session_state.chat_system.chunks))
                            else:
                                chunks_used = min(5, len(st.session_state.chat_system.chunks))
                            
                            # Display answer
                            st.markdown(answer)
                            
                            # Add assistant message to chat history
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

