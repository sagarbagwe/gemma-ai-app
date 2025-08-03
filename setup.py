# File: enhanced_app.py

import streamlit as st
import os
import tempfile
import cv2
import numpy as np
from PIL import Image
import torch
from transformers import TextStreamer
import yt_dlp
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import unsloth first for optimizations
from unsloth import FastModel

# Fix PyTorch recompilation issues - MUST BE SET EARLY
torch._dynamo.config.cache_size_limit = 1000  # Increase cache size
torch._dynamo.config.suppress_errors = True   # Suppress compilation errors
torch._dynamo.reset()  # Reset any existing compilation state

# Page configuration for the Streamlit app
st.set_page_config(
    page_title="Gemma 3N Conversational AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a better look and feel
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .stChatMessage {
        background-color: #f8f9fa;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 1rem;
    }
    .sidebar-section {
        margin-bottom: 1rem;
        padding: 1rem;
        background-color: #f0f2f6;
        border-radius: 8px;
    }
    .error-message {
        background-color: #ffebee;
        color: #c62828;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #c62828;
    }
    .success-message {
        background-color: #e8f5e8;
        color: #2e7d32;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2e7d32;
    }
    .feature-card {
        background-color: #f8f9fa;
        border: 2px solid #e9ecef;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    .feature-card:hover {
        border-color: #667eea;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.15);
    }
    .feature-selected {
        border-color: #667eea !important;
        background-color: #f0f4ff !important;
    }
</style>
""", unsafe_allow_html=True)

# --- SESSION STATE INITIALIZATION ---
def initialize_session_state():
    """Initialize all session state variables"""
    default_states = {
        "model_loaded": False,
        "model": None,
        "tokenizer": None,
        "current_feature": "📝 Text → Text",
        "chat_histories": {
            "📝 Text → Text": [],
            "📸 Image + Text → Text": [],
            "🎥 Video + Text → Text": [],
            "🎵 Audio + Text → Text": [],
            "🎬 Video + Audio → Text": [],
            "🎭 Video + Audio + Text → Text": [],
        },
        "media_content": None,
        "inference_count": 0
    }
    
    for key, default_value in default_states.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

initialize_session_state()

# --- MODEL LOADING FUNCTION ---
@st.cache_resource(show_spinner="🧠 Loading Gemma 3N Model... This may take a few minutes on first run.")
def load_gemma_model():
    """
    Loads and caches the Unsloth model and tokenizer.
    This function runs only once when the app starts.
    """
    try:
        logger.info("Starting model loading process...")
        
        # Load model with optimized settings
        model, tokenizer = FastModel.from_pretrained(
            model_name="unsloth/gemma-3n-E4B-it",
            dtype=None, 
            max_seq_length=2048, 
            load_in_4bit=True,
        )
        
        # Set model to eval mode to prevent training-related recompilations
        model.eval()
        
        # Check if model is already on CUDA (quantized models are automatically placed)
        device = next(model.parameters()).device
        logger.info(f"Model loaded on device: {device}")
        
        # Warm up the model with a dummy input to trigger initial compilation
        try:
            dummy_input = tokenizer("Hello", return_tensors="pt", padding=True)
            device = next(model.parameters()).device
            if "cuda" in str(device):
                dummy_input = {k: v.to(device) for k, v in dummy_input.items()}
            
            with torch.inference_mode():
                _ = model.generate(**dummy_input, max_new_tokens=1, do_sample=False)
            logger.info("Model warmup completed")
        except Exception as warmup_error:
            logger.warning(f"Model warmup failed: {warmup_error}")
        
        logger.info("Model loading completed successfully")
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        st.error(f"Failed to load model: {e}")
        st.stop()

# --- UTILITY FUNCTIONS ---
@st.cache_data(show_spinner="📥 Downloading YouTube video...")
def download_youtube_video(url):
    """Download YouTube video with error handling"""
    try:
        temp_dir = tempfile.gettempdir()
        video_path = os.path.join(temp_dir, f"youtube_{os.urandom(8).hex()}.mp4")
        
        ydl_opts = {
            'format': 'best[ext=mp4][height<=720]/best[ext=mp4]/best',
            'outtmpl': video_path,
            'quiet': True,
            'no_warnings': True,
            'merge_output_format': 'mp4'
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
            
        if os.path.exists(video_path):
            logger.info(f"YouTube video downloaded: {video_path}")
            return video_path
        else:
            raise Exception("Video file not found after download")
            
    except Exception as e:
        logger.error(f"YouTube download failed: {e}")
        st.error(f"YouTube download failed: {e}")
        return None

@st.cache_data(show_spinner="🎬 Extracting video frames...")
def extract_video_frames(video_path, max_frames=8):
    """Extract frames from video with improved error handling"""
    frames = []
    cap = None
    
    try:
        if not os.path.exists(video_path):
            raise Exception("Video file not found")
            
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise Exception("Could not open video file")
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        logger.info(f"Video info: {total_frames} frames, {fps} FPS")
        
        if total_frames == 0:
            raise Exception("Video has no frames")
            
        frame_interval = max(1, total_frames // max_frames)
        count, extracted = 0, 0
        
        while cap.isOpened() and extracted < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            if count % frame_interval == 0:
                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_frame)
                
                # Resize frame if too large
                max_size = 512
                if max(pil_image.size) > max_size:
                    pil_image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                
                frames.append(pil_image)
                extracted += 1
                
            count += 1
            
        logger.info(f"Extracted {len(frames)} frames from video")
        
    except Exception as e:
        logger.error(f"Frame extraction failed: {e}")
        st.error(f"Frame extraction failed: {e}")
        frames = []
        
    finally:
        if cap:
            cap.release()
            
    return frames

def do_gemma_inference(messages, max_new_tokens, temperature):
    """
    Perform model inference with comprehensive error handling and optimization
    """
    try:
        st.session_state.inference_count += 1
        logger.info(f"Starting inference #{st.session_state.inference_count}")
        
        # Validate inputs
        if not messages:
            return "I need a message to respond to."
            
        # Prepare inputs
        inputs = st.session_state.tokenizer.apply_chat_template(
            messages, 
            add_generation_prompt=True, 
            tokenize=True, 
            return_dict=True, 
            return_tensors="pt"
        )
        
        # Move to the same device as the model
        model_device = next(st.session_state.model.parameters()).device
        inputs = {k: v.to(model_device) for k, v in inputs.items()}
        
        input_ids_length = inputs['input_ids'].shape[1]
        
        # Check sequence length
        if input_ids_length > 2048:
            logger.warning(f"Input sequence too long: {input_ids_length}")
            return "Your message is too long. Please try a shorter message."
        
        logger.info(f"Input length: {input_ids_length} tokens")
        
        # Generation configuration
        generation_config = {
            "max_new_tokens": int(max_new_tokens),
            "temperature": float(temperature),
            "do_sample": True if temperature > 0.1 else False,
            "pad_token_id": st.session_state.tokenizer.eos_token_id,
            "eos_token_id": st.session_state.tokenizer.eos_token_id,
            "use_cache": True,
        }
        
        # Perform inference with proper context management
        with torch.inference_mode():
            outputs = st.session_state.model.generate(**inputs, **generation_config)
        
        # Extract and decode new tokens
        new_tokens = outputs[0, input_ids_length:]
        response = st.session_state.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        
        if not response:
            response = "I apologize, but I couldn't generate a response. Please try again."
            
        logger.info(f"Generated response length: {len(response)} characters")
        return response
        
    except torch.cuda.OutOfMemoryError:
        logger.error("CUDA out of memory")
        torch.cuda.empty_cache()
        return "I'm sorry, but I'm running low on memory. Please try a shorter message or restart the app."
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Inference failed: {error_msg}")
        return f"An error occurred during inference: {error_msg}"

# --- MAIN HEADER ---
st.markdown('<h1 class="main-header">🤖 Gemma 3N Conversational AI Assistant</h1>', unsafe_allow_html=True)

# --- SIDEBAR FOR MODEL LOADING AND SETTINGS ---
with st.sidebar:
    st.header("🔧 Model Configuration")
    
    if not st.session_state.model_loaded:
        st.warning("⚠️ Model is not loaded.")
        if st.button("🚀 Load Gemma 3N Model", type="primary", use_container_width=True):
            with st.spinner("Loading model... This may take a few minutes..."):
                try:
                    model, tokenizer = load_gemma_model()
                    st.session_state.model = model
                    st.session_state.tokenizer = tokenizer
                    st.session_state.model_loaded = True
                    st.success("✅ Model loaded successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Model loading failed: {e}")
    else:
        st.success("✅ Model is loaded and ready!")
        
        # System status
        col1, col2 = st.columns(2)
        with col1:
            st.metric("💬 Messages", len(st.session_state.chat_histories[st.session_state.current_feature]))
        with col2:
            st.metric("🔄 Inferences", st.session_state.inference_count)

    st.divider()
    
    # Generation settings
    st.header("⚙️ Generation Settings")
    max_tokens = st.slider("Max New Tokens", 50, 1024, 384, help="Maximum number of tokens to generate")
    temperature = st.slider("Temperature", 0.1, 1.5, 0.9, 0.05, help="Controls randomness: lower = more focused, higher = more creative")
    
    st.divider()
    
    # Instructions
    st.header("💡 How to Use")
    st.info(
        "1. **Load Model**: Click the load button above\n"
        "2. **Choose Mode**: Select your analysis type below\n"
        "3. **Upload Media**: Add files if needed\n"
        "4. **Chat**: Ask questions and get AI responses"
    )
    
    st.divider()
    
    # System info
    st.header("📊 System Info")
    if st.session_state.model_loaded:
        try:
            model_device = next(st.session_state.model.parameters()).device
            if "cuda" in str(model_device):
                gpu_info = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory // (1024**3)
                st.success(f"🖥️ GPU: {gpu_info}\n💾 Memory: {gpu_memory}GB")
            else:
                st.info(f"🖥️ Model Device: {model_device}")
        except Exception as e:
            st.warning(f"⚠️ Device info unavailable: {e}")

# --- MAIN INTERFACE ---
if not st.session_state.model_loaded:
    st.info("👋 Welcome! Please load the Gemma model from the sidebar to begin.")
else:
    # Feature selection
    st.subheader("🎯 Choose Analysis Mode")
    
    features = list(st.session_state.chat_histories.keys())
    
    # Create feature selection cards
    cols = st.columns(3)
    for i, feature in enumerate(features):
        with cols[i % 3]:
            is_selected = st.session_state.current_feature == feature
            card_class = "feature-card feature-selected" if is_selected else "feature-card"
            
            if st.button(
                feature, 
                key=f"feature_{i}",
                use_container_width=True,
                type="primary" if is_selected else "secondary"
            ):
                st.session_state.current_feature = feature
                st.rerun()
    
    st.divider()
    
    # Current feature display
    current_feature = st.session_state.current_feature
    history = st.session_state.chat_histories[current_feature]
    
    st.subheader(f"💬 {current_feature}")
    
    # Clear chat button
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("🗑️ New Chat", use_container_width=True):
            st.session_state.chat_histories[current_feature] = []
            st.session_state.media_content = None
            st.rerun()
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for message in history:
            with st.chat_message(message["role"]):
                # Display images if present
                images = [part["image"] for part in message["content"] if part["type"] == "image"]
                if images:
                    if len(images) == 1:
                        st.image(images[0], caption="📸 Attached Image", width=300)
                    else:
                        with st.expander(f"🖼️ View {len(images)} attached frames"):
                            cols = st.columns(min(len(images), 4))
                            for idx, image in enumerate(images):
                                with cols[idx % 4]:
                                    st.image(image, use_column_width=True, caption=f"Frame {idx+1}")
                
                # Display text content
                text_content = next((part["text"] for part in message["content"] if part["type"] == "text"), "")
                if text_content:
                    st.markdown(text_content)
    
    # Media upload section (only show for new chats or text-only mode)
    if not history or "Text" in current_feature:
        media_uploaded = False
        temp_files = []
        
        # Image upload
        if "Image" in current_feature and not history:
            st.subheader("📸 Upload Image")
            uploaded_image = st.file_uploader(
                "Choose an image file", 
                type=['png', 'jpg', 'jpeg', 'webp', 'bmp'],
                key="image_upload"
            )
            if uploaded_image:
                image = Image.open(uploaded_image)
                st.image(image, caption="Uploaded Image", width=300)
                st.session_state.media_content = {"type": "image", "content": image}
                media_uploaded = True
        
        # Video upload
        if "Video" in current_feature and not history:
            st.subheader("🎬 Upload Video")
            tab1, tab2 = st.tabs(["📁 Upload File", "🌐 YouTube URL"])
            
            with tab1:
                uploaded_video = st.file_uploader(
                    "Choose a video file", 
                    type=['mp4', 'mov', 'avi', 'mkv', 'webm'],
                    key="video_upload"
                )
                if uploaded_video:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
                        tfile.write(uploaded_video.read())
                        video_path = tfile.name
                    temp_files.append(video_path)
                    
                    frames = extract_video_frames(video_path)
                    if frames:
                        st.success(f"✅ Extracted {len(frames)} frames")
                        with st.expander("🔍 Preview Frames"):
                            cols = st.columns(min(4, len(frames)))
                            for idx, frame in enumerate(frames):
                                with cols[idx % 4]:
                                    st.image(frame, use_column_width=True, caption=f"Frame {idx+1}")
                        st.session_state.media_content = {"type": "video", "content": frames}
                        media_uploaded = True
            
            with tab2:
                youtube_url = st.text_input("Enter YouTube URL", key="youtube_input")
                if st.button("🎬 Process YouTube Video") and youtube_url:
                    video_path = download_youtube_video(youtube_url)
                    if video_path:
                        temp_files.append(video_path)
                        frames = extract_video_frames(video_path)
                        if frames:
                            st.success(f"✅ Extracted {len(frames)} frames")
                            st.session_state.media_content = {"type": "video", "content": frames}
                            media_uploaded = True
        
        # Audio upload
        if "Audio" in current_feature and not history:
            st.subheader("🎵 Upload Audio")
            uploaded_audio = st.file_uploader(
                "Choose an audio file", 
                type=['mp3', 'wav', 'ogg', 'm4a', 'flac'],
                key="audio_upload"
            )
            if uploaded_audio:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tfile:
                    tfile.write(uploaded_audio.read())
                    audio_path = tfile.name
                temp_files.append(audio_path)
                st.audio(uploaded_audio, format="audio/mp3")
                st.session_state.media_content = {"type": "audio", "content": audio_path}
                media_uploaded = True
    
    # Chat input
    if prompt := st.chat_input("💭 Type your message here..."):
        if not history and "Text" not in current_feature and not st.session_state.media_content:
            st.warning("Please upload the required media for a new chat.")
        else:
            # Prepare user content
            user_content = [{"type": "text", "text": prompt}]
            
            # Add media content if this is the first message
            if not history and st.session_state.media_content:
                media_type = st.session_state.media_content["type"]
                media_data = st.session_state.media_content["content"]
                
                if media_type == "image":
                    user_content.insert(0, {"type": "image", "image": media_data})
                elif media_type == "video":
                    # Add frames to content
                    frame_content = [{"type": "image", "image": frame} for frame in media_data[:8]]
                    user_content = frame_content + user_content
                elif media_type == "audio":
                    user_content.insert(0, {"type": "audio", "audio": media_data})
            
            # Add user message to history
            st.session_state.chat_histories[current_feature].append({
                "role": "user", 
                "content": user_content
            })
            
            # Display user message
            with st.chat_message("user"):
                if st.session_state.media_content and not history:
                    media_type = st.session_state.media_content["type"]
                    media_data = st.session_state.media_content["content"]
                    
                    if media_type == "image":
                        st.image(media_data, caption="📸 Uploaded Image", width=300)
                    elif media_type == "video":
                        with st.expander(f"🎬 View {len(media_data)} extracted frames"):
                            cols = st.columns(min(4, len(media_data)))
                            for idx, frame in enumerate(media_data):
                                with cols[idx % 4]:
                                    st.image(frame, use_column_width=True, caption=f"Frame {idx+1}")
                    elif media_type == "audio":
                        st.audio(st.session_state.media_content["content"], format="audio/mp3")
                
                st.markdown(prompt)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("🧠 Thinking..."):
                    response = do_gemma_inference(
                        st.session_state.chat_histories[current_feature], 
                        max_tokens, 
                        temperature
                    )
                    st.markdown(response)
            
            # Add assistant response to history
            st.session_state.chat_histories[current_feature].append({
                "role": "assistant", 
                "content": [{"type": "text", "text": response}]
            })
            
            # Clear media content after first use
            if st.session_state.media_content:
                st.session_state.media_content = None
            
            # Clean up temp files
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    try:
                        os.unlink(temp_file)
                    except:
                        pass
            
            st.rerun()

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.8em;'>
    🤖 Powered by Gemma 3N via Unsloth | Built with Streamlit<br>
    💡 Tip: Use the "New Chat" button to start fresh conversations with different media
</div>
""", unsafe_allow_html=True)