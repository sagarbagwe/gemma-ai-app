# File: app.py

import streamlit as st
import os
import tempfile
import cv2
import numpy as np
from PIL import Image
import torch
from transformers import TextStreamer
import yt_dlp
from unsloth import FastModel
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Fix PyTorch recompilation issues - MUST BE SET EARLY
torch._dynamo.config.cache_size_limit = 1000  # Increase cache size
torch._dynamo.config.suppress_errors = True   # Suppress compilation errors
torch._dynamo.reset()  # Reset any existing compilation state

# Alternative: Uncomment to completely disable dynamo if issues persist
# torch._dynamo.config.disable = True
# os.environ['PYTORCH_DISABLE_DYNAMO'] = '1'

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
</style>
""", unsafe_allow_html=True)

# --- MODEL LOADING (AUTOMATIC ON STARTUP) ---
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
        
        # Ensure model is on the correct device
        if torch.cuda.is_available():
            model = model.cuda()
            logger.info("Model loaded on CUDA")
        else:
            logger.warning("CUDA not available, using CPU")
        
        # Warm up the model with a dummy input to trigger initial compilation
        try:
            dummy_input = tokenizer("Hello", return_tensors="pt", padding=True)
            if torch.cuda.is_available():
                dummy_input = {k: v.cuda() for k, v in dummy_input.items()}
            
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

# Load the model immediately. Streamlit shows a spinner and caches the result.
try:
    model, tokenizer = load_gemma_model()
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Model loading failed: {e}")
    st.stop()

# --- SESSION STATE INITIALIZATION ---
def initialize_session_state():
    """Initialize all session state variables"""
    default_states = {
        "messages": [],
        "media_content": None,
        "model_ready": True,
        "inference_count": 0
    }
    
    for key, default_value in default_states.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

initialize_session_state()

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

@st.cache_data(show_spinner=False, hash_funcs={"builtins.list": lambda x: hash(str(x)[:1000])})
def prepare_tokenized_inputs(messages_for_api):
    """Prepare and cache tokenized inputs to avoid recompilation"""
    try:
        inputs = tokenizer.apply_chat_template(
            messages_for_api, 
            add_generation_prompt=True, 
            tokenize=True, 
            return_dict=True, 
            return_tensors="pt"
        )
        return inputs
    except Exception as e:
        logger.error(f"Tokenization failed: {e}")
        return None

# --- INFERENCE FUNCTION ---
def do_gemma_inference(messages_for_api, max_new_tokens, temperature):
    """
    Perform model inference with comprehensive error handling and optimization
    """
    try:
        st.session_state.inference_count += 1
        logger.info(f"Starting inference #{st.session_state.inference_count}")
        
        # Validate inputs
        if not messages_for_api:
            return "I need a message to respond to."
            
        # Prepare inputs with caching
        inputs = prepare_tokenized_inputs(messages_for_api)
        if inputs is None:
            return "Sorry, I couldn't process your message. Please try again."
            
        # Move to GPU if available
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
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
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "use_cache": True,
            "return_dict_in_generate": False,
            "output_attentions": False,
            "output_hidden_states": False
        }
        
        # Perform inference with proper context management
        with torch.inference_mode():
            try:
                outputs = model.generate(**inputs, **generation_config)
                logger.info("Generation completed successfully")
                
            except RuntimeError as e:
                if "recompile_limit" in str(e) or "compilation" in str(e):
                    logger.warning("Recompilation limit reached, using fallback")
                    
                    # Fallback: disable dynamo for this inference
                    with torch._dynamo.config.disable():
                        outputs = model.generate(**inputs, **generation_config)
                else:
                    raise e
        
        # Extract and decode new tokens
        new_tokens = outputs[0, input_ids_length:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        
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
        
        # Provide user-friendly error messages
        if "recompile_limit" in error_msg:
            return "The model is optimizing itself. This response might be slower, but functionality isn't affected."
        elif "CUDA" in error_msg:
            return "There was a GPU-related issue. Please try again."
        elif "memory" in error_msg.lower():
            return "Memory issue encountered. Please try a shorter message."
        else:
            return f"An unexpected error occurred. Please try again or contact support if the issue persists."

# --- MAIN HEADER ---
st.markdown('<h1 class="main-header">🤖 Gemma 3N Multimodal Assistant</h1>', unsafe_allow_html=True)

# Add system status
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("🧠 Model Status", "Ready" if st.session_state.model_ready else "Loading")
with col2:
    st.metric("💬 Conversations", len(st.session_state.messages))
with col3:
    st.metric("🔄 Inferences", st.session_state.inference_count)

# --- SIDEBAR FOR CONTROLS ---
with st.sidebar:
    st.header("⚙️ Controls")
    
    # Mode selection
    selected_mode = st.radio(
        "Choose an analysis mode:",
        ["📝 Text Only", "🖼️ Image Analysis", "🎬 Video Analysis", "🎵 Audio Analysis"],
        help="Select the type of content you want to analyze"
    )

    # Clear chat button
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.media_content = None
        st.session_state.inference_count = 0
        # Clear caches
        st.cache_data.clear()
        st.success("Chat history cleared!")
        st.rerun()

    st.divider()
    
    # Instructions
    st.header("💡 How to Use")
    st.info(
        "1. **Select Mode**: Choose your content type\n"
        "2. **Upload File**: Add your image/video/audio\n"
        "3. **Ask Questions**: Chat about your content\n"
        "4. **Get Insights**: Receive AI analysis"
    )
    
    st.divider()
    
    # Generation parameters
    st.header("🎛️ Generation Settings")
    max_tokens = st.slider(
        "Max New Tokens:", 
        min_value=50, 
        max_value=2048, 
        value=512, 
        step=50,
        help="Maximum number of tokens to generate"
    )
    
    temperature = st.slider(
        "Temperature:", 
        min_value=0.1, 
        max_value=1.5, 
        value=0.9, 
        step=0.05,
        help="Controls randomness: lower = more focused, higher = more creative"
    )
    
    st.divider()
    
    # System info
    st.header("📊 System Info")
    if torch.cuda.is_available():
        gpu_info = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory // (1024**3)
        st.success(f"🖥️ GPU: {gpu_info}\n💾 Memory: {gpu_memory}GB")
    else:
        st.warning("🖥️ Running on CPU")

# --- MEDIA UPLOAD & PROCESSING AREA ---
if not st.session_state.media_content:
    st.subheader("📁 Upload Content")
    
    if selected_mode == "🖼️ Image Analysis":
        uploaded_file = st.file_uploader(
            "Upload an Image", 
            type=['png', 'jpg', 'jpeg', 'webp', 'bmp'],
            help="Supported formats: PNG, JPG, JPEG, WebP, BMP"
        )
        if uploaded_file:
            try:
                image = Image.open(uploaded_file)
                # Resize if too large
                max_size = 1024
                if max(image.size) > max_size:
                    image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                
                st.session_state.media_content = {"type": "image", "content": image}
                st.success("✅ Image uploaded successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to process image: {e}")
    
    elif selected_mode == "🎬 Video Analysis":
        tab1, tab2 = st.tabs(["📁 Upload File", "🌐 YouTube URL"])
        
        with tab1:
            uploaded_file = st.file_uploader(
                "Upload a Video", 
                type=['mp4', 'mov', 'avi', 'mkv', 'webm'],
                help="Supported formats: MP4, MOV, AVI, MKV, WebM"
            )
            if uploaded_file:
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
                        tfile.write(uploaded_file.read())
                        video_path = tfile.name
                    
                    frames = extract_video_frames(video_path)
                    os.unlink(video_path)  # Clean up temp file
                    
                    if frames:
                        st.session_state.media_content = {"type": "video", "content": frames}
                        st.success(f"✅ Video processed! Extracted {len(frames)} frames.")
                        st.rerun()
                    else:
                        st.error("Failed to extract frames from video.")
                        
                except Exception as e:
                    st.error(f"Failed to process video: {e}")
        
        with tab2:
            youtube_url = st.text_input(
                "Enter YouTube URL",
                placeholder="https://www.youtube.com/watch?v=...",
                help="Paste a YouTube video URL"
            )
            if st.button("🎬 Process YouTube Video", use_container_width=True):
                if youtube_url:
                    video_path = download_youtube_video(youtube_url)
                    if video_path:
                        frames = extract_video_frames(video_path)
                        try:
                            os.unlink(video_path)  # Clean up
                        except:
                            pass
                        
                        if frames:
                            st.session_state.media_content = {"type": "video", "content": frames}
                            st.success(f"✅ YouTube video processed! Extracted {len(frames)} frames.")
                            st.rerun()
                        else:
                            st.error("Failed to extract frames from YouTube video.")
                else:
                    st.warning("Please enter a YouTube URL.")

    elif selected_mode == "🎵 Audio Analysis":
        uploaded_file = st.file_uploader(
            "Upload an Audio File", 
            type=['mp3', 'wav', 'ogg', 'm4a', 'flac'],
            help="Supported formats: MP3, WAV, OGG, M4A, FLAC"
        )
        if uploaded_file:
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tfile:
                    tfile.write(uploaded_file.read())
                    audio_path = tfile.name
                
                st.session_state.media_content = {"type": "audio", "content": audio_path}
                st.success("✅ Audio uploaded successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to process audio: {e}")

    elif selected_mode == "📝 Text Only":
        st.info("💬 Text mode selected. You can start chatting directly below!")

# --- CHAT INTERFACE ---
st.subheader("💬 Conversation")

# Display chat messages from history
chat_container = st.container()
with chat_container:
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Display media if it has been processed
if st.session_state.media_content:
    with st.chat_message("user"):
        media_type = st.session_state.media_content["type"]
        media_data = st.session_state.media_content["content"]
        
        if media_type == "image":
            st.image(media_data, caption="📸 Uploaded Image", width=300)
            
        elif media_type == "video":
            st.info(f"🎬 Video Analysis Ready - {len(media_data)} frames extracted")
            with st.expander("🔍 View Extracted Frames"):
                cols = st.columns(min(4, len(media_data)))
                for idx, frame in enumerate(media_data):
                    with cols[idx % 4]:
                        st.image(frame, caption=f"Frame {idx+1}", width=150)
                        
        elif media_type == "audio":
            st.audio(media_data, format="audio/mp3")
            st.info("🎵 Audio file ready for analysis")

# Chat input
if prompt := st.chat_input("💭 Ask a question about your content or start a conversation..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Prepare messages for the API
    messages_for_api = []
    
    # Handle media content
    if st.session_state.media_content:
        media_type = st.session_state.media_content["type"]
        media_data = st.session_state.media_content["content"]
        
        if media_type == "image":
            messages_for_api.append({
                "role": "user", 
                "content": [
                    {"type": "image", "image": media_data}, 
                    {"type": "text", "text": prompt}
                ]
            })
        elif media_type == "video":
            # Combine frames and text for video analysis
            content_list = [{"type": "image", "image": frame} for frame in media_data[:8]]  # Limit frames
            content_list.append({"type": "text", "text": f"This is a video with {len(media_data)} frames. {prompt}"})
            messages_for_api.append({"role": "user", "content": content_list})
        elif media_type == "audio":
            messages_for_api.append({
                "role": "user", 
                "content": [
                    {"type": "audio", "audio": media_data}, 
                    {"type": "text", "text": prompt}
                ]
            })
        
        # Clear media after first use
        st.session_state.media_content = None
    
    else:
        # Text-only conversation
        for msg in st.session_state.messages:
            messages_for_api.append({
                "role": msg["role"], 
                "content": [{"type": "text", "text": msg["content"]}]
            })

    # Generate and display response
    with st.chat_message("assistant"):
        with st.spinner("🧠 Thinking... This may take a moment."):
            response = do_gemma_inference(messages_for_api, max_tokens, temperature)
            st.markdown(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.8em;'>
    🤖 Powered by Gemma 3N via Unsloth | Built with Streamlit<br>
    💡 Tip: Clear chat history if you encounter memory issues
</div>
""", unsafe_allow_html=True)