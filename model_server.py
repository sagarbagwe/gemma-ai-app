import streamlit as st
import os
import tempfile
import cv2
import numpy as np
from PIL import Image
import torch
from transformers import pipeline # Updated import
import yt_dlp
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import unsloth first for optimizations
from unsloth import FastModel

# Advanced PyTorch optimizations for faster inference
torch._dynamo.config.cache_size_limit = 1000  # Increase cache size
torch._dynamo.config.suppress_errors = True   # Suppress compilation errors
torch._dynamo.reset()  # Reset any existing compilation state

# Enable additional optimizations
torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
torch.backends.cuda.matmul.allow_tf32 = True  # Enable TF32 for faster matmul
torch.backends.cudnn.allow_tf32 = True  # Enable TF32 for convolutions

# Set optimal threading for inference
torch.set_num_threads(4)  # Adjust based on your CPU cores

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
        "models_loaded": False,
        "model": None,
        "tokenizer": None,
        "asr_pipeline": None, # For audio transcription
        "current_feature": "📄 Text → Text",
        "chat_histories": {
            "📄 Text → Text": [],
            "🖼️ Image + Text → Text": [],
            "🎞️ Video + Text → Text": [],
            "🎵 Audio + Text → Text": [],
        },
        "media_content": None,
        "inference_count": 0
    }
    
    for key, default_value in default_states.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

initialize_session_state()

# --- UTILITY FUNCTIONS ---
def truncate_conversation_history(messages, max_tokens=1500):
    """Truncate conversation history to prevent token overflow"""
    if not messages: return messages
    if len(messages) <= 4: return messages
    
    truncated = []
    if messages[0].get("role") == "system":
        truncated.append(messages[0])
        start_idx = 1
    else:
        start_idx = 0
    
    recent_messages = messages[max(start_idx, len(messages) - 6):]
    truncated.extend(recent_messages)
    logger.info(f"Truncated conversation from {len(messages)} to {len(truncated)} messages")
    return truncated

# --- MODEL LOADING FUNCTIONS ---
@st.cache_resource(show_spinner="🧠 Loading Gemma 3N Model...")
def load_gemma_model():
    """Loads and caches the Unsloth model and tokenizer."""
    try:
        logger.info("Starting Gemma model loading process...")
        model, tokenizer = FastModel.from_pretrained(
            model_name="unsloth/gemma-3n-E4B-it",
            dtype=None,
            max_seq_length=2048,
            load_in_4bit=True,
            attn_implementation="eager",
        )
        model.eval()
        model.config.use_cache = True
        logger.info("Gemma model loaded successfully.")
        return model, tokenizer
    except Exception as e:
        logger.error(f"Failed to load Gemma model: {e}")
        st.error(f"Failed to load Gemma model: {e}")
        st.stop()

@st.cache_resource(show_spinner="🔊 Loading Whisper ASR Model...")
def load_whisper_model():
    """Loads and caches the Whisper model for audio transcription."""
    try:
        logger.info("Loading Whisper ASR pipeline...")
        # Use a distilled model for speed and efficiency on consumer GPUs
        asr_pipeline = pipeline(
            "automatic-speech-recognition",
            model="distil-whisper/distil-large-v2",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device="cuda:0" if torch.cuda.is_available() else "cpu",
        )
        logger.info("Whisper ASR pipeline loaded successfully.")
        return asr_pipeline
    except Exception as e:
        logger.error(f"Failed to load Whisper model: {e}")
        st.error(f"Could not load the audio transcription model: {e}")
        return None

# --- MEDIA PROCESSING FUNCTIONS ---
@st.cache_data(show_spinner="🎤 Transcribing audio... this may take a moment.")
def transcribe_audio(_audio_bytes, asr_pipeline):
    """Transcribes audio using the Whisper pipeline. Caches result based on audio bytes."""
    if asr_pipeline is None:
        st.error("Audio transcription model is not loaded.")
        return None
    try:
        logger.info("Starting audio transcription...")
        # Write bytes to a temporary file for the pipeline
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tfile:
            tfile.write(_audio_bytes)
            audio_filename = tfile.name
        
        outputs = asr_pipeline(
            audio_filename,
            chunk_length_s=30,
            batch_size=8,
            return_timestamps=False,
        )
        transcription = outputs["text"]
        os.unlink(audio_filename) # Clean up the temp file
        logger.info(f"Transcription successful. Length: {len(transcription)} chars.")
        return transcription
    except Exception as e:
        logger.error(f"Audio transcription failed: {e}")
        st.error(f"Audio transcription failed: {e}")
        return None

@st.cache_data(show_spinner="📥 Downloading YouTube video...")
def download_youtube_video(url):
    """Download YouTube video with error handling"""
    try:
        temp_dir = tempfile.gettempdir()
        video_path = os.path.join(temp_dir, f"youtube_{os.urandom(8).hex()}.mp4")
        ydl_opts = {'format': 'best[ext=mp4][height<=720]/best', 'outtmpl': video_path, 'quiet': True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        return video_path if os.path.exists(video_path) else None
    except Exception as e:
        logger.error(f"YouTube download failed: {e}")
        st.error(f"YouTube download failed: {e}")
        return None

@st.cache_data(show_spinner="🎞️ Extracting video frames...")
def extract_video_frames(video_path, max_frames=50):
    """Extract frames from video with improved error handling"""
    frames = []
    cap = None
    try:
        if not video_path or not os.path.exists(video_path): raise FileNotFoundError("Video file not found")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): raise IOError("Could not open video file")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0: raise ValueError("Video has no frames")

        frame_interval = max(1, total_frames // max_frames)
        count, extracted = 0, 0
        while cap.isOpened() and extracted < max_frames:
            ret, frame = cap.read()
            if not ret: break
            if count % frame_interval == 0:
                pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                pil_image.thumbnail((384, 384), Image.Resampling.LANCZOS)
                frames.append(pil_image)
                extracted += 1
            count += 1
        logger.info(f"Extracted {len(frames)} frames from video")
    except Exception as e:
        logger.error(f"Frame extraction failed: {e}")
        st.error(f"Frame extraction failed: {e}")
    finally:
        if cap: cap.release()
    return frames

# --- INFERENCE FUNCTION ---
def do_gemma_inference(messages, max_new_tokens, temperature):
    """High-performance model inference with speed optimizations"""
    try:
        start_time = time.time()
        st.session_state.inference_count += 1
        logger.info(f"Starting inference #{st.session_state.inference_count}")
        
        if not messages: return "I need a message to respond to."
        
        truncated_messages = truncate_conversation_history(messages, max_tokens=1500)
        
        inputs = st.session_state.tokenizer.apply_chat_template(
            truncated_messages, add_generation_prompt=True, tokenize=True, 
            return_dict=True, return_tensors="pt"
        )
        
        model_device = next(st.session_state.model.parameters()).device
        inputs = {k: v.to(model_device) for k, v in inputs.items()}
        input_ids_length = inputs['input_ids'].shape[1]

        if input_ids_length > 1800:
            logger.warning(f"Input sequence too long: {input_ids_length}, using emergency fallback")
            last_message = truncated_messages[-1:]
            inputs = st.session_state.tokenizer.apply_chat_template(
                last_message, add_generation_prompt=True, tokenize=True, 
                return_dict=True, return_tensors="pt"
            ).to(model_device)
            input_ids_length = inputs['input_ids'].shape[1]

        generation_config = {
            "max_new_tokens": min(int(max_new_tokens), 512),
            "temperature": float(temperature),
            "do_sample": True if temperature > 0.1 else False,
            "pad_token_id": st.session_state.tokenizer.eos_token_id,
            "eos_token_id": st.session_state.tokenizer.eos_token_id,
            "use_cache": True,
            "num_beams": 1,
        }
        
        with torch.inference_mode():
            outputs = st.session_state.model.generate(**inputs, **generation_config)
        
        new_tokens = outputs[0, input_ids_length:]
        response = st.session_state.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        
        if not response: response = "I couldn't generate a response. Please try again."

        inference_time = time.time() - start_time
        tokens_per_sec = len(new_tokens) / inference_time if inference_time > 0 else 0
        logger.info(f"Inference time: {inference_time:.2f}s, Speed: {tokens_per_sec:.1f} tokens/sec")
        
        return response
        
    except torch.cuda.OutOfMemoryError:
        logger.error("CUDA out of memory")
        torch.cuda.empty_cache()
        return "I'm running low on GPU memory. Please try a shorter message or clear the chat."
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        return f"An error occurred during inference: {str(e)[:100]}..."

# --- MAIN HEADER ---
st.markdown('<h1 class="main-header">🤖 Gemma 3N Conversational AI Assistant</h1>', unsafe_allow_html=True)

# --- SIDEBAR FOR MODEL LOADING AND SETTINGS ---
with st.sidebar:
    st.header("🛠️ Model Configuration")
    
    if not st.session_state.models_loaded:
        st.warning("⚠️ Models are not loaded.")
        if st.button("🚀 Load AI Models", type="primary", use_container_width=True):
            st.session_state.model, st.session_state.tokenizer = load_gemma_model()
            st.session_state.asr_pipeline = load_whisper_model()
            
            if st.session_state.model and st.session_state.tokenizer and st.session_state.asr_pipeline:
                st.session_state.models_loaded = True
                st.success("✅ All models loaded successfully!")
                st.rerun()
            else:
                st.error("❌ Model loading failed. Check logs.")
    else:
        st.success("✅ All models are loaded!")
        col1, col2 = st.columns(2)
        col1.metric("💬 Messages", len(st.session_state.chat_histories[st.session_state.current_feature]))
        col2.metric("💡 Inferences", st.session_state.inference_count)

    st.divider()
    
    st.header("⚙️ Generation Settings")
    max_tokens = st.slider("Max New Tokens", 50, 512, 256)
    temperature = st.slider("Temperature", 0.1, 1.2, 0.7, 0.05)
    
    st.divider()
    
    st.header("ℹ️ System Info")
    if st.session_state.models_loaded:
        try:
            model_device = next(st.session_state.model.parameters()).device
            if "cuda" in str(model_device):
                gpu_info = torch.cuda.get_device_name(0)
                st.success(f"GPU: {gpu_info}")
                if st.button("🧹 Clear GPU Cache"):
                    torch.cuda.empty_cache()
                    st.success("GPU cache cleared!")
            else:
                st.info(f"Device: {model_device}")
        except Exception as e:
            st.warning(f"Device info unavailable: {e}")

# --- MAIN INTERFACE ---
if not st.session_state.models_loaded:
    st.info("👋 Welcome! Please load the AI models from the sidebar to begin.")
else:
    st.subheader("🎯 Choose Analysis Mode")
    features = list(st.session_state.chat_histories.keys())
    cols = st.columns(len(features))
    for i, feature in enumerate(features):
        with cols[i]:
            if st.button(feature, key=f"feature_{i}", use_container_width=True, type="primary" if st.session_state.current_feature == feature else "secondary"):
                st.session_state.current_feature = feature
                st.rerun()
    
    st.divider()
    
    current_feature = st.session_state.current_feature
    history = st.session_state.chat_histories[current_feature]
    
    st.subheader(f"💬 {current_feature}")
    
    if st.button("🗑️ New Chat", use_container_width=True):
        st.session_state.chat_histories[current_feature] = []
        st.session_state.media_content = None
        st.rerun()
    
    # Display chat history
    for message in history:
        with st.chat_message(message["role"]):
            images = [part["image"] for part in message["content"] if part["type"] == "image"]
            if images:
                with st.expander(f"🖼️ View {len(images)} attached frames/images"):
                    st.image(images, width=150)
            text_content = next((part["text"] for part in message["content"] if part["type"] == "text"), "")
            if text_content: st.markdown(text_content)
    
    # Media upload section (only for new chats)
    if not history:
        temp_files = []
        
        if "Image" in current_feature:
            uploaded_image = st.file_uploader("Choose an image", type=['png', 'jpg', 'jpeg'])
            if uploaded_image:
                image = Image.open(uploaded_image)
                image.thumbnail((512, 512), Image.Resampling.LANCZOS)
                st.image(image, caption="Uploaded Image", width=300)
                st.session_state.media_content = {"type": "image", "content": image}

        if "Video" in current_feature:
            st.subheader("🎞️ Upload Video")
            tab1, tab2 = st.tabs(["📤 Upload File", "🔗 YouTube URL"])
            video_path = None
            with tab1:
                uploaded_video = st.file_uploader("Choose a video file", type=['mp4', 'mov', 'avi'])
                if uploaded_video:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
                        tfile.write(uploaded_video.read())
                        video_path = tfile.name
                        temp_files.append(video_path)
            with tab2:
                youtube_url = st.text_input("Enter YouTube URL")
                if st.button("Process YouTube Video") and youtube_url:
                    video_path = download_youtube_video(youtube_url)
                    if video_path: temp_files.append(video_path)
            
            if video_path:
                frames = extract_video_frames(video_path)
                if frames:
                    st.success(f"✅ Extracted {len(frames)} frames.")
                    st.session_state.media_content = {"type": "video", "content": frames}

        if "Audio" in current_feature:
            st.subheader("🎵 Upload Audio")
            uploaded_audio = st.file_uploader("Choose an audio file", type=['mp3', 'wav', 'ogg', 'm4a'])
            if uploaded_audio:
                st.audio(uploaded_audio)
                audio_bytes = uploaded_audio.getvalue()
                transcription = transcribe_audio(audio_bytes, st.session_state.asr_pipeline)
                if transcription:
                    st.success("✅ Transcription Complete!")
                    with st.expander("📝 View Transcription"):
                        st.write(transcription)
                    st.session_state.media_content = {"type": "audio", "content": transcription}

    # Chat input
    if prompt := st.chat_input("✏️ Type your message here..."):
        if not history and "Text" not in current_feature and not st.session_state.media_content:
            st.warning("Please upload the required media for a new chat.")
        else:
            user_message_content = []
            if not history and st.session_state.media_content:
                media_type = st.session_state.media_content["type"]
                media_data = st.session_state.media_content["content"]
                
                if media_type == "image":
                    user_message_content.append({"type": "image", "image": media_data})
                    user_message_content.append({"type": "text", "text": prompt})
                elif media_type == "video":
                    user_message_content.extend([{"type": "image", "image": f} for f in media_data[:4]]) # Limit frames
                    user_message_content.append({"type": "text", "text": prompt})
                elif media_type == "audio":
                    combined_text = (
                        f"Please analyze the following audio transcription.\n\n"
                        f"**Transcription:**\n```\n{media_data}\n```\n\n"
                        f"**My Request:**\n{prompt}"
                    )
                    user_message_content.append({"type": "text", "text": combined_text})
            else:
                user_message_content.append({"type": "text", "text": prompt})
            
            st.session_state.chat_histories[current_feature].append({"role": "user", "content": user_message_content})
            
            with st.chat_message("assistant"):
                with st.spinner("🤔 Thinking..."):
                    response = do_gemma_inference(history, max_tokens, temperature)
                    st.markdown(response)
            
            st.session_state.chat_histories[current_feature].append({"role": "assistant", "content": [{"type": "text", "text": response}]})
            
            if st.session_state.media_content:
                st.session_state.media_content = None
            
            # Clean up temp files
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    try: os.unlink(temp_file)
                    except: pass
            
            st.rerun()