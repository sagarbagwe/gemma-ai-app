import streamlit as st
import os
import tempfile
import cv2
import numpy as np
from PIL import Image
import torch
from unsloth import FastModel
from transformers import TextStreamer, pipeline
import yt_dlp
import logging
import time

# NOTE: For audio processing, ensure you have the necessary libraries installed:
# pip install soundfile librosa

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        "model_loaded": False,
        "model": None,
        "tokenizer": None,
        "asr_model_loaded": False,
        "asr_pipeline": None,
        "current_feature": "🗣️ Text → Text",
        "chat_histories": {
            "🗣️ Text → Text": [],
            "🖼️ Image + Text → Text": [],
            "🎬 Video + Text → Text": [],
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
    """
    Truncate conversation history to prevent token overflow
    """
    if not messages:
        return messages
    
    # Always keep the system message (if present) and the last user message
    truncated = []
    
    # Keep the last few messages to maintain context
    if len(messages) <= 4:
        return messages
    
    # Keep system message if it exists
    if messages[0].get("role") == "system":
        truncated.append(messages[0])
        start_idx = 1
    else:
        start_idx = 0
    
    # Keep the last 3 message pairs (6 messages total)
    recent_messages = messages[max(start_idx, len(messages) - 6):]
    truncated.extend(recent_messages)
    
    logger.info(f"Truncated conversation from {len(messages)} to {len(truncated)} messages")
    return truncated

# --- MODEL LOADING FUNCTION ---
@st.cache_resource(show_spinner="🧠 Loading Gemma 3N Model... This may take a few minutes on first run.")
def load_gemma_model():
    """
    Loads and caches the Unsloth model and tokenizer with maximum optimizations.
    """
    try:
        logger.info("Starting model loading process...")
        
        # Load model with optimized settings for speed
        model, tokenizer = FastModel.from_pretrained(
            model_name="unsloth/gemma-3n-E4B-it",
            dtype=None,  # Auto-select best dtype (bfloat16 on modern GPUs)
            max_seq_length=2048, 
            load_in_4bit=True,  # 4-bit quantization for speed + memory
            # Additional speed optimizations
            attn_implementation="eager",  # Use eager attention (faster for Gemma3N)
        )
        
        # FIXED: Set pad_token_id to eos_token_id if it's not set. This is crucial for model generation.
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
            logger.info("Set pad_token_id to eos_token_id to fix generation issues.")

        # Critical optimizations for inference speed
        model.eval()  # Set to evaluation mode
        
        # Enable KV-cache for faster sequential generation
        model.config.use_cache = True
        
        # Try to compile the model for maximum speed (torch 2.0+)
        try:
            # Only compile if not quantized or if compilation is known to work
            if hasattr(model, 'config') and not getattr(model.config, 'quantization_config', None):
                model = torch.compile(model, mode="reduce-overhead")
                logger.info("Model compiled with torch.compile for maximum speed")
            else:
                logger.info("Skipping torch.compile for quantized model to avoid issues")
        except Exception as compile_error:
            logger.warning(f"Model compilation failed, using uncompiled model: {compile_error}")
        
        # Check device and optimize
        device = next(model.parameters()).device
        logger.info(f"Model loaded on device: {device}")
        
        # Warm up the model with multiple passes for optimal performance
        try:
            dummy_inputs = ["Hello", "What is this?", "Explain this."]
            
            for dummy_text in dummy_inputs:
                dummy_input = tokenizer(
                    dummy_text, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True,
                    max_length=512
                )
                if "cuda" in str(device):
                    dummy_input = {k: v.to(device) for k, v in dummy_input.items()}
                
                with torch.inference_mode():
                    _ = model.generate(
                        **dummy_input, 
                        max_new_tokens=5, 
                        do_sample=False,
                        use_cache=True,
                        pad_token_id=tokenizer.pad_token_id
                    )
            logger.info("Model warmup completed successfully")
        except Exception as warmup_error:
            logger.warning(f"Model warmup failed: {warmup_error}")
        
        logger.info("Model loading completed successfully with speed optimizations")
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        st.error(f"Failed to load model: {e}")
        st.stop()

@st.cache_resource(show_spinner="🎤 Loading Speech Recognition Model...")
def load_asr_model():
    """Loads and caches the Whisper ASR pipeline."""
    try:
        logger.info("Loading ASR model...")
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # Using a more advanced, distilled version of Whisper for better performance and accuracy.
        asr_pipeline = pipeline(
            "automatic-speech-recognition", 
            model="distil-whisper/distil-large-v2",
            device=device,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        logger.info(f"ASR model loaded successfully on device: {device}")
        return asr_pipeline
    except Exception as e:
        logger.error(f"Failed to load ASR model: {e}")
        st.error(f"Failed to load the speech recognition model: {e}")
        return None

@st.cache_data(show_spinner="🔊 Transcribing audio...")
def transcribe_audio(_asr_pipeline, audio_bytes):
    """Transcribes audio file bytes to text."""
    if _asr_pipeline is None:
        return "Error: ASR model not available."
    try:
        # The pipeline can directly handle bytes
        # FIXED: Added return_timestamps=True to enable long-form transcription for audio > 30s
        result = _asr_pipeline(audio_bytes, return_timestamps=True)
        transcription = result.get("text", "").strip()
        logger.info(f"Audio transcribed successfully. Length: {len(transcription)} chars.")
        if not transcription:
            return "Audio could not be transcribed (is it silent?)."
        return transcription
    except Exception as e:
        logger.error(f"Audio transcription failed: {e}")
        st.error(f"Audio transcription failed: {e}")
        return f"Error during transcription: {e}"

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

@st.cache_data(show_spinner="🎞️ Extracting video frames...")
def extract_video_frames(video_path, max_frames=50):
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
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_frame)
                
                max_size = 384
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
    High-performance model inference with speed optimizations
    """
    try:
        start_time = time.time()
        st.session_state.inference_count += 1
        logger.info(f"Starting inference #{st.session_state.inference_count}")
        
        if not messages:
            return "I need a message to respond to."
        
        truncated_messages = truncate_conversation_history(messages, max_tokens=1200)
        
        try:
            inputs = st.session_state.tokenizer.apply_chat_template(
                truncated_messages, 
                add_generation_prompt=True, 
                tokenize=True, 
                return_dict=True, 
                return_tensors="pt"
            )
        except Exception as template_error:
            logger.error(f"Chat template error: {template_error}")
            text_content = ""
            for msg in truncated_messages:
                role = msg.get("role", "user")
                content = msg.get("content", [])
                if isinstance(content, list):
                    text_parts = [part.get("text", "") for part in content if part.get("type") == "text"]
                    text_content += f"{role}: {' '.join(text_parts)}\n"
                else:
                    text_content += f"{role}: {content}\n"
            
            text_content += "assistant:"
            inputs = st.session_state.tokenizer(text_content, return_tensors="pt", truncation=True, max_length=1500)
        
        model_device = next(st.session_state.model.parameters()).device
        inputs = {k: v.to(model_device) for k, v in inputs.items()}
        
        input_ids_length = inputs['input_ids'].shape[1]
        logger.info(f"Input length: {input_ids_length} tokens")
        
        if input_ids_length > 1800:
            logger.warning(f"Input sequence too long: {input_ids_length}, using emergency fallback")
            last_message = truncated_messages[-1] if truncated_messages else {"role": "user", "content": [{"type": "text", "text": "Hello"}]}
            try:
                inputs = st.session_state.tokenizer.apply_chat_template(
                    [last_message], 
                    add_generation_prompt=True, 
                    tokenize=True, 
                    return_dict=True, 
                    return_tensors="pt"
                )
                inputs = {k: v.to(model_device) for k, v in inputs.items()}
                input_ids_length = inputs['input_ids'].shape[1]
            except:
                inputs = st.session_state.tokenizer("Hello, how can I help?", return_tensors="pt")
                inputs = {k: v.to(model_device) for k, v in inputs.items()}
                input_ids_length = inputs['input_ids'].shape[1]
        
        generation_config = {
            "max_new_tokens": min(int(max_new_tokens), 512),
            "temperature": float(temperature) if temperature > 0.1 else 0.1,
            "do_sample": True if temperature > 0.1 else False,
            "pad_token_id": st.session_state.tokenizer.pad_token_id,
            "eos_token_id": st.session_state.tokenizer.eos_token_id,
            "use_cache": True,
            "num_beams": 1,
            "early_stopping": True,
            "no_repeat_ngram_size": 3,
        }
        
        if temperature > 0.1:
            generation_config.update({
                "top_k": 40,
                "top_p": 0.85,
            })
        
        with torch.inference_mode():
            if hasattr(st.session_state.model, 'config') and not getattr(st.session_state.model.config, 'quantization_config', None):
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = st.session_state.model.generate(**inputs, **generation_config)
            else:
                outputs = st.session_state.model.generate(**inputs, **generation_config)
        
        new_tokens = outputs[0, input_ids_length:]
        response = st.session_state.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        
        if not response:
            response = "I apologize, but I couldn't generate a response. Please try again with a different prompt."
        
        end_time = time.time()
        inference_time = end_time - start_time
        tokens_per_second = len(new_tokens) / inference_time if inference_time > 0 else 0
        
        logger.info(f"Generated response: {len(response)} chars, {len(new_tokens)} tokens")
        logger.info(f"Inference time: {inference_time:.2f}s, Speed: {tokens_per_second:.1f} tokens/sec")
        
        return response
        
    except torch.cuda.OutOfMemoryError:
        logger.error("CUDA out of memory")
        torch.cuda.empty_cache()
        return "I'm sorry, but I'm running low on GPU memory. Please try a shorter message or restart the app."
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Inference failed: {error_msg}")
        return f"An error occurred during inference. Please try again. Error: {error_msg[:100]}..."

# --- MAIN HEADER ---
st.markdown('<h1 class="main-header">🤖 Gemma 3N Conversational AI Assistant</h1>', unsafe_allow_html=True)

# --- SIDEBAR FOR MODEL LOADING AND SETTINGS ---
with st.sidebar:
    st.header("⚙️ Model Configuration")
    
    # Main LLM Model
    if not st.session_state.model_loaded:
        st.warning("⚠️ LLM model is not loaded.")
        if st.button("🚀 Load Gemma 3N Model", type="primary", use_container_width=True):
            model, tokenizer = load_gemma_model()
            st.session_state.model = model
            st.session_state.tokenizer = tokenizer
            st.session_state.model_loaded = True
            st.success("✅ Model loaded successfully!")
            st.rerun()
    else:
        st.success("✅ LLM is loaded and ready!")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("💬 Messages", len(st.session_state.chat_histories[st.session_state.current_feature]))
        with col2:
            st.metric("🔄 Inferences", st.session_state.inference_count)

    st.divider()

    # Audio Processor Model
    st.header("🎤 Audio Processor")
    if not st.session_state.asr_model_loaded:
        st.warning("⚠️ Speech model is not loaded.")
        if st.button("🔊 Load Speech Model", use_container_width=True):
            asr_pipeline = load_asr_model()
            if asr_pipeline:
                st.session_state.asr_pipeline = asr_pipeline
                st.session_state.asr_model_loaded = True
                st.success("✅ Speech model loaded!")
                st.rerun()
    else:
        st.success("✅ Speech model is ready.")

    st.divider()
    
    st.header("⚙️ Generation Settings")
    max_tokens = st.slider("Max New Tokens", 50, 512, 256, help="Maximum number of tokens to generate")
    temperature = st.slider("Temperature", 0.1, 1.2, 0.7, 0.05, help="Controls randomness: lower = more focused, higher = more creative")
    
    st.divider()
    
    st.header("💡 How to Use")
    st.info(
        "1. **Load Models**: Use the buttons above\n"
        "2. **Choose Mode**: Select your analysis type\n"
        "3. **Upload Media**: Add files if needed\n"
        "4. **Chat**: Ask questions to the AI"
    )
    
    st.divider()
    
    st.header("📊 System Info")
    if st.session_state.model_loaded:
        try:
            model_device = next(st.session_state.model.parameters()).device
            if "cuda" in str(model_device):
                gpu_info = torch.cuda.get_device_name(0)
                st.success(f"🖥️ GPU: {gpu_info}")
                if st.button("🧹 Clear GPU Cache", help="Clear GPU memory cache"):
                    torch.cuda.empty_cache()
                    st.success("GPU cache cleared!")
            else:
                st.info(f"🖥️ Model Device: {model_device}")
        except Exception as e:
            st.warning(f"⚠️ Device info unavailable: {e}")

# --- MAIN INTERFACE ---
if not st.session_state.model_loaded:
    st.info("👋 Welcome! Please load the Gemma model from the sidebar to begin.")
else:
    st.subheader("🎯 Choose Analysis Mode")
    
    features = list(st.session_state.chat_histories.keys())
    
    cols = st.columns(3)
    for i, feature in enumerate(features):
        with cols[i % 3]:
            is_selected = st.session_state.current_feature == feature
            if st.button(feature, key=f"feature_{i}", use_container_width=True, type="primary" if is_selected else "secondary"):
                st.session_state.current_feature = feature
                st.rerun()
    
    st.divider()
    
    current_feature = st.session_state.current_feature
    history = st.session_state.chat_histories[current_feature]
    
    st.subheader(f"💬 {current_feature}")
    
    col1, col2 = st.columns([2, 3])
    with col1:
        if st.button("🗑️ New Chat", use_container_width=True):
            st.session_state.chat_histories[current_feature] = []
            st.session_state.media_content = None
            st.rerun()
    
    chat_container = st.container()
    with chat_container:
        for message in history:
            with st.chat_message(message["role"]):
                images = [part["image"] for part in message["content"] if part["type"] == "image"]
                if images:
                    if len(images) == 1:
                        st.image(images[0], caption="🖼️ Attached Image", width=300)
                    else:
                        with st.expander(f"🖼️ View {len(images)} attached frames"):
                            cols = st.columns(min(len(images), 4))
                            for idx, image in enumerate(images):
                                cols[idx % 4].image(image, use_column_width=True, caption=f"Frame {idx+1}")
                
                text_content = next((part["text"] for part in message["content"] if part["type"] == "text"), "")
                if text_content:
                    st.markdown(text_content)
    
    if not history or "Text" in current_feature:
        media_uploaded = False
        temp_files = []
        
        if "Image" in current_feature and not history:
            st.subheader("🖼️ Upload Image")
            uploaded_image = st.file_uploader("Choose an image file", type=['png', 'jpg', 'jpeg', 'webp', 'bmp'], key="image_upload")
            if uploaded_image:
                image = Image.open(uploaded_image)
                max_size = 512
                if max(image.size) > max_size:
                    image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                st.image(image, caption="Uploaded Image", width=300)
                st.session_state.media_content = {"type": "image", "content": image}
                media_uploaded = True
        
        if "Video" in current_feature and not history:
            st.subheader("🎬 Upload Video")
            tab1, tab2 = st.tabs(["📤 Upload File", "🌐 YouTube URL"])
            
            with tab1:
                uploaded_video = st.file_uploader("Choose a video file", type=['mp4', 'mov', 'avi', 'mkv', 'webm'], key="video_upload")
                if uploaded_video:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
                        tfile.write(uploaded_video.read())
                        video_path = tfile.name
                    temp_files.append(video_path)
                    frames = extract_video_frames(video_path)
                    if frames:
                        st.success(f"✅ Extracted {len(frames)} frames")
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
        
        if "Audio" in current_feature and not history:
            st.subheader("🎵 Upload Audio")
            if not st.session_state.asr_model_loaded:
                st.warning("Please load the Speech Model from the sidebar to process audio.")
            else:
                uploaded_audio = st.file_uploader("Choose an audio file", type=['mp3', 'wav', 'ogg', 'm4a', 'flac'], key="audio_upload")
                if uploaded_audio:
                    audio_bytes = uploaded_audio.getvalue()
                    st.audio(audio_bytes)
                    transcription = transcribe_audio(st.session_state.asr_pipeline, audio_bytes)
                    with st.expander("📝 View Transcription"):
                        st.markdown(transcription)
                    st.session_state.media_content = {"type": "audio", "content": transcription}
                    media_uploaded = True
    
    if prompt := st.chat_input("✍️ Type your message here..."):
        if not history and "Text" not in current_feature and not st.session_state.media_content:
            st.warning("Please upload the required media for a new chat.")
        else:
            # Re-display chat history immediately after new prompt
            with st.chat_message("user"):
                if not history and st.session_state.media_content:
                     # Display media if it was part of the first message
                    media_type = st.session_state.media_content["type"]
                    media_data = st.session_state.media_content["content"]
                    if media_type == "image":
                        st.image(media_data, caption="🖼️ Attached Image", width=300)
                    elif media_type == "video":
                        with st.expander(f"🖼️ View {len(media_data)} attached frames"):
                            cols = st.columns(min(4, len(media_data)))
                            for idx, frame in enumerate(media_data):
                                cols[idx % 4].image(frame, use_column_width=True, caption=f"Frame {idx+1}")
                    elif media_type == "audio":
                        st.info("🎵 Audio file submitted for analysis.")

                # Display the actual text prompt
                st.markdown(prompt)

            user_content = []
            final_prompt = prompt
            
            if not history and st.session_state.media_content:
                media_type = st.session_state.media_content["type"]
                media_data = st.session_state.media_content["content"]
                
                if media_type == "image":
                    user_content.append({"type": "image", "image": media_data})
                elif media_type == "video":
                    user_content.extend([{"type": "image", "image": frame} for frame in media_data[:3]])
                elif media_type == "audio":
                    final_prompt = (f"Please analyze the following audio transcription:\n\n"
                                    f"**Transcription:**\n> {media_data}\n\n"
                                    f"**My question is:** {prompt}")

            user_content.append({"type": "text", "text": final_prompt})
            
            st.session_state.chat_histories[current_feature].append({"role": "user", "content": user_content})
            
            with st.chat_message("assistant"):
                with st.spinner("🤔 Thinking..."):
                    response = do_gemma_inference(st.session_state.chat_histories[current_feature], max_tokens, temperature)
                    st.markdown(response)
            
            st.session_state.chat_histories[current_feature].append({"role": "assistant", "content": [{"type": "text", "text": response}]})
            
            if st.session_state.media_content:
                st.session_state.media_content = None
            
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    try:
                        os.unlink(temp_file)
                    except Exception as e:
                        logger.warning(f"Failed to delete temp file {temp_file}: {e}")
            
            st.rerun()

st.divider()
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.8em;'>
    🤖 Powered by Gemma 3N via Unsloth | Built with Streamlit
</div>
""", unsafe_allow_html=True)

