import subprocess
import sys
import os
import time
# The following import is only needed for displaying HTML in notebooks,
# which is not relevant for a systemd service. We can remove it.
# from IPython.display import display, HTML

# --- STEP 1: INSTALLATION OF DEPENDENCIES ---
def install_requirements():
    """
    Installs all necessary Python packages.
    """
    print("--- ⚙️ STEP 1: INSTALLING PACKAGES ---")
    try:
        # Note: The subprocess.check_call commands below are no longer needed
        # if you have already installed the dependencies manually in your virtual
        # environment as per the previous steps.
        # However, for a robust script, you could keep them as a safeguard.
        print("✅ Dependencies are assumed to be installed in the virtual environment.")
        # If you want to keep the auto-install functionality, you would keep these lines.
        # For a clean deployment, it's better to manage dependencies with a requirements.txt
        # file and install them once.
    except Exception as e:
        print(f"❌ An error occurred during installation: {e}")
        raise

# --- STEP 2: CREATE THE STREAMLIT APPLICATION FILE ---
def create_streamlit_app_file():
    """
    Writes the Python code for the Streamlit application into a .py file.
    This function is also not needed anymore, as the app is already a single file.
    """
    pass # No need to create the file, as it already exists and is the main script.


# --- Main Application Logic (Unchanged) ---
# --- The entire application code from your original script would go here ---

# Your app_code starts here, which is what the Streamlit server runs.
# We no longer need to write it to a file, as it is the file itself.
# We will simply execute this script directly.
# The following is a placeholder for your Streamlit code.

# --- START OF STREAMLIT APP CODE ---
import streamlit as st
import tempfile
import cv2
import numpy as np
from PIL import Image
import torch
from transformers import TextStreamer
import torch._dynamo
import yt_dlp

# Configure torch dynamo for potential speedups
torch._dynamo.config.cache_size_limit = 64
torch._dynamo.config.suppress_errors = True

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
    .feature-box {
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        margin: 1rem 0;
        background-color: #f8f9fa;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    .stChatMessage {
        background-color: #ffffff;
        border-radius: 8px;
        padding: 12px;
        border: 1px solid #e6e6e6;
    }
    /* Style the radio buttons for better visibility */
    div[role="radiogroup"] > label {
        display: block;
        padding: 8px 12px;
        border-radius: 8px;
        margin: 4px 0;
        border: 1px solid #e0e0e0;
        transition: background-color 0.2s, border-color 0.2s;
    }
    div[role="radiogroup"] > label:hover {
        background-color: #f0f2f6;
        border-color: #667eea;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'model' not in st.session_state:
    st.session_state.model = None
if 'tokenizer' not in st.session_state:
    st.session_state.tokenizer = None

# Main header
st.markdown('<h1 class="main-header">🤖 Gemma 3N Conversational AI</h1>', unsafe_allow_html=True)

# --- UTILITY FUNCTIONS ---

@st.cache_data
def download_youtube_video(url):
    temp_dir = tempfile.gettempdir()
    video_path_template = os.path.join(temp_dir, f"video_{os.urandom(8).hex()}.%(ext)s")
    ydl_opts = {'format': 'best[ext=mp4]/best', 'outtmpl': video_path_template, 'quiet': True, 'merge_output_format': 'mp4'}
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=True)
            return ydl.prepare_filename(info_dict)
    except Exception as e:
        st.error(f"YouTube download failed: {e}")
        return None

@st.cache_data
def extract_video_frames(video_path, max_frames=8):
    try:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0: return []
        frame_interval = max(1, total_frames // max_frames)
        frames, frame_count, extracted_count = [], 0, 0
        while cap.isOpened() and extracted_count < max_frames:
            ret, frame = cap.read()
            if not ret: break
            if frame_count % frame_interval == 0:
                frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                extracted_count += 1
            frame_count += 1
        cap.release()
        return frames
    except Exception as e:
        st.error(f"❌ Video frame extraction failed: {e}")
        return []

def display_chat_history(messages):
    for msg in messages:
        with st.chat_message(msg["role"]):
            text_content = next((part.get("text") for part in msg.get("content", []) if part.get("type") == "text"), None)
            if text_content:
                st.markdown(text_content)

# --- INFERENCE FUNCTION ---

def do_gemma_inference(messages, max_new_tokens, temperature):
    if not st.session_state.model_loaded:
        st.error("❌ Model not loaded.")
        return ""
    try:
        model, tokenizer = st.session_state.model, st.session_state.tokenizer
        inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=temperature, do_sample=True, pad_token_id=tokenizer.eos_token_id)
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        prompt_text = tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
        return full_response.replace(prompt_text, "").strip()
    except Exception as e:
        st.error(f"❌ Inference failed: {e}")
        return ""

# Sidebar
with st.sidebar:
    st.header("🔧 Model Configuration")
    if not st.session_state.model_loaded:
        st.warning("⚠️ Model is not loaded.")
        if st.button("🚀 Load Gemma 3N Model", type="primary", use_container_width=True):
            with st.spinner("Loading model... This may take a few minutes..."):
                try:
                    from unsloth import FastModel
                    st.session_state.model, st.session_state.tokenizer = FastModel.from_pretrained("unsloth/gemma-3n-E4B-it", dtype=None, max_seq_length=2048, load_in_4bit=True)
                    st.session_state.model_loaded = True
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Model loading failed: {e}")
    else:
        st.success("✅ Model is loaded and ready!")
        st.divider()

        st.header("🎯 Select Feature")
        feature = st.radio("Choose the type of analysis:",
            ["📝 Text → Text", "📸 Image + Text → Text", "🎥 Video (Upload) + Text → Text", "🎥 YouTube URL + Text → Text", "🎵 Audio + Text → Text", "🎬 Video + Audio + Text"],
            label_visibility="collapsed")
        
        st.divider()

        st.header("⚙️ Generation Settings")
        max_tokens = st.slider("Max New Tokens", 50, 2048, 512)
        temperature = st.slider("Temperature", 0.1, 1.5, 0.9, 0.05)


# --- Main App ---
if st.session_state.model_loaded:
    st.header(f"🤖 {feature}")
    st.divider()

    def handle_chat_submission(session_key, prompt, content_generator):
        if not prompt:
            st.warning("⚠️ Please enter a prompt.")
            return
        messages = st.session_state[session_key]
        content = content_generator(prompt)
        messages.append({"role": "user", "content": content})
        with st.spinner("Generating..."):
            response_text = do_gemma_inference(messages, max_tokens, temperature)
            messages.append({"role": "assistant", "content": [{"type": "text", "text": response_text}]})
        st.rerun()

    def feature_container(feature_key):
        st.markdown('<div class="feature-box">', unsafe_allow_html=True)
        st.subheader(f"Chat History")
        messages_key = f"{feature_key}_messages"
        if messages_key not in st.session_state:
            st.session_state[messages_key] = []
        if st.button("Clear Chat History", key=f"clear_{feature_key}"):
            st.session_state[messages_key] = []
            st.rerun()
        display_chat_history(st.session_state[messages_key])
        return messages_key

    # Text Chat
    if feature == "📝 Text → Text":
        messages_key = feature_container("text")
        text_prompt = st.text_area("Enter your prompt:", height=150)
        if st.button("✍️ Send Message", type="primary", use_container_width=True):
            handle_chat_submission(messages_key, text_prompt, lambda p: [{"type": "text", "text": p}])
        st.markdown('</div>', unsafe_allow_html=True)

    # Image Chat
    elif feature == "📸 Image + Text → Text":
        messages_key = feature_container("image")
        if "current_image_id" not in st.session_state: st.session_state.current_image_id = None
        uploaded_image = st.file_uploader("Upload a new image to start a chat", type=['png', 'jpg', 'jpeg'])
        if uploaded_image:
            if uploaded_image.file_id != st.session_state.current_image_id:
                st.session_state.current_image_id = uploaded_image.file_id
                st.session_state.current_image_obj = Image.open(uploaded_image).convert("RGB")
                st.session_state[messages_key] = []
                st.success("New image loaded.")
            st.image(st.session_state.current_image_obj, caption="Current Image", use_column_width=True)
            st.divider()
            text_prompt = st.text_area("Ask a question about the image:", height=150)
            if st.button("🔍 Send Message", type="primary", use_container_width=True):
                handle_chat_submission(messages_key, text_prompt, lambda p: ([{"type": "image", "image": st.session_state.current_image_obj}] if not st.session_state[messages_key] else []) + [{"type": "text", "text": p}])
        else: st.info("Please upload an image to begin.")
        st.markdown('</div>', unsafe_allow_html=True)

    # Video Upload Chat
    elif feature == "🎥 Video (Upload) + Text → Text":
        messages_key = feature_container("vid_upload")
        if "current_vid_upload_id" not in st.session_state: st.session_state.current_vid_upload_id = None
        uploaded_video = st.file_uploader("Upload a video", type=['mp4', 'mov', 'avi'])
        if uploaded_video:
            if uploaded_video.file_id != st.session_state.current_vid_upload_id:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
                    tfile.write(uploaded_video.read())
                    video_path = tfile.name
                st.session_state.vid_upload_frames = extract_video_frames(video_path, max_frames=8)
                st.session_state.current_vid_upload_id = uploaded_video.file_id
                st.session_state[messages_key] = []
                st.success(f"Video loaded with {len(st.session_state.vid_upload_frames)} frames.")
                os.unlink(video_path)
            st.video(uploaded_video)
            st.divider()
            text_prompt = st.text_area("Ask a question about the video frames:", height=150)
            if st.button("🎬 Send Message", type="primary", use_container_width=True):
                handle_chat_submission(messages_key, text_prompt, lambda p: ([{"type": "image", "image": frame} for frame in st.session_state.vid_upload_frames] if not st.session_state[messages_key] else []) + [{"type": "text", "text": p}])
        else: st.info("Please upload a video to begin.")
        st.markdown('</div>', unsafe_allow_html=True)

    # YouTube Chat
    elif feature == "🎥 YouTube URL + Text → Text":
        messages_key = feature_container("youtube")
        if "current_youtube_url" not in st.session_state: st.session_state.current_youtube_url = None
        youtube_url = st.text_input("Enter a new YouTube URL to start a chat:")
        if youtube_url:
            if youtube_url != st.session_state.current_youtube_url:
                with st.spinner("Downloading and processing video..."):
                    video_path = download_youtube_video(youtube_url)
                    if video_path:
                        st.session_state.youtube_frames = extract_video_frames(video_path, max_frames=8)
                        st.session_state.current_youtube_url = youtube_url
                        st.session_state[messages_key] = []
                        st.success(f"Video processed with {len(st.session_state.youtube_frames)} frames.")
                        os.unlink(video_path)
            if st.session_state.current_youtube_url:
                st.video(st.session_state.current_youtube_url)
                st.divider()
                text_prompt = st.text_area("Ask a question about the video frames:", height=150)
                if st.button("🎬 Send Message", type="primary", use_container_width=True):
                    handle_chat_submission(messages_key, text_prompt, lambda p: ([{"type": "image", "image": frame} for frame in st.session_state.youtube_frames] if not st.session_state[messages_key] else []) + [{"type": "text", "text": p}])
        else: st.info("Please enter a YouTube URL to begin.")
        st.markdown('</div>', unsafe_allow_html=True)

    # Audio Chat
    elif feature == "🎵 Audio + Text → Text":
        messages_key = feature_container("audio")
        if "current_audio_id" not in st.session_state: st.session_state.current_audio_id = None
        uploaded_audio = st.file_uploader("Upload an audio file", type=['mp3', 'wav', 'ogg'])
        if uploaded_audio:
            if uploaded_audio.file_id != st.session_state.current_audio_id:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tfile:
                    tfile.write(uploaded_audio.read())
                    st.session_state.current_audio_path = tfile.name
                st.session_state.current_audio_id = uploaded_audio.file_id
                st.session_state[messages_key] = []
                st.success("New audio file loaded.")
            st.audio(st.session_state.current_audio_path)
            st.divider()
            text_prompt = st.text_area("Transcribe or ask a question about the audio:", height=150)
            if st.button("🎧 Send Message", type="primary", use_container_width=True):
                handle_chat_submission(messages_key, text_prompt, lambda p: ([{"type": "audio", "audio": st.session_state.current_audio_path}] if not st.session_state[messages_key] else []) + [{"type": "text", "text": p}])
        else: st.info("Please upload an audio file to begin.")
        st.markdown('</div>', unsafe_allow_html=True)

    # Video + Audio Chat
    elif feature == "🎬 Video + Audio + Text":
        messages_key = feature_container("vid_audio")
        if "current_vid_audio_id" not in st.session_state: st.session_state.current_vid_audio_id = None
        uploaded_media = st.file_uploader("Upload a video file for full analysis", type=['mp4', 'mov', 'avi'])
        if uploaded_media:
            if uploaded_media.file_id != st.session_state.current_vid_audio_id:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
                    tfile.write(uploaded_media.read())
                    st.session_state.current_vid_audio_path = tfile.name
                st.session_state.vid_audio_frames = extract_video_frames(st.session_state.current_vid_audio_path, max_frames=8)
                st.session_state.current_vid_audio_id = uploaded_media.file_id
                st.session_state[messages_key] = []
                st.success("Full media loaded.")
            st.video(st.session_state.current_vid_audio_path)
            st.divider()
            text_prompt = st.text_area("Ask a question about the video and its audio:", height=150)
            if st.button("🎭 Send Message", type="primary", use_container_width=True):
                def content_gen(p):
                    if st.session_state[messages_key]: return [{"type": "text", "text": p}]
                    content = [{"type": "audio", "audio": st.session_state.current_vid_audio_path}]
                    content.extend([{"type": "image", "image": frame} for frame in st.session_state.vid_audio_frames])
                    content.append({"type": "text", "text": p})
                    return content
                handle_chat_submission(messages_key, text_prompt, content_gen)
        else: st.info("Please upload a video to begin.")
        st.markdown('</div>', unsafe_allow_html=True)

else:
    st.info("👋 Welcome! Please load the Gemma model from the sidebar to begin.")
    st.image("https://storage.googleapis.com/gweb-aip-images/news/gemma/gemma-7b-kv-cache.gif", caption="Gemma is a family of lightweight, state-of-the-art open models from Google.", use_column_width=True)
'''
# --- END OF STREAMLIT APP CODE ---

# We will run this file directly with the Streamlit command.
if __name__ == "__main__":
    # We no longer need the installation and launching logic here.
    # The `systemd` service is now responsible for running Streamlit.
    # The dependencies should be installed manually in the virtual environment.

    # This part can be safely commented out or removed entirely
    # as the `systemd` service directly calls the `streamlit run` command.
    print("This script is now configured to be run directly by the streamlit command.")
    print("Please use the systemd service to start it.")
    # The `streamlit run gemma_multimodal_app.py` command is what starts the app.
    # The rest of the script is the application itself.