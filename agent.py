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
</style>
""", unsafe_allow_html=True)

# --- MODEL LOADING (AUTOMATIC ON STARTUP) ---
@st.cache_resource(show_spinner="🧠 Loading Gemma 3N Model...")
def load_gemma_model():
    """
    Loads and caches the Unsloth model and tokenizer.
    This function runs only once when the app starts.
    """
    model, tokenizer = FastModel.from_pretrained(
        model_name="unsloth/gemma-3n-E4B-it",
        dtype=None, max_seq_length=2048, load_in_4bit=True,
    )
    return model, tokenizer

# Load the model immediately. Streamlit shows a spinner and caches the result.
model, tokenizer = load_gemma_model()


# --- SESSION STATE INITIALIZATION ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "media_content" not in st.session_state:
    st.session_state.media_content = None


# --- UTILITY FUNCTIONS ---
@st.cache_data(show_spinner="Downloading YouTube video...")
def download_youtube_video(url):
    temp_dir = tempfile.gettempdir()
    video_path = os.path.join(temp_dir, f"{os.urandom(8).hex()}.mp4")
    ydl_opts = {'format': 'best[ext=mp4]/best', 'outtmpl': video_path, 'quiet': True, 'merge_output_format': 'mp4'}
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        return video_path
    except Exception as e:
        st.error(f"YouTube download failed: {e}")
        return None

@st.cache_data
def extract_video_frames(video_path, max_frames=8):
    frames = []
    try:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0: return []
        frame_interval = max(1, total_frames // max_frames)
        count, extracted = 0, 0
        while cap.isOpened() and extracted < max_frames:
            ret, frame = cap.read()
            if not ret: break
            if count % frame_interval == 0:
                frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                extracted += 1
            count += 1
    finally:
        cap.release()
    return frames


# --- INFERENCE FUNCTION ---
def do_gemma_inference(messages_for_api, max_new_tokens, temperature):
    try:
        inputs = tokenizer.apply_chat_template(
            messages_for_api, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        ).to("cuda")
        
        input_ids_length = inputs['input_ids'].shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=max_new_tokens, temperature=temperature,
                do_sample=True, pad_token_id=tokenizer.eos_token_id
            )
        
        new_tokens = outputs[0, input_ids_length:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True)
    except Exception as e:
        st.error(f"An error occurred during inference: {e}", icon="🔥")
        return ""

# --- MAIN HEADER ---
st.markdown('<h1 class="main-header">🤖 Gemma 3N Multimodal Assistant</h1>', unsafe_allow_html=True)


# --- SIDEBAR FOR CONTROLS ---
with st.sidebar:
    st.header("⚙️ Controls")
    
    selected_mode = st.radio(
        "Choose an analysis mode:",
        ["📝 Text", "🖼️ Image", "🎬 Video", "🎵 Audio"]
    )

    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.session_state.media_content = None
        st.rerun()

    st.divider()
    st.header("💡 How to Use")
    st.info(
        "1. Select a mode (e.g., Image).\n"
        "2. Upload your file in the main window.\n"
        "3. Ask a question about the file."
    )
    st.divider()
    max_tokens = st.slider("Max New Tokens:", 50, 2048, 512)
    temperature = st.slider("Temperature:", 0.1, 1.5, 0.9, 0.05)


# --- MEDIA UPLOAD & PROCESSING AREA ---
media_uploader_container = st.container()
if not st.session_state.media_content:
    with media_uploader_container:
        if selected_mode == "🖼️ Image":
            uploaded_file = st.file_uploader("Upload an Image", type=['png', 'jpg', 'jpeg'])
            if uploaded_file:
                st.session_state.media_content = {"type": "image", "content": Image.open(uploaded_file)}
                st.rerun()
        
        elif selected_mode == "🎬 Video":
            vid_tab1, vid_tab2 = st.tabs(["Upload", "YouTube"])
            with vid_tab1:
                uploaded_file = st.file_uploader("Upload a Video", type=['mp4', 'mov', 'avi'])
                if uploaded_file:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
                        tfile.write(uploaded_file.read())
                        video_path = tfile.name
                    frames = extract_video_frames(video_path)
                    os.unlink(video_path)
                    st.session_state.media_content = {"type": "video", "content": frames}
                    st.rerun()
            with vid_tab2:
                youtube_url = st.text_input("Or enter a YouTube URL")
                if st.button("Process YouTube Video"):
                    video_path = download_youtube_video(youtube_url)
                    if video_path:
                        frames = extract_video_frames(video_path)
                        os.unlink(video_path)
                        st.session_state.media_content = {"type": "video", "content": frames}
                        st.rerun()

        elif selected_mode == "🎵 Audio":
            uploaded_file = st.file_uploader("Upload an Audio File", type=['mp3', 'wav', 'ogg'])
            if uploaded_file:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tfile:
                    tfile.write(uploaded_file.read())
                    st.session_state.media_content = {"type": "audio", "content": tfile.name}
                st.rerun()


# --- CHAT INTERFACE ---
# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Display media if it has been processed
if st.session_state.media_content:
    with st.chat_message("user"):
        media_type = st.session_state.media_content["type"]
        media_data = st.session_state.media_content["content"]
        
        if media_type == "image":
            st.image(media_data, caption="You uploaded this image.", width=250)
        elif media_type == "video":
            st.info(f"You attached a video ({len(media_data)} frames).")
            with st.expander("View attached frames"):
                st.image(media_data, width=150)
        elif media_type == "audio":
            st.audio(media_data)
            st.info("You attached this audio file.")

# Get new user input
if prompt := st.chat_input("Ask a question about the media or start a text chat..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Prepare message for the API
    messages_for_api = []
    
    # Add media to the beginning of the prompt for the model
    if st.session_state.media_content:
        media_type = st.session_state.media_content["type"]
        media_data = st.session_state.media_content["content"]
        
        if media_type == "image":
            messages_for_api.append({"role": "user", "content": [{"type": "image", "image": media_data}, {"type": "text", "text": prompt}]})
        elif media_type == "video":
            # Combine frames and text for the first message
            content_list = [{"type": "image", "image": frame} for frame in media_data]
            content_list.append({"type": "text", "text": prompt})
            messages_for_api.append({"role": "user", "content": content_list})
        elif media_type == "audio":
            messages_for_api.append({"role": "user", "content": [{"type": "audio", "audio": media_data}, {"type": "text", "text": prompt}]})
    
        # Clear media after it has been sent once
        st.session_state.media_content = None
    
    else: # If no media, just use the text history
        for msg in st.session_state.messages:
            messages_for_api.append({"role": msg["role"], "content": [{"type": "text", "text": msg["content"]}]})


    # Display assistant response
    with st.chat_message("assistant"):
        with st.spinner("🧠 Thinking..."):
            response = do_gemma_inference(messages_for_api, max_tokens, temperature)
            st.markdown(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})