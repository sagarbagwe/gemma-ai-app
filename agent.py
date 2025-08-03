# File: agent.py

import streamlit as st
import requests  # We use requests to call our API
import tempfile
import cv2
import numpy as np
from PIL import Image
import os
import yt_dlp

# --- No model loading here! ---

st.set_page_config(
    page_title="Gemma 3N Conversational AI",
    page_icon="🤖",
    layout="wide"
)

# --- INFERENCE FUNCTION (Updated to call the API) ---
def do_gemma_inference(messages, max_new_tokens, temperature):
    # This is the address of your model_server.py
    api_url = "http://127.0.0.1:8000/generate"
    payload = {
        "messages": messages,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature
    }
    try:
        response = requests.post(api_url, json=payload)
        response.raise_for_status()  # Raise an exception for bad status codes
        data = response.json()
        if "response" in data:
            return data["response"]
        else:
            return f"API Error: {data.get('error', 'Unknown error')}"
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Could not connect to the model server. Is it running? Error: {e}")
        return ""

# --- UTILITY FUNCTIONS (from your original notebook) ---
@st.cache_data
def download_youtube_video(url):
    temp_dir = tempfile.gettempdir()
    video_path = os.path.join(temp_dir, f"{os.urandom(8).hex()}.mp4")
    ydl_opts = {'format': 'best[ext=mp4]/best', 'outtmpl': video_path, 'quiet': True, 'merge_output_format': 'mp4'}
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            st.info("Downloading video...")
            ydl.download([url])
        st.success("Download complete.")
        return video_path
    except Exception as e:
        st.error(f"YouTube download failed: {e}")
        return None

@st.cache_data
def extract_video_frames(video_path, max_frames=8):
    # This function is kept for UI purposes, but the frames won't be sent to the current text-only API
    try:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = max(1, total_frames // max_frames)
        frames, count, extracted = [], 0, 0
        while cap.isOpened() and extracted < max_frames:
            ret, frame = cap.read()
            if not ret: break
            if count % frame_interval == 0:
                frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                extracted += 1
            count += 1
        cap.release()
        return frames
    except Exception as e:
        st.error(f"Frame extraction failed: {e}")
        return []


# --- UI CODE ---
# Custom CSS
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: bold; text-align: center; margin-bottom: 2rem; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .feature-box { padding: 1.5rem; border-radius: 10px; border: 1px solid #e0e0e0; margin: 1rem 0; background-color: #f8f9fa; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }
    .stChatMessage { background-color: #ffffff; border-radius: 8px; padding: 12px; border: 1px solid #e6e6e6; }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown('<h1 class="main-header">🤖 Gemma 3N Conversational AI</h1>', unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.header("🔧 Model Status")
    st.success("✅ Ready to generate via API")
    st.divider()
    st.header("🎯 Select Feature")
    # Note: The model server only supports text, so other features are disabled.
    feature = st.radio(
        "Choose analysis type:",
        ["📝 Text → Text", "📸 Image + Text", "🎥 Video + Text", "🎵 Audio + Text"]
    )
    st.divider()
    st.header("⚙️ Generation Settings")
    max_tokens = st.slider("Max New Tokens", 50, 2048, 512)
    temperature = st.slider("Temperature", 0.1, 1.5, 0.9, 0.05)

# --- Main App ---
st.header(f"Mode: {feature}")
st.divider()

if feature == "📝 Text → Text":
    st.markdown('<div class="feature-box">', unsafe_allow_html=True)
    messages_key = "text_messages"
    if messages_key not in st.session_state:
        st.session_state[messages_key] = []

    # Display chat history
    for msg in st.session_state[messages_key]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    # Get user input
    prompt = st.chat_input("Enter your prompt...")
    if prompt:
        # Add user message to history and display it
        st.session_state[messages_key].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("🧠 Thinking..."):
                # We send the whole history to the API for context
                api_messages = [{"role": msg["role"], "content": msg["content"]} for msg in st.session_state[messages_key]]
                response_text = do_gemma_inference(api_messages, max_tokens, temperature)
                st.markdown(response_text)
        
        # Add assistant response to history
        st.session_state[messages_key].append({"role": "assistant", "content": response_text})
    st.markdown('</div>', unsafe_allow_html=True)
else:
    # Show a warning for multimodal features since the API doesn't support them
    st.warning(f"⚠️ The **{feature}** mode is not supported by the current model server.")
    st.info("The provided `model_server.py` can only process text. To enable image or audio features, the server code would need to be updated to handle file uploads and multimodal inference.")