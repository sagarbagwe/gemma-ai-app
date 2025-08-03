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

# The rest of the Streamlit code is mostly the same,
# it just doesn't handle model loading.

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
    feature = st.radio("Choose analysis type:", ["📝 Text → Text", "📸 Image Upload (Not Supported via API)", "🎥 Video Upload (Not Supported via API)"])
    st.divider()
    st.header("⚙️ Generation Settings")
    max_tokens = st.slider("Max New Tokens", 50, 2048, 512)
    temperature = st.slider("Temperature", 0.1, 1.5, 0.9, 0.05)

# --- Main App ---
st.header(f"🤖 {feature}")
st.divider()

if feature == "📝 Text → Text":
    st.markdown('<div class="feature-box">', unsafe_allow_html=True)
    messages_key = "text_messages"
    if messages_key not in st.session_state:
        st.session_state[messages_key] = []

    for msg in st.session_state[messages_key]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    prompt = st.chat_input("Enter your prompt:")
    if prompt:
        st.session_state[messages_key].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Generating..."):
                # Note: We send the whole history for context
                api_messages = [{"role": msg["role"], "content": msg["content"]} for msg in st.session_state[messages_key]]
                response_text = do_gemma_inference(api_messages, max_tokens, temperature)
                st.markdown(response_text)
        
        st.session_state[messages_key].append({"role": "assistant", "content": response_text})
    st.markdown('</div>', unsafe_allow_html=True)
else:
    st.warning("This feature is not supported in the API version of the app yet. Please select Text → Text.")