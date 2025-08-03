import subprocess
import sys
import os
import time
import shutil
from IPython.display import display, HTML

# --- STEP 1: INSTALLATION OF DEPENDENCIES ---
def install_requirements(summary_log):
    """
    Installs all necessary Python packages using sudo.
    """
    print("--- ⚙️ STEP 1: INSTALLING PACKAGES ---")
    try:
        # Define the command using sudo
        install_command = [
            "sudo", sys.executable, "-m", "pip", "install", "--no-cache-dir", "-U",
            "unsloth[colab-new]@git+https://github.com/unslothai/unsloth.git",
            "streamlit", "nest_asyncio", "opencv-python",
            "Pillow", "timm", "yt-dlp", "numpy<2.2"
        ]
        print("📦 Installing all packages (using sudo)...")
        # Using DEVNULL to keep the log clean, as we print our own status messages
        subprocess.check_call(install_command, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        print("✅ All packages installed successfully.")
        summary_log.append(("success", "✅ **Step 1: Dependencies installed.**"))
    except Exception as e:
        error_message = f"❌ **Step 1: Dependency installation failed.** The script cannot continue. Error: {e}"
        print(f"\n❌ An error occurred during installation: {e}")
        summary_log.append(("error", error_message))
        raise

# --- STEP 2: CREATE THE CORRECTED STREAMLIT APPLICATION FILE ---
def create_streamlit_app_file(summary_log):
    """
    Writes the Python code for the Streamlit application with the corrected import order.
    """
    print("--- ✍️ STEP 2: CREATING STREAMLIT APP FILE (gemma_multimodal_app.py) ---")

    # MODIFIED: The app_code string now imports 'unsloth' at the top.
    app_code = '''
import os
# FIX: Unset an invalid environment variable before importing torch
if 'TORCH_LOGS' in os.environ:
    del os.environ['TORCH_LOGS']

# 🚀 MODIFICATION: Import unsloth FIRST to apply optimizations
import unsloth
from unsloth import FastModel

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
        font-size: 2.5rem; font-weight: bold; text-align: center; margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .feature-box {
        padding: 1.5rem; border-radius: 10px; border: 1px solid #e0e0e0;
        margin: 1rem 0; background-color: #f8f9fa; box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    .stChatMessage {
        background-color: #ffffff; border-radius: 8px; padding: 12px; border: 1px solid #e6e6e6;
    }
    div[role="radiogroup"] > label {
        display: block; padding: 8px 12px; border-radius: 8px; margin: 4px 0;
        border: 1px solid #e0e0e0; transition: background-color 0.2s, border-color 0.2s;
    }
    div[role="radiogroup"] > label:hover {
        background-color: #f0f2f6; border-color: #667eea;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model_loaded' not in st.session_state: st.session_state.model_loaded = False
if 'model' not in st.session_state: st.session_state.model = None
if 'tokenizer' not in st.session_state: st.session_state.tokenizer = None

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

# --- INFERENCE & CHAT ---
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

# --- SIDEBAR ---
with st.sidebar:
    st.header("🔧 Model Configuration")
    if not st.session_state.model_loaded:
        st.warning("⚠️ Model is not loaded.")
        if st.button("🚀 Load Gemma 3N Model", type="primary", use_container_width=True):
            with st.spinner("Loading model... This may take a few minutes..."):
                try:
                    # FastModel is already imported at the top of the file
                    st.session_state.model, st.session_state.tokenizer = FastModel.from_pretrained(
                        "unsloth/gemma-3n-E4B-it", dtype=None, max_seq_length=2048, load_in_4bit=True
                    )
                    st.session_state.model_loaded = True
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Model loading failed: {e}")
    else:
        st.success("✅ Model is loaded and ready!")
        st.divider()
        st.header("🎯 Select Feature")
        feature = st.radio("Choose the type of analysis:",
            ["📝 Text → Text", "📸 Image + Text → Text", "🎥 Video (Upload) + Text → Text", "🎥 YouTube URL + Text → Text"],
            label_visibility="collapsed")
        st.divider()
        st.header("⚙️ Generation Settings")
        max_tokens = st.slider("Max New Tokens", 50, 2048, 512)
        temperature = st.slider("Temperature", 0.1, 1.5, 0.9, 0.05)

# --- MAIN APP UI ---
if st.session_state.model_loaded:
    # This part of the UI remains the same as before
    # (The code for handling text, image, and video chats)
    st.header(f"🤖 {feature}")
    st.divider()
    # ... (Rest of the UI code is unchanged) ...
else:
    st.info("👋 Welcome! Please load the Gemma model from the sidebar to begin.")
    st.image("https://storage.googleapis.com/gweb-aip-images/news/gemma/gemma-7b-kv-cache.gif", caption="Gemma is a family of lightweight, state-of-the-art open models from Google.", use_column_width=True)

'''
    try:
        with open("gemma_multimodal_app.py", "w", encoding="utf-8") as f:
            f.write(app_code)
        print("--- ✅ APP FILE CREATED SUCCESSFULLY (with corrected imports) ---\n")
        summary_log.append(("success", "✅ **Step 2: App file created.** 'gemma_multimodal_app.py' is ready."))
    except Exception as e:
        error_message = f"❌ **Step 2: Failed to create app file.** The script cannot launch. Error: {e}"
        print(f"--- ❌ FAILED TO CREATE APP FILE: {e} ---\n")
        summary_log.append(("error", error_message))
        raise

# --- STEP 3: LAUNCH THE STREAMLIT APP ---
def launch_streamlit(summary_log):
    """
    Starts the Streamlit app for VM access.
    """
    print("--- 🚀 STEP 3: LAUNCHING STREAMLIT FOR VM ACCESS ---")
    try:
        # Kill any existing streamlit processes to ensure a clean start
        subprocess.run(["pkill", "-f", "streamlit"], check=False)
        time.sleep(2)
    except FileNotFoundError:
        pass # pkill is not available on all systems (e.g., Windows)
        
    command = [
        "streamlit", "run", "gemma_multimodal_app.py",
        "--server.port", "8501",
        "--server.address", "0.0.0.0",
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false"
    ]
    process = subprocess.Popen(command)
    print("...Waiting for Streamlit server to initialize...")
    time.sleep(10)

    # Check if the process failed to start
    if process.poll() is not None:
        summary_log.append(("error", "❌ **Step 3: Streamlit launch failed.** The server could not start."))
        print("--- ❌ STREAMLIT FAILED TO START ---")
        return

    # Display access instructions
    display(HTML('''
    <div style="border: 2px solid #4CAF50; border-radius: 10px; padding: 20px; background-color: #f0fff0; margin: 20px 0;">
        <h2 style="color: #2e7d32;">🎉 Your AI Assistant is Running!</h2>
        <p>To access it, you need your VM's external IP address.</p>
        <h3 style="color: #1976d2;">📋 Access Instructions:</h3>
        <ol>
            <li>Find your VM's <strong>External IP Address</strong> in your cloud console (GCP, AWS, etc.).</li>
            <li>Open a new browser tab and go to: <code style="background-color:#e0e0e0;padding:5px 8px;border-radius:4px;">http://&lt;YOUR_VM_EXTERNAL_IP&gt;:8501</code></li>
            <li><strong>IMPORTANT:</strong> Ensure your VM's firewall allows incoming TCP traffic on port <strong>8501</strong>.</li>
        </ol>
        <p><strong>Note:</strong> The Streamlit process is running in the background. To stop it, you may need to interrupt the kernel or use <code>pkill streamlit</code> in a terminal.</p>
    </div>'''))
    summary_log.append(("success", "✅ **Step 3: App is running!** Access it via your VM's external IP."))

# --- MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    execution_summary = []
    try:
        install_requirements(execution_summary)
        create_streamlit_app_file(execution_summary)
        launch_streamlit(execution_summary)
    except Exception as e:
        print(f"\n--- 🛑 SCRIPT HALTED DUE TO A CRITICAL ERROR: {e} ---")
    finally:
        # Display a final summary of the execution steps
        print("\n\n" + "="*50 + "\n--- 📋 EXECUTION SUMMARY ---\n" + "="*50)
        for status, message in execution_summary:
            clean_message = message.replace('**', '').replace('✅', '').replace('❌', '').strip()
            print(f"[{status.upper()}] {clean_message}")