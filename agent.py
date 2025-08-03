import subprocess
import sys
import os
import time
from IPython.display import display, HTML
import socket

# --- STEP 1: INSTALLATION OF DEPENDENCIES ---
def install_requirements(summary_log):
    """
    Installs all necessary Python packages and logs the result.
    """
    print("--- ⚙️ STEP 1: INSTALLING PACKAGES ---")
    try:
        print("📦 Installing unsloth for dedicated VMs...")
        # --- CORRECTED: Use standard unsloth installation for VMs ---
        unsloth_command = [
            sys.executable, "-m", "pip", "install", "unsloth"
        ]
        subprocess.check_call(unsloth_command, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        print("✅ Unsloth installed successfully.")

        print("\n📦 Installing remaining application packages...")
        app_packages = [
            "streamlit", "nest_asyncio", "opencv-python",
            "Pillow", "timm", "yt-dlp", "numpy<2.2"
        ]
        # --- CORRECTED: Removed the '-U' (upgrade) flag to prevent conflicts ---
        app_command = [sys.executable, "-m", "pip", "install"] + app_packages
        subprocess.check_call(app_command, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        print("✅ Application packages installed successfully.")

        print("\n--- ✅ INSTALLATION COMPLETE ---\n")
        summary_log.append(("success", "✅ **Step 1: Dependencies installed.** All required packages are ready."))
    except Exception as e:
        error_message = f"❌ **Step 1: Dependency installation failed.** The script cannot continue. Error: {e}"
        print(f"\n❌ An error occurred during installation: {e}")
        summary_log.append(("error", error_message))
        raise

# --- STEP 2: CREATE THE STREAMLIT APPLICATION FILE ---
def create_streamlit_app_file(summary_log):
    """
    Writes the Python code for the Streamlit application into a .py file.
    """
    print("--- ✍️ STEP 2: CREATING STREAMLIT APP FILE (gemma_multimodal_app.py) ---")
    app_code = '''
import os
# FIX: Unset the invalid environment variable before importing torch
if 'TORCH_LOGS' in os.environ:
    del os.environ['TORCH_LOGS']

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
    page_title="Gemma Conversational AI",
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
st.markdown('<h1 class="main-header">🤖 Gemma Conversational AI</h1>', unsafe_allow_html=True)

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
    st.header(f"{feature}")
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

    if feature == "📝 Text → Text":
        messages_key = feature_container("text")
        text_prompt = st.text_area("Enter your prompt:", height=150)
        if st.button("✍️ Send Message", type="primary", use_container_width=True):
            handle_chat_submission(messages_key, text_prompt, lambda p: [{"type": "text", "text": p}])
        st.markdown('</div>', unsafe_allow_html=True)

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

    # Add other elif blocks for other features here...

else:
    st.info("👋 Welcome! Please load the Gemma model from the sidebar to begin.")
    st.image("https://storage.googleapis.com/gweb-aip-images/news/gemma/gemma-7b-kv-cache.gif", caption="Gemma is a family of lightweight, state-of-the-art open models from Google.", use_column_width=True)
'''
    try:
        with open("gemma_multimodal_app.py", "w", encoding="utf-8") as f:
            f.write(app_code)
        print("--- ✅ APP FILE CREATED SUCCESSFULLY ---\n")
        summary_log.append(("success", "✅ **Step 2: App file created.** 'gemma_multimodal_app.py' is ready."))
    except Exception as e:
        error_message = f"❌ **Step 2: Failed to create app file.** Error: {e}"
        print(f"--- ❌ FAILED TO CREATE APP FILE: {e} ---\n")
        summary_log.append(("error", error_message))
        raise

# --- STEP 3: LAUNCH THE STREAMLIT APP DIRECTLY ---
def launch_streamlit(summary_log):
    """
    Kills any old Streamlit process and starts a new one directly.
    """
    print("--- 🚀 STEP 3: LAUNCHING STREAMLIT APPLICATION ---")
    PORT = 8501

    try:
        subprocess.run(["pkill", "-f", "streamlit run gemma_multimodal_app.py"], capture_output=True)
        print("...Terminated any old Streamlit processes.")
        time.sleep(2)
    except FileNotFoundError:
        print("...`pkill` not found, skipping (normal on Windows).")

    command = ["streamlit", "run", "gemma_multimodal_app.py", "--server.port", str(PORT), "--server.headless", "true", "--browser.gatherUsageStats", "false"]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print("...Streamlit server is starting in the background.")
    time.sleep(10)

    if process.poll() is not None:
        print("--- ❌ STREAMLIT FAILED TO START ---")
        stdout, stderr = process.communicate()
        print("--- Streamlit stdout ---\n", stdout.decode())
        print("--- Streamlit stderr ---\n", stderr.decode())
        summary_log.append(("error", "❌ **Step 3: Streamlit launch failed.** The server could not start."))
        return

    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        vm_ip = s.getsockname()[0]
        s.close()
    except Exception:
        vm_ip = "YOUR_VM_IP_ADDRESS"

    # --- MODIFIED: Display direct access instructions for ipython ---
    access_message = f'''
    <div style="border: 2px solid #4CAF50; border-radius: 10px; padding: 20px; background-color: #f0fff0; margin: 20px 0;">
        <h2 style="color: #2e7d32;">🎉 Your AI Assistant is Live!</h2>
        <p>You can now access the application directly in your browser using your VM's IP address.</p>
        <p>Open this URL in a new tab:</p>
        <a href="http://{vm_ip}:{PORT}" target="_blank" style="display: inline-block; padding: 12px 24px; background-color: #4CAF50; color: white; text-decoration: none; border-radius: 5px; font-size: 18px; font-weight: bold;">
            http://{vm_ip}:{PORT}
        </a>
    </div>
    '''
    display(HTML(access_message))
    summary_log.append(("success", f"✅ **Step 3: App is live!** Access it at http://{vm_ip}:{PORT}"))

# --- STEP 4: DISPLAY EXECUTION SUMMARY ---
def display_execution_summary(summary_log):
    """
    Displays a final, formatted summary of the script's execution.
    """
    print("\n\n" + "="*80)
    print("--- 📋 EXECUTION SUMMARY ---")
    print("="*80)

    if not summary_log:
        print("No actions were performed.")
        return

    for status, message in summary_log:
        clean_message = message.replace("**", "").replace("✅", "[SUCCESS]").replace("❌", "[ERROR]")
        print(f"- {clean_message}")
    print("="*80)


# --- MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    execution_summary = []
    try:
        install_requirements(execution_summary)
        create_streamlit_app_file(execution_summary)
        launch_streamlit(execution_summary)
        print("\n--- SCRIPT COMPLETE ---")
        print("The Streamlit application is running in the background.")
        print("To stop the app, you can close this terminal or find and kill the process.")

    except Exception as e:
        print(f"\n--- 🛑 SCRIPT HALTED DUE TO A CRITICAL ERROR: {e} ---")
    finally:
        display_execution_summary(execution_summary)
