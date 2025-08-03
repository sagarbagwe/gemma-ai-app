import subprocess
import sys
import os
import time
import shutil
from IPython.display import display, HTML

# --- NEW STEP 0: CLEAR CACHES ---
def clear_caches(summary_log):
    """
    Clears pip and local Python __pycache__ directories.
    """
    print("--- 🗑️ STEP 0: CLEARING CACHES ---")
    try:
        # Clear pip cache
        print("... Clearing pip cache...")
        pip_cache_command = [sys.executable, "-m", "pip", "cache", "purge"]
        subprocess.check_call(pip_cache_command, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        print("✅ Pip cache cleared successfully.")

        # Clear local __pycache__ directories
        print("... Clearing local __pycache__ directories...")
        for root, dirs, files in os.walk("."):
            if "__pycache__" in dirs:
                pycache_path = os.path.join(root, "__pycache__")
                print(f"   - Removing {pycache_path}")
                shutil.rmtree(pycache_path)
        print("✅ Local __pycache__ directories cleared.")
        
        print("\n--- ✅ CACHE CLEARING COMPLETE ---\n")
        summary_log.append(("success", "✅ **Step 0: Caches cleared.** Pip and local __pycache__ have been removed."))
    except Exception as e:
        # This step is not critical, so we'll log a warning instead of an error.
        error_message = f"⚠️ **Step 0: Cache clearing failed.** Proceeding anyway. Error: {e}"
        print(f"\n⚠️ An error occurred during cache clearing (non-critical): {e}")
        summary_log.append(("warning", error_message))

# --- STEP 1: INSTALLATION OF DEPENDENCIES ---
def install_requirements(summary_log):
    """
    Installs all necessary Python packages and logs the result.
    """
    print("--- ⚙️ STEP 1: INSTALLING PACKAGES (Official Unsloth Method with Forced Reinstall) ---")
    try:
        print("📦 Forcing re-installation of unsloth and its core dependencies...")
        unsloth_command = [
            sys.executable, "-m", "pip", "install", "--no-cache-dir", "--force-reinstall",
            "unsloth[colab-new]@git+https://github.com/unslothai/unsloth.git"
        ]
        subprocess.check_call(unsloth_command, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        print("✅ Unsloth and core AI libraries installed successfully.")

        print("\n📦 Pinning NumPy version to prevent conflicts...")
        numpy_command = [sys.executable, "-m", "pip", "install", "--no-cache-dir", "numpy<2.2"]
        subprocess.check_call(numpy_command, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        print("✅ NumPy version pinned successfully.")

        print("\n📦 Installing remaining application packages (streamlit, yt-dlp, etc.)...")
        # MODIFIED: Removed 'pyngrok' from the list of packages.
        app_packages = [
            "streamlit", "nest_asyncio", "opencv-python",
            "Pillow", "timm", "yt-dlp"
        ]
        app_command = [sys.executable, "-m", "pip", "install", "--no-cache-dir", "-U"] + app_packages
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
    Writes the Python code for the Streamlit application into a .py file and logs the result.
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
    try:
        with open("gemma_multimodal_app.py", "w", encoding="utf-8") as f:
            f.write(app_code)
        print("--- ✅ APP FILE CREATED SUCCESSFULLY ---\n")
        summary_log.append(("success", "✅ **Step 2: App file created.** 'gemma_multimodal_app.py' is ready."))
    except Exception as e:
        error_message = f"❌ **Step 2: Failed to create app file.** The script cannot launch. Error: {e}"
        print(f"--- ❌ FAILED TO CREATE APP FILE: {e} ---\n")
        summary_log.append(("error", error_message))
        raise

# --- MODIFIED STEP 3: LAUNCH THE STREAMLIT APP FOR VM ---
def launch_streamlit(summary_log):
    """
    Kills any old Streamlit process and starts a new one accessible on the VM's network.
    """
    print("--- 🚀 STEP 3: LAUNCHING STREAMLIT FOR VM ACCESS ---")

    try:
        # Terminate any old Streamlit processes to ensure a clean start
        subprocess.run(["pkill", "-f", "streamlit"], capture_output=True)
        print("...Terminated any old Streamlit processes.")
        time.sleep(2)
    except FileNotFoundError:
        print("...`pkill` not found, skipping (normal on Windows).")

    # MODIFIED: Command now binds to 0.0.0.0 to be accessible externally via the VM's IP.
    command = [
        "streamlit", "run", "gemma_multimodal_app.py",
        "--server.port", "8501",
        "--server.address", "0.0.0.0",
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false"
    ]
    
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print("...Waiting for Streamlit server to initialize (approx. 10 seconds)...")
    time.sleep(10)

    # Check if the process terminated prematurely
    if process.poll() is not None:
        print("--- ❌ STREAMLIT FAILED TO START ---")
        stdout, stderr = process.communicate()
        print("--- Streamlit stdout ---\n", stdout.decode())
        print("--- Streamlit stderr ---\n", stderr.decode())
        summary_log.append(("error", "❌ **Step 3: Streamlit launch failed.** The server could not start."))
        return

    # MODIFIED: Removed ngrok logic and replaced with instructions for VM access.
    display(HTML('''
    <div style="border: 2px solid #4CAF50; border-radius: 10px; padding: 20px; background-color: #f0fff0; margin: 20px 0;">
        <h2 style="color: #2e7d32;">🎉 Your AI Assistant is Running!</h2>
        <p>The Streamlit server has been started in the background. To access it, you need your VM's external IP address.</p>
        
        <h3 style="color: #1976d2;">📋 Access Instructions:</h3>
        <ol>
            <li><strong>Find your VM's External IP Address.</strong> You can usually find this in your cloud provider's console (GCP, AWS, Azure, etc.).</li>
            <li>
                <strong>Construct the URL:</strong> Open a web browser on your local machine and go to:
                <br>
                <code style="background-color: #e0e0e0; padding: 5px 8px; border-radius: 4px; font-size: 16px;">
                    http://&lt;YOUR_VM_EXTERNAL_IP&gt;:8501
                </code>
                <br>
                <small>(Replace <code>&lt;YOUR_VM_EXTERNAL_IP&gt;</code> with the actual IP address).</small>
            </li>
            <li><strong>Firewall Rules:</strong> Ensure that your VM's firewall allows incoming TCP traffic on port <strong>8501</strong>. You may need to add a new firewall rule in your cloud provider's settings.</li>
        </ol>

        <hr style="margin: 20px 0;">
        <h3 style="color: #1976d2;">🚀 Next Steps in the App:</h3>
        <ol>
            <li>In the app's sidebar, click <strong>"Load Gemma 3N Model"</strong>. This can take a few minutes.</li>
            <li>Once loaded, select a feature from the list in the sidebar.</li>
            <li>Upload your media, type your question, and click "Send"!</li>
        </ol>
        <p><strong>Note:</strong> The Streamlit process is running in the background. To stop it, you will need to manually terminate the process (e.g., by interrupting the kernel or using <code>pkill streamlit</code> in a new terminal).</p>
    </div>'''))
    summary_log.append(("success", "✅ **Step 3: App is running!** Access it via your VM's external IP on port 8501."))

# --- STEP 4: DISPLAY EXECUTION SUMMARY ---
def display_execution_summary(summary_log):
    """
    Displays a final, formatted summary of the script's execution.
    """
    print("\n\n" + "="*80)
    print("--- 📋 EXECUTION SUMMARY ---")
    print("="*80)

    summary_html = """
    <div style="border: 2px solid #1976d2; border-radius: 10px; padding: 20px; background-color: #e3f2fd; margin: 20px 0; font-family: sans-serif;">
        <h2 style="color: #1565c0; border-bottom: 2px solid #bbdefb; padding-bottom: 10px;">Execution Report</h2>
        <ul style="list-style-type: none; padding-left: 0;">
    """

    if not summary_log:
        summary_html += '<li style="padding: 10px; color: #555;">No actions were performed.</li>'
    else:
        for status, message in summary_log:
            bg_color = "#e8f5e9" if status == "success" else ("#fff8e1" if status == "warning" else "#ffcdd2")
            text_color = "#2e7d32" if status == "success" else ("#ff8f00" if status == "warning" else "#c62828")
            summary_html += f'<li style="padding: 10px; border-radius: 5px; margin-top: 8px; background-color: {bg_color}; color: {text_color}; font-size: 16px;">{message}</li>'

    summary_html += "</ul></div>"
    display(HTML(summary_html))

# --- MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    execution_summary = []
    try:
        # Each step is now self-contained and reports its status to the summary list.
        clear_caches(execution_summary)
        install_requirements(execution_summary)
        create_streamlit_app_file(execution_summary)
        launch_streamlit(execution_summary)
    except Exception:
        print("\n--- 🛑 SCRIPT HALTED DUE TO A CRITICAL ERROR ---")
    finally:
        # This block will always run, ensuring the summary is displayed regardless of errors.
        display_execution_summary(execution_summary)