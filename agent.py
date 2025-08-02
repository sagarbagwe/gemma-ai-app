import subprocess
import sys
import os
import time
from IPython.display import display, HTML

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
        # Using DEVNULL to keep the log clean, as we print our own status messages
        subprocess.check_call(unsloth_command, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        print("✅ Unsloth and core AI libraries installed successfully.")

        print("\n📦 Pinning NumPy version to prevent conflicts...")
        numpy_command = [sys.executable, "-m", "pip", "install", "numpy<2.2"]
        subprocess.check_call(numpy_command, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        print("✅ NumPy version pinned successfully.")

        print("\n📦 Installing remaining application packages (streamlit, ngrok, yt-dlp, etc.)...")
        app_packages = [
            "streamlit", "nest_asyncio", "pyngrok", "opencv-python",
            "Pillow", "timm", "yt-dlp"
        ]
        app_command = [sys.executable, "-m", "pip", "install", "-U"] + app_packages
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
        # --- MODIFICATION START: Changed st.selectbox to st.radio ---
        feature = st.radio("Choose the type of analysis:",
            ["📝 Text → Text", "📸 Image + Text → Text", "🎥 Video (Upload) + Text → Text", "🎥 YouTube URL + Text → Text", "🎵 Audio + Text → Text", "🎬 Video + Audio + Text"],
            label_visibility="collapsed")
        # --- MODIFICATION END ---
        
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

# --- STEP 3: LAUNCH THE STREAMLIT APP ---
def launch_streamlit(summary_log):
    """
    Kills any old Streamlit process, starts a new one, creates a public URL using pyngrok, and logs the result.
    """
    print("--- 🚀 STEP 3: LAUNCHING STREAMLIT & CREATING PUBLIC URL ---")

    # ⚠️ IMPORTANT: Paste your ngrok authtoken here if the one below doesn't work.
    NGROK_AUTH_TOKEN = "30YBmNDW7XOji1e6BYS7lO1DCea_2YiLgw4UwGFWeW6NwNaiS"

    try:
        subprocess.run(["pkill", "-f", "streamlit"], capture_output=True)
        print("...Terminated any old Streamlit processes.")
        time.sleep(2)
    except FileNotFoundError:
        print("...`pkill` not found, skipping (normal on Windows).")

    command = ["streamlit", "run", "gemma_multimodal_app.py", "--server.port", "8501", "--server.headless", "true", "--browser.gatherUsageStats", "false"]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    print("...Waiting for Streamlit server to initialize...")
    time.sleep(10)

    if process.poll() is not None:
        print("--- ❌ STREAMLIT FAILED TO START ---")
        stdout, stderr = process.communicate()
        print("--- Streamlit stdout ---\n", stdout.decode())
        print("--- Streamlit stderr ---\n", stderr.decode())
        summary_log.append(("error", "❌ **Step 3: Streamlit launch failed.** The server could not start."))
        return

    print("...Streamlit server running. Connecting ngrok...")
    try:
        from pyngrok import ngrok
        if not NGROK_AUTH_TOKEN or "PASTE_YOUR" in NGROK_AUTH_TOKEN:
            display(HTML("""
            <div style="border: 2px solid #ffc107; border-radius: 10px; padding: 20px; background-color: #fff8e1; margin: 20px 0;">
                <h2 style="color: #ff8f00;">Action Required: Add Your ngrok Authtoken</h2>
                <p>To create a public URL, you need a free ngrok authtoken.</p>
                <ol>
                    <li>Go to <a href="https://dashboard.ngrok.com/get-started/your-authtoken" target="_blank">your ngrok dashboard</a>.</li>
                    <li>Copy your authtoken.</li>
                    <li>Paste it into the `NGROK_AUTH_TOKEN` variable in this script and re-run the cell.</li>
                </ol>
            </div>"""))
            summary_log.append(("error", "❌ **Step 3: ngrok launch failed.** Authtoken is missing."))
            return

        ngrok.set_auth_token(NGROK_AUTH_TOKEN)
        for tunnel in ngrok.get_tunnels():
            ngrok.disconnect(tunnel.public_url)
        public_tunnel = ngrok.connect(8501)
        public_url = public_tunnel.public_url
        
        # --- MODIFIED: ADDED A PRINT STATEMENT FOR TERMINAL VIEW ---
        print("\n" + "="*80)
        print("🎉 Your AI Assistant is Live! 🎉")
        print(f"Open this URL in your web browser: {public_url}")
        print("="*80 + "\n")
        # --- END MODIFIED SECTION ---

        # Original HTML display (for compatibility with interactive notebooks)
        display(HTML(f'''
        <div style="border: 2px solid #4CAF50; border-radius: 10px; padding: 20px; background-color: #f0fff0; margin: 20px 0;">
            <h2 style="color: #2e7d32;">🎉 Your AI Assistant is Live!</h2>
            <p>Click the link below to open the application in a new tab.</p>
            <a href="{public_url}" target="_blank" style="display: inline-block; padding: 12px 24px; background-color: #4CAF50; color: white; text-decoration: none; border-radius: 5px; font-size: 16px; font-weight: bold;">
                🚀 Open Gemma Conversational App
            </a>
            <hr style="margin: 20px 0;">
            <h3 style="color: #1976d2;">📋 Instructions:</h3>
            <ol>
                <li>Click the link above to open the app.</li>
                <li>In the app's sidebar, click <strong>"Load Gemma 3N Model"</strong> (this can take a few minutes).</li>
                <li>Once loaded, select a feature from the <strong>list in the sidebar</strong> by clicking it.</li>
                <li>Upload your media, type a question, and click "Send".</li>
                <li>Ask follow-up questions to continue the conversation!</li>
            </ol>
        </div>'''))
        summary_log.append(("success", "✅ **Step 3: App is live!** Streamlit and ngrok launched successfully."))
    except Exception as e:
        error_message = f"❌ **Step 3: Failed to create public URL.** Error: {e}"
        print(f"--- ❌ FAILED TO CREATE PUBLIC URL ---")
        print(f"Error details: {e}")
        summary_log.append(("error", error_message))

# --- NEW STEP 4: DISPLAY EXECUTION SUMMARY ---
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
            bg_color = "#e8f5e9" if status == "success" else "#ffcdd2"
            text_color = "#2e7d32" if status == "success" else "#c62828"
            summary_html += f'<li style="padding: 10px; border-radius: 5px; margin-top: 8px; background-color: {bg_color}; color: {text_color}; font-size: 16px;">{message}</li>'

    summary_html += "</ul></div>"
    display(HTML(summary_html))

# --- MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    execution_summary = []
    try:
        install_requirements(execution_summary)
        create_streamlit_app_file(execution_summary)
        launch_streamlit(execution_summary)
    except Exception:
        print("\n--- 🛑 SCRIPT HALTED DUE TO A CRITICAL ERROR ---")
    finally:
        display_execution_summary(execution_summary)