# File: agent.py

import streamlit as st
import requests

st.set_page_config(
    page_title="Gemma 3N Conversational AI",
    page_icon="🤖",
    layout="wide"
)

# This function calls our API server to get a response
def do_gemma_inference(messages, max_new_tokens, temperature):
    api_url = "http://127.0.0.1:8000/generate"
    payload = {
        "messages": messages,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature
    }
    try:
        response = requests.post(api_url, json=payload, timeout=120) # Added timeout
        response.raise_for_status()
        data = response.json()
        return data.get("response", f"API Error: {data.get('error', 'Unknown error')}")
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Could not connect to the model server. Please ensure it's running. Error: {e}")
        return ""

# --- UI CODE ---
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: bold; text-align: center; margin-bottom: 2rem; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .stChatMessage { background-color: #ffffff; border-radius: 8px; padding: 12px; border: 1px solid #e6e6e6; }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🤖 Gemma 3N Conversational AI</h1>', unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.header("🔧 Model Status")
    st.success("✅ Ready to generate via API")
    st.divider()
    st.header("🎯 Select Feature")
    feature = st.radio(
        "Choose analysis type:",
        ["📝 Text → Text", "📸 Image + Text (Not Implemented)", "🎥 Video + Text (Not Implemented)"]
    )
    st.divider()
    st.header("⚙️ Generation Settings")
    max_tokens = st.slider("Max New Tokens", 50, 2048, 512)
    temperature = st.slider("Temperature", 0.1, 1.5, 0.9, 0.05)

# --- Main App Body ---
st.header(f"Mode: {feature}")
st.divider()

if feature == "📝 Text → Text":
    messages_key = "text_messages"
    if messages_key not in st.session_state:
        st.session_state[messages_key] = []

    # Display chat history
    for msg in st.session_state[messages_key]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    # Get user input
    if prompt := st.chat_input("Enter your prompt here..."):
        st.session_state[messages_key].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("🧠 Thinking..."):
                # We send the whole history to the API for context
                api_messages = st.session_state[messages_key]
                response_text = do_gemma_inference(api_messages, max_tokens, temperature)
                st.markdown(response_text)
        
        st.session_state[messages_key].append({"role": "assistant", "content": response_text})
else:
    st.warning(f"⚠️ The **{feature}** mode is not supported by the current API server.")
    st.info("The `model_server.py` script currently only handles text. To enable other features, the server code would need to be updated.")