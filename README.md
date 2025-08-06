

# Gemma 3N: First Multimodal AI Assistant


This project is a powerful, offline-first conversational AI assistant built with Streamlit. It leverages the multimodal capabilities of the `unsloth/gemma-3n-E4B-it` model to understand and process text, images, video, and audio inputs. The entire application is optimized for high-performance, on-device inference, ensuring privacy and accessibility even without an internet connection.

 ---

## 🚀 Challenge & Real-World Impact

### 🧠 Problem Statement

In low-connectivity or offline regions around the world, millions of people lack access to real-time assistance for health, education, accessibility, and crisis response. Standard AI tools that require internet connectivity exclude vulnerable populations from the benefits of smart technologies.

### 🎯 Our Mission

Our project leverages the Gemma 3N model to create an offline-first, multimodal assistant that runs on-device. By combining vision, audio, and text inputs, we empower users to access intelligent, private, and real-time support without internet access, making AI truly inclusive and life-changing.

### 💡 The Impact We Aim to Create

  * **📚 Education in Remote Areas:** An interactive tutor that uses voice + image input from a book or whiteboard and provides spoken explanations — all offline. Ideal for underserved classrooms with no connectivity.
  * **🦻 Accessibility for the Hearing-Impaired:** Live speech-to-text transcription and gesture recognition using a webcam or phone camera to help users understand conversations around them, with no privacy risk because everything stays on-device.
  * **🧘 Mental Health & Wellness:** A personal wellness companion that analyzes voice tone + facial emotion to detect stress and suggest grounding exercises — functioning offline for true privacy.
  * **🌱 Agriculture & Sustainability:** A camera-based tool that helps farmers detect plant diseases or pests using Gemma 3N’s vision capabilities and receive advice based on symptoms — entirely offline.
  * **🌪️ Disaster & Crisis Support:** A lightweight mobile app that uses voice, photo, and GPS (if available) to help users navigate emergency instructions, even in areas where the network is down.
  * **🧓 Elderly Care Support:** An offline companion for seniors that uses speech and facial recognition to monitor emotional well-being, remind them to take medication, and detect falls or confusion — all privately on-device, without internet reliance.
  * **🎓 Vocational Training:** Offline technical training assistant that guides users through physical tasks (like plumbing, electrical work, or farming techniques) by analyzing images or videos of their work and giving real-time feedback — ideal for low-resource settings.
  * **🧭 Travel Assistant for the Visually Impaired:** A wearable or phone-based tool that uses Gemma 3N’s image + audio understanding to describe surroundings, read signs, and provide navigation prompts — enabling independent mobility in unfamiliar areas, all offline.
  * **🛠️ Field Engineer Toolkit:** For workers in mining, construction, or oil fields, an on-device assistant that can recognize equipment from images and explain troubleshooting steps using natural language, even when cut off from the cloud.
  * **👮 Border & Law Enforcement Support:** An offline translator + communicator that uses Gemma 3N to enable real-time multilingual communication and document understanding in remote, no-network border or law enforcement scenarios — promoting clarity and safety.
  * **📈 Financial Literacy Coach:** An offline agent that helps underserved populations understand loans, savings, or insurance by reading printed materials aloud, explaining them in local languages, and answering voice queries using on-device processing.
  * **🔧 DIY Help Desk:** A tool for rural users that uses photo + voice input to recognize broken appliances (like fans, switches, pipes) and explain step-by-step repairs using voice and visuals — a fully local, low-cost home repair guide.

### ✨ Why Gemma 3N?

Gemma 3N is uniquely suited for this challenge because:

  - **Multimodal:** Understands images, text, and audio together.
  - **Offline-first:** Can run entirely on-device with no cloud dependency.
  - **Private & Secure:** User data never leaves the device.
  - **Customizable:** Can be fine-tuned or distilled for specific tasks (e.g., disease detection, wellness tracking, etc.).

We believe this model allows us to bridge the digital divide and reach users who are typically left behind.

-----

## ✨ Key Features

  - **🗣️ Text → Text:** Standard conversational chat with history management.
  - **🖼️ Image + Text → Text:** Upload an image and ask questions about its content.
  - **🎬 Video + Text → Text:** Analyze videos from a local file or a YouTube URL. The app extracts keyframes for the model to "see" the video content.
  - **🎵 Audio + Text → Text:** Upload an audio file, get an automatic transcription using a distilled Whisper model, and ask the AI to analyze or summarize the transcript.
  - **🚀 High-Performance Inference:** Heavily optimized for speed and low memory usage using:
      - [Unsloth](https://github.com/unslothai/unsloth) for 2x faster, memory-efficient fine-tuning and inference.
      - 4-bit quantization (`load_in_4bit=True`).
      - Advanced PyTorch optimizations (`torch.compile`, `torch.inference_mode`, `cudnn.benchmark`).
  - **🔒 Privacy-Focused:** All models run locally on your machine. No data is sent to external servers.
  - **🌐 Interactive UI:** A clean and user-friendly interface built with Streamlit.

-----

## 🛠️ Setup and Installation

This application requires a local machine with a **CUDA-enabled NVIDIA GPU** for optimal performance.

### 1\. Clone the Repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2\. Create a Virtual Environment

It's highly recommended to use a virtual environment.

```bash
# For Linux/macOS
python -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
.\venv\Scripts\activate
```

### 3\. Install PyTorch with CUDA

Install PyTorch according to the instructions on the [official website](https://pytorch.org/get-started/locally/). Choose the appropriate CUDA version for your system. For example:

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 4\. Install Dependencies

Create a `requirements.txt` file with the following content:

**`requirements.txt`**

```
streamlit
unsloth
transformers
accelerate
bitsandbytes
opencv-python-headless
numpy
Pillow
yt-dlp
librosa
soundfile
sentencepiece
torch
```

Then, install all the required packages:

```bash
pip install -r requirements.txt
```

**Note:** The `unsloth` package will handle `xformers` and other dependencies automatically.

-----

## 🏃‍♀️ How to Run the Application

Once the setup is complete, you can run the Streamlit app with the following command:

```bash
streamlit run app.py
```

*(Assuming your Python script is named `app.py`)*

Your web browser should automatically open a new tab with the application running.

### Usage Guide

1.  **Load Models:** Upon first launch, use the buttons in the left sidebar to load the **Gemma 3N Model** and the **Speech Model**. This may take a few minutes and will download the models to your local cache.
2.  **Choose Mode:** Select an analysis mode from the main screen (e.g., "🖼️ Image + Text → Text").
3.  **Upload Media:** If you choose a media-based mode, an uploader will appear. Provide the image, video, or audio file.
4.  **Start Chatting:** Type your question in the chat input box at the bottom and press Enter. The AI will process your input and generate a response.
5.  **New Chat:** To start a new conversation or analyze a new piece of media, click the "🗑️ New Chat" button.

-----

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.

-----

## 🙏 Acknowledgements

  - The **Unsloth team** for their incredible work on making LLMs faster and more accessible.
  - **Google** for the powerful open-source Gemma models.
  - The teams behind **Hugging Face Transformers**, **PyTorch**, and **Streamlit** for their foundational libraries.
