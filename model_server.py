# File: model_server.py

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from unsloth import FastModel

print("--- 🤖 Loading model... This will take a moment. ---")

# Load the model and tokenizer ONCE when the server starts
model, tokenizer = FastModel.from_pretrained(
    "unsloth/gemma-3n-E4B-it",
    dtype=None,
    max_seq_length=2048,
    load_in_4bit=True,
)
print("--- ✅ Model loaded successfully! ---")

# Initialize the FastAPI app
app = FastAPI()

# Define the structure of the request we expect
class InferenceRequest(BaseModel):
    messages: list
    max_new_tokens: int = 512
    temperature: float = 0.9

# Define an endpoint for inference
@app.post("/generate")
def do_inference(req: InferenceRequest):
    try:
        # Get the data from the request
        messages = req.messages
        max_new_tokens = req.max_new_tokens
        temperature = req.temperature

        # Prepare input for the model
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to("cuda")

        # Generate output
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode the full response
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Get the original prompt text
        prompt_text = tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
        # Isolate just the newly generated text
        response_only = full_response.replace(prompt_text, "").strip()

        return {"response": response_only}

    except Exception as e:
        return {"error": str(e)}

# --- Main entry point to run the server ---
if __name__ == "__main__":
    print("--- 🚀 Starting API server on http://127.0.0.1:8000 ---")
    uvicorn.run(app, host="127.0.0.1", port=8000)