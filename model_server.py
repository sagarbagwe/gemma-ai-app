# File: model_server.py

import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
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

# This defines the structure of the data we expect to receive
class InferenceRequest(BaseModel):
    messages: list
    max_new_tokens: int = 512
    temperature: float = 0.9

# This is the main endpoint for running inference
@app.post("/generate")
async def do_inference(request: Request):
    try:
        # Manually parse the JSON from the request for robustness
        json_payload = await request.json()
        print(f"DEBUG: Received payload: {json_payload}") # For debugging

        # Validate the payload using our Pydantic model
        req = InferenceRequest(**json_payload)

        # Prepare input for the model
        inputs = tokenizer.apply_chat_template(
            req.messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to("cuda")

        # Generate output
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=req.max_new_tokens,
                temperature=req.temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode the full response
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Get the original prompt text to isolate the new response
        prompt_text = tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
        response_only = full_response.replace(prompt_text, "").strip()

        return JSONResponse(content={"response": response_only})

    except Exception as e:
        print(f"ERROR: An error occurred during inference: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Main entry point to run the server
if __name__ == "__main__":
    print("--- 🚀 Starting API server on http://127.0.0.1:8000 ---")
    uvicorn.run(app, host="127.0.0.1", port=8000)