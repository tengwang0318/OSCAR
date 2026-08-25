#!/usr/bin/env python3

import argparse
import base64
import io
import math
import time
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
import uvicorn

app = FastAPI(title="VLM Ranker Server (CIR)")

# Global model (initialized in main)
model = None
processor = None
tokenizer = None
yes_token = None
no_token = None
sampling_params = None


class RankCIRRequest(BaseModel):
    query: str
    reference_image: str  # base64 encoded reference image
    candidate_images: List[str]  # base64 encoded candidate images


class RankCIRResponse(BaseModel):
    scores: List[float]
    tokens: List[str]


def init_model(model_path: str, tensor_parallel_size: int, max_model_len: int, gpu_memory_utilization: float, mm_processor_cache_gb: float, disable_mm_cache: bool = False):
    """Initialize model (called once at startup)."""
    global model, processor, tokenizer, yes_token, no_token, sampling_params
    
    from vllm import LLM, SamplingParams
    from transformers import AutoProcessor
    
    print(f"🔧 Loading VLM Ranker (CIR): {model_path}")
    print(f"   GPUs: {tensor_parallel_size} | Max len: {max_model_len} | MM cache: {mm_processor_cache_gb}GB | Disable cache: {disable_mm_cache}")
    
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = processor.tokenizer
    yes_token = tokenizer("yes", add_special_tokens=False).input_ids[0]
    no_token = tokenizer("no", add_special_tokens=False).input_ids[0]
    
    # Prepare mm_processor_kwargs
    mm_kwargs = {"cache_size_gb": mm_processor_cache_gb}
    if disable_mm_cache:
        # Disable caching entirely to avoid cache miss errors (trades speed for stability)
        mm_kwargs["cache_size_gb"] = 0
    
    model = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=True,
        limit_mm_per_prompt={"image": 2},  # CIR needs 2 images per request
        mm_processor_kwargs=mm_kwargs,
    )
    
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=1,
        logprobs=2,
        allowed_token_ids=[yes_token, no_token],
    )
    
    print("✅ VLM Ranker Server (CIR) ready")


def decode_image(img_b64: str) -> Optional[Image.Image]:
    """Decode base64 image."""
    try:
        img_data = base64.b64decode(img_b64)
        return Image.open(io.BytesIO(img_data)).convert("RGB")
    except Exception:
        return None


def compute_score(logprobs: dict) -> tuple:
    """Compute P(yes) from logprobs."""
    yes_lp = logprobs.get(yes_token)
    no_lp = logprobs.get(no_token)
    
    yes_lp = yes_lp.logprob if yes_lp else -10.0
    no_lp = no_lp.logprob if no_lp else -10.0
    
    yes_score = math.exp(yes_lp)
    no_score = math.exp(no_lp)
    score = yes_score / (yes_score + no_score)
    
    token = "yes" if yes_score > no_score else "no"
    return score, token


@app.post("/rank_cir", response_model=RankCIRResponse)
async def rank_cir(request: RankCIRRequest):
    """Rank candidate images against reference image and query. Returns P(yes) scores."""
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    # Decode reference image
    ref_image = decode_image(request.reference_image)
    if ref_image is None:
        raise HTTPException(status_code=400, detail="Failed to decode reference image")
    
    # Build prompt for CIR
    messages = [{
        "role": "user",
        "content": [
            {"type": "image"},  # Reference image
            {"type": "image"},  # Candidate image
            {"type": "text", "text": (
                f"Modification Query: '{request.query}'\n\n"
                "The first image is the reference image. "
                "The second image is a candidate image.\n\n"
                "Does the candidate image match what you would expect after applying "
                f"the modification query to the reference image? Modification Query: '{request.query}'.\n\n"
                "Answer with 'yes' or 'no' only."
            )}
        ]
    }]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Decode candidate images and build batch inputs
    inputs = []
    valid_indices = []
    
    for i, img_b64 in enumerate(request.candidate_images):
        cand_image = decode_image(img_b64)
        if cand_image is not None:
            inputs.append({
                "prompt": prompt, 
                "multi_modal_data": {"image": [ref_image, cand_image]}  # 2 images
            })
            valid_indices.append(i)
    
    if not inputs:
        raise HTTPException(status_code=400, detail="No valid candidate images")
    
    # Batch inference with retry mechanism for cache errors
    max_retries = 3
    last_error = None
    
    for attempt in range(max_retries):
        try:
            outputs = model.generate(inputs, sampling_params)
            break  # Success, exit retry loop
        except AssertionError as e:
            if "Expected a cached item for mm_hash" in str(e):
                last_error = e
                if attempt < max_retries - 1:
                    # Cache miss, wait and retry
                    wait_time = 0.5 * (2 ** attempt)  # Exponential backoff
                    print(f"⚠️  Cache miss detected (attempt {attempt + 1}/{max_retries}), retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    # Max retries reached
                    print(f"❌ Cache error persisted after {max_retries} attempts: {e}")
                    raise HTTPException(
                        status_code=503, 
                        detail=f"Multimodal cache error after {max_retries} retries. Try reducing concurrent requests or increasing cache size."
                    )
            else:
                # Different assertion error, re-raise
                raise
        except Exception as e:
            # Other errors, re-raise immediately
            raise HTTPException(status_code=500, detail=f"Model inference error: {str(e)}")
    
    # Extract scores
    scores = [0.0] * len(request.candidate_images)
    tokens = ["error"] * len(request.candidate_images)
    
    for j, output in enumerate(outputs):
        idx = valid_indices[j]
        logprobs = output.outputs[0].logprobs[-1] if output.outputs[0].logprobs else {}
        score, token = compute_score(logprobs)
        scores[idx] = score
        tokens[idx] = token
    
    return RankCIRResponse(scores=scores, tokens=tokens)


@app.get("/health")
async def health():
    """Health check."""
    return {"status": "ok", "model_loaded": model is not None, "mode": "CIR"}


def main():
    parser = argparse.ArgumentParser(description="VLM Ranker Server (CIR)")
    parser.add_argument("--model", default="")
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--max-model-len", type=int, default=100000)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--mm-processor-cache-gb", type=float, default=50, 
                        help="Multimodal processor cache size in GB (default: 50)")
    parser.add_argument("--disable-mm-cache", action="store_true",
                        help="Disable multimodal cache entirely (slower but more stable)")
    parser.add_argument("--host", default="")
    parser.add_argument("--port", type=int, default=8005)
    
    args = parser.parse_args()
    
    # Initialize model
    init_model(
        model_path=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        mm_processor_cache_gb=args.mm_processor_cache_gb,
        disable_mm_cache=args.disable_mm_cache,
    )
    
    # Start server
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

