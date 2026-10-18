# new_app.py

import os
import gc
from pathlib import Path
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from transformers import AutoTokenizer

from src.models.raise_absa import RaiseAbsaModel
from src.train_source import decode_bio_spans

# Clear system memory before initialization
gc.collect()

app = FastAPI(
    title="ReviewLens: RAISE-ABSA FastAPI Engine",
    description="Risk-Aware Aspect-Based Sentiment Analysis High-Performance REST API",
    version="2.0",
)

PROJECT_ROOT = Path(__file__).resolve().parent
TOKENIZER_DIR = PROJECT_ROOT / "models" / "deberta-v3-base-tokenizer"
CHECKPOINT_DIR = PROJECT_ROOT / "models" / "checkpoints" / "raise_absa_deberta_small"
CHECKPOINT_PATH = CHECKPOINT_DIR / "best_raise_absa_deberta.pt"

SENTIMENT_MAP = {0: "negative", 1: "neutral", 2: "positive", 3: "conflict"}

# Safely use CPU
device = torch.device("cpu")
print(f"Loading RAISE-ABSA model on {device} for FastAPI...")

if not TOKENIZER_DIR.exists() or not CHECKPOINT_PATH.exists():
    raise FileNotFoundError("Model checkpoint or tokenizer not found. Ensure training is completed.")

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR, local_files_only=True)

# Load checkpoint safely with explicit memory mapping
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)

model = RaiseAbsaModel("microsoft/deberta-v3-small", use_gradient_checkpointing=False).to(device).float()

state_dict = checkpoint.get("teacher_state_dict", checkpoint.get("student_state_dict"))
model.load_state_dict(state_dict)
model.eval()
print("FastAPI Model loaded successfully on CPU!")


class ReviewRequest(BaseModel):
    text: str


def run_inference_on_text(text: str):
    encoding = tokenizer(
        text,
        add_special_tokens=True,
        truncation=True,
        max_length=96,
        return_offsets_mapping=True,
        return_tensors="pt",
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    offset_mapping = encoding["offset_mapping"][0].cpu().tolist()

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        bio_preds = outputs["bio_logits"].argmax(dim=-1).cpu().numpy()
        decoded_spans = decode_bio_spans(bio_preds, attention_mask.cpu().numpy())[0]

        extracted_results = []
        if decoded_spans:
            p_b_idx = torch.zeros(len(decoded_spans), dtype=torch.long, device=device)
            p_starts = torch.tensor([s for s, e in decoded_spans], dtype=torch.long, device=device)
            p_ends = torch.tensor([e for s, e in decoded_spans], dtype=torch.long, device=device)

            pred_sent_logits = model.classify_spans(
                outputs["sequence_output"], p_b_idx, p_starts, p_ends
            )
            pred_sent_ids = pred_sent_logits.argmax(dim=-1).cpu().tolist()
            pred_probs = torch.softmax(pred_sent_logits.float(), dim=-1).cpu().numpy()

            for (s, e), sent_id, probs in zip(decoded_spans, pred_sent_ids, pred_probs):
                if s >= len(offset_mapping) or e > len(offset_mapping) or s >= e:
                    continue
                
                char_start = offset_mapping[s][0]
                char_end = offset_mapping[min(e - 1, len(offset_mapping) - 1)][1]
                aspect_term = text[char_start:char_end].strip()

                if not aspect_term or aspect_term == "" or len(aspect_term) < 2 or aspect_term.isspace():
                    continue

                sentiment_label = SENTIMENT_MAP.get(sent_id, "neutral")
                confidence = float(probs[sent_id])

                extracted_results.append({
                    "aspect": aspect_term,
                    "sentiment": sentiment_label,
                    "confidence": round(confidence, 4),
                })

    return extracted_results


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ReviewLens | FastAPI Neural Hub</title>
    <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-dark: #090d16;
            --panel-bg: rgba(16, 24, 39, 0.75);
            --border-glow: rgba(56, 189, 248, 0.2);
            --accent-blue: #38bdf8;
            --accent-purple: #c084fc;
            --text-main: #f1f5f9;
            --text-muted: #94a3b8;
        }

        body {
            font-family: 'Space Grotesk', sans-serif;
            background-color: var(--bg-dark);
            background-image: radial-gradient(circle at 10% 20%, rgba(56, 189, 248, 0.08) 0%, transparent 40%),
                              radial-gradient(circle at 90% 80%, rgba(192, 132, 252, 0.08) 0%, transparent 40%);
            color: var(--text-main);
            min-height: 100vh;
            margin: 0;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }

        .container {
            width: 100%;
            max-width: 850px;
            background: var(--panel-bg);
            backdrop-filter: blur(20px);
            border: 1px solid var(--border-glow);
            border-radius: 24px;
            padding: 40px;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.7), 0 0 30px rgba(56, 189, 248, 0.1);
        }

        .header {
            text-align: center;
            margin-bottom: 30px;
        }

        h1 {
            font-size: 2.4rem;
            margin: 0 0 8px 0;
            background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 700;
        }

        .tagline {
            color: var(--text-muted);
            font-size: 0.95rem;
            letter-spacing: 1px;
            text-transform: uppercase;
        }

        .preset-container {
            display: flex;
            gap: 12px;
            margin-bottom: 15px;
            flex-wrap: wrap;
        }

        .preset-btn {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.1);
            color: var(--text-muted);
            padding: 8px 16px;
            border-radius: 8px;
            font-size: 0.85rem;
            cursor: pointer;
            transition: all 0.3s ease;
        }

        .preset-btn:hover {
            background: rgba(56, 189, 248, 0.15);
            color: var(--accent-blue);
            border-color: var(--accent-blue);
        }

        textarea {
            width: 100%;
            height: 120px;
            background: rgba(5, 8, 15, 0.8);
            border: 1px solid rgba(255, 255, 255, 0.1);
            color: var(--text-main);
            padding: 16px;
            border-radius: 14px;
            font-size: 1rem;
            font-family: 'Space Grotesk', sans-serif;
            resize: vertical;
            box-sizing: border-box;
            outline: none;
            transition: border-color 0.3s ease;
            margin-bottom: 20px;
        }

        textarea:focus {
            border-color: var(--accent-blue);
            box-shadow: 0 0 15px rgba(56, 189, 248, 0.2);
        }

        .action-btn {
            background: linear-gradient(135deg, var(--accent-blue), #2563eb);
            color: #0f172a;
            border: none;
            padding: 16px;
            font-size: 1rem;
            font-weight: 700;
            border-radius: 14px;
            cursor: pointer;
            width: 100%;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
            box-shadow: 0 10px 20px -5px rgba(56, 189, 248, 0.4);
        }

        .action-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 15px 25px -5px rgba(56, 189, 248, 0.6);
        }

        .results-box {
            margin-top: 30px;
            display: flex;
            flex-direction: column;
            gap: 12px;
        }

        .result-card {
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid rgba(255, 255, 255, 0.08);
            padding: 18px 22px;
            border-radius: 14px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            animation: fadeIn 0.4s ease forwards;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .aspect-title {
            font-size: 1.1rem;
            font-weight: 600;
        }

        .badge {
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .badge-positive { background: rgba(34, 197, 94, 0.2); color: #4ade80; border: 1px solid rgba(34, 197, 94, 0.4); }
        .badge-negative { background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.4); }
        .badge-neutral { background: rgba(59, 130, 246, 0.2); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.4); }
        .badge-conflict { background: rgba(249, 115, 22, 0.2); color: #fb923c; border: 1px solid rgba(249, 115, 22, 0.4); }

        .api-docs-link {
            text-align: center;
            margin-top: 25px;
            font-size: 0.85rem;
            color: var(--text-muted);
        }

        .api-docs-link a {
            color: var(--accent-blue);
            text-decoration: none;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>ReviewLens: FastAPI Hub</h1>
            <div class="tagline">High-Performance Neural Aspect Sentiment Engine</div>
        </div>

        <div class="preset-container">
            <button class="preset-btn" onclick="setPreset(1)">English Hardware Review</button>
            <button class="preset-btn" onclick="setPreset(2)">Spanglish Review</button>
            <button class="preset-btn" onclick="setPreset(3)">Hinglish Review</button>
        </div>

        <textarea id="reviewInput">The battery life is exceptionally long, but the screen resolution is quite disappointing.</textarea>
        <button class="action-btn" onclick="executePrediction()">⚡ Execute FastAPI Inference</button>

        <div class="results-box" id="resultsArea"></div>

        <div class="api-docs-link">
            Interactive Swagger API Docs available at <a href="/docs" target="_blank">/docs</a> &bull; ReDoc at <a href="/redoc" target="_blank">/redoc</a>
        </div>
    </div>

    <script>
        const samples = {
            1: "The battery life is exceptionally long, but the screen resolution is quite disappointing.",
            2: "La battery life is amazing, pero el display screen se rompió muy rápido.",
            3: "Dell G15 laptop gaming ke liye accha hai, lekin battery satisfying nahi hai. Build quality acchi hai aur zyada reliable bhi."
        };

        function setPreset(id) {
            document.getElementById('reviewInput').value = samples[id];
        }

        async function executePrediction() {
            const text = document.getElementById('reviewInput').value;
            const area = document.getElementById('resultsArea');
            area.innerHTML = "<p style='text-align:center; color: var(--text-muted);'>Computing neural tensors & token mappings...</p>";

            try {
                const response = await fetch('/api/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text: text })
                });

                const data = await response.json();
                area.innerHTML = "";

                if (data.results.length === 0) {
                    area.innerHTML = "<p style='text-align:center; color: var(--text-muted);'>No valid aspect terms detected.</p>";
                    return;
                }

                data.results.forEach(item => {
                    const card = document.createElement('div');
                    card.className = "result-card";
                    card.innerHTML = `
                        <div>
                            <div class="aspect-title">Aspect: &ldquo;${item.aspect}&rdquo;</div>
                            <div style="font-size: 0.8rem; color: var(--text-muted); margin-top: 4px;">Confidence: ${item.confidence * 100}%</div>
                        </div>
                        <span class="badge badge-${item.sentiment}">${item.sentiment}</span>
                    `;
                    area.appendChild(card);
                });
            } catch (err) {
                area.innerHTML = "<p style='text-align:center; color: #f87171;'>Error connecting to FastAPI backend.</p>";
            }
        }
    </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def read_root():
    return HTML_TEMPLATE


@app.post("/api/predict")
def predict_endpoint(payload: ReviewRequest):
    if not payload.text.strip():
        return {"results": []}
    results = run_inference_on_text(payload.text)
    return {"results": results}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("new_app:app", host="127.0.0.1", port=8000, reload=False)