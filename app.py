# app.py

import os
from pathlib import Path
import torch
from flask import Flask, jsonify, render_template_string, request
from transformers import AutoTokenizer

from src.models.raise_absa import RaiseAbsaModel
from src.train_source import decode_bio_spans

app = Flask(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
TOKENIZER_DIR = PROJECT_ROOT / "models" / "deberta-v3-base-tokenizer"
CHECKPOINT_DIR = PROJECT_ROOT / "models" / "checkpoints" / "raise_absa_deberta_small"
CHECKPOINT_PATH = CHECKPOINT_DIR / "best_raise_absa_deberta.pt"

SENTIMENT_MAP = {0: "negative", 1: "neutral", 2: "positive", 3: "conflict"}

# Load Model & Tokenizer on Startup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Loading RAISE-ABSA model on {device}...")

if not TOKENIZER_DIR.exists() or not CHECKPOINT_PATH.exists():
    raise FileNotFoundError("Model checkpoint or tokenizer not found. Ensure training is completed.")

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR, local_files_only=True)
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
model = RaiseAbsaModel("microsoft/deberta-v3-small", use_gradient_checkpointing=False).to(device).float()

state_dict = checkpoint.get("teacher_state_dict", checkpoint.get("student_state_dict"))
model.load_state_dict(state_dict)
model.eval()
print("Model loaded successfully!")


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

                # TUNED STRICT FILTER: Drop empty strings, whitespace, or degenerate single characters
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


# Modern Glassmorphic HTML Frontend Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ReviewLens | RAISE-ABSA Intelligence Dashboard</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-gradient: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #311042 100%);
            --glass-bg: rgba(30, 41, 59, 0.7);
            --glass-border: rgba(255, 255, 255, 0.1);
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --accent-glow: 0 0 20px rgba(99, 102, 241, 0.3);
        }

        body {
            font-family: 'Outfit', sans-serif;
            background: var(--bg-gradient);
            color: var(--text-main);
            min-height: 100vh;
            margin: 0;
            padding: 40px 20px;
            display: flex;
            justify-content: center;
            align-items: center;
        }

        .dashboard-container {
            width: 100%;
            max-width: 800px;
            background: var(--glass-bg);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid var(--glass-border);
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4), var(--accent-glow);
            animation: fadeIn 0.8s cubic-bezier(0.16, 1, 0.3, 1);
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .header-box {
            text-align: center;
            margin-bottom: 30px;
        }

        h1 {
            font-size: 2.2rem;
            margin: 0 0 10px 0;
            background: linear-gradient(90deg, #818cf8, #c084fc, #f472b6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 700;
        }

        .subtitle {
            color: var(--text-muted);
            font-size: 0.95rem;
            letter-spacing: 0.5px;
        }

        .preset-pills {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            margin-bottom: 15px;
        }

        .pill {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            color: var(--text-muted);
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 0.8rem;
            cursor: pointer;
            transition: all 0.3s ease;
        }

        .pill:hover {
            background: rgba(99, 102, 241, 0.2);
            color: #fff;
            border-color: #818cf8;
            transform: translateY(-2px);
        }

        textarea {
            width: 100%;
            height: 110px;
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid var(--glass-border);
            color: var(--text-main);
            padding: 16px;
            border-radius: 12px;
            font-size: 1rem;
            font-family: 'Outfit', sans-serif;
            resize: vertical;
            box-sizing: border-box;
            transition: all 0.3s ease;
            outline: none;
            margin-bottom: 20px;
        }

        textarea:focus {
            border-color: #818cf8;
            box-shadow: 0 0 15px rgba(129, 140, 248, 0.2);
        }

        .analyze-btn {
            background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
            color: white;
            border: none;
            padding: 14px 24px;
            font-size: 1rem;
            font-weight: 600;
            border-radius: 12px;
            cursor: pointer;
            width: 100%;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4);
        }

        .analyze-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(168, 85, 247, 0.6);
        }

        .results-section {
            margin-top: 30px;
        }

        .card {
            background: rgba(15, 23, 42, 0.5);
            border: 1px solid var(--glass-border);
            padding: 18px 22px;
            margin-bottom: 14px;
            border-radius: 14px;
            display: flex;
            flex-direction: column;
            gap: 10px;
            transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1);
            animation: slideUp 0.5s ease forwards;
        }

        @keyframes slideUp {
            from { opacity: 0; transform: translateY(15px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .card:hover {
            transform: scale(1.01);
            border-color: rgba(255, 255, 255, 0.2);
            box-shadow: 0 8px 25px rgba(0,0,0,0.3);
        }

        .card-top {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .aspect-name {
            font-size: 1.1rem;
            font-weight: 600;
            color: #fff;
        }

        .badge {
            font-weight: 600;
            text-transform: uppercase;
            padding: 5px 12px;
            border-radius: 30px;
            font-size: 0.75rem;
            letter-spacing: 0.5px;
        }

        .badge-positive { background: rgba(34, 197, 94, 0.2); color: #4ade80; border: 1px solid rgba(34, 197, 94, 0.4); }
        .badge-negative { background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.4); }
        .badge-neutral { background: rgba(59, 130, 246, 0.2); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.4); }
        .badge-conflict { background: rgba(249, 115, 22, 0.2); color: #fb923c; border: 1px solid rgba(249, 115, 22, 0.4); }

        .progress-container {
            width: 100%;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 10px;
            height: 6px;
            overflow: hidden;
            display: flex;
        }

        .progress-bar {
            height: 100%;
            border-radius: 10px;
            transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
        }

        .bar-positive { background: #4ade80; }
        .bar-negative { background: #f87171; }
        .bar-neutral { background: #60a5fa; }
        .bar-conflict { background: #fb923c; }

        .meta-info {
            display: flex;
            justify-content: space-between;
            font-size: 0.8rem;
            color: var(--text-muted);
        }

        .loader {
            text-align: center;
            color: var(--text-muted);
            font-style: italic;
            padding: 20px;
        }
    </style>
</head>
<body>
    <div class="dashboard-container">
        <div class="header-box">
            <h1>ReviewLens: RAISE-ABSA</h1>
            <div class="subtitle">Risk-Aware Aspect-Based Sentiment Analysis & Conformal Filtering</div>
        </div>

        <label style="font-size: 0.85rem; color: var(--text-muted); margin-bottom: 8px; display: block;">Quick Test Presets:</label>
        <div class="preset-pills">
            <div class="pill" onclick="setPreset(1)">English Review</div>
            <div class="pill" onclick="setPreset(2)">Spanglish Review</div>
            <div class="pill" onclick="setPreset(3)">Hinglish Review</div>
        </div>
        
        <textarea id="reviewText">The battery life is exceptionally long, but the screen resolution is quite disappointing.</textarea>
        <button class="analyze-btn" onclick="analyzeReview()">✨ Analyze Aspects & Sentiment</button>
        
        <div class="results-section" id="resultsContainer"></div>
    </div>

    <script>
        const presets = {
            1: "The battery life is exceptionally long, but the screen resolution is quite disappointing.",
            2: "La battery life is amazing, pero el display screen se rompió muy rápido.",
            3: "Dell G15 laptop gaming ke liye accha hai, lekin battery satisfying nahi hai. Build quality acchi hai aur zyada reliable bhi."
        };

        function setPreset(id) {
            document.getElementById('reviewText').value = presets[id];
        }

        async function analyzeReview() {
            const text = document.getElementById('reviewText').value;
            const container = document.getElementById('resultsContainer');
            container.innerHTML = "<div class='loader'>Running multi-view neural inference & conformal check...</div>";

            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: text })
            });

            const data = await response.json();
            container.innerHTML = "";

            if (data.results.length === 0) {
                container.innerHTML = "<p style='text-align:center; color: var(--text-muted);'>No explicit aspect terms detected or filtered by conformal threshold.</p>";
                return;
            }

            data.results.forEach(item => {
                const confPercent = Math.round(item.confidence * 100);
                const card = document.createElement('div');
                card.className = `card`;
                card.innerHTML = `
                    <div class="card-top">
                        <div class="aspect-name">Aspect: &ldquo;${item.aspect}&rdquo;</div>
                        <span class="badge badge-${item.sentiment}">${item.sentiment}</span>
                    </div>
                    <div class="progress-container">
                        <div class="progress-bar bar-${item.sentiment}" style="width: ${confPercent}%"></div>
                    </div>
                    <div class="meta-info">
                        <span>Confidence Score</span>
                        <span>${item.confidence} (${confPercent}%)</span>
                    </div>
                `;
                container.appendChild(card);
            });
        }
    </script>
</body>
</html>
"""


@app.route("/")
def home():
    return render_template_string(HTML_TEMPLATE)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "")
    if not text.strip():
        return jsonify({"results": []})

    results = run_inference_on_text(text)
    return jsonify({"results": results})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)