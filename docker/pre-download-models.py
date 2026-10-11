"""Pre-download models so API startup does not block on network downloads."""
import os
import sys

import open_clip
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Use the same model the app will use at runtime.
name = os.environ.get("TEXT_MODEL_NAME", "distilbert-base-uncased")

# Set download timeouts so the build does not hang forever.
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")

MAX_RETRIES = 3

for attempt in range(1, MAX_RETRIES + 1):
    try:
        print(f"[attempt {attempt}/{MAX_RETRIES}] Downloading {name} tokenizer...")
        AutoTokenizer.from_pretrained(name)
        print(f"[attempt {attempt}/{MAX_RETRIES}] Downloading {name} model with 3 labels...")
        AutoModelForSequenceClassification.from_pretrained(
            name, num_labels=3, problem_type="multi_label_classification"
        )
        if os.environ.get("IMAGE_MODEL_PROVIDER", "nudenet") in {"clip", "ensemble"}:
            print("Downloading image model...")
            open_clip.create_model_and_transforms(
                os.environ.get("IMAGE_MODEL_NAME", "ViT-B-32"),
                pretrained=os.environ.get("IMAGE_MODEL_PRETRAINED", "openai"),
            )
        else:
            print("Using NudeNet package-bundled image weights.")
        print("All models downloaded.")
        sys.exit(0)
    except Exception as e:
        print(f"Attempt {attempt}/{MAX_RETRIES} failed: {e}")
        if attempt == MAX_RETRIES:
            print("All attempts failed; model download is required for API startup.")
            sys.exit(1)

print("All models downloaded.")
