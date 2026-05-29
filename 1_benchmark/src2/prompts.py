"""Prompt templates for pneumothorax detection — MedGemma only in Experiment 2."""

from PIL import Image

PNEUMOTHORAX_TASK = """You are a radiologist analyzing a chest X-ray image.

Task: Determine whether pneumothorax is present in this chest X-ray.

Respond ONLY with valid JSON in this exact format:
{
  "diagnosis": "YES",
  "confidence": 8,
  "findings": "Brief description of relevant radiographic findings (1-3 sentences).",
  "reasoning": "Short clinical reasoning for your diagnosis."
}

Rules:
- "diagnosis" must be exactly "YES" (pneumothorax present) or "NO" (not present)
- "confidence" is an integer from 1 (very uncertain) to 10 (very certain)
- "findings" describes what you see in the image related to pneumothorax
- "reasoning" explains why you concluded YES or NO
- Output ONLY the JSON object, no additional text before or after"""


MEDSIGLIP_TEXT_POSITIVE = "chest radiograph showing pneumothorax with pleural air"
MEDSIGLIP_TEXT_NEGATIVE = "normal chest radiograph without pneumothorax"


def build_medgemma_messages(image: Image.Image) -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text": PNEUMOTHORAX_TASK},
            ],
        }
    ]
