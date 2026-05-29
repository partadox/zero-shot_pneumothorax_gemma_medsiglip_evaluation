"""
MedSigLIP zero-shot classification engine.

Cosine similarity between image embedding and two text embeddings:
  - "chest radiograph showing pneumothorax with pleural air"
  - "normal chest radiograph without pneumothorax"
"""

from __future__ import annotations
import gc
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, AutoModel

from .prompts import MEDSIGLIP_TEXT_POSITIVE, MEDSIGLIP_TEXT_NEGATIVE


def _extract_features(output) -> torch.Tensor:
    """Handle both direct tensor and BaseModelOutputWithPooling."""
    if isinstance(output, torch.Tensor):
        return output.float()
    if hasattr(output, 'pooler_output') and output.pooler_output is not None:
        return output.pooler_output.float()
    if hasattr(output, 'last_hidden_state'):
        return output.last_hidden_state[:, 0, :].float()
    raise ValueError(f"Cannot extract features from {type(output)}")


class MedSigLIPRunner:

    def __init__(self, model_path: str):
        self.model_path = model_path
        print(f"[MedSigLIP] Loading: {model_path}")
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="auto",
        )
        self.model.eval()
        self._text_embeds = self._encode_texts([MEDSIGLIP_TEXT_NEGATIVE, MEDSIGLIP_TEXT_POSITIVE])
        vram = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
        print(f"[MedSigLIP] Ready. VRAM: {vram:.2f} GB")

    def _encode_texts(self, texts: list[str]) -> torch.Tensor:
        inputs = self.processor(text=texts, return_tensors="pt", padding=True).to(self.model.device)
        with torch.inference_mode():
            embeds = _extract_features(self.model.get_text_features(**inputs))
        return F.normalize(embeds.float(), dim=-1)

    def _encode_image(self, image: Image.Image) -> torch.Tensor:
        inputs = self.processor(images=image, return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            embed = _extract_features(self.model.get_image_features(**inputs))
        return F.normalize(embed.float(), dim=-1)

    def classify(self, image: Image.Image) -> dict:
        t0 = time.time()
        img_embed = self._encode_image(image)
        sims = (img_embed @ self._text_embeds.T).squeeze(0)
        probs = torch.softmax(sims * 100, dim=0).cpu()
        score_normal = float(probs[0])
        score_ptx    = float(probs[1])
        diagnosis    = 1 if score_ptx >= 0.5 else 0
        elapsed      = time.time() - t0
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {
            "diagnosis":           diagnosis,
            "confidence_explicit": round(score_ptx, 4),
            "score_normal":        round(score_normal, 4),
            "score_ptx":           round(score_ptx, 4),
            "findings":            "",
            "reasoning":           "",
            "parse_success":       True,
            "raw_output":          f"score_ptx={score_ptx:.4f} score_normal={score_normal:.4f}",
            "inference_time":      round(elapsed, 3),
            "model":               "medsiglip",
        }

    def run_dataset(self, df, output_path: Path, dataset_name: str, done_ids: set[str]) -> None:
        from .utils import append_result
        from .data import load_image

        n = len(df)
        skipped = 0

        for i, row in df.iterrows():
            fid = row["file_id"]
            if fid in done_ids:
                skipped += 1
                continue

            print(f"  [{i+1}/{n}] {fid}", end=" ... ", flush=True)
            try:
                image = load_image(row["image_path"])
                result = self.classify(image)
            except Exception as e:
                print(f"ERROR: {e}")
                result = {
                    "diagnosis": -1, "confidence_explicit": None,
                    "score_normal": None, "score_ptx": None,
                    "findings": "", "reasoning": "", "parse_success": False,
                    "raw_output": f"ERROR: {e}", "inference_time": 0.0, "model": "medsiglip",
                }

            record = {"file_id": fid, "label_gt": int(row["label"]), "dataset": dataset_name, **result}
            append_result(output_path, record)
            done_ids.add(fid)

            p = result.get("score_ptx")
            t = result.get("inference_time", 0)
            print(f"diag={result['diagnosis']} P(ptx)={p} ({t}s)")

        print(f"\n[MedSigLIP] Done. Skipped {skipped}/{n} (already done).")

    def free(self):
        del self.model, self.processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
