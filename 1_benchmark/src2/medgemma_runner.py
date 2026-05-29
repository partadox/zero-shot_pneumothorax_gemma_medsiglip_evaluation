"""
MedGemma 1.5 4B inference engine.

CRITICAL: bfloat16 wajib untuk Gemma family.
float16 → semua output PAD token (lesson dari P1).
"""

from __future__ import annotations
import gc
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

from .parser import parse_output, extract_logit_confidence
from .prompts import build_medgemma_messages


class MedGemmaRunner:

    def __init__(self, model_path: str, max_new_tokens: int = 512):
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        print(f"[MedGemma] Loading: {model_path}")
        print(f"[MedGemma] dtype: bfloat16 (mandatory for Gemma family)")
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.model.eval()
        vram = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
        print(f"[MedGemma] Ready. VRAM: {vram:.2f} GB")

    def generate(self, image: Image.Image) -> dict:
        messages = build_medgemma_messages(image)
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=text, images=[image], return_tensors="pt"
        ).to(self.model.device)

        t0 = time.time()
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
            )
        elapsed = time.time() - t0

        input_len = inputs["input_ids"].shape[1]
        generated_ids = output.sequences[:, input_len:]
        raw_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        parsed = parse_output(raw_text)
        conf_logit = extract_logit_confidence(output.scores, self.processor.tokenizer)

        del inputs, generated_ids
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {
            **parsed,
            "confidence_logit": conf_logit,
            "inference_time":   round(elapsed, 2),
            "model":            "medgemma-1.5-4b-it",
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
                result = self.generate(image)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                gc.collect()
                print("OOM — skipped")
                result = _error_record("OOM")
            except Exception as e:
                print(f"ERROR: {e}")
                result = _error_record(str(e))

            record = {"file_id": fid, "label_gt": int(row["label"]), "dataset": dataset_name, **result}
            append_result(output_path, record)
            done_ids.add(fid)

            status = "✓" if result.get("parse_success") else "✗"
            t = result.get("inference_time", 0)
            print(f"{status} diag={result.get('diagnosis')} conf={result.get('confidence_explicit')} ({t}s)")

        print(f"\n[MedGemma] Done. Skipped {skipped}/{n} (already done).")

    def free(self):
        del self.model, self.processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


def _error_record(msg: str) -> dict:
    return {
        "diagnosis": -1, "confidence_explicit": None, "confidence_logit": None,
        "findings": "", "reasoning": "", "parse_success": False,
        "raw_output": f"ERROR: {msg}", "inference_time": 0.0,
        "model": "medgemma-1.5-4b-it",
    }
