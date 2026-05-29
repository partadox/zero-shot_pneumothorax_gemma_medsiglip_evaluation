"""Output parsing for MedGemma generative responses."""

from __future__ import annotations
import json
import re

_JSON_BLOCK = re.compile(r"\{[^{}]*\}", re.DOTALL)


def parse_output(raw_text: str) -> dict:
    result = {
        "diagnosis":            -1,
        "confidence_explicit":  None,
        "findings":             "",
        "reasoning":            "",
        "parse_success":        False,
        "raw_output":           raw_text,
    }
    if not raw_text or not raw_text.strip():
        return result

    parsed = _try_json(raw_text.strip())
    if parsed:
        result["diagnosis"]           = _extract_diagnosis(parsed.get("diagnosis", ""))
        result["confidence_explicit"] = _extract_confidence(parsed.get("confidence"))
        result["findings"]            = str(parsed.get("findings", "")).strip()
        result["reasoning"]           = str(parsed.get("reasoning", "")).strip()
        result["parse_success"]       = result["diagnosis"] != -1
        return result

    result["diagnosis"] = _regex_diagnosis(raw_text)
    result["parse_success"] = result["diagnosis"] != -1
    return result


def _try_json(text: str) -> dict | None:
    try:
        return json.loads(text)
    except Exception:
        pass
    m = _JSON_BLOCK.search(text)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    return None


def _extract_diagnosis(value) -> int:
    if not value:
        return -1
    v = str(value).strip().upper()
    if v == "YES":
        return 1
    if v == "NO":
        return 0
    return -1


def _extract_confidence(value) -> int | None:
    if value is None:
        return None
    try:
        c = int(str(value).strip())
        return c if 1 <= c <= 10 else None
    except Exception:
        return None


_YES_PAT = re.compile(r"\b(yes|pneumothorax\s+present|pneumothorax\s+detected)\b", re.I)
_NO_PAT  = re.compile(r"\b(no\s+pneumothorax|pneumothorax\s+absent|not\s+detected|not\s+present)\b", re.I)


def _regex_diagnosis(text: str) -> int:
    if _NO_PAT.search(text):
        return 0
    if _YES_PAT.search(text):
        return 1
    return -1


def extract_logit_confidence(scores, tokenizer, yes_tokens=None, no_tokens=None):
    import torch
    if not scores:
        return None
    if yes_tokens is None:
        yes_tokens = ["YES", "yes", " YES", " yes", "Yes"]
    if no_tokens is None:
        no_tokens = ["NO", "no", " NO", " no", "No"]
    try:
        first_logits = scores[0][0]
        probs = torch.softmax(first_logits.float(), dim=-1)
        yes_ids = [i for i in [tokenizer.convert_tokens_to_ids(t) for t in yes_tokens]
                   if i != tokenizer.unk_token_id]
        no_ids  = [i for i in [tokenizer.convert_tokens_to_ids(t) for t in no_tokens]
                   if i != tokenizer.unk_token_id]
        p_yes = sum(probs[i].item() for i in yes_ids)
        p_no  = sum(probs[i].item() for i in no_ids)
        total = p_yes + p_no
        if total < 1e-8:
            return None
        return float(p_yes / total)
    except Exception:
        return None
