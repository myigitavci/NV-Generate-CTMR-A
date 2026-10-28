"""Structured conditioning from the *same* texts used to precompute MR-CLIP.

No learned text encoder is used. Categories are one-hot encoded (with explicit
missing/unknown buckets); numerical values are standardized, with missingness
indicators. Vocabulary and statistics are fitted on training indices only.
Timing values retain the units in the source text (normally seconds).
"""
from __future__ import annotations

import math
import re

import numpy as np

CATEGORICAL = ("plane", "manufacturer", "model", "description", "sequence", "variant")
NUMERICAL = ("field_strength", "echo_time", "repetition_time", "inversion_time", "flip_angle")
MISSING = "<missing>"
UNKNOWN = "<unknown>"


def normalize_sequence_label(value: str) -> str:
    """Accept coarse contrast names only, never protocol IDs or descriptions."""
    if not isinstance(value, str):
        raise ValueError("Sequence conditioning requires coarse string labels from PKL 'sequences'")
    label = re.sub(r"[\s_\-]", "", value.upper()).replace("*", "STAR")
    label = {"T1": "T1W", "T2": "T2W", "PD": "PDW", "T2STAR": "T2STARW"}.get(label, label)
    if label not in ("T1W", "T2W", "FLAIR", "PDW", "T2STARW"):
        raise ValueError(f"Unsupported high-level sequence label: {value!r}. "
                         "Use T1W, T2W, FLAIR, PDW, or T2STARW.")
    return label


class SequenceLabelEncoder:
    """Training-only class vocabulary with one-hot inputs to the shared MLP."""

    def __init__(self, schema: dict):
        if schema.get("version") != 1 or schema.get("source") != "sequences":
            raise ValueError("Unsupported sequence-label schema")
        classes = schema["classes"]
        if not classes or len(set(classes)) != len(classes):
            raise ValueError("Sequence classes must be nonempty and unique")
        if any(normalize_sequence_label(label) != label for label in classes):
            raise ValueError("Saved sequence classes must use canonical names")
        self.schema = schema
        self.input_dim = len(classes)

    @classmethod
    def fit(cls, labels: list[str]) -> "SequenceLabelEncoder":
        if not labels:
            raise ValueError("Cannot fit sequence vocabulary without training labels")
        return cls({"version": 1, "source": "sequences",
                    "classes": sorted({normalize_sequence_label(label) for label in labels})})

    def transform(self, labels: list[str]) -> np.ndarray:
        lookup = {label: i for i, label in enumerate(self.schema["classes"])}
        result = np.zeros((len(labels), self.input_dim), dtype=np.float32)
        for i, value in enumerate(labels):
            label = normalize_sequence_label(value)
            if label not in lookup:
                raise ValueError(f"Sequence {label!r} was absent from the training vocabulary")
            result[i, lookup[label]] = 1.0
        return result


def _category(value: str) -> str:
    value = " ".join(value.strip().lower().split())
    return MISSING if value in ("", "none", "nan", "n/a", "null") else value


def _number(value: str) -> float | None:
    if _category(value) == MISSING:
        return None
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Non-finite numerical metadata: {value!r}")
    return number


def parse_metadata(text: str) -> dict:
    """Parse the canonical prompt, including parentheses in descriptions.

    Fail on unrecognized text instead of silently discarding conditioning fields.
    """
    pattern = (
        r"A brain MRI,\s*plane\s+(.*?),\s*"
        r"Scanner\s*\(Manufacturer, Model, Field Strength\):\s*\((.*?)\),\s*"
        r"Acquisition\s*\(Description, Sequence, Variant\):\s*\((.*?)\),\s*"
        r"Imaging Parameters\s*\(Echo Time, Repetition Time, Inversion Time, Flip Angle\):\s*\((.*?)\)"
    )
    match = re.fullmatch(pattern, text.strip(), flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Unsupported metadata text format: {text[:160]!r}")
    plane, scanner, acquisition, imaging = match.groups()
    manufacturer, model, field_strength = scanner.rsplit(",", 2)
    description, sequence, variant = acquisition.rsplit(",", 2)
    te, tr, ti, fa = imaging.split(",")
    return dict(zip(CATEGORICAL, map(_category, (plane, manufacturer, model, description, sequence, variant))),
                **dict(zip(NUMERICAL, map(_number, (field_strength, te, tr, ti, fa)))))


class MetadataEncoder:
    """Serializable preprocessing; the trainable MLP lives in the UNet wrapper."""

    def __init__(self, schema: dict):
        if schema.get("version") != 1:
            raise ValueError("Unsupported metadata schema version")
        self.schema = schema
        self.input_dim = sum(len(schema["categories"][key]) for key in CATEGORICAL) + 2 * len(NUMERICAL)

    @classmethod
    def fit(cls, texts: list[str]) -> "MetadataEncoder":
        if not texts:
            raise ValueError("Cannot fit metadata preprocessing without training rows")
        rows = [parse_metadata(text) for text in texts]
        categories = {
            key: [MISSING, UNKNOWN] + sorted({row[key] for row in rows} - {MISSING, UNKNOWN})
            for key in CATEGORICAL
        }
        statistics = {}
        for key in NUMERICAL:
            values = np.asarray([row[key] for row in rows if row[key] is not None], dtype=np.float64)
            mean = float(values.mean()) if len(values) else 0.0
            std = float(values.std()) if len(values) else 1.0
            statistics[key] = {"mean": mean, "std": std if std > 1e-8 else 1.0}
        return cls({"version": 1, "categories": categories, "statistics": statistics})

    def transform(self, texts: list[str]) -> np.ndarray:
        result = np.zeros((len(texts), self.input_dim), dtype=np.float32)
        lookups = {key: {value: i for i, value in enumerate(self.schema["categories"][key])}
                   for key in CATEGORICAL}
        for i, text in enumerate(texts):
            row = parse_metadata(text)
            offset = 0
            for key in CATEGORICAL:
                vocab = lookups[key]
                result[i, offset + vocab.get(row[key], vocab[UNKNOWN])] = 1.0
                offset += len(vocab)
            for key in NUMERICAL:
                value = row[key]
                if value is None:
                    result[i, offset + 1] = 1.0
                else:
                    stats = self.schema["statistics"][key]
                    result[i, offset] = (value - stats["mean"]) / stats["std"]
                offset += 2
        if not np.isfinite(result).all():
            raise ValueError("Non-finite structured metadata features")
        return result
