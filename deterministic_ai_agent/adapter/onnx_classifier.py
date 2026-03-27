import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import onnxruntime as ort

if TYPE_CHECKING:
    from numpy.typing import NDArray


class OnnxIntentClassifier:
    """
    Lightweight ONNX inference version of IntentAdapter.
    ... (docstring truncated)
    """

    def __init__(self, model_path: str | Path, metadata_path: str | Path):
        """
        Args:
            model_path: Path to the .onnx model file.
            metadata_path: Path to the .json metadata containing centroids.
        """
        # Inference session (uses CPU by default for stability in OT)
        self.session = ort.InferenceSession(str(model_path))

        # Load metadata (centroids, num_classes)
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
            # Convert list back to numpy array for fast computation
            self.centroids = np.array(metadata["centroids"])
            self.num_classes = metadata["num_classes"]

    def _softmax(self, x: "NDArray[Any]") -> "NDArray[Any]":
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)

    def _cosine_similarity(self, a: "NDArray[Any]", b: "NDArray[Any]") -> "NDArray[Any]":
        """Calculate cosine similarity between a (batch, dim) and b (num_classes, dim)."""
        # Normalizing vectors
        a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
        b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
        # Dot product
        return np.dot(a_norm, b_norm.T)

    def predict_with_confidence(self, x: "NDArray[Any]") -> tuple[int, float]:
        """
        L1 & L2 Implementation using ONNX and Numpy.
        """
        # Ensure embedding is 2D
        if x.ndim == 1:
            x = x[np.newaxis, :]

        # Step 1: Run ONNX inference
        inputs = {self.session.get_inputs()[0].name: x.astype(np.float32)}
        logits = self.session.run(None, inputs)[0]

        # Step 2: Softmax & Argmax
        probs = self._softmax(logits)
        intent_id = int(np.argmax(probs, axis=1)[0])
        confidence = float(probs[0, intent_id])

        return intent_id, confidence

    def get_ood_score(self, x: "NDArray[Any]") -> float:
        """
        L3 Implementation using Numpy.
        """
        if x.ndim == 1:
            x = x[np.newaxis, :]

        # Check if centroids are initialized (norm > 0)
        if np.linalg.norm(self.centroids) < 1e-6:
            return 0.0

        # Matrix multiplication for cosine similarity
        similarities = self._cosine_similarity(x, self.centroids)
        max_similarity = float(np.max(similarities))

        return max_similarity
