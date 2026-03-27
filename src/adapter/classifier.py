from typing import TYPE_CHECKING, Any, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from numpy.typing import NDArray


class IntentAdapter(nn.Module):
    """
    Lightweight MLP adapter that maps embedding vectors to intent/action IDs.
    ... (docstring truncated)
    """

    centroids: torch.Tensor

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()
        # Dropout is disabled automatically in eval() mode.
        self.dropout = nn.Dropout(0.1)

        # Centroids for OOD detection (L3)
        self.register_buffer("centroids", torch.zeros(num_classes, input_dim))
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Input : (batch, input_dim) tensor from EmbeddingEncoder
        Output: (batch, num_classes) raw logits
        """
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        logits: torch.Tensor = self.fc2(x)
        return logits

    def _ensure_tensor(self, x: "Union[torch.Tensor, NDArray[Any]]") -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x
        return torch.from_numpy(x).float()

    # ------------------------------------------------------------------
    # L1: Deterministic prediction (argmax)
    # ------------------------------------------------------------------
    def predict(self, x: "Union[torch.Tensor, NDArray[Any]]") -> int:
        """
        Return the most probable intent ID.
        Deterministic: same input always yields the same output.
        """
        self.eval()
        t_x = self._ensure_tensor(x)
        with torch.no_grad():
            logits = self.forward(t_x)
            return int(torch.argmax(logits, dim=-1).item())

    # ------------------------------------------------------------------
    # L2: Confidence-aware prediction
    # ------------------------------------------------------------------
    def predict_with_confidence(self, x: "Union[torch.Tensor, NDArray[Any]]") -> tuple[int, float]:
        """
        Return (intent_id, confidence) where confidence is the softmax
        probability of the winning class (0.0 – 1.0).
        """
        self.eval()
        t_x = self._ensure_tensor(x)
        with torch.no_grad():
            logits = self.forward(t_x)
            probs = F.softmax(logits, dim=-1)
            # Use argmax on the last dimension
            intent_id = int(torch.argmax(probs, dim=-1).item())
            # Access the probability safely regardless of batch dimension
            confidence = float(probs.view(-1)[intent_id].item())
        return intent_id, confidence

    # ------------------------------------------------------------------
    # L3: OOD Detection (Centroid Similarity)
    # ------------------------------------------------------------------
    def update_centroids(self, vectors: torch.Tensor, labels: torch.Tensor) -> None:
        """
        Calculate and store class centroids for OOD detection.
        Expects vectors to be (N, input_dim) and labels to be (N,).
        """
        self.eval()
        with torch.no_grad():
            for i in range(self.num_classes):
                mask = labels == i
                if mask.any():
                    # Mean of all embeddings for this class
                    class_vectors = vectors[mask]
                    self.centroids[i] = class_vectors.mean(dim=0)

    def get_ood_score(self, x: "Union[torch.Tensor, NDArray[Any]]") -> float:
        """
        Calculate the maximum cosine similarity to any class centroid.
        """
        self.eval()
        t_x = self._ensure_tensor(x)
        with torch.no_grad():
            # Check if centroids have been populated (norm > 0)
            if torch.norm(self.centroids) < 1e-6:
                return 0.0

            # Ensure input is 2D
            if t_x.dim() == 1:
                t_x = t_x.unsqueeze(0)

            # Normalized centroids and input for cosine similarity
            norm_centroids = F.normalize(self.centroids, p=2.0, dim=1)
            norm_x = F.normalize(t_x, p=2.0, dim=1)

            # Cosine similarity to each centroid (matrix mult: (1, dim) @ (dim, num_classes))
            similarities = torch.mm(norm_x, norm_centroids.t())
            max_similarity = float(torch.max(similarities).item())

        return max_similarity

    # ------------------------------------------------------------------
    # Training helpers
    # ------------------------------------------------------------------
    def train_one_epoch(
        self,
        vectors: torch.Tensor,
        labels: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        """
        Run one training epoch.

        Args:
            vectors : (N, input_dim) float tensor — pre-computed embeddings
            labels  : (N,) long tensor — ground-truth intent IDs
            optimizer: any torch optimizer (e.g. Adam)

        Returns:
            Average cross-entropy loss for this epoch.
        """
        self.train()
        optimizer.zero_grad()
        logits = self.forward(vectors)
        loss = nn.CrossEntropyLoss()(logits, labels)
        loss.backward()
        optimizer.step()
        return float(loss.item())

    def save(self, path: str) -> None:
        """Persist adapter weights to disk."""
        torch.save(self.state_dict(), path)

    def load(self, path: str) -> None:
        """Load adapter weights from disk."""
        self.load_state_dict(torch.load(path, weights_only=True))
        self.eval()

    def export_to_onnx(self, path: str, input_dim: int) -> None:
        """
        Export the adapter model to ONNX format.

        Args:
            path: Target file path (.onnx)
            input_dim: The dimension of input embedding vectors.
        """
        self.eval()
        dummy_input = torch.randn(1, input_dim)
        torch.onnx.export(
            self,
            (dummy_input,),
            path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
        )


if __name__ == "__main__":
    input_dim = 384
    num_intents = 3

    adapter = IntentAdapter(input_dim, num_intents)
    dummy_input = torch.randn(1, input_dim)

    intent_id = adapter.predict(dummy_input)
    intent_id_conf, conf = adapter.predict_with_confidence(dummy_input)
    print(f"Predicted Intent ID : {intent_id}")
    print(f"With confidence     : id={intent_id_conf}, confidence={conf:.4f}")
