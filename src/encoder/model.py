import torch
from sentence_transformers import SentenceTransformer


class EmbeddingEncoder:
    """
    Encoder class to convert unstructured text into fixed-size embedding vectors.

    Uses a pre-trained SentenceTransformer model with frozen weights.
    Domain adaptation is handled entirely by the downstream IntentAdapter,
    keeping this layer stateless and side-effect-free.

    Default model: intfloat/multilingual-e5-small
      - 384-dim output, ~117MB, CPU-friendly
      - Supports Japanese and English (suitable for OT log messages)
      - Fully offline-capable once the model file is downloaded
    """

    def __init__(self, model_name: str = "intfloat/multilingual-e5-small"):
        # Weights are frozen; no training occurs in this layer.
        # Lightweight multilingual models are suitable for edge/OT environments.
        self.model = SentenceTransformer(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def encode(self, text: str) -> torch.Tensor:
        """
        Encode input text into a vector.
        """
        # Convert to torch tensor for downstream adapter processing.
        embeddings = self.model.encode(text, convert_to_tensor=True)
        return embeddings

    @property
    def embedding_dimension(self) -> int:
        """
        Return the output dimension of the embedding model.
        """
        dim = self.model.get_sentence_embedding_dimension()
        assert dim is not None, "Could not determine embedding dimension."
        return int(dim)


if __name__ == "__main__":
    # Simple test to verify embedding generation.
    encoder = EmbeddingEncoder()
    vector = encoder.encode("Conveyor stopped. Error code E-01")
    print(f"Embedding Shape: {vector.shape}")
    print(f"Device: {encoder.device}")
