from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

if TYPE_CHECKING:
    from numpy.typing import NDArray


class OnnxEmbeddingEncoder:
    """
    Lightweight ONNX inference version of EmbeddingEncoder.
    ... (docstring truncated)
    """

    def __init__(self, model_path: str | Path, tokenizer_path: str | Path):
        """
        Args:
            model_path: Path to the model.onnx file.
            tokenizer_path: Path to the tokenizer.json file.
        """
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        # Inference session
        self.session = ort.InferenceSession(str(model_path))

        # Get input names from ONNX session
        self.input_names = [i.name for i in self.session.get_inputs()]

    def encode(self, text: str) -> "NDArray[Any]":
        """
        Encode text into a vector using ONNX.
        """
        # Step 1: Tokenize
        encoded = self.tokenizer.encode(text)

        # Prepare inputs for BERT-based models
        # (Usually: input_ids, attention_mask, token_type_ids)
        inputs = {
            "input_ids": np.array([encoded.ids], dtype=np.int64),
            "attention_mask": np.array([encoded.attention_mask], dtype=np.int64),
            "token_type_ids": np.array([encoded.type_ids], dtype=np.int64),
        }

        # Filter only the inputs required by the model
        model_inputs = {k: v for k, v in inputs.items() if k in self.input_names}

        # Step 2: Run ONNX Inference
        # Output is usually [last_hidden_state]
        outputs = self.session.run(None, model_inputs)
        last_hidden_state = outputs[0]  # shape: (1, seq_len, 384)

        # Step 3: Mean Pooling (SentenceTransformer equivalent)
        # We average across the sequence length (axis=1)
        mask = np.array(encoded.attention_mask)
        # Apply mask to handle variable sequence lengths correctly
        # input: (1, seq_len, dim) * (1, seq_len, 1)
        mask_expanded = np.expand_dims(mask, (0, 2)).astype(float)
        sum_embeddings = np.sum(last_hidden_state * mask_expanded, axis=1)
        sum_mask = np.clip(mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)

        sentence_embedding = sum_embeddings / sum_mask

        # Step 4: L2 Normalization (Required for E5 models)
        norm = np.linalg.norm(sentence_embedding, axis=1, keepdims=True)
        normalized_embedding = sentence_embedding / norm

        return cast("NDArray[Any]", normalized_embedding.astype(np.float32))

    @property
    def embedding_dimension(self) -> int:
        """
        Return the output dimension from the ONNX session.
        """
        # Shape: [batch_size, sequence_length, hidden_dim] or [batch_size, hidden_dim]
        # We need the last dimension
        return int(self.session.get_outputs()[0].shape[-1])
