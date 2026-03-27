
from src.encoder.model import EmbeddingEncoder
import torch

encoder = EmbeddingEncoder()
vec = encoder.encode("test")
print(f"Shape: {vec.shape}")
print(f"Dimensions: {vec.dim()}")
