import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import Literal
def format_sample(sample):
    trace = sample['trace']
    annotation_str = ', '.join([f"{k}: {v}" for k, v in sample['mast_annotation'].items()])
    return (
        f"mas_name: {sample['mas_name']},"
        f"llm_name: {sample['llm_name']},"
        f"benchmark_name: {sample['benchmark_name']}\n"
        f"trace_id: {sample['trace_id']},"
        f"trace.key: {trace['key']},"
        f"trace.index: {trace['index']}\n"
        f"trace.trajectory: {trace['trajectory']}\n"
        "===================="
    )
class ChunkedTextClassifier(nn.Module):
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 512,
        stride: int = 256,
        aggregation: Literal["mean", "max"] = "mean",
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.chunk_size = chunk_size
        self.stride = stride
        self.aggregation = aggregation

 
        self.encoder = SentenceTransformer(model_name)
        #self.encoder.eval()  
        for param in self.encoder.parameters():
            param.requires_grad = False

        embedding_dim = self.encoder.get_sentence_embedding_dimension()

      
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)  
        )

    def embed_batch_texts(self, texts: list[str]) -> torch.Tensor:
        embeddings = []
        for text in texts:
            text = format_sample(text)
            emb = self.embed_long_text(text)
            embeddings.append(emb)
        return torch.tensor(embeddings, dtype=torch.float32)
    def embed_long_text(self, text: str) -> np.ndarray:
        words = text.split()
        chunks = [
            " ".join(words[i:i + self.chunk_size])
            for i in range(0, len(words), self.stride)
            if i + self.chunk_size <= len(words) or i == 0
        ]

        embeddings = self.encoder.encode(chunks, normalize_embeddings=True)

        if self.aggregation == "mean":
            final_embedding = np.mean(embeddings, axis=0)
        elif self.aggregation == "max":
            final_embedding = np.max(embeddings, axis=0)
        else:
            raise ValueError("Unsupported aggregation method.")

        return final_embedding

    def forward(self, text: str) -> torch.Tensor:
        text = format_sample(text)
        emb = self.embed_long_text(text)
        emb_tensor = torch.tensor(emb, dtype=torch.float32).unsqueeze(0).to(next(self.classifier.parameters()).device)
        return self.classifier(emb_tensor)

