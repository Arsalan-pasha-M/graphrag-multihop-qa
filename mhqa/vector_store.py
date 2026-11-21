from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from . import config

class VectorStore:
    def __init__(self):
        self.client = QdrantClient(url=config.QDRANT_URL)
        self.model = SentenceTransformer(config.EMBED_MODEL)

    def reset(self):
        dim = self.model.get_sentence_embedding_dimension()
        if self.client.collection_exists(config.COLLECTION):
            self.client.delete_collection(config.COLLECTION)
        self.client.create_collection(config.COLLECTION,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE))

    def index(self, corpus: dict):
        titles = list(corpus)
        texts = [f"{t}. {corpus[t]}" for t in titles]
        vecs = self.model.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=True)
        points = [PointStruct(id=i, vector=v.tolist(), payload={"title": t, "text": corpus[t]})
                  for i, (t, v) in enumerate(zip(titles, vecs))]
        for i in range(0, len(points), 256):
            self.client.upsert(config.COLLECTION, points[i:i + 256])

    def search(self, query: str, k: int = config.TOP_K):
        vec = self.model.encode(query, normalize_embeddings=True).tolist()
        hits = self.client.query_points(config.COLLECTION, query=vec, limit=k).points
        return [{"title": h.payload["title"], "text": h.payload["text"], "score": h.score} for h in hits]
