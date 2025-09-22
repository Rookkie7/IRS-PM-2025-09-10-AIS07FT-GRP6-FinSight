from app.ports.vector_index import VectorIndexPort
from app.adapters.db.mongo_client import db
from app.model.models import EmbeddingVector

class MongoVectorIndex(VectorIndexPort):
    def __init__(self, collection_name: str):
        self.col = db[collection_name]