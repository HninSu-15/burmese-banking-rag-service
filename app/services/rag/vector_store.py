# app/services/rag/vector_store.py
import os
import logging
from typing import Optional, List, Dict, Any
from pathlib import Path

from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.schema import BaseNode
from llama_index.vector_stores.chroma import ChromaVectorStore
from core.config import settings

# Load environment variables
load_dotenv()

# Logger setup
logger = logging.getLogger(__name__)

# Constants with environment variable fallbacks
DEFAULT_PERSIST_DIR = os.getenv("VECTOR_STORE_PATH", "./data/chroma_db")
DEFAULT_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "burmese_banking_knowledge")


class ChromaVectorStoreService:
    """
    Production-grade Vector Store Service utilizing ChromaDB and LlamaIndex.
    Handles persistent storage, index creation, node insertion, and similarity queries.
    """

    def __init__(
        self,
        persist_dir: str = DEFAULT_PERSIST_DIR,
        collection_name: str = DEFAULT_COLLECTION_NAME,
    ) -> None:
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        
        # Ensure target directory exists for local persistence
        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)

        logger.info(f"Initializing ChromaDB PersistentClient at: {self.persist_dir}")
        
        # 1. Initialize Persistent ChromaDB Client
        self._chroma_client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        # 2. Get or create Chroma collection (using cosine similarity metric)
        logger.info(f"Connecting to Chroma Collection: '{self.collection_name}'")
        self._chroma_collection = self._chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        # 3. Instantiate LlamaIndex Vector Store Connector
        self._vector_store = ChromaVectorStore(chroma_collection=self._chroma_collection)

        # 4. Create LlamaIndex Storage Context
        self._storage_context = StorageContext.from_defaults(vector_store=self._vector_store)

    def delete_document(self, doc_name: str) -> int:
        """
        Delete all chunks belonging to a specific document by its filename.
        Returns the number of chunks deleted.
        """
        try:
            results = self._chroma_collection.get(where={"doc_name": doc_name})
            
            if results and results['ids']:
                self._chroma_collection.delete(ids=results['ids'])
                deleted_count = len(results['ids'])
                logger.info(f" Deleted {deleted_count} chunks for document: {doc_name}")
                return deleted_count
            else:
                logger.info(f" No existing chunks found for document: {doc_name}")
                return 0
        except Exception as e:
            logger.error(f" Failed to delete document {doc_name}: {e}")
            return 0


    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
        threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks for a query.
        
        Args:
            query: User query text
            top_k: Number of results (default: from config)
            filter_metadata: Metadata filters
            threshold: Minimum relevance score (default: from config)
        """
        from app.services.rag.embedder_service import get_embedder_service
        
        if not query or not query.strip():
            return []
        
        # Use config values if not provided
        if top_k is None:
            top_k = settings.TOP_K  # ← Config ကနေယူတယ်
        if threshold is None:
            threshold = settings.SIMILARITY_THRESHOLD  # ← Config ကနေယူတယ်
        
        try:
            embedder = get_embedder_service()
            query_vector = embedder.get_query_embedding(query)
            
            # Get more results for filtering
            results = self._chroma_collection.query(
                query_embeddings=[query_vector],
                n_results=top_k * 3,
                where=filter_metadata
            )
            
            formatted = []
            if results and results.get("documents") and results["documents"][0]:
                for i in range(len(results["documents"][0])):
                    distance = results["distances"][0][i] if "distances" in results else None
                    relevance_score = 1.0 - distance if distance is not None else None
                    
                    # Filter by threshold
                    if relevance_score is not None and relevance_score < threshold:
                        continue
                    
                    formatted.append({
                        "chunk_id": results["ids"][0][i],
                        "text": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "distance": distance,
                        "relevance_score": relevance_score
                    })
            
            return formatted[:top_k]
            
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return []
    
    @property
    def vector_store(self) -> ChromaVectorStore:
        """Returns the LlamaIndex ChromaVectorStore instance."""
        return self._vector_store

    @property
    def storage_context(self) -> StorageContext:
        """Returns the LlamaIndex StorageContext instance."""
        return self._storage_context

    def get_index(self, embed_model: Any = None) -> VectorStoreIndex:
        """
        Retrieves or initializes a VectorStoreIndex from the stored embeddings.
        """
        logger.info("Initializing VectorStoreIndex from persistent Chroma vector store...")
        return VectorStoreIndex.from_vector_store(
            vector_store=self._vector_store,
            embed_model=embed_model,
        )

    def add_nodes(self, nodes: List[BaseNode], embed_model: Any = None) -> None:
        """
        Inserts new embedded document nodes into the persistent Vector Database.
        
        Note: If nodes already have embeddings, they are inserted directly.
        If embed_model is provided, it will be used to generate embeddings.
        """
        if not nodes:
            logger.warning("No nodes provided for insertion.")
            return

        logger.info(f"Inserting {len(nodes)} document nodes into vector store...")
        
        # Check if nodes already have embeddings
        has_embeddings = all(hasattr(node, 'embedding') and node.embedding is not None for node in nodes)
        
        if has_embeddings:
            # ✅ FIX: Direct insertion with metadata
            logger.info("Nodes already have embeddings. Inserting directly into ChromaDB...")
            
            # Extract data for ChromaDB
            ids = [node.id_ for node in nodes]
            documents = [node.text for node in nodes]
            embeddings = [node.embedding for node in nodes]
            metadatas = [node.metadata for node in nodes]  # ← Metadata 
            
            self._chroma_collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas
            )
            logger.info(f"Successfully added {len(nodes)} nodes to ChromaDB.")
        else:
            # Fallback: Use VectorStoreIndex (requires embed_model)
            logger.info("Nodes do not have embeddings. Using VectorStoreIndex...")
            index = VectorStoreIndex(
                nodes=nodes,
                storage_context=self._storage_context,
                embed_model=embed_model,
            )
            logger.info("Successfully added nodes to ChromaDB.")

    def count_documents(self) -> int:
        """
        Returns total number of vector items currently stored in the collection.
        """
        return self._chroma_collection.count()

    def clear_collection(self) -> None:
        """
        Resets and clears all vector embeddings from the collection.
        """
        logger.warning(f"Clearing all documents in collection '{self.collection_name}'...")
        self._chroma_client.delete_collection(name=self.collection_name)
        # Re-create fresh collection
        self._chroma_collection = self._chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        self._vector_store = ChromaVectorStore(chroma_collection=self._chroma_collection)
        self._storage_context = StorageContext.from_defaults(vector_store=self._vector_store)
        logger.info("Collection successfully reset.")


# Singleton Instance Pattern for Dependency Injection
_vector_store_instance: Optional[ChromaVectorStoreService] = None


def get_vector_store_service(
    persist_dir: str = DEFAULT_PERSIST_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> ChromaVectorStoreService:
    """
    Factory function providing a Singleton instance of ChromaVectorStoreService.
    Prevents opening multiple DB client locks on the persistent storage file.
    """
    global _vector_store_instance
    if _vector_store_instance is None:
        _vector_store_instance = ChromaVectorStoreService(
            persist_dir=persist_dir,
            collection_name=collection_name,
        )
    return _vector_store_instance

