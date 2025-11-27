# Arquivo: jade/shorestone.py

import chromadb
import uuid
import logging
import time
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer

# Configura um logger para nosso módulo de memória
logging.basicConfig(level=logging.INFO, format="%(asctime)s - SHORESTONE - %(levelname)s - %(message)s")

class ShoreStoneMemory:
    """
    Sistema de memória vetorial persistente e enriquecido com metadados para curadoria.
    Salva os dados em disco e gerencia o ciclo de vida das memórias.
    """
    def __init__(self, model_name='all-MiniLM-L6-v2', storage_path="./jade_memory_db", pca_model_path="pca_model.joblib"):
        """
        Inicializa o ShoreStone.
        :param model_name: O modelo de embedding a ser usado.
        :param storage_path: O diretório no disco onde a memória será salva.
        :param pca_model_path: (Opcional) Caminho para um modelo PCA treinado para compressão de vetores.
        """
        logging.info("Iniciando o motor de memória ShoreStone...")
        self.model = SentenceTransformer(model_name)
        # Usa PersistentClient para salvar os dados em disco
        self.client = chromadb.PersistentClient(path=storage_path)
        self.collection = None

        # Tenta carregar o modelo PCA para compressão de vetores
        try:
            logging.info(f"Tentando carregar modelo PCA de '{pca_model_path}'...")
            self.pca = joblib.load(pca_model_path)
            logging.info("Modelo PCA carregado. A compressão de vetores está ATIVA.")
        except FileNotFoundError:
            logging.warning(f"Arquivo do modelo PCA '{pca_model_path}' não encontrado. A compressão de vetores está DESATIVADA.")
            self.pca = None
        
        logging.info("Motor ShoreStone pronto.")

    def load_or_create_session(self, session_name: str):
        """
        Carrega uma sessão de memória existente ou cria uma nova.
        :param session_name: O nome da sessão (coleção no ChromaDB).
        """
        logging.info(f"Carregando ou criando a sessão de memória: '{session_name}'")
        self.collection = self.client.get_or_create_collection(name=session_name)
        logging.info(f"Sessão '{session_name}' pronta com {self.collection.count()} memórias.")

    def _get_compressed_embedding(self, text: str) -> list:
        """Gera um embedding e o comprime com PCA, se o modelo PCA estiver carregado."""
        embedding = self.model.encode(text)
        if self.pca:
            embedding_2d = embedding.reshape(1, -1)
            compressed_embedding = self.pca.transform(embedding_2d)
            return compressed_embedding.flatten().tolist()
        return embedding.tolist()

    def memorize(self, user_input: str, assistant_response: str):
        """Adiciona uma nova memória à sessão ativa, enriquecida com metadados para curadoria."""
        if not self.collection:
            logging.error("Nenhuma sessão de memória ativa. Use 'load_or_create_session' primeiro.")
            return

        text = f"O usuário disse: '{user_input}'. A IA respondeu: '{assistant_response}'"
        embedding = self._get_compressed_embedding(text)
        doc_id = str(uuid.uuid4())
        
        current_timestamp = time.time()
        metadata = {
            "created_at": current_timestamp,
            "last_accessed_at": current_timestamp,
            "access_count": 0,
            "text": text  # Armazena o texto completo nos metadados para o Curador
        }

        self.collection.add(
            embeddings=[embedding],
            metadatas=[metadata],
            documents=[text.split('.')[0] + "..."], # Documento pode ser um resumo para buscas rápidas
            ids=[doc_id]
        )
        logging.info(f"Nova memória adicionada à sessão '{self.collection.name}'.")

    def remember(self, query: str, n_results: int = 3) -> str | None:
        """
        Busca as memórias mais relevantes e ATUALIZA seus metadados de acesso.
        """
        if not self.collection or self.collection.count() == 0:
            return None

        query_embedding = self._get_compressed_embedding(query)
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["metadatas", "documents"] # Essencial para obter os metadados para atualização
        )

        if not (results and results.get('ids') and results['ids'][0]):
            return None

        # Atualiza os metadados das memórias que foram recuperadas (acessadas)
        ids_to_update = results['ids'][0]
        metadatas_to_update = results['metadatas'][0]
        
        current_timestamp = time.time()
        updated_metadatas = []
        for meta in metadatas_to_update:
            meta['access_count'] = meta.get('access_count', 0) + 1
            meta['last_accessed_at'] = current_timestamp
            updated_metadatas.append(meta)

        if ids_to_update:
            self.collection.update(ids=ids_to_update, metadatas=updated_metadatas)
            logging.info(f"{len(ids_to_update)} memórias tiveram seus metadados de acesso atualizados.")

        retrieved_docs = "\n".join(results['documents'][0])
        return retrieved_docs