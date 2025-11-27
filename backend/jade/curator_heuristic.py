import time
import logging
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Configura um logger específico para o Curador
logging.basicConfig(level=logging.INFO, format="%(asctime)s - CURATOR - %(levelname)s - %(message)s")

class MemoryCuratorHeuristic:
    """
    Implementa a curadoria de memória baseada em uma fórmula heurística (RFR-Score),
    conforme a arquitetura proposta por Ada.
    Este sistema é determinístico, rápido e escalável.
    """
    def __init__(self, shorestone_memory, llm_agent=None, alpha=0.5, beta=0.3, gamma=0.2, lambda_decay=0.01):
        """
        Inicializa o Curador.
        :param shorestone_memory: A instância ativa do sistema de memória ShoreStone.
        :param llm_agent: (Opcional) A instância do agente LLM para o modo híbrido.
        :param alpha: Peso da Frequência (F).
        :param beta: Peso da Recência (R).
        :param gamma: Peso da Relevância Geométrica (G).
        :param lambda_decay: Taxa de decaimento para o score de Recência. Controla quão rápido o esquecimento acontece.
        """
        self.memory = shorestone_memory
        self.llm_agent = llm_agent
        
        # Pesos da fórmula de Score: S = α*F + β*R + γ*G
        self.ALPHA = alpha
        self.BETA = beta
        self.GAMMA = gamma
        self.LAMBDA = lambda_decay # Um lambda pequeno significa um esquecimento mais lento

        logging.info("Curador de Memória Heurístico iniciado com sucesso.")

    def _calculate_score(self, mem_meta, mem_embedding, all_embeddings, all_ids):
        """Calcula o RFR-Score para uma única memória."""
        
        # 1. Frequência (F) - Usando escala logarítmica
        access_count = mem_meta.get('access_count', 0)
        f_score = np.log(1 + access_count)

        # 2. Recência (R) - Usando decaimento exponencial
        now = time.time()
        last_accessed = mem_meta.get('last_accessed_at', now)
        days_since_access = (now - last_accessed) / (24 * 60 * 60)
        r_score = np.exp(-self.LAMBDA * days_since_access)

        # 3. Relevância Geométrica (G) - Coerência com vizinhos próximos (k-NN)
        # Esta é uma abordagem robusta: uma memória é relevante se estiver em um "bairro" coeso.
        # Encontra os 5 vizinhos mais próximos (excluindo ela mesma).
        similarities = cosine_similarity(mem_embedding.reshape(1, -1), all_embeddings)[0]
        # Pega os índices dos mais similares, em ordem decrescente
        nearest_indices = np.argsort(similarities)[::-1][1:6] # [1:6] para ignorar a si mesma
        
        if len(nearest_indices) > 0:
            # A relevância é a similaridade média com seus vizinhos mais próximos
            g_score = np.mean(similarities[nearest_indices])
        else:
            g_score = 0 # Se não tiver vizinhos, a relevância é zero

        # Fórmula Final Ponderada
        final_score = (self.ALPHA * f_score) + (self.BETA * r_score) + (self.GAMMA * g_score)
        
        return final_score

    def run_maintenance_cycle(self, t_delete=0.2, t_archive=0.4):
        """
        Executa um ciclo de verificação, pontuação e poda da memória.
        """
        logging.info("Iniciando ciclo de manutenção da memória...")
        collection = self.memory.collection
        if not collection or collection.count() == 0:
            logging.info("Nenhuma memória para manter.")
            return

        # Pega TODAS as memórias, incluindo seus embeddings e metadados. Essencial para o cálculo de relevância.
        all_mems = collection.get(include=["metadatas", "embeddings"])
        
        if len(all_mems['ids']) < 10: # Não roda a manutenção em memórias muito recentes
             logging.info("Poucas memórias, pulando ciclo de manutenção.")
             return

        all_ids = all_mems['ids']
        all_embeddings = np.array(all_mems['embeddings'])
        all_metadatas = all_mems['metadatas']
        
        ids_to_delete = []
        
        logging.info(f"Avaliando {len(all_ids)} memórias...")
        for i in range(len(all_ids)):
            mem_id = all_ids[i]
            mem_meta = all_metadatas[i]
            mem_embedding = all_embeddings[i]
            
            score = self._calculate_score(mem_meta, mem_embedding, all_embeddings, all_ids)
            
            # --- Lógica de Decisão ---
            if score < t_delete:
                # MODO HÍBRIDO: Verificação de segurança para memórias fundamentais
                if "meu criador" in mem_meta.get('text', '').lower() or "gabriel yogi" in mem_meta.get('text', '').lower():
                    logging.warning(f"SCORE BAIXO [{score:.2f}] para memória potencialmente fundamental '{mem_id}'. IGNORANDO EXCLUSÃO.")
                    continue
                
                logging.info(f"Memória '{mem_id}' marcada para exclusão com score [{score:.2f}]")
                ids_to_delete.append(mem_id)
            
            # A lógica de arquivamento (mover para outra coleção) pode ser adicionada aqui.
            # elif t_delete <= score < t_archive:
            #     ...

        if ids_to_delete:
            logging.info(f"Excluindo {len(ids_to_delete)} memórias obsoletas...")
            collection.delete(ids=ids_to_delete)
            logging.info("Exclusão concluída.")
        else:
            logging.info("Nenhuma memória atingiu o threshold para exclusão. Ciclo concluído.")