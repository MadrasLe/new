import os
import json
import concurrent.futures
from typing import Optional, List, Iterator, Tuple
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from datasets import load_dataset

class DataStarIngest:
    """
    Módulo de Ingestão de Alta Performance do DataStar.

    Responsável por:
    1. Streaming de datasets massivos (HuggingFace/Parquet).
    2. Tokenização paralela (Multi-core CPU).
    3. Normalização de texto preservando estrutura (JSONL).
    """

    def __init__(self, tokenizer_name: str, max_workers: Optional[int] = None, batch_size: int = 4096):
        """
        Inicializa o motor de ingestão.

        Args:
            tokenizer_name (str): ID do modelo no HuggingFace (ex: 'mistralai/Mistral-7B-v0.1').
            max_workers (int, optional): Número de núcleos de CPU. Padrão: todos disponíveis.
            batch_size (int): Tamanho do lote para envio aos processos filhos.
        """
        print(f"🌟 [DataStar Ingest] Inicializando com tokenizer: {tokenizer_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.batch_size = batch_size
        self.max_workers = max_workers or os.cpu_count()
        print(f"🚀 [DataStar Ingest] CPU Power: {self.max_workers} núcleos ativos.")

    def _process_batch(self, batch_texts: List[str]) -> Tuple[int, List[str]]:
        """
        Método interno executado pelos workers.
        Tokeniza e formata um lote de textos.
        """
        processed_lines = []
        batch_tokens_count = 0

        # Tokenização vetorizada (Rust-speed)
        encodings = self.tokenizer(batch_texts, add_special_tokens=False, return_length=True)

        for txt, length in zip(batch_texts, encodings['length']):
            if length == 0: continue

            # Normalização DataStar: JSON para preservar newlines e estrutura
            clean_txt = json.dumps(txt.strip(), ensure_ascii=False)

            processed_lines.append(clean_txt)
            batch_tokens_count += length

        return batch_tokens_count, processed_lines

    def stream_from_huggingface(self, repo_id: str, split: str = "train") -> Iterator[str]:
        """
        Gerador robusto que carrega datasets ignorando erros de schema (DataStar Mode).
        """
        print(f"📡 [DataStar Ingest] Conectando via Parquet Mode ao {repo_id}...")
        try:
            # Estratégia "Bare Metal": Lê os parquets direto, ignorando metadados quebrados
            ds = load_dataset(
                "parquet",
                data_files={split: f"hf://datasets/{repo_id}/data/**/*.parquet"},
                split=split,
                streaming=True,
                columns=["text"] # Projeção de coluna para economizar banda
            )
            for ex in ds:
                yield ex["text"]
        except Exception as e:
            raise RuntimeError(f"❌ Falha crítica no streaming do DataStar: {e}")

    def run_pipeline(self, source_iterator: Iterator[str], output_path: str, quota_tokens: int, min_chars: int = 100):
        """
        Executa o pipeline completo de ingestão -> processamento -> disco.

        Args:
            source_iterator: Iterador de strings (use stream_from_huggingface ou sua própria fonte).
            output_path: Onde salvar o arquivo .txt ou .jsonl final.
            quota_tokens: Meta de tokens para processar.
            min_chars: Filtro mínimo de caracteres por documento.
        """
        collected_tokens = 0
        buffer_texts = []

        print(f"💾 [DataStar Ingest] Gravando saída em: {output_path}")

        with open(output_path, "w", encoding="utf-8") as f_out:
            with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                futures = []
                pbar = tqdm(total=quota_tokens, unit="tok", desc="🔥 DataStar Processing")

                try:
                    for txt in source_iterator:
                        # Filtro rápido (pré-tokenização)
                        if not txt or len(txt) < min_chars:
                            continue

                        buffer_texts.append(txt)

                        # Disparo de lote
                        if len(buffer_texts) >= self.batch_size:
                            # Envia cópia para worker
                            future = executor.submit(self._process_batch, list(buffer_texts))
                            futures.append(future)
                            buffer_texts = []

                            # Backpressure: Evita estourar RAM acumulando tarefas
                            if len(futures) >= self.max_workers * 2:
                                done, _ = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)
                                for fut in done:
                                    futures.remove(fut)
                                    count, lines = fut.result()
                                    if lines:
                                        f_out.write("\n".join(lines) + "\n")
                                    collected_tokens += count
                                    pbar.update(count)

                                if collected_tokens >= quota_tokens:
                                    break

                        if collected_tokens >= quota_tokens:
                            break

                except Exception as e:
                    print(f"⚠️ [DataStar Ingest] Aviso: Stream interrompido: {e}")

                # Drenagem final dos processos
                print("⏳ [DataStar Ingest] Finalizando tarefas pendentes...")
                for fut in concurrent.futures.as_completed(futures):
                    count, lines = fut.result()
                    if lines:
                        f_out.write("\n".join(lines) + "\n")
                    collected_tokens += count
                    pbar.update(count)

                # Buffer residual
                if buffer_texts and collected_tokens < quota_tokens:
                    count, lines = self._process_batch(buffer_texts)
                    if lines:
                        f_out.write("\n".join(lines) + "\n")
                    collected_tokens += count
                    pbar.update(count)

                pbar.close()

        print(f"✅ [DataStar Ingest] Concluído! Total: {collected_tokens/1e6:.2f}M tokens.")
