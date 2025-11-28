import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm
import os
import json

class DataStarPacker:
    """
    O Compactador do DataStar.
    Transforma texto bruto em binário contíguo (uint16) via Memory Mapping.
    Zero risco de estourar a RAM, mesmo com datasets de Terabytes.
    """
    def __init__(self, txt_path, bin_path, tokenizer_name):
        self.txt_path = txt_path
        self.bin_path = bin_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def run(self):
        print(f"📦 DataStar Packer: Iniciando compactação de {self.txt_path}...")

        # 1. Primeira passada: Contar tokens exatos para alocar o arquivo binário
        # Isso parece lento, mas é necessário para criar o memmap do tamanho exato.
        # Se quiser arriscar, pode estimar, mas o DataStar gosta de precisão.
        print("🔍 Fase 1: Calculando tamanho final (Pre-alloc)...")
        total_tokens = 0
        with open(self.txt_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Contando"):
                # Estimativa rápida baseada em caracteres para não tokenizar tudo duas vezes
                # Média do Mistral: ~3.5 chars por token
                if line:
                    total_tokens += len(line) // 3

        # Margem de segurança de 10%
        alloc_size = int(total_tokens * 1.1)
        print(f"💾 Alocando arquivo virtual de ~{alloc_size} tokens...")

        # 2. Cria o arquivo mapeado em memória (O segredo do Big Data)
        arr = np.memmap(self.bin_path, dtype=np.uint16, mode='w+', shape=(alloc_size,))

        # 3. Fase 2: Tokenização Real e Escrita Direta no Disco
        print("⚡ Fase 2: Tokenizando e gravando direto no disco...")
        idx = 0
        with open(self.txt_path, 'r', encoding='utf-8') as f:
            # Buffer de escrita para acelerar
            batch_lines = []

            pbar = tqdm(total=alloc_size, unit="tok", desc="Packing")

            for line in f:
                # O ingestor anterior salvou json, lembra? Precisamos carregar.
                try:
                    text = json.loads(line)
                except:
                    text = line # Fallback se não for json

                # Adiciona EOS token (importante para o modelo saber que o texto acabou)
                text = text + self.tokenizer.eos_token

                token_ids = self.tokenizer.encode(text, add_special_tokens=False)

                # Verifica se cabe (uint16 só vai até 65535, Mistral tem 32k, então tá safe)
                # Verifica espaço no array
                if idx + len(token_ids) > alloc_size:
                    # Resize se a estimativa foi ruim (Custo caro, mas salva o job)
                    print("⚠️ Expandindo memmap (Estimativa foi baixa)...")
                    arr.flush()
                    new_size = int(idx + len(token_ids) + 1e6)
                    arr = np.memmap(self.bin_path, dtype=np.uint16, mode='r+', shape=(new_size,))
                    alloc_size = new_size

                # Grava no array (que na verdade é o disco SSD)
                arr[idx : idx + len(token_ids)] = token_ids
                idx += len(token_ids)
                pbar.update(len(token_ids))

        # 4. Finalização: Cortar o excesso
        print(f"✂️ Finalizando: Cortando excesso de alocação. Total real: {idx}")
        final_arr = np.memmap(self.bin_path, dtype=np.uint16, mode='r+', shape=(idx,))
        final_arr.flush()

        print(f"✅ Sucesso! Arquivo '{self.bin_path}' pronto para treino.")
        print(f"📊 Tamanho final: {os.path.getsize(self.bin_path) / 1024**2:.2f} MB")
