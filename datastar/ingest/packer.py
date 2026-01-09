import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm
import os

class DataStarPacker:
    """
    O Compactador do DataStar.
    Transforma texto bruto em binário contíguo (uint16 ou uint32) via Memory Mapping.
    Zero risco de estourar a RAM, mesmo com datasets de Terabytes.
    """
    def __init__(self, txt_path, bin_path, tokenizer_name):
        self.txt_path = txt_path
        self.bin_path = bin_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def run(self):
        print(f"📦 DataStar Packer: Iniciando compactação de {self.txt_path}...")

        # Determine appropriate dtype based on vocab size
        vocab_size = self.tokenizer.vocab_size
        if vocab_size < 65535:
            self.dtype = np.uint16
            bytes_per_token = 2
            print(f"📊 Vocab size: {vocab_size} -> Using uint16")
        else:
            self.dtype = np.uint32
            bytes_per_token = 4
            print(f"📊 Vocab size: {vocab_size} -> Using uint32")

        # 1. Primeira passada: Contar tokens exatos para alocar o arquivo binário
        print("🔍 Fase 1: Calculando tamanho final (Pre-alloc)...")
        total_tokens = 0
        if not os.path.exists(self.txt_path):
             raise FileNotFoundError(f"{self.txt_path} not found")

        with open(self.txt_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Contando"):
                if line:
                    # Estimativa rápida
                    total_tokens += len(line) // 3

        # Margem de segurança de 10%
        alloc_size = int(total_tokens * 1.1)
        if alloc_size == 0:
            alloc_size = 1024 # Min alloc

        print(f"💾 Alocando arquivo virtual de ~{alloc_size} tokens...")

        # 2. Cria o arquivo mapeado em memória
        arr = np.memmap(self.bin_path, dtype=self.dtype, mode='w+', shape=(alloc_size,))

        # 3. Fase 2: Tokenização Real e Escrita Direta no Disco
        print("⚡ Fase 2: Tokenizando e gravando direto no disco...")
        idx = 0
        with open(self.txt_path, 'r', encoding='utf-8') as f:
            pbar = tqdm(total=alloc_size, unit="tok", desc="Packing")

            for line in f:
                try:
                    import json
                    text = json.loads(line)
                except:
                    text = line # Fallback se não for json

                text = text + self.tokenizer.eos_token
                token_ids = self.tokenizer.encode(text, add_special_tokens=False)

                # Verifica espaço no array
                if idx + len(token_ids) > alloc_size:
                    print("⚠️ Expandindo memmap (Estimativa foi baixa)...")
                    arr.flush()
                    new_size = int(idx + len(token_ids) + 1e6)

                    del arr
                    # Extend file
                    with open(self.bin_path, "ab") as fb:
                         fb.write(b'\x00' * (new_size - alloc_size) * bytes_per_token)

                    arr = np.memmap(self.bin_path, dtype=self.dtype, mode='r+', shape=(new_size,))
                    alloc_size = new_size

                # Grava no array
                arr[idx : idx + len(token_ids)] = token_ids
                idx += len(token_ids)
                pbar.update(len(token_ids))

        # 4. Finalização: Cortar o excesso
        print(f"✂️ Finalizando: Cortando excesso de alocação. Total real: {idx}")
        del arr

        # Truncate file
        with open(self.bin_path, "r+b") as f:
             f.truncate(idx * bytes_per_token)

        print(f"✅ Sucesso! Arquivo '{self.bin_path}' pronto para treino.")
        print(f"📊 Tamanho final: {os.path.getsize(self.bin_path) / 1024**2:.2f} MB")
