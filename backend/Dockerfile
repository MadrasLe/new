# Usa uma imagem Python leve e moderna
FROM python:3.10-slim

# Define a pasta de trabalho dentro do container
WORKDIR /app

# Copia os requisitos primeiro (para aproveitar o cache do Docker e ser mais rápido)
COPY requirements.txt .

# Instala as dependências
# A Berta adicionou o --no-cache-dir para o container ficar menorzinho
RUN pip install --no-cache-dir -r requirements.txt

# Copia todo o resto do seu código para dentro do container
COPY . .

# Cria a pasta para o banco de dados e dá permissão (importante no Linux do HF)
RUN mkdir -p /app/jade_memory_db && chmod 777 /app/jade_memory_db

# Comando para ligar o servidor
# O Hugging Face sempre espera a porta 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]