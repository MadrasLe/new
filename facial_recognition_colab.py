# -*- coding: utf-8 -*-
"""facial_recognition_colab.py

Este script foi projetado para rodar no Google Colab.
Ele implementa reconhecimento facial usando a biblioteca DeepFace.

Instruções:
1. Copie o código abaixo para uma célula no Google Colab.
2. Execute a célula. Se for a primeira vez, ele vai instalar as dependências.
3. Use a função `registrar_usuario()` para adicionar faces ao banco de dados.
4. Use a função `iniciar_reconhecimento()` para começar o reconhecimento via webcam.

"""

import os
import sys
import time
from base64 import b64decode
import logging

# Tenta importar bibliotecas do Colab. Se falhar, assume que não está no Colab ou precisa instalar.
try:
    from IPython.display import display, Javascript, Image, clear_output
    from google.colab.output import eval_js
except ImportError:
    print("Aviso: Bibliotecas do Google Colab não encontradas. Este script é otimizado para rodar no Colab.")

# Variáveis globais para as bibliotecas que serão carregadas sob demanda
np = None
cv2 = None
DeepFace = None

def verificar_e_instalar():
    """Verifica e instala dependências necessárias."""
    global np, cv2, DeepFace

    print("Verificando dependências...")
    precisa_reiniciar = False

    # Verifica DeepFace
    try:
        from deepface import DeepFace as DF
        DeepFace = DF
    except ImportError:
        print("Instalando DeepFace...")
        os.system("pip install deepface tf-keras opencv-python matplotlib")
        precisa_reiniciar = True

    # Verifica OpenCV e Numpy (geralmente já tem no Colab)
    try:
        import cv2 as cv
        import numpy as npy
        cv2 = cv
        np = npy
    except ImportError:
        print("Instalando OpenCV/Numpy...")
        os.system("pip install opencv-python numpy")
        precisa_reiniciar = True

    if precisa_reiniciar:
        print("\nDependências instaladas!")
        print("IMPORTANTE: Pode ser necessário reiniciar o Runtime do Colab para que as bibliotecas carreguem corretamente.")
        print("Se der erro de importação, vá em 'Runtime' > 'Restart Runtime' e rode o script novamente.")

        # Tenta importar novamente após instalação
        try:
            from deepface import DeepFace as DF
            import cv2 as cv
            import numpy as npy
            DeepFace = DF
            cv2 = cv
            np = npy
        except ImportError:
            print("Não foi possível carregar as bibliotecas automaticamente. Por favor, reinicie o Runtime.")
            return False

    # Configuração de Logs
    if os.environ.get('TF_CPP_MIN_LOG_LEVEL') != '2':
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    if DeepFace:
        logging.getLogger('deepface').setLevel(logging.ERROR)

    # Cria pasta para o banco de dados de faces
    if not os.path.exists("db"):
        os.makedirs("db")
        print("Pasta 'db' criada para armazenar as fotos.")

    return True

# --- Funções Auxiliares para Webcam (JavaScript Injection) ---

def js_take_photo(quality=0.8):
  js = Javascript('''
    async function takePhoto(quality) {
      const div = document.createElement('div');
      const capture = document.createElement('button');
      capture.textContent = 'Capturar Foto';
      div.appendChild(capture);

      const video = document.createElement('video');
      video.style.display = 'block';
      const stream = await navigator.mediaDevices.getUserMedia({video: true});

      document.body.appendChild(div);
      div.appendChild(video);
      video.srcObject = stream;
      await video.play();

      // Redimensiona o output para caber no vídeo
      google.colab.output.setIframeHeight(document.documentElement.scrollHeight, true);

      // Espera pelo clique
      await new Promise((resolve) => capture.onclick = resolve);

      const canvas = document.createElement('canvas');
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      canvas.getContext('2d').drawImage(video, 0, 0);
      stream.getVideoTracks()[0].stop();
      div.remove();
      return canvas.toDataURL('image/jpeg', quality);
    }
    ''')
  display(js)
  data = eval_js('takePhoto({})'.format(quality))
  return data

def js_video_stream_start():
  js = Javascript('''
    async function startVideo() {
        const video = document.createElement('video');
        video.id = 'video-stream';
        video.style.display = 'block';
        video.width = 640;
        video.height = 480;
        const stream = await navigator.mediaDevices.getUserMedia({video: true});
        video.srcObject = stream;
        await video.play();
        document.body.appendChild(video);
    }
    startVideo();
  ''')
  display(js)

def js_capture_frame():
    script = '''
    (function() {
      const video = document.getElementById('video-stream');
      if (!video) return null;
      const canvas = document.createElement('canvas');
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      canvas.getContext('2d').drawImage(video, 0, 0);
      return canvas.toDataURL('image/jpeg', 0.8);
    })()
    '''
    return eval_js(script)

def js_stop_video():
    display(Javascript('''
        const video = document.getElementById('video-stream');
        if (video) {
            video.srcObject.getTracks().forEach(track => track.stop());
            video.remove();
        }
    '''))

# --- Funções Principais ---

def registrar_usuario():
    """
    Captura uma foto da webcam e salva na pasta 'db' com o nome fornecido.
    """
    if not verificar_e_instalar():
        return

    print("\n--- Registro de Usuário ---")
    nome = input("Digite o nome da pessoa (sem espaços ou caracteres especiais preferencialmente): ")
    if not nome:
        print("Nome inválido.")
        return

    print("Por favor, olhe para a câmera e aguarde o botão aparecer...")
    try:
        data = js_take_photo()
        binary = b64decode(data.split(',')[1])

        filename = f"db/{nome}.jpg"
        with open(filename, 'wb') as f:
            f.write(binary)

        print(f"Foto salva com sucesso em: {filename}")
        display(Image(filename))

        # Remove arquivo de representações antigo para atualizar o banco
        if os.path.exists("db/representations_vgg_face.pkl"):
            os.remove("db/representations_vgg_face.pkl")

    except Exception as e:
        print(f"Erro ao capturar foto: {e}")

def iniciar_reconhecimento():
    """
    Inicia o stream de vídeo e tenta reconhecer faces em tempo real.
    """
    if not verificar_e_instalar():
        return

    # Verifica se há fotos no DB
    if not os.path.exists("db") or not os.listdir("db"):
        print("A pasta 'db' está vazia ou não existe. Use registrar_usuario() primeiro.")
        return

    print("\n--- Iniciando Reconhecimento Facial ---")
    print("O vídeo será exibido abaixo. O reconhecimento roda em loop.")
    print("Pressione o botão 'Interromper' (Stop) do Colab para finalizar.")

    js_video_stream_start()

    try:
        while True:
            # Captura frame do vídeo
            frame_data = js_capture_frame()
            if not frame_data:
                time.sleep(1)
                continue

            # Converte base64 para imagem OpenCV
            image_bytes = b64decode(frame_data.split(',')[1])
            jpg_as_np = np.frombuffer(image_bytes, dtype=np.uint8)
            img = cv2.imdecode(jpg_as_np, flags=1)

            try:
                # DeepFace.find busca a face na imagem contra o banco de dados em "db"
                # model_name='VGG-Face' é o padrão
                dfs = DeepFace.find(
                    img_path=img,
                    db_path="db",
                    model_name="VGG-Face",
                    detector_backend="opencv",
                    enforce_detection=False,
                    silent=True
                )

                reconhecido = False
                nome_reconhecido = "Desconhecido"

                if len(dfs) > 0:
                    for df in dfs:
                        if not df.empty:
                            match = df.iloc[0]
                            caminho_identidade = match['identity']
                            nome_reconhecido = os.path.basename(caminho_identidade).split('.')[0]
                            reconhecido = True
                            break

                # Feedback no output
                clear_output(wait=True)
                if reconhecido:
                    print(f"✅ IDENTIFICADO: {nome_reconhecido}")
                else:
                    print("⚠️ Rosto não identificado ou nenhum rosto detectado.")

            except Exception as e:
                clear_output(wait=True)
                print(f"Processando... (Erro: {str(e)})")

            # Pausa para não sobrecarregar
            time.sleep(2)

    except KeyboardInterrupt:
        print("\nReconhecimento encerrado pelo usuário.")
        js_stop_video()

if __name__ == "__main__":
    # Verifica instalação ao carregar
    verificar_e_instalar()
    print("\nScript pronto! Execute:\n -> registrar_usuario()\n -> iniciar_reconhecimento()")
