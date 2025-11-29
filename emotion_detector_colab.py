# emotion_detector_colab.py

"""
Emotion Detector for Google Colab
=================================

Este script permite detecção de emoções em tempo real usando a webcam no Google Colab.
Ele usa a biblioteca DeepFace para análise de sentimentos.

Instruções de uso:
1. Abra um notebook no Google Colab (https://colab.research.google.com/).
2. Crie uma célula de código.
3. Copie e cole todo o conteúdo deste arquivo na célula.
4. Execute a célula.
5. Permita o acesso à câmera quando solicitado pelo navegador.
"""

import sys
import subprocess

# Função para instalar dependências automaticamente
def install_dependencies():
    print("📦 Instalando dependências necessárias (pode demorar alguns minutos)...")
    packages = ["deepface", "opencv-python-headless", "tf-keras", "tensorflow", "ultralytics"]
    for package in packages:
        try:
            # Check for opencv specifically
            if "opencv" in package:
                import cv2
            else:
                __import__(package.replace("-", "_").replace("headless", ""))
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    print("✅ Dependências instaladas!")

# Executa instalação
try:
    import cv2
    import numpy as np
    from deepface import DeepFace
    from google.colab.patches import cv2_imshow
    from google.colab import output
    import base64
    from IPython.display import display, Javascript, Image
except ImportError:
    install_dependencies()
    import cv2
    import numpy as np
    from deepface import DeepFace
    from google.colab.patches import cv2_imshow
    from google.colab import output
    import base64
    from IPython.display import display, Javascript, Image

# JavaScript para capturar stream da webcam
def start_webcam_js():
    js = Javascript('''
    async function startCamera() {
      const div = document.createElement('div');
      const video = document.createElement('video');
      video.style.display = 'block';
      const stream = await navigator.mediaDevices.getUserMedia({video: true});
      document.body.appendChild(div);
      div.appendChild(video);
      video.srcObject = stream;
      await video.play();

      // Resize the output to fit the video element.
      google.colab.output.setIframeHeight(document.documentElement.scrollHeight, true);

      return video;
    }

    async function captureFrame(video) {
        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        canvas.getContext('2d').drawImage(video, 0, 0);
        const result = canvas.toDataURL('image/jpeg', 0.8);
        return result;
    }

    window.videoElement = null;

    window.startWebcam = async function() {
        window.videoElement = await startCamera();
    }

    window.getFrame = async function() {
        if (!window.videoElement) return null;
        return await captureFrame(window.videoElement);
    }
    ''')
    display(js)

def data_uri_to_cv2_img(uri):
    encoded_data = uri.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def emotion_detection_loop():
    print("🎥 Iniciando webcam...")
    start_webcam_js()
    output.eval_js('startWebcam()')

    print("🧠 Carregando modelo de emoções (DeepFace) e YOLOv8... Isso pode demorar na primeira vez.")

    # Configuração do Backend de Detecção
    # Tenta usar YOLOv8 se disponível (mais moderno), senão cai para OpenCV (mais rápido/padrão)
    detector_backend = 'opencv'
    try:
        import ultralytics
        detector_backend = 'yolov8'
        print("✅ Usando YOLOv8 para detecção facial (Maior precisão).")
    except ImportError:
        print("⚠️ YOLOv8 não encontrado ou falhou. Usando OpenCV para detecção (Modo Rápido).")

    # Carrega modelo dummy para inicializar pesos
    try:
        DeepFace.analyze(np.zeros((100, 100, 3), dtype=np.uint8),
                        actions=['emotion'],
                        detector_backend=detector_backend,
                        enforce_detection=False,
                        silent=True)
    except Exception as e:
        print(f"Nota: Inicialização falhou levemente ({e}), mas prosseguindo...")
        pass

    print("🚀 Detectando emoções! (Pressione o botão de parar no Colab para encerrar)")

    frame_count = 0
    last_emotion = "Analisando..."

    while True:
        try:
            # Captura frame do JS
            data_uri = output.eval_js('getFrame()')
            if not data_uri:
                break

            img = data_uri_to_cv2_img(data_uri)

            # Otimização: Detectar a cada 5 frames para não travar o stream
            if frame_count % 5 == 0:
                try:
                    # Tenta detecção com o backend escolhido (YOLO ou OpenCV)
                    try:
                        result = DeepFace.analyze(img,
                                                actions=['emotion'],
                                                detector_backend=detector_backend,
                                                enforce_detection=False,
                                                silent=True)
                    except Exception as e_det:
                        # Se YOLO falhar (ex: erro de CUDA ou memória), tenta fallback para OpenCV
                        if detector_backend == 'yolov8':
                             result = DeepFace.analyze(img,
                                                actions=['emotion'],
                                                detector_backend='opencv',
                                                enforce_detection=False,
                                                silent=True)
                        else:
                            raise e_det

                    if result and isinstance(result, list):
                        result = result[0]

                    if result and 'dominant_emotion' in result:
                        emotion = result['dominant_emotion']
                        # Tradução simples
                        translations = {
                            'angry': 'Raiva', 'disgust': 'Nojo', 'fear': 'Medo',
                            'happy': 'Feliz', 'sad': 'Triste', 'surprise': 'Surpresa',
                            'neutral': 'Neutro'
                        }
                        last_emotion = translations.get(emotion, emotion)

                        # Desenha retângulo no rosto se disponível
                        if 'region' in result:
                            region = result['region']
                            x, y, w, h = region['x'], region['y'], region['w'], region['h']
                            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

                except Exception as e:
                    pass # Ignora erros de detecção pontuais

            # Escreve a emoção na imagem
            cv2.putText(img, f"Emocao: {last_emotion}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Mostra imagem atualizada e limpa a anterior para efeito de vídeo
            # Nota: cv2_imshow no loop pode causar flicker, mas é o padrão do Colab
            output.clear(wait=True)
            cv2_imshow(img)

            frame_count += 1

        except Exception as e:
            print(f"Erro: {e}")
            break

if __name__ == "__main__":
    emotion_detection_loop()
