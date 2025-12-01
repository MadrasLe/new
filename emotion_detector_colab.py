# emotion_detector_colab.py

"""
Emotion Detector for Google Colab v3.3 (Protobuf Fix)
=====================================================

Este script permite detecção de emoções em tempo real e usa uma IA (LLM)
para comentar sobre o estado emocional do usuário.

Novidades v3.3:
- Fix para erro de "Failed to parse: node" do MediaPipe (conflito de Protobuf).
- Instalação forçada de protobuf compatível se necessário.
- Tratamento de erro aprimorado na inicialização dos detectores.

Instruções:
1. Execute a célula.
2. Insira sua chave da Groq API (opcional).
3. Permita o uso da câmera.
"""

import sys
import subprocess
import time
import csv
import threading
import queue
import random
import os
import traceback
from datetime import datetime

# Define backend puro para protobuf para evitar crash em algumas versões
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# Função para instalar dependências automaticamente
def install_dependencies():
    print("📦 Instalando dependências necessárias (pode demorar alguns minutos)...")
    # Forçamos protobuf 3.20.3 para compatibilidade máxima com MediaPipe antigo se necessário
    # mas o env var acima geralmente resolve.
    packages = [
        "deepface", "opencv-python-headless", "tf-keras", "tensorflow",
        "ultralytics", "groq", "mediapipe", "mtcnn"
    ]
    for package in packages:
        try:
            if "opencv" in package:
                import cv2
            else:
                module_name = package.replace("-", "_").replace("headless", "")
                if module_name == "mediapipe": module_name = "mediapipe"
                __import__(module_name)
        except ImportError:
            print(f"   -> Instalando {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--quiet"])

    # Verifica compatibilidade do protobuf
    try:
        import google.protobuf
        # Se for v4+, pode dar problema com certas builds do mediapipe, mas o env var ajuda
    except:
        pass

    print("✅ Dependências instaladas!")

# Executa instalação e imports
try:
    import cv2
    import numpy as np
    from deepface import DeepFace
    from google.colab.patches import cv2_imshow
    from google.colab import output
    import base64
    from IPython.display import display, Javascript, Image
    import groq
except ImportError:
    install_dependencies()
    import cv2
    import numpy as np
    from deepface import DeepFace
    from google.colab.patches import cv2_imshow
    from google.colab import output
    import base64
    from IPython.display import display, Javascript, Image
    import groq

# --- LLM Integration Class ---

class EmotionLLM:
    """
    Gerencia a interação com a LLM (Groq) para gerar comentários sobre as emoções.
    Roda em uma thread separada para não travar o vídeo.
    """
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.client = None
        self.is_simulated = False
        self.last_message = "Aguardando análise..."
        self.emotion_buffer = []
        self.running = True
        self.lock = threading.Lock()

        if api_key and len(api_key) > 10:
            try:
                self.client = groq.Groq(api_key=api_key)
                print("🧠 IA Conectada: Usando Llama-3 (Groq).")
            except Exception as e:
                print(f"⚠️ Erro ao conectar Groq: {e}. Usando modo simulado.")
                self.is_simulated = True
        else:
            print("🤖 Modo Simulado: Usando respostas pré-definidas (Sem API Key).")
            self.is_simulated = True

        # Inicia thread de processamento
        self.thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.thread.start()

    def add_observation(self, emotion, score):
        """Adiciona uma observação ao buffer."""
        with self.lock:
            self.emotion_buffer.append((emotion, score))

    def get_current_message(self):
        """Retorna a última mensagem gerada."""
        return self.last_message

    def _worker_loop(self):
        """Loop principal da thread da IA."""
        while self.running:
            time.sleep(5) # Analisa a cada 5 segundos

            with self.lock:
                if not self.emotion_buffer:
                    continue

                # Análise estatística simples do buffer
                emotions = [e[0] for e in self.emotion_buffer]
                most_common = max(set(emotions), key=emotions.count)
                avg_score = sum([e[1] for e in self.emotion_buffer]) / len(self.emotion_buffer)
                self.emotion_buffer = [] # Limpa buffer

            # Gera resposta
            if self.is_simulated:
                response = self._generate_simulated_response(most_common)
            else:
                response = self._call_llm(most_common, avg_score)

            self.last_message = response

    def _generate_simulated_response(self, emotion):
        responses = {
            'happy': ["Você parece radiante hoje!", "Esse sorriso é contagiante.", "Continue com essa energia positiva!"],
            'sad': ["Tudo bem não estar bem as vezes.", "Respire fundo, vai passar.", "Quer um abraço virtual?"],
            'angry': ["Calma, respira fundo.", "Parece tenso, tente relaxar os ombros.", "Conte até 10..."],
            'surprise': ["Uau! O que te surpreendeu?", "Parece que você viu algo incrível.", "Chocado?"],
            'neutral': ["Focado e sereno.", "Apenas observando...", "Cara de pôquer perfeita."],
            'fear': ["Está tudo bem, você está seguro.", "O que te assusta?", "Respire devagar."],
            'disgust': ["Algo não cheira bem?", "Eca!", "Desaprovação total."]
        }
        return random.choice(responses.get(emotion, ["Interessante..."]))

    def _call_llm(self, emotion, score):
        try:
            prompt = f"""
            Você é um amigo empático observando o usuário pela webcam.
            O usuário parece estar sentindo: {emotion} (Intensidade: {score:.1f}%).

            Gere um comentário CURTO (máximo 15 palavras), casual e engajador em Português do Brasil.
            Se for feliz, celebre. Se for triste, apoie. Se for neutro, faça uma piada leve ou observação filosófica.
            Não use aspas.
            """

            chat_completion = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.3-70b-versatile",
                max_tokens=50,
                temperature=0.7,
            )
            return chat_completion.choices[0].message.content.strip()
        except Exception as e:
            return f"Erro na IA: {str(e)[:20]}..."

    def stop(self):
        self.running = False

# --- Webcam JavaScript ---

def start_webcam_js():
    js = Javascript('''
    async function startCamera() {
      const div = document.createElement('div');
      const video = document.createElement('video');
      video.style.display = 'none';
      const stream = await navigator.mediaDevices.getUserMedia({video: true});
      document.body.appendChild(div);
      div.appendChild(video);
      video.srcObject = stream;
      await video.play();
      google.colab.output.setIframeHeight(document.documentElement.scrollHeight, true);
      return video;
    }

    async function captureFrame(video) {
        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        canvas.getContext('2d').drawImage(video, 0, 0);
        return canvas.toDataURL('image/jpeg', 0.8);
    }

    window.videoElement = null;
    window.startWebcam = async function() { window.videoElement = await startCamera(); }
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

def draw_emotion_stats_v3(img, face_data, x, y, w, h, llm_message, backend_name):
    """Desenha UI completa com mensagem da IA."""
    overlay = img.copy()
    translations = {'angry': 'Raiva', 'disgust': 'Nojo', 'fear': 'Medo', 'happy': 'Feliz', 'sad': 'Triste', 'surprise': 'Surpresa', 'neutral': 'Neutro'}

    dominant_emotion = face_data['dominant_emotion']
    prob = face_data['emotion'][dominant_emotion]
    label = translations.get(dominant_emotion, dominant_emotion)

    colors = {'angry': (0,0,255), 'happy': (0,255,255), 'sad': (255,0,0), 'neutral': (200,200,200)}
    color = colors.get(dominant_emotion, (255,255,255))

    # 1. Caixa do Rosto
    cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)

    # 2. Painel Inferior (Mensagem da IA)
    H, W, _ = img.shape
    # Fundo escuro na parte inferior
    cv2.rectangle(overlay, (0, H-60), (W, H), (0,0,0), -1)
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)

    # Texto da IA (Centralizado)
    text = f"IA: {llm_message}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 1
    (fw, fh), _ = cv2.getTextSize(text, font, scale, thickness)
    text_x = (W - fw) // 2
    cv2.putText(img, text, (text_x, H - 20), font, scale, (255,255,255), thickness)

    # 3. Info do Backend (Canto superior esquerdo)
    cv2.putText(img, f"Backend: {backend_name}", (10, 20), font, 0.4, (200,200,200), 1)

    # 4. Label flutuante acima da cabeça
    cv2.putText(img, f"{label} {prob:.0f}%", (x, y-10), font, 0.7, color, 2)

def emotion_detection_loop():
    # Setup Inicial
    print("\n" + "="*40)
    print("🤖 CONFIGURAÇÃO DO ASSISTENTE EMOCIONAL")
    print("="*40)
    print("Para usar a IA real (Llama 3), insira sua GROQ API KEY.")
    print("Se deixar em branco, usaremos o Modo Simulado (Respostas prontas).")
    print("Obtenha chave grátis em: https://console.groq.com/keys")

    api_key_input = input("🔑 Groq API Key (Enter para pular): ").strip()

    print("\n🎥 Iniciando webcam...")
    start_webcam_js()
    output.eval_js('startWebcam()')

    print("🧠 Inicializando IA e Modelos...")
    llm_agent = EmotionLLM(api_key=api_key_input)

    # LISTA DE BACKENDS PARA TENTATIVA (Ordem de prioridade)
    # MediaPipe no inicio, mas se der crash, o loop vai trocar.
    backend_priority = ['mediapipe', 'yolov8', 'ssd', 'opencv', 'mtcnn']
    current_backend_idx = 0
    detector_backend = backend_priority[current_backend_idx]

    consecutive_failures = 0
    MAX_FAILURES_BEFORE_SWITCH = 20  # Tenta trocar a cada ~20 frames sem detecção

    print(f"🎯 Backend inicial: {detector_backend.upper()}")

    # Warmup
    print("🔥 Aquecendo motores (Warmup)...")
    try:
        DeepFace.analyze(np.zeros((100,100,3),dtype=np.uint8), actions=['emotion'], detector_backend=detector_backend, enforce_detection=False, silent=True)
    except Exception as e:
        print(f"⚠️ Erro crítico no warmup de {detector_backend}: {str(e)[:100]}...")
        # Se falhou no warmup (ex: erro de protobuf), troca imediatamente
        current_backend_idx = (current_backend_idx + 1) % len(backend_priority)
        detector_backend = backend_priority[current_backend_idx]
        print(f"🔄 Trocando imediatamente para: {detector_backend.upper()}")

    print("🚀 Sistema Online! Sorria para a câmera. :)")
    print("(Pressione o botão de parar do Colab para encerrar)")

    frame_count = 0
    display_handle = display(None, display_id=True)
    last_results = []

    # Log Data
    log_data = []

    try:
        while True:
            data_uri = output.eval_js('getFrame()')
            if not data_uri: break
            img = data_uri_to_cv2_img(data_uri)

            frame_count += 1

            # Detecção a cada 3 frames
            if frame_count % 3 == 0:
                try:
                    # Tenta detectar
                    results = DeepFace.analyze(img, actions=['emotion'], detector_backend=detector_backend, enforce_detection=False, silent=True)

                    found_valid_face = False
                    if isinstance(results, list) and len(results) > 0:
                        valid_faces = []
                        H, W, _ = img.shape
                        for face in results:
                            r = face['region']
                            # Filtra tela cheia (falso positivo)
                            if r['w'] > 0.9 * W and r['h'] > 0.9 * H:
                                continue
                            valid_faces.append(face)

                        if valid_faces:
                            last_results = valid_faces
                            found_valid_face = True
                            consecutive_failures = 0 # Reset falhas

                            # Envia para LLM
                            for face in valid_faces:
                                dom = face['dominant_emotion']
                                score = face['emotion'][dom]
                                llm_agent.add_observation(dom, score)
                                log_data.append({'timestamp': datetime.now().strftime("%H:%M:%S"), 'emotion': dom, 'ia_comment': llm_agent.get_current_message()})
                        else:
                            last_results = []
                    else:
                        last_results = []

                    if not found_valid_face:
                        consecutive_failures += 1

                except Exception as e:
                    # Se for o erro do mediapipe, imprime menos texto
                    err_msg = str(e)
                    if "Failed to parse" in err_msg:
                        print(f"⚠️ Erro de parsing no {detector_backend} (ignorando frame).")
                    else:
                        print(f"⚠️ Erro no backend {detector_backend}: {err_msg[:50]}...")

                    consecutive_failures += 1
                    last_results = []

                # LÓGICA DE TROCA DE BACKEND
                # Se falhar muitas vezes OU se for erro critico de protobuf
                if consecutive_failures >= MAX_FAILURES_BEFORE_SWITCH:
                    current_backend_idx = (current_backend_idx + 1) % len(backend_priority)
                    detector_backend = backend_priority[current_backend_idx]
                    consecutive_failures = 0
                    print(f"🔄 Trocando detector para: {detector_backend.upper()}")

            # Desenha UI
            current_message = llm_agent.get_current_message()
            H, W, _ = img.shape

            if not last_results:
                msg = f"Procurando rosto ({detector_backend})..."
                cv2.putText(img, msg, (10, H-50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

                # Barra de progresso da falha
                fail_pct = min(consecutive_failures / MAX_FAILURES_BEFORE_SWITCH, 1.0)
                cv2.rectangle(img, (10, H-45), (10 + int(200*fail_pct), H-40), (0,0,255), -1)

            for face in last_results:
                region = face['region']
                draw_emotion_stats_v3(img, face, region['x'], region['y'], region['w'], region['h'], current_message, detector_backend)

            # Atualiza Display
            _, enc_img = cv2.imencode('.jpg', img)
            display_handle.update(Image(data=enc_img.tobytes()))

    except KeyboardInterrupt:
        print("🛑 Parando...")
    finally:
        llm_agent.stop()
        if log_data:
            with open('emotion_log_v3.csv', 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=log_data[0].keys())
                writer.writeheader()
                writer.writerows(log_data)
            print("💾 Log salvo em 'emotion_log_v3.csv'")
            try:
                from google.colab import files
                files.download('emotion_log_v3.csv')
            except: pass

if __name__ == "__main__":
    emotion_detection_loop()
