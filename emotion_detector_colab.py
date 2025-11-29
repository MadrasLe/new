# emotion_detector_colab.py

"""
Emotion Detector for Google Colab v2.2 (Data Logging)
=====================================================

Este script permite detecção de emoções em tempo real usando a webcam no Google Colab.
Versão 2.2: Adiciona salvamento automático de dados (CSV) ao final da sessão.

Instruções de uso:
1. Abra um notebook no Google Colab (https://colab.research.google.com/).
2. Crie uma célula de código.
3. Copie e cole todo o conteúdo deste arquivo na célula.
4. Execute a célula.
5. Permita o acesso à câmera quando solicitado pelo navegador.
6. Ao parar a execução (botão Stop), os dados serão salvos em 'emotion_log.csv'.
"""

import sys
import subprocess
import time
import csv
from datetime import datetime

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
      video.style.display = 'none'; // Oculta o vídeo raw para mostrar apenas o processado
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

def draw_transparent_overlay(img, alpha=0.6):
    overlay = img.copy()
    return overlay

def draw_emotion_stats(img, face_data, x, y, w, h):
    """
    Desenha estatísticas de emoção, barra de confiança e gráfico ao lado do rosto.
    """

    # Traduções
    translations = {
        'angry': 'Raiva', 'disgust': 'Nojo', 'fear': 'Medo',
        'happy': 'Feliz', 'sad': 'Triste', 'surprise': 'Surpresa',
        'neutral': 'Neutro'
    }

    # 1. Dados Básicos
    emotions = face_data['emotion']
    dominant_emotion = face_data['dominant_emotion']
    dominant_prob = emotions[dominant_emotion]
    label = translations.get(dominant_emotion, dominant_emotion)

    # Cores (BGR)
    colors = {
        'angry': (0, 0, 255), 'disgust': (0, 255, 0), 'fear': (255, 0, 255),
        'happy': (0, 255, 255), 'sad': (255, 0, 0), 'surprise': (0, 165, 255),
        'neutral': (200, 200, 200)
    }
    main_color = colors.get(dominant_emotion, (255, 255, 255))

    # Configuração do Overlay
    overlay = img.copy()

    # --- A. Caixa do Rosto ---
    cv2.rectangle(overlay, (x, y), (x+w, y+h), main_color, -1) # Preenchimento suave
    cv2.addWeighted(overlay, 0.2, img, 0.8, 0, img) # Aplica transparência
    cv2.rectangle(img, (x, y), (x+w, y+h), main_color, 2) # Borda sólida

    # --- B. Painel Lateral (Gráfico) ---
    # Desenha ao lado direito do rosto
    panel_x = x + w + 10
    panel_y = y
    panel_w = 160
    panel_h = 150 # Altura fixa para caber todas as emoções

    # Limitações de borda da tela
    height, width, _ = img.shape
    if panel_x + panel_w > width:
        panel_x = x - panel_w - 10 # Desenha na esquerda se não couber na direita

    if panel_x < 0: panel_x = 0 # Fallback extremo

    # Fundo do painel transparente
    overlay_panel = img.copy()
    cv2.rectangle(overlay_panel, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (50, 50, 50), -1)
    cv2.addWeighted(overlay_panel, 0.7, img, 0.3, 0, img)

    # Título do Painel: Emoção Dominante
    text = f"{label}: {dominant_prob:.1f}%"
    cv2.putText(img, text, (panel_x + 5, panel_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Barra de Confiança (Principal)
    bar_width = int((panel_w - 10) * (dominant_prob / 100))
    cv2.rectangle(img, (panel_x + 5, panel_y + 25), (panel_x + 5 + bar_width, panel_y + 35), main_color, -1)

    # Gráfico de Probabilidade (Todas as emoções)
    # Lista ordenada por probabilidade decrescente
    sorted_emotions = sorted(emotions.items(), key=lambda item: item[1], reverse=True)

    y_offset = panel_y + 55
    for emotion_key, score in sorted_emotions[:5]: # Top 5 emoções
        # Nome curto
        name = translations.get(emotion_key, emotion_key)[:3].upper()

        # Barra
        bar_len = int((panel_w - 40) * (score / 100))
        ecolor = colors.get(emotion_key, (200, 200, 200))

        # Texto da Label
        cv2.putText(img, name, (panel_x + 5, y_offset + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        # Barra
        cv2.rectangle(img, (panel_x + 35, y_offset), (panel_x + 35 + bar_len, y_offset + 8), ecolor, -1)

        y_offset += 18

def draw_fps(img, fps):
    overlay = img.copy()
    cv2.rectangle(overlay, (10, 10), (120, 50), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    cv2.putText(img, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

def emotion_detection_loop():
    print("🎥 Iniciando webcam...")
    start_webcam_js()
    output.eval_js('startWebcam()')

    print("🧠 Carregando modelo de emoções (DeepFace) e YOLOv8... Isso pode demorar na primeira vez.")

    detector_backend = 'opencv'
    try:
        import ultralytics
        detector_backend = 'yolov8'
        print("✅ Usando YOLOv8 para detecção facial (Maior precisão).")
    except ImportError:
        print("⚠️ YOLOv8 não encontrado ou falhou. Usando OpenCV para detecção (Modo Rápido).")

    # Warmup
    try:
        DeepFace.analyze(np.zeros((100, 100, 3), dtype=np.uint8),
                        actions=['emotion'],
                        detector_backend=detector_backend,
                        enforce_detection=False,
                        silent=True)
    except Exception as e:
        print(f"Nota: Inicialização falhou levemente ({e}), mas prosseguindo...")
        pass

    print("🚀 Detectando emoções v2.2! (Pressione o botão de parar no Colab para encerrar e salvar os dados)")

    frame_count = 0
    fps_start_time = time.time()
    fps_frame_counter = 0 # To count frames within the second
    fps = 0

    # Dados para salvamento
    log_data = []
    log_filename = "emotion_log.csv"

    # Armazena últimos resultados para desenhar nos frames intermediários (otimização)
    last_results = []

    # Display Handle para atualização eficiente
    display_handle = display(None, display_id=True)

    try:
        while True:
            data_uri = output.eval_js('getFrame()')
            if not data_uri:
                break

            img = data_uri_to_cv2_img(data_uri)
            height, width, _ = img.shape

            # FPS Calculation
            frame_count += 1
            fps_frame_counter += 1
            elapsed = time.time() - fps_start_time
            if elapsed > 1.0:
                fps = fps_frame_counter / elapsed
                fps_frame_counter = 0
                fps_start_time = time.time()

            # Detecção (A cada 5 frames para maior estabilidade)
            if frame_count % 5 == 0:
                try:
                    # Tenta detecção com o backend escolhido
                    try:
                        results = DeepFace.analyze(img,
                                                actions=['emotion'],
                                                detector_backend=detector_backend,
                                                enforce_detection=False,
                                                silent=True)
                    except Exception as e_det:
                        # Fallback se YOLOv8 falhar
                        if detector_backend == 'yolov8':
                             results = DeepFace.analyze(img,
                                                actions=['emotion'],
                                                detector_backend='opencv',
                                                enforce_detection=False,
                                                silent=True)
                        else:
                            raise e_det

                    if results and isinstance(results, list):
                        last_results = results

                        # LOG DATA
                        for face in results:
                            try:
                                entry = {
                                    'timestamp': time.time(),
                                    'datetime': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    'dominant_emotion': face['dominant_emotion'],
                                    'score': face['emotion'][face['dominant_emotion']],
                                    'box_x': face['region']['x'],
                                    'box_y': face['region']['y'],
                                    'box_w': face['region']['w'],
                                    'box_h': face['region']['h']
                                }
                                log_data.append(entry)
                            except Exception as log_err:
                                pass # Ignora erro de log individual
                    else:
                        last_results = []

                except Exception as e:
                    last_results = [] # Nenhuma face detectada ou erro

            # Desenha resultados (Overlay UI)
            if not last_results and frame_count > 10:
                # Feedback visual se não encontrar rosto após warmup
                cv2.putText(img, "Procurando rosto...", (10, height - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            for face in last_results:
                if 'region' in face and 'dominant_emotion' in face:
                    region = face['region']
                    x, y, w, h = region['x'], region['y'], region['w'], region['h']

                    # Desenha estatísticas avançadas (Gráfico, Barra, Transparência)
                    draw_emotion_stats(img, face, x, y, w, h)

            # Desenha FPS
            draw_fps(img, fps)

            # Atualiza o display SEM limpar a saída (preserva JS e Video Element)
            # Encode para JPEG
            _, enc_img = cv2.imencode('.jpg', img)
            display_handle.update(Image(data=enc_img.tobytes()))

    except KeyboardInterrupt:
        print("🛑 Interrompido pelo usuário.")
    except Exception as e:
        print(f"Erro Fatal: {e}")
    finally:
        # Salva os dados ao encerrar
        print("\n💾 Salvando dados coletados...")
        if log_data:
            try:
                keys = log_data[0].keys()
                with open(log_filename, 'w', newline='') as output_file:
                    dict_writer = csv.DictWriter(output_file, fieldnames=keys)
                    dict_writer.writeheader()
                    dict_writer.writerows(log_data)
                print(f"✅ Dados salvos com sucesso em: {log_filename} ({len(log_data)} registros)")

                # Tenta download automático
                try:
                    from google.colab import files
                    print("📥 Tentando iniciar download automático...")
                    files.download(log_filename)
                except Exception as e_dl:
                    print(f"⚠️ Não foi possível iniciar download automático (Erro: {e_dl}).")
                    print(f"👉 Você pode baixar o arquivo '{log_filename}' manualmente na aba 'Arquivos' do Colab.")
            except Exception as e_save:
                print(f"❌ Erro ao salvar arquivo CSV: {e_save}")
        else:
            print("📭 Nenhum rosto foi detectado para salvar.")

if __name__ == "__main__":
    emotion_detection_loop()
