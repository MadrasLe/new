# app.py - VERSÃO COMPLETA COM VOZ (BASE64) E VISÃO
import os
import base64
import io
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel 
from PIL import Image
from jade.core import JadeAgent 

print("Iniciando a J.A.D.E. com FastAPI...")
agent = JadeAgent() 
print("J.A.D.E. pronta para receber requisições.")

app = FastAPI(title="J.A.D.E. API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# Ensure generated directory exists
GENERATED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generated")
if not os.path.exists(GENERATED_DIR):
    os.makedirs(GENERATED_DIR)

# Mount static files
app.mount("/generated", StaticFiles(directory=GENERATED_DIR), name="generated")

# Dicionário global para armazenar sessões de usuários
user_sessions = {}

class UserRequest(BaseModel):
    user_input: str
    image_base64: str | None = None
    user_id: str | None = None
    agent_mode: str | None = "jade"  # "jade" or "scholar"

@app.post("/chat")
def handle_chat(request: UserRequest):
    try:
        # Gerenciamento de sessão
        user_id = request.user_id if request.user_id else "default_user"
        if user_id not in user_sessions:
            print(f"Nova sessão criada para: {user_id}")
            # Inicializa com o prompt do sistema do agente
            user_sessions[user_id] = [agent.system_prompt]
        
        current_history = user_sessions[user_id]

        vision_context = None
        if request.image_base64:
            try:
                header, encoded_data = request.image_base64.split(",", 1)
                image_bytes = base64.b64decode(encoded_data)
                pil_image = Image.open(io.BytesIO(image_bytes))
                vision_context = agent.image_handler.process_pil_image(pil_image)
            except Exception as img_e:
                print(f"Erro ao processar imagem Base64: {img_e}")
                vision_context = "Houve um erro ao analisar a imagem."

        final_user_input = request.user_input if request.user_input else "Descreva a imagem em detalhes."

        # Pass agent_mode to respond
        bot_response_text, audio_path, updated_history, attachments = agent.respond(
            history=current_history,
            user_input=final_user_input,
            user_id=user_id,
            vision_context=vision_context,
            agent_mode=request.agent_mode
        )
        
        # Atualiza o histórico da sessão
        user_sessions[user_id] = updated_history

        # LÓGICA DO ÁUDIO: Converte o arquivo MP3 gerado para Base64
        # Note: If Scholar returns an audio attachment URL, we might still want to return audio_base64 for auto-play?
        # Or let frontend handle attachments differently.
        # JadeAgent.respond returns audio_path for TTS. Scholar might return it via attachments.
        
        audio_base64 = None
        if audio_path and os.path.exists(audio_path):
            print(f"Codificando arquivo de áudio: {audio_path}")
            with open(audio_path, "rb") as audio_file:
                audio_bytes = audio_file.read()
                audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
            # Only remove if it's a temp file. Scholar generated files might persist.
            # Jade TTS creates temp files. Scholar creates persistent files in /generated.
            if "/generated/" not in audio_path:
                os.remove(audio_path) 
        
        return {
            "success": True,
            "bot_response": bot_response_text,
            "audio_base64": audio_base64, # Envia o áudio como texto para o front-end
            "attachments": attachments
        }
    except Exception as e:
        print(f"Erro crítico no endpoint /chat: {e}")
        return {"success": False, "error": str(e)}

@app.get("/")
def root():
    return {"message": "Servidor J.A.D.E. com FastAPI está online."}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    print(f"Iniciando o servidor Uvicorn em http://0.0.0.0:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
