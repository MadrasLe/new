import json
import logging
import os
import sys
import time
import uuid

from groq import Groq

# Importa nossos módulos customizados
from .handlers import ImageHandler
from .tts import TTSPlayer
from .utils import slim_history
from .shorestone import ShoreStoneMemory
from .curator_heuristic import MemoryCuratorHeuristic
from .scholar import ScholarAgent  # Import Scholar Agent

# Configura o logger principal
logging.basicConfig(level=logging.INFO, format="%(asctime)s - JADE - %(levelname)s - %(message)s")

class JadeAgent:
    def __init__(self, config_path="jade/config.json"):
        # Carrega configurações
        # Try to load from absolute path first, then relative
        try:
            with open(config_path) as f:
                self.cfg = json.load(f)
        except FileNotFoundError:
             # Fallback: try to find it relative to this file
            base_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(base_dir, "config.json")
            with open(config_path) as f:
                self.cfg = json.load(f)

        # --- Configuração da API Groq ---
        logging.info("Iniciando J.A.D.E. em modo API (Groq)...")
        self.api_key = self._get_api_key()
        self.client = Groq(api_key=self.api_key)
        self.model_name = self.cfg.get("groq_model", "meta-llama/llama-4-maverick-17b-128e-instruct")

        # System Prompt Base
        self.system_prompt = {"role": "system", "content": "Você é J.A.D.E., uma IA multimodal calma e inteligente. Seja direta. Responda de forma concisa e natural. NÃO explique seu processo de pensamento. Apenas responda à pergunta."}
        
        # --- Inicialização dos Módulos ---
        logging.info("Carregando módulos de percepção e memória...")
        
        # Visão e Fala
        self.image_handler = ImageHandler(self.cfg.get("caption_model", "Salesforce/blip-image-captioning-large"))
        self.tts = TTSPlayer(lang=self.cfg.get("language", "pt"))
        
        # 1. Memória ShoreStone (Persistente)
        self.memory = ShoreStoneMemory()
        # Inicializa com sessão padrão, mas será trocada dinamicamente no respond()
        self.memory.load_or_create_session("sessao_padrao_gabriel")
        
        # 2. Curador Heurístico (Manutenção Automática)
        self.curator = MemoryCuratorHeuristic(shorestone_memory=self.memory)
        self.response_count = 0
        self.maintenance_interval = 10 # Executar a manutenção a cada 10 interações
        
        # 3. Scholar Agent
        try:
            self.scholar = ScholarAgent(api_key=self.api_key)
            logging.info("Scholar Agent inicializado com sucesso.")
        except Exception as e:
            logging.error(f"Erro ao inicializar Scholar Agent: {e}")
            self.scholar = None

        logging.info(f"J.A.D.E. pronta e conectada ao modelo {self.model_name}.")

    def _get_api_key(self):
        """Recupera a chave da API do ambiente de forma segura."""
        key = os.getenv("GROQ_API_KEY")
        if not key:
            logging.error("Chave GROQ_API_KEY não encontrada nas variáveis de ambiente.")
            # For development, try to warn but not crash if possible, but Groq needs it.
            # raise RuntimeError("❌ GROQ_API_KEY não encontrada. Defina a variável de ambiente.")
            print("WARNING: GROQ_API_KEY not found.")
        return key

    def _chat(self, messages):
        """Envia as mensagens para a Groq e retorna a resposta."""
        try:
            chat = self.client.chat.completions.create(
                messages=messages, 
                model=self.model_name,
                temperature=0.7, # Criatividade balanceada
                max_tokens=1024  # Limite de resposta razoável
            )
            return chat.choices[0].message.content.strip()
        except Exception as e:
            logging.error(f"Erro na comunicação com a Groq: {e}")
            return "Desculpe, tive um problema ao me conectar com meu cérebro na nuvem."

    def respond(self, history, user_input, user_id="default", vision_context=None, agent_mode="jade"):
        """Processo principal de raciocínio: Lembrar -> Ver -> Responder -> Memorizar -> Manter."""
        
        # Attachments list to return
        attachments = []
        
        # SCHOLAR AGENT ROUTING
        if agent_mode == "scholar" and self.scholar:
            scholar_response = self.scholar.process_request(user_input, user_id)
            resposta = scholar_response.get("text", "")
            attachments = scholar_response.get("attachments", [])
            
            # Add to history
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": resposta})
            
            # Check for audio attachment to set as primary audio response for autoplay
            audio_path = None
            for att in attachments:
                if att.get("type") == "audio":
                    # Attachments have URL /generated/filename.mp3
                    # We need the full file path for TTSPlayer logic in app.py if we wanted to read bytes
                    # But app.py logic reads file at `audio_path`.
                    # So we need to convert URL back to path or change app.py logic.
                    # App.py expects a file path.
                    
                    # Convert /generated/foo.mp3 -> backend/generated/foo.mp3
                    url = att.get("url", "")
                    filename = os.path.basename(url)
                    # We know where generated dir is relative to this file? 
                    # scholar.py defines generated dir.
                    # It's better if app.py handles the URL if provided.
                    
                    # Hack: app.py expects audio_path to read bytes. 
                    # Let's reconstruct path for now.
                    # Assuming cwd is repo root
                    possible_path = os.path.join("backend", "generated", filename)
                    if os.path.exists(possible_path):
                        audio_path = possible_path
                    else:
                        # Try absolute
                        possible_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "generated", filename)
                        if os.path.exists(possible_path):
                            audio_path = possible_path
            
            # If no audio generated by scholar, maybe TTS the text?
            # Scholar usually generates its own audio for podcasts/debates.
            # If it's just text response, maybe we should TTS it?
            if not audio_path and resposta:
                 try:
                    audio_path = self.tts.save_audio_to_file(resposta)
                 except Exception as e:
                    logging.warning(f"TTS falhou: {e}")

            return resposta, audio_path, history, attachments

        # NORMAL JADE AGENT FLOW
        
        # TROCA A SESSÃO DA MEMÓRIA PARA O USUÁRIO ATUAL
        session_name = f"user_{user_id}"
        self.memory.load_or_create_session(session_name)
        
        messages = history[:]
        
        # 1. Lembrar (Recuperação de Contexto)
        memories = self.memory.remember(user_input)
        if memories:
            memory_context = f"--- MEMÓRIAS RELEVANTES (ShoreStone) ---\n{memories}\n--- FIM DAS MEMÓRIAS ---"
            # Inserimos as memórias como contexto de sistema para guiar a resposta
            messages.append({"role": "system", "content": memory_context})

        # 2. Ver (Contexto Visual)
        if vision_context:
            messages.append({"role": "system", "content": f"Contexto visual da imagem que o usuário enviou: {vision_context}"})

        # Adiciona a pergunta atual ao histórico temporário e ao prompt
        history.append({"role": "user", "content": user_input})
        messages.append({"role": "user", "content": user_input})

        # 3. Responder (Geração)
        resposta = self._chat(messages)
        
        # Atualiza histórico
        history.append({"role": "assistant", "content": resposta})
        history = slim_history(history, keep=self.cfg.get("max_context", 12))
        
        # 4. Memorizar (Armazenamento Persistente)
        self.memory.memorize(user_input, resposta)

        print(f"\n🤖 J.A.D.E.: {resposta}")
        
        # Falar (TTS) - Modified for Backend compatibility
        audio_path = None
        try:
            # Uses the TTSPlayer from tts.py which has save_audio_to_file
            audio_path = self.tts.save_audio_to_file(resposta)
        except Exception as e:
            logging.warning(f"TTS falhou (silenciado): {e}")
            
        # 5. Manter (Ciclo de Curadoria Automática)
        self.response_count += 1
        if self.response_count % self.maintenance_interval == 0:
            logging.info(f"Ciclo de manutenção agendado (interação {self.response_count}). Verificando saúde da memória...")
            try:
                self.curator.run_maintenance_cycle()
            except Exception as e:
                logging.error(f"Erro no Curador de Memória: {e}")
        
        return resposta, audio_path, history, attachments
