# backend/jade/scholar.py
import os
import json
import re
import random
import logging
from io import BytesIO
from typing import List, Dict, Any, Optional
import numpy as np

# Third-party imports
import groq
import pypdf
import faiss
import graphviz
import genanki
import requests
from bs4 import BeautifulSoup
from youtube_transcript_api import YouTubeTranscriptApi
from sentence_transformers import SentenceTransformer
from fpdf import FPDF
from duckduckgo_search import DDGS
from gtts import gTTS
from pydub import AudioSegment

# Setup logging
logger = logging.getLogger(__name__)

# Constants
GENERATED_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "generated")
if not os.path.exists(GENERATED_DIR):
    os.makedirs(GENERATED_DIR)

class ToolBox:
    """Caixa de ferramentas para os agentes."""
    
    @staticmethod
    def get_file_path(filename: str) -> str:
        return os.path.join(GENERATED_DIR, filename)

    @staticmethod
    def read_pdf(filepath: str) -> str:
        try:
            logger.info(f"📄 [Ferramenta] Lendo PDF: {filepath}...")
            reader = pypdf.PdfReader(filepath)
            text = "".join([p.extract_text() or "" for p in reader.pages])
            return re.sub(r'\s+', ' ', text).strip()
        except Exception as e:
            return f"Erro ao ler PDF: {str(e)}"

    @staticmethod
    def scrape_web(url: str) -> str:
        try:
            logger.info(f"🌐 [Ferramenta] Acessando URL: {url}...")
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            for script in soup(["script", "style", "header", "footer", "nav"]):
                script.extract()
            text = soup.get_text()
            return re.sub(r'\s+', ' ', text).strip()[:40000]
        except Exception as e:
            logger.error(f"Erro ao acessar {url}: {e}")
            return ""

    @staticmethod
    def search_topic(topic: str) -> List[str]:
        """Pesquisa no DuckDuckGo e retorna URLs."""
        logger.info(f"🔎 [Ferramenta] Pesquisando na Web sobre: '{topic}'...")
        urls = []
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(topic, max_results=3))
                for r in results:
                    urls.append(r['href'])
        except Exception as e:
            logger.error(f"Erro na busca: {e}")
        return urls

    @staticmethod
    def get_youtube_transcript(url: str) -> str:
        try:
            logger.info(f"📺 [Ferramenta] Extraindo legendas do YouTube: {url}...")
            video_id = url.split("v=")[-1].split("&")[0]
            transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['pt', 'en'])
            text = " ".join([t['text'] for t in transcript])
            return text
        except Exception as e:
            return f"Erro ao pegar legendas do YouTube: {str(e)}"

    @staticmethod
    def generate_audio_mix(script: List[Dict], filename="aula_podcast.mp3"):
        logger.info("🎙️ [Estúdio] Produzindo áudio imersivo...")
        combined = AudioSegment.silent(duration=500)
        
        for line in script:
            speaker = line.get("speaker", "Narrador").upper()
            text = line.get("text", "")
            
            lang = 'pt'
            tld = 'com.br'
            
            if "BERTA" in speaker or "PROFESSORA" in speaker or "AGENT B" in speaker:
                tld = 'pt' # Portuguese accent
            
            try:
                tts = gTTS(text=text, lang=lang, tld=tld, slow=False)
                fp = BytesIO()
                tts.write_to_fp(fp)
                fp.seek(0)
                
                segment = AudioSegment.from_file(fp, format="mp3")
                combined += segment
                combined += AudioSegment.silent(duration=300)
            except Exception as e: 
                logger.error(f"Error generating audio segment: {e}")

        filepath = ToolBox.get_file_path(filename)
        combined.export(filepath, format="mp3")
        return filepath

    @staticmethod
    def generate_mindmap_image(dot_code: str, filename="mapa_mental"):
        try:
            logger.info("🗺️ [Design] Renderizando Mapa Mental...")
            clean_dot = dot_code.replace("```dot", "").replace("```", "").strip()
            filepath = ToolBox.get_file_path(filename)
            # Graphviz adds extension automatically, so we remove it from filename if present
            # But Source.render expects filename without extension if we want exact control or with extension?
            # actually render(filename=...) saves as filename.format
            
            src = graphviz.Source(clean_dot)
            src.format = 'png'
            output_path = src.render(filename=filepath, view=False, cleanup=True)
            return output_path
        except Exception as e:
            logger.error(f"Erro ao gerar gráfico: {e}")
            return None
            
    @staticmethod
    def generate_anki_deck(qa_pairs: List[Dict], deck_name="ScholarGraph Deck"):
        logger.info("🧠 [Anki] Criando arquivo de Flashcards (.apkg)...")
        try:
            model_id = random.randrange(1 << 30, 1 << 31)
            deck_id = random.randrange(1 << 30, 1 << 31)
            
            my_model = genanki.Model(
              model_id,
              'Simple Model',
              fields=[{'name': 'Question'}, {'name': 'Answer'}],
              templates=[{
                'name': 'Card 1',
                'qfmt': '{{Question}}',
                'afmt': '{{FrontSide}}<hr id="answer">{{Answer}}',
              }]
            )
            
            my_deck = genanki.Deck(deck_id, deck_name)
            
            for item in qa_pairs:
                my_deck.add_note(genanki.Note(
                  model=my_model,
                  fields=[item['question'], item['answer']]
                ))
            
            filename = f"flashcards_{deck_id}.apkg"
            filepath = ToolBox.get_file_path(filename)
            genanki.Package(my_deck).write_to_file(filepath)
            return filepath
        except Exception as e:
            logger.error(f"Erro ao criar Anki deck: {e}")
            return None

class VectorMemory:
    def __init__(self):
        logger.info("🧠 [Memória] Inicializando Banco de Vetores (RAG)...")
        # Modelo leve para embeddings
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = None
        self.chunks = []
    
    def ingest(self, text: str, chunk_size=500):
        words = text.split()
        # Cria chunks sobrepostos para melhor contexto
        self.chunks = [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), int(chunk_size*0.8))]
        
        logger.info(f"🧠 [Memória] Vetorizando {len(self.chunks)} fragmentos...")
        if not self.chunks: return

        embeddings = self.model.encode(self.chunks)
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings).astype('float32'))
        logger.info("🧠 [Memória] Indexação concluída.")

    def retrieve(self, query: str, k=3) -> str:
        if not self.index: return ""
        query_vec = self.model.encode([query])
        D, I = self.index.search(np.array(query_vec).astype('float32'), k)
        
        results = [self.chunks[i] for i in I[0] if i < len(self.chunks)]
        return "\n\n".join(results)

class GraphState:
    def __init__(self):
        self.raw_content: str = ""
        self.summary: str = ""
        self.script: List[Dict] = []
        self.quiz_data: List[Dict] = []
        self.mindmap_path: str = ""
        self.flashcards: List[Dict] = []

class LLMEngine:
    def __init__(self, api_key: str):
        self.client = groq.Groq(api_key=api_key)
        self.model = "llama-3.3-70b-versatile"
    
    def chat(self, messages: List[Dict], json_mode=False) -> str:
        try:
            kwargs = {"messages": messages, "model": self.model, "temperature": 0.6}
            if json_mode: kwargs["response_format"] = {"type": "json_object"}
            return self.client.chat.completions.create(**kwargs).choices[0].message.content
        except Exception as e:
            return f"Erro na IA: {e}"

# --- Agentes ---

class ResearcherAgent:
    def deep_research(self, topic: str) -> str:
        logger.info(f"🕵️ [Pesquisador] Iniciando Deep Research sobre: {topic}")
        urls = ToolBox.search_topic(topic)
        if not urls:
            return f"Não encontrei informações sobre {topic}."
            
        full_text = ""
        for url in urls:
            content = ToolBox.scrape_web(url)
            if content:
                full_text += f"\n\n--- Fonte: {url} ---\n{content[:10000]}"
        
        return full_text

class FlashcardAgent:
    def __init__(self, llm: LLMEngine):
        self.llm = llm

    def create_deck(self, content: str) -> List[Dict]:
        logger.info("🃏 [Flashcard] Gerando pares Pergunta-Resposta...")
        prompt = f"""
        Crie 10 Flashcards (Pergunta e Resposta) sobre o conteúdo para memorização.
        SAÍDA JSON: {{ "cards": [ {{ "question": "...", "answer": "..." }} ] }}
        Conteúdo: {content[:15000]}
        """
        try:
            resp = self.llm.chat([{"role": "user", "content": prompt}], json_mode=True)
            return json.loads(resp).get("cards", [])
        except: return []

class IngestAgent:
    def __init__(self, researcher: ResearcherAgent):
        self.researcher = researcher

    def process(self, user_input: str) -> str:
        # Se for arquivo PDF (assume path local se existir, mas no contexto web pode ser mais complexo)
        # Aqui vamos simplificar: se for URL ou topico.
        # Se o backend salvar arquivos de upload, poderiamos passar o path aqui.
        
        if user_input.lower().endswith(".pdf") and os.path.exists(user_input):
            return ToolBox.read_pdf(user_input)
        elif "youtube.com" in user_input or "youtu.be" in user_input:
            return ToolBox.get_youtube_transcript(user_input)
        elif user_input.startswith("http"):
            return ToolBox.scrape_web(user_input)
        else:
            logger.info("🔍 Entrada detectada como Tópico. Ativando ResearcherAgent...")
            return self.researcher.deep_research(user_input)

class ProfessorAgent:
    def __init__(self, llm: LLMEngine):
        self.llm = llm
    
    def summarize(self, full_text: str) -> str:
        logger.info("🧠 [Professor] Gerando resumo estratégico...")
        prompt = f"""
        Você é um Professor Universitário. Crie um resumo estruturado e profundo.
        Texto: {full_text[:25000]}
        Formato: # Título / ## Introdução / ## Pontos Chave / ## Conclusão
        """
        return self.llm.chat([{"role": "user", "content": prompt}])

class VisualizerAgent:
    def __init__(self, llm: LLMEngine):
        self.llm = llm
        
    def create_mindmap(self, text: str) -> str:
        logger.info("🎨 [Visualizador] Projetando Mapa Mental...")
        prompt = f"""
        Crie um código GRAPHVIZ (DOT) para um mapa mental deste conteúdo.
        Use formas coloridas. NÃO explique, apenas dê o código DOT dentro de ```dot ... ```.
        Texto: {text[:15000]}
        """
        response = self.llm.chat([{"role": "user", "content": prompt}])
        match = re.search(r'```dot(.*?)```', response, re.DOTALL)
        if match: return match.group(1).strip()
        return response

class ScriptwriterAgent:
    def __init__(self, llm: LLMEngine):
        self.llm = llm

    def create_script(self, content: str, mode="lecture") -> List[Dict]:
        if mode == "debate":
            logger.info("🔥 [Roteirista] Criando DEBATE INTENSO...")
            prompt = f"""
            Crie um DEBATE acalorado mas intelectual entre dois agentes (8 falas).
            Personagens: 
            - AGENT A (Gabriel): A favor / Otimista / Pragmático.
            - AGENT B (Berta): Contra / Cética / Filosófica.
            
            SAÍDA JSON: {{ "dialogue": [ {{"speaker": "Agent A", "text": "..."}}, {{"speaker": "Agent B", "text": "..."}} ] }}
            Tema Base: {content[:15000]}
            """
        else:
            logger.info("✍️ [Roteirista] Escrevendo roteiro de aula...")
            prompt = f"""
            Crie um roteiro de podcast (8 falas).
            Personagens: GABRIEL (Aluno BR) e BERTA (Professora PT).
            SAÍDA JSON: {{ "dialogue": [ {{"speaker": "Gabriel", "text": "..."}}, ...] }}
            Base: {content[:15000]}
            """
        
        try:
            resp = self.llm.chat([{"role": "user", "content": prompt}], json_mode=True)
            return json.loads(resp).get("dialogue", [])
        except: return []

class ExaminerAgent:
    def __init__(self, llm: LLMEngine):
        self.llm = llm
        
    def generate_quiz(self, content: str) -> List[Dict]:
        logger.info("📝 [Examinador] Criando Prova Gamificada...")
        prompt = f"""
        Crie 5 perguntas de múltipla escolha (Difíceis).
        SAÍDA JSON: {{ "quiz": [ {{ "question": "...", "options": ["A)..."], "correct_option": "A", "explanation": "..." }} ] }}
        Base: {content[:15000]}
        """
        try:
            resp = self.llm.chat([{"role": "user", "content": prompt}], json_mode=True)
            return json.loads(resp).get("quiz", [])
        except: return []

class PublisherAgent:
    def create_handout(self, state: GraphState, filename="Apostila_Estudos.pdf"):
        logger.info("📚 [Editora] Diagramando Apostila PDF...")
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, "Apostila de Estudos - Scholar Graph", ln=True, align='C')
        pdf.ln(10)
        pdf.set_font("Arial", size=11)
        safe_summary = state.summary.encode('latin-1', 'replace').decode('latin-1')
        pdf.multi_cell(0, 7, safe_summary)
        if state.mindmap_path and os.path.exists(state.mindmap_path):
            pdf.add_page()
            # FPDF expects path to image
            pdf.image(state.mindmap_path, x=10, y=30, w=190)
        
        filepath = ToolBox.get_file_path(filename)
        pdf.output(filepath)
        return filepath

class ScholarAgent:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY is required for ScholarAgent")
            
        self.llm = LLMEngine(self.api_key)
        self.memory = VectorMemory()
        
        self.researcher = ResearcherAgent()
        self.ingestor = IngestAgent(self.researcher)
        
        self.professor = ProfessorAgent(self.llm)
        self.visualizer = VisualizerAgent(self.llm)
        self.scriptwriter = ScriptwriterAgent(self.llm)
        self.examiner = ExaminerAgent(self.llm)
        self.flashcarder = FlashcardAgent(self.llm)
        self.publisher = PublisherAgent()
        
        # In a real multi-user app, state should be managed externally or per-session.
        # For this integration, we'll keep a simple session mapping if needed, 
        # or just pass state around.
        self.sessions: Dict[str, GraphState] = {}

    def get_or_create_state(self, user_id: str) -> GraphState:
        if user_id not in self.sessions:
            self.sessions[user_id] = GraphState()
        return self.sessions[user_id]

    def process_request(self, user_input: str, user_id: str = "default") -> Dict[str, Any]:
        """
        Process user input and return a dictionary with response text and optional attachments.
        Structure:
        {
            "text": "...",
            "attachments": [
                {"type": "image", "url": "...", "title": "..."},
                {"type": "audio", "url": "...", "title": "..."},
                {"type": "file", "url": "...", "title": "..."}
            ]
        }
        """
        state = self.get_or_create_state(user_id)
        
        # Simple command parsing logic
        cmd = user_input.lower().strip()
        
        if not state.raw_content and not cmd.startswith("scholar:"):
            # Assume it's a topic or URL to ingest
            content = self.ingestor.process(user_input)
            if not content or len(content) < 50:
                 return {"text": f"Não consegui encontrar conteúdo suficiente sobre '{user_input}'. Tente ser mais específico ou fornecer uma URL válida."}
            
            state.raw_content = content
            self.memory.ingest(content)
            
            return {
                "text": (
                    f"🎓 Conteúdo sobre '{user_input}' processado com sucesso!\n\n"
                    "**Menu Scholar Graph:**\n"
                    "1. Resumo Estratégico\n"
                    "2. Mapa Mental\n"
                    "3. Podcast (Aula)\n"
                    "4. Debate (Prós/Contras)\n"
                    "5. Quiz Gamificado\n"
                    "6. Flashcards (Anki)\n"
                    "7. Gerar Apostila PDF\n\n"
                    "Digite o número ou o nome da opção."
                )
            }

        # Menu handling
        if "resumo" in cmd or cmd == "1":
            state.summary = self.professor.summarize(state.raw_content)
            return {"text": f"## 📝 Resumo Estratégico\n\n{state.summary}"}
            
        elif "mapa" in cmd or "mental" in cmd or cmd == "2":
            dot = self.visualizer.create_mindmap(state.raw_content)
            path = ToolBox.generate_mindmap_image(dot)
            if path:
                state.mindmap_path = path
                filename = os.path.basename(path)
                return {
                    "text": "Aqui está o mapa mental do conteúdo:",
                    "attachments": [{
                        "type": "image", 
                        "url": f"/generated/{filename}",
                        "title": "Mapa Mental"
                    }]
                }
            return {"text": "Desculpe, não consegui gerar o mapa mental."}
            
        elif "podcast" in cmd or "aula" in cmd or cmd == "3":
            script = self.scriptwriter.create_script(state.raw_content, mode="lecture")
            path = ToolBox.generate_audio_mix(script, filename=f"podcast_{user_id}.mp3")
            filename = os.path.basename(path)
            return {
                "text": "🎙️ Aula preparada! Ouça abaixo:",
                "attachments": [{
                    "type": "audio",
                    "url": f"/generated/{filename}",
                    "title": "Aula Podcast"
                }]
            }
            
        elif "debate" in cmd or cmd == "4":
            script = self.scriptwriter.create_script(state.raw_content, mode="debate")
            path = ToolBox.generate_audio_mix(script, filename=f"debate_{user_id}.mp3")
            filename = os.path.basename(path)
            return {
                "text": "🔥 Debate gerado! Ouça abaixo:",
                "attachments": [{
                    "type": "audio",
                    "url": f"/generated/{filename}",
                    "title": "Debate Intenso"
                }]
            }
            
        elif "quiz" in cmd or cmd == "5":
            quiz = self.examiner.generate_quiz(state.raw_content)
            state.quiz_data = quiz
            text = "## 🎮 Quiz Gamificado\n\n"
            for i, q in enumerate(quiz):
                text += f"**{i+1}. {q['question']}**\n"
                for opt in q['options']:
                    text += f"- {opt}\n"
                text += f"\n*(Resposta correta: {q['correct_option']})*\n\n"
            return {"text": text}
            
        elif "flashcard" in cmd or "anki" in cmd or cmd == "6":
            cards = self.flashcarder.create_deck(state.raw_content)
            if cards:
                path = ToolBox.generate_anki_deck(cards)
                filename = os.path.basename(path)
                return {
                    "text": "✅ Deck do Anki criado com sucesso!",
                    "attachments": [{
                        "type": "file",
                        "url": f"/generated/{filename}",
                        "title": "Flashcards.apkg"
                    }]
                }
            return {"text": "Falha ao criar flashcards."}
            
        elif "apostila" in cmd or "pdf" in cmd or cmd == "7":
            if not state.summary:
                state.summary = self.professor.summarize(state.raw_content)
            
            path = self.publisher.create_handout(state)
            filename = os.path.basename(path)
            return {
                "text": "📚 Apostila completa gerada!",
                "attachments": [{
                    "type": "file",
                    "url": f"/generated/{filename}",
                    "title": "Apostila de Estudos.pdf"
                }]
            }
        
        elif cmd == "sair" or cmd == "8":
            # Reset state?
            self.sessions[user_id] = GraphState()
            return {"text": "Saindo do modo Scholar. Envie um novo tópico para começar."}

        else:
            # New topic? Or unrecognized command
             return {
                "text": (
                    "Não entendi o comando. Se quiser iniciar um novo tópico, apenas digite o tema ou URL.\n"
                    "Se estiver tentando acessar o menu, use os números (1-7)."
                )
            }
