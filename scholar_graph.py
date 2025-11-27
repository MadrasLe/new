# @title 🎓 Scholar Graph Agent - O Professor IA (Versão GOD MODE)
# Este arquivo implementa um sistema de Agentes em Grafo para ensino autônomo
# com suporte a RAG, YouTube, Deep Research, Flashcards (Anki), Mapas Mentais e Debates.

import os
import sys
import json
import time
import re
import random
import textwrap
from io import BytesIO
from typing import List, Dict, Any, Optional
import numpy as np

# --- 1. Setup e Dependências (Auto-Instalação para Colab) ---
def setup_environment():
    packages = [
        "groq", "pypdf", "gtts", "pydub", "beautifulsoup4", "requests", 
        "fpdf", "youtube_transcript_api", "sentence-transformers", 
        "faiss-cpu", "graphviz", "duckduckgo-search", "genanki"
    ]
    missing = []
    
    # Verifica pacotes Python
    for pkg in packages:
        module_name = pkg.replace("-", "_")
        if pkg == "beautifulsoup4": module_name = "bs4"
        if pkg == "faiss-cpu": module_name = "faiss"
        if pkg == "sentence-transformers": module_name = "sentence_transformers"
        if pkg == "duckduckgo-search": module_name = "duckduckgo_search"
        
        try:
            __import__(module_name)
        except ImportError:
            missing.append(pkg)
    
    if missing:
        print(f"📦 Instalando dependências ausentes: {', '.join(missing)}...")
        os.system(f"pip install -q {' '.join(missing)}")
    
    # Verifica FFmpeg e Graphviz (Sistema)
    print("🔧 Verificando dependências do sistema (FFmpeg, Graphviz)...")
    if not os.path.exists("/usr/bin/ffmpeg") or not os.path.exists("/usr/bin/dot"):
        os.system('apt-get install -q ffmpeg graphviz')
    
    print("✅ Ambiente configurado com sucesso! GOD MODE ATIVADO.")

# Executa setup se rodar direto
try:
    import groq
    import pypdf
    import faiss
    import graphviz
    import genanki
    from gtts import gTTS
    from pydub import AudioSegment
    import requests
    from bs4 import BeautifulSoup
    from youtube_transcript_api import YouTubeTranscriptApi
    from sentence_transformers import SentenceTransformer
    from fpdf import FPDF
    from duckduckgo_search import DDGS
except ImportError:
    setup_environment()
    import groq
    import pypdf
    import faiss
    import graphviz
    import genanki
    from gtts import gTTS
    from pydub import AudioSegment
    import requests
    from bs4 import BeautifulSoup
    from youtube_transcript_api import YouTubeTranscriptApi
    from sentence_transformers import SentenceTransformer
    from fpdf import FPDF
    from duckduckgo_search import DDGS

from IPython.display import Audio, display, clear_output, Image

# --- 2. Configuração Global ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "INSIRA_SUA_CHAVE_AQUI")

# --- 3. Camada de Ferramentas (Tooling Layer) ---

class ToolBox:
    """Caixa de ferramentas para os agentes."""
    
    @staticmethod
    def read_pdf(filepath: str) -> str:
        try:
            print(f"📄 [Ferramenta] Lendo PDF: {filepath}...")
            reader = pypdf.PdfReader(filepath)
            text = "".join([p.extract_text() or "" for p in reader.pages])
            return re.sub(r'\s+', ' ', text).strip()
        except Exception as e:
            return f"Erro ao ler PDF: {str(e)}"

    @staticmethod
    def scrape_web(url: str) -> str:
        try:
            print(f"🌐 [Ferramenta] Acessando URL: {url}...")
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            for script in soup(["script", "style", "header", "footer", "nav"]):
                script.extract()
            text = soup.get_text()
            return re.sub(r'\s+', ' ', text).strip()[:40000]
        except Exception as e:
            print(f"Erro ao acessar {url}: {e}")
            return ""

    @staticmethod
    def search_topic(topic: str) -> List[str]:
        """Pesquisa no DuckDuckGo e retorna URLs."""
        print(f"🔎 [Ferramenta] Pesquisando na Web sobre: '{topic}'...")
        urls = []
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(topic, max_results=3))
                for r in results:
                    urls.append(r['href'])
        except Exception as e:
            print(f"Erro na busca: {e}")
        return urls

    @staticmethod
    def get_youtube_transcript(url: str) -> str:
        try:
            print(f"📺 [Ferramenta] Extraindo legendas do YouTube: {url}...")
            video_id = url.split("v=")[-1].split("&")[0]
            transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['pt', 'en'])
            text = " ".join([t['text'] for t in transcript])
            return text
        except Exception as e:
            return f"Erro ao pegar legendas do YouTube: {str(e)}"

    @staticmethod
    def generate_audio_mix(script: List[Dict], filename="aula_podcast.mp3"):
        print("🎙️ [Estúdio] Produzindo áudio imersivo...")
        combined = AudioSegment.silent(duration=500)
        
        for line in script:
            speaker = line.get("speaker", "Narrador").upper()
            text = line.get("text", "")
            
            if "BERTA" in speaker or "PROFESSORA" in speaker or "AGENT B" in speaker:
                tts = gTTS(text=text, lang='pt', tld='pt', slow=False)
            else:
                # Gabriel / Agent A
                tts = gTTS(text=text, lang='pt', tld='com.br', slow=False)
            
            fp = BytesIO()
            tts.write_to_fp(fp)
            fp.seek(0)
            
            try:
                segment = AudioSegment.from_file(fp, format="mp3")
                combined += segment
                combined += AudioSegment.silent(duration=300)
            except: pass

        combined.export(filename, format="mp3")
        return filename

    @staticmethod
    def generate_mindmap_image(dot_code: str, filename="mapa_mental"):
        try:
            print("🗺️ [Design] Renderizando Mapa Mental...")
            clean_dot = dot_code.replace("```dot", "").replace("```", "").strip()
            src = graphviz.Source(clean_dot)
            src.format = 'png'
            filepath = src.render(filename, view=False)
            return filepath
        except Exception as e:
            print(f"Erro ao gerar gráfico: {e}")
            return None
            
    @staticmethod
    def generate_anki_deck(qa_pairs: List[Dict], deck_name="ScholarGraph Deck"):
        print("🧠 [Anki] Criando arquivo de Flashcards (.apkg)...")
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
            
            filename = "flashcards.apkg"
            genanki.Package(my_deck).write_to_file(filename)
            return filename
        except Exception as e:
            print(f"Erro ao criar Anki deck: {e}")
            return None

# --- 4. Vector Store (RAG) ---

class VectorMemory:
    def __init__(self):
        print("🧠 [Memória] Inicializando Banco de Vetores (RAG)...")
        # Modelo leve para embeddings
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = None
        self.chunks = []
    
    def ingest(self, text: str, chunk_size=500):
        words = text.split()
        # Cria chunks sobrepostos para melhor contexto
        self.chunks = [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), int(chunk_size*0.8))]
        
        print(f"🧠 [Memória] Vetorizando {len(self.chunks)} fragmentos...")
        if not self.chunks: return

        embeddings = self.model.encode(self.chunks)
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings).astype('float32'))
        print("🧠 [Memória] Indexação concluída.")

    def retrieve(self, query: str, k=3) -> str:
        if not self.index: return ""
        query_vec = self.model.encode([query])
        D, I = self.index.search(np.array(query_vec).astype('float32'), k)
        
        results = [self.chunks[i] for i in I[0] if i < len(self.chunks)]
        return "\n\n".join(results)

# --- 5. Estado e LLM ---

class GraphState:
    def __init__(self):
        self.raw_content: str = ""
        self.summary: str = ""
        self.script: List[Dict] = []
        self.quiz_data: List[Dict] = []
        self.mindmap_path: str = ""
        self.flashcards: List[Dict] = []

class LLMEngine:
    def __init__(self):
        api_key = os.environ.get("GROQ_API_KEY", GROQ_API_KEY)
        self.client = groq.Groq(api_key=api_key)
        self.model = "llama-3.3-70b-versatile"
    
    def chat(self, messages: List[Dict], json_mode=False) -> str:
        try:
            kwargs = {"messages": messages, "model": self.model, "temperature": 0.6}
            if json_mode: kwargs["response_format"] = {"type": "json_object"}
            return self.client.chat.completions.create(**kwargs).choices[0].message.content
        except Exception as e:
            return f"Erro na IA: {e}"

# --- 6. Agentes Avançados (GOD MODE) ---

class ResearcherAgent:
    """Agente que pesquisa na web se o input for um tópico."""
    def deep_research(self, topic: str) -> str:
        print(f"🕵️ [Pesquisador] Iniciando Deep Research sobre: {topic}")
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
    """Agente focado em memorização (Anki)."""
    def __init__(self, llm: LLMEngine):
        self.llm = llm

    def create_deck(self, content: str) -> List[Dict]:
        print("🃏 [Flashcard] Gerando pares Pergunta-Resposta...")
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
        # Se for arquivo
        if user_input.lower().endswith(".pdf") and os.path.exists(user_input):
            return ToolBox.read_pdf(user_input)
        # Se for URL
        elif "youtube.com" in user_input or "youtu.be" in user_input:
            return ToolBox.get_youtube_transcript(user_input)
        elif user_input.startswith("http"):
            return ToolBox.scrape_web(user_input)
        # Se não for URL nem arquivo, assume que é Tópico para Pesquisa
        else:
            print("🔍 Entrada detectada como Tópico. Ativando ResearcherAgent...")
            return self.researcher.deep_research(user_input)

class ProfessorAgent:
    def __init__(self, llm: LLMEngine):
        self.llm = llm
    
    def summarize(self, full_text: str) -> str:
        print("🧠 [Professor] Gerando resumo estratégico...")
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
        print("🎨 [Visualizador] Projetando Mapa Mental...")
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
            print("🔥 [Roteirista] Criando DEBATE INTENSO...")
            prompt = f"""
            Crie um DEBATE acalorado mas intelectual entre dois agentes (8 falas).
            Personagens: 
            - AGENT A (Gabriel): A favor / Otimista / Pragmático.
            - AGENT B (Berta): Contra / Cética / Filosófica.
            
            SAÍDA JSON: {{ "dialogue": [ {{"speaker": "Agent A", "text": "..."}}, {{"speaker": "Agent B", "text": "..."}} ] }}
            Tema Base: {content[:15000]}
            """
        else:
            print("✍️ [Roteirista] Escrevendo roteiro de aula...")
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
        print("📝 [Examinador] Criando Prova Gamificada...")
        prompt = f"""
        Crie 5 perguntas de múltipla escolha (Difíceis).
        SAÍDA JSON: {{ "quiz": [ {{ "question": "...", "options": ["A)..."], "correct_option": "A", "explanation": "..." }} ] }}
        Base: {content[:15000]}
        """
        try:
            resp = self.llm.chat([{"role": "user", "content": prompt}], json_mode=True)
            return json.loads(resp).get("quiz", [])
        except: return []

    def run_session(self, quiz_data: List[Dict]):
        xp = 0
        print("\n🎮 MODO GAMIFICADO: Responda para ganhar XP!")
        for i, q in enumerate(quiz_data):
            print(f"\n🔹 QUESTÃO {i+1} (Valendo 100 XP): {q['question']}")
            for opt in q['options']: print(opt)
            ans = input("👉 Sua resposta: ").strip().upper()
            if ans and ans[0] == q['correct_option'][0]:
                print(f"✨ ACERTOU! +100 XP.")
                xp += 100
            else:
                print(f"💀 Errou... Era {q['correct_option']}.")
        print(f"\n🏆 TOTAL DE XP: {xp}")

class PublisherAgent:
    def create_handout(self, state: GraphState, filename="Apostila_Estudos.pdf"):
        print("📚 [Editora] Diagramando Apostila PDF...")
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
            pdf.image(state.mindmap_path, x=10, y=30, w=190)
        pdf.output(filename)
        return filename

# --- 7. Orquestrador Principal ---

class ScholarGraphGodMode:
    def __init__(self):
        self.state = GraphState()
        self.memory = VectorMemory()
        
        try:
            self.llm = LLMEngine()
        except: self.llm = None
        
        self.researcher = ResearcherAgent()
        self.ingestor = IngestAgent(self.researcher)
        
        if self.llm:
            self.professor = ProfessorAgent(self.llm)
            self.visualizer = VisualizerAgent(self.llm)
            self.scriptwriter = ScriptwriterAgent(self.llm)
            self.examiner = ExaminerAgent(self.llm)
            self.flashcarder = FlashcardAgent(self.llm)
            self.publisher = PublisherAgent()

    def run(self):
        print("\n🌌 SCHOLAR GRAPH - GOD MODE ATIVADO")
        print("Deep Research • RAG • Flashcards • Debates • Mapas Mentais")
        
        if "GROQ_API_KEY" not in os.environ and GROQ_API_KEY == "INSIRA_SUA_CHAVE_AQUI":
            key = input("🔑 Cole sua GROQ_API_KEY: ").strip()
            if key: 
                os.environ["GROQ_API_KEY"] = key
                self.llm = LLMEngine()
                # Recarregar agentes
                self.professor = ProfessorAgent(self.llm)
                self.visualizer = VisualizerAgent(self.llm)
                self.scriptwriter = ScriptwriterAgent(self.llm)
                self.examiner = ExaminerAgent(self.llm)
                self.flashcarder = FlashcardAgent(self.llm)
            else: return

        # 1. Input Inteligente
        print("\n💡 DICA: Você pode colar um Link, um PDF ou apenas um TÓPICO (ex: 'Revolução Francesa')")
        target = input("📂 Entrada: ").strip()
        if not target: return

        # Processamento (Ingestão ou Pesquisa)
        content = self.ingestor.process(target)
        if not content or len(content) < 50:
            print("❌ Falha ao obter conteúdo suficiente.")
            return
            
        self.state.raw_content = content
        self.memory.ingest(content) # RAG Ingest
        
        while True:
            print("\n" + "━"*30)
            print(" 🎓 MENU GOD MODE")
            print("━"*30)
            print("1. 🧠 Resumo Estratégico")
            print("2. 🗺️ Mapa Mental Visual")
            print("3. 🎧 Podcast (Aula Didática)")
            print("4. 🔥 DEBATE IA (Visões Opostas)")
            print("5. 🎮 Quiz Gamificado")
            print("6. 🃏 Gerar Flashcards (Anki .apkg)")
            print("7. 📚 Baixar Apostila Completa")
            print("8. 🚪 Sair")
            
            opt = input("\n👉 Escolha: ").strip()
            
            if opt == "1":
                self.state.summary = self.professor.summarize(self.state.raw_content)
                print("\n📝 RESUMO:\n" + self.state.summary)
                input("⏎")
            elif opt == "2":
                dot = self.visualizer.create_mindmap(self.state.raw_content)
                path = ToolBox.generate_mindmap_image(dot)
                if path:
                    self.state.mindmap_path = path
                    try: display(Image(path))
                    except: pass
                input("⏎")
            elif opt == "3":
                script = self.scriptwriter.create_script(self.state.raw_content, mode="lecture")
                f = ToolBox.generate_audio_mix(script)
                try: display(Audio(f, autoplay=True))
                except: print(f"Salvo: {f}")
            elif opt == "4":
                script = self.scriptwriter.create_script(self.state.raw_content, mode="debate")
                f = ToolBox.generate_audio_mix(script, filename="debate.mp3")
                try: display(Audio(f, autoplay=True))
                except: print(f"Salvo: {f}")
            elif opt == "5":
                quiz = self.examiner.generate_quiz(self.state.raw_content)
                if quiz: self.examiner.run_session(quiz)
                input("⏎")
            elif opt == "6":
                cards = self.flashcarder.create_deck(self.state.raw_content)
                if cards:
                    f = ToolBox.generate_anki_deck(cards)
                    print(f"✅ Deck gerado: {f} (Baixe e importe no Anki)")
                input("⏎")
            elif opt == "7":
                if self.state.summary:
                    f = self.publisher.create_handout(self.state)
                    print(f"📚 Apostila salva: {f}")
                else: print("⚠️ Gere o Resumo primeiro!")
                input("⏎")
            elif opt == "8": break

if __name__ == "__main__":
    app = ScholarGraphGodMode()
    app.run()
