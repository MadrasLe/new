from gtts import gTTS
import tempfile

class TTSPlayer:
    def __init__(self, lang="pt"):
        self.lang = lang

    def save_audio_to_file(self, text):
        """
        Gera o áudio a partir do texto e o salva em um arquivo MP3 temporário.
        Retorna o caminho (path) para o arquivo de áudio gerado.
        """
        try:
            tts = gTTS(text, lang=self.lang, slow=False)
            
            # Cria um arquivo temporário com a extensão .mp3
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as fp:
                temp_filename = fp.name
            
            # Salva o áudio no arquivo temporário
            tts.save(temp_filename)
            
            return temp_filename
        except Exception as e:
            print(f"Erro ao gerar arquivo de áudio TTS: {e}")
            return None