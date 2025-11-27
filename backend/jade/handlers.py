from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import torch

class TextHandler:
    def process(self):
        return input("⌨️ Digite sua mensagem: ").strip()

class AudioHandler:
    def __init__(self, client, audio_model):
        self.client = client
        self.audio_model = audio_model
        
class ImageHandler:
    def __init__(self, model_name):
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()

    def process_pil_image(self, pil_image: Image.Image):
        """Processa um objeto PIL.Image vindo diretamente do Gradio."""
        if not isinstance(pil_image, Image.Image):
            raise TypeError("A entrada deve ser um objeto PIL.Image.")
        return self._generate_caption(pil_image.convert("RGB"))

    def _generate_caption(self, img):
        """Lógica de geração de legenda reutilizável usando Florence-2."""
        # Prompt para descrição detalhada
        prompt = "<MORE_DETAILED_CAPTION>"
        
        with torch.no_grad():
            inputs = self.processor(text=prompt, images=img, return_tensors="pt").to(self.device)
            
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                do_sample=False,
                num_beams=3,
            )
            
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            
            # O Florence-2 requer pós-processamento para extrair a resposta limpa
            parsed_answer = self.processor.post_process_generation(
                generated_text, 
                task=prompt, 
                image_size=(img.width, img.height)
            )
            
            # parsed_answer retorna um dict, ex: {'<MORE_DETAILED_CAPTION>': 'texto da legenda'}
            return parsed_answer.get(prompt, "")
