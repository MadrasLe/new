
import unittest
import os
import sys
import shutil
from unittest.mock import MagicMock, patch

# Mock dependencies that might be heavy or require API keys
sys.modules['groq'] = MagicMock()
sys.modules['sentence_transformers'] = MagicMock()
sys.modules['faiss'] = MagicMock()
sys.modules['pypdf'] = MagicMock()
sys.modules['genanki'] = MagicMock()
sys.modules['youtube_transcript_api'] = MagicMock()
sys.modules['gtts'] = MagicMock()
sys.modules['pydub'] = MagicMock()
sys.modules['graphviz'] = MagicMock()
sys.modules['duckduckgo_search'] = MagicMock()

# Import after mocking
from backend.jade.scholar import ScholarAgent, ToolBox, GraphState

class TestScholarAgent(unittest.TestCase):
    def setUp(self):
        self.mock_api_key = "test_key"
        with patch.dict(os.environ, {"GROQ_API_KEY": self.mock_api_key}):
            self.agent = ScholarAgent(api_key=self.mock_api_key)
        
        # Ensure generated dir exists
        self.generated_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "generated")
        if not os.path.exists(self.generated_dir):
            os.makedirs(self.generated_dir)

    def test_initialization(self):
        self.assertIsNotNone(self.agent)
        self.assertIsNotNone(self.agent.llm)

    def test_get_or_create_state(self):
        state = self.agent.get_or_create_state("user1")
        self.assertIsInstance(state, GraphState)
        
        state2 = self.agent.get_or_create_state("user1")
        self.assertEqual(state, state2)
        
        state3 = self.agent.get_or_create_state("user2")
        self.assertNotEqual(state, state3)

    @patch.object(ToolBox, 'search_topic')
    @patch.object(ToolBox, 'scrape_web')
    def test_process_request_new_topic(self, mock_scrape, mock_search):
        mock_search.return_value = ["http://example.com"]
        mock_scrape.return_value = "Content about topic"
        
        response = self.agent.process_request("Physics", "user1")
        
        self.assertIn("text", response)
        self.assertIn("Conteúdo sobre 'Physics' processado", response["text"])
        self.assertEqual(self.agent.sessions["user1"].raw_content, "\n\n--- Fonte: http://example.com ---\nContent about topic")

    def test_process_request_menu_command(self):
        # Setup state
        state = self.agent.get_or_create_state("user1")
        state.raw_content = "Some content"
        
        # Mock professor summarize
        self.agent.professor.summarize = MagicMock(return_value="Summary of content")
        
        response = self.agent.process_request("1", "user1")
        
        self.assertIn("text", response)
        self.assertIn("Resumo Estratégico", response["text"])
        self.assertIn("Summary of content", response["text"])
        self.assertEqual(state.summary, "Summary of content")

    def test_process_request_unknown_command(self):
        # Set state to simulate that we have content, so it should treat input as command
        state = self.agent.get_or_create_state("user1")
        state.raw_content = "Some content"
        
        response = self.agent.process_request("unknown command", "user1")
        self.assertIn("text", response)
        self.assertIn("Não entendi o comando", response["text"])
        
    def tearDown(self):
        pass

if __name__ == '__main__':
    unittest.main()
