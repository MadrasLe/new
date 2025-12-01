import os
import sys
import subprocess
import traceback
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from pyngrok import ngrok

# Initialize Flask App
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# --- Agent / LLM Integration ---
# Try to import groq if available, otherwise use mock
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    print("Groq library not found. Running in mock mode.")

def get_agent_response(message, code_context):
    """
    Generates a response using Groq or a mock agent.
    """
    system_prompt = (
        "You are an expert Python developer and data scientist assistant. "
        "You are running inside a Google Colab environment via a custom browser IDE. "
        "Your goal is to help the user write, debug, and understand Python code. "
        "Always provide the full corrected or new code in a markdown block, e.g., ```python ... ```. "
        "If the user asks to run something, remind them to click the 'Run' button. "
        "Be concise and helpful. Speak Portuguese as requested by the user."
    )

    user_content = f"User Request: {message}\n\nCurrent Code:\n```python\n{code_context}\n```"

    if GROQ_AVAILABLE and os.environ.get("GROQ_API_KEY"):
        try:
            client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                model="llama3-70b-8192", # Using a capable model
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            return f"Error communicating with Groq: {str(e)}"
    else:
        # Mock Response
        return (
            f"**Simulated Agent Response** (Groq API Key missing or library not installed)\n\n"
            f"I received your request: *{message}*\n\n"
            f"Here is a template code you might need:\n"
            f"```python\n"
            f"# Auto-generated code based on: {message}\n"
            f"import os\n"
            f"print('Hello from the Colab Agent!')\n"
            f"print(f'Current working directory: {{os.getcwd()}}')\n"
            f"```"
        )

# --- Routes ---

@app.route('/')
def home():
    return "Colab IDE Backend is Running! Connect via the Frontend."

@app.route('/chat', methods=['POST'])
def chat_endpoint():
    try:
        data = request.json
        message = data.get('message', '')
        code = data.get('current_file_content', '')

        response_text = get_agent_response(message, code)
        return jsonify({'response': response_text})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'response': f"Internal Server Error: {str(e)}"}), 500

@app.route('/run', methods=['POST'])
def run_endpoint():
    """
    Executes the provided Python code and returns stdout/stderr.
    """
    data = request.json
    code = data.get('code', '')

    if not code:
        return jsonify({'stdout': '', 'stderr': 'No code provided', 'returncode': 1})

    filename = 'temp_script.py'

    try:
        # Save code to file
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(code)

        # Execute code
        # We run it in a subprocess.
        # Note: In Colab, we might want to run it in the same process to share state
        # if we were using a more advanced kernel manager (like jupyter_client),
        # but for this simple "tunnel" logic, subprocess is safer and easier to manage.
        result = subprocess.run(
            [sys.executable, filename],
            capture_output=True,
            text=True,
            timeout=60 # 60 seconds timeout
        )

        return jsonify({
            'stdout': result.stdout,
            'stderr': result.stderr,
            'returncode': result.returncode
        })
    except subprocess.TimeoutExpired:
        return jsonify({
            'stdout': '',
            'stderr': 'Execution timed out (limit: 60s)',
            'returncode': -1
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'stdout': '',
            'stderr': f"Execution Error: {str(e)}",
            'returncode': 1
        })

def start_ngrok():
    """
    Starts ngrok tunnel and prints the public URL.
    """
    # Check if we are running in Colab to auto-authenticate if needed
    # (Users usually authenticate manually or via config)

    try:
        # Kill previous tunnels
        ngrok.kill()

        # Connect to port 5000
        tunnel = ngrok.connect(5000)
        public_url = tunnel.public_url
        print(f" * Public URL: {public_url}")
        print(" * Copy this URL to your Browser IDE.")
        return public_url
    except Exception as e:
        print(f" * Error starting ngrok: {e}")
        return None

if __name__ == '__main__':
    print(" * Starting Colab Backend...")

    # Attempt to open tunnel if in a colab-like environment or requested
    # We just always try to open it for convenience in this context
    public_url = start_ngrok()

    app.run(port=5000)
