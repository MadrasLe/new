import os
import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    print(f"Iniciando o servidor Uvicorn em http://0.0.0.0:{port}")
    # Import app from backend.app (module path)
    uvicorn.run("backend.app:app", host="0.0.0.0", port=port, reload=True)
