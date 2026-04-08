
from typing import Optional
from dataclasses import dataclass, field
import requests
@dataclass
class OllamaConfig:
    """Configuration for Ollama LLM endpoint."""
    base_url: str = "http://127.0.0.1:11434"
    model: str = "qwen3:4b"
    temperature: float = 0.1
    max_tokens: int = 512
    timeout: int = 600


class OllamaClient:
    """
    LLM Client that connects to Ollama endpoint.
    Prerequisites:
        1. Install Ollama: https://ollama.ai
        2. Run: ollama serve
        3. Pull model: ollama pull llama2
    """
    
    def __init__(self, config: Optional[OllamaConfig] = None):
        self.config = config or OllamaConfig()
        self._session = None
        
    def _get_session(self):
        if self._session is None:
            try:
                self._session = requests.Session()
            except ImportError:
                raise ImportError("Please install requests: pip install requests")
        return self._session
    
    def generate(self, prompt: str, system: Optional[str] = None) -> str:
        """Generate a response from the Ollama model."""
        session = self._get_session()
        url = f"{self.config.base_url}/api/generate"
        
        payload = {
            "model": self.config.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
            }
        }
        
        if system:
            payload["system"] = system
        
        try:
            response = session.post(url, json=payload, timeout=self.config.timeout)
            response.raise_for_status()
            return response.json().get("response", "").strip()
        except Exception as e:
            return f"[Error: {str(e)}]"
        


    def check_health(self) -> bool:
        try:
            session = self._get_session()
            response = session.get(f"{self.config.base_url}/api/tags", timeout=5)
            print("OLLAMA is Running")
            # Add this print to see what's happening
            if response.status_code != 200:
                print(f"Server responded with: {response.status_code}")
            return response.status_code == 200
        except Exception as e:
            print(f"Connection Error: {e}") # This will tell you the TRUE reason
            return False
