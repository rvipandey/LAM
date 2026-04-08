LAM-Core is a sophisticated Agentic AI framework designed to run complex reasoning tasks on consumer-grade hardware. By leveraging the power of Ollama for local inference, it eliminates API costs and data privacy concerns while providing advanced capabilities typically found in enterprise-grade cloud solutions.

This project addresses the "computation problem" by optimizing token usage and memory management, allowing models like Llama3 or Qwen to perform multi-step reasoning, retrieval-augmented generation, and exploratory problem solving without crashing.

✨ Key Features
🤖 1. Large Action Model (LAM) Core
A robust implementation of the ReAct (Reason + Act) pattern. The agent breaks down user queries into distinct Thought ➔ Action ➔ Observation cycles, enabling dynamic tool use and self-correction.

Dynamic Tool Selection: Automatically chooses between Calculator, Search, and Custom Tools.
Auto-Recovery: Built-in retry mechanisms for empty responses or API timeouts.
🔗 2. Retrieval-Augmented Generation (RAG)
Don't let your model hallucinate. LAM-Core integrates seamlessly with Vector Memory.

Embedding Integration: Automatically chunks and embeds documents into a local vector store.
Context Injection: Retrieves relevant semantic chunks to ground the LLM's answers in your specific data.
🌲 3. Advanced Reasoning Strategies
Go beyond simple linear thinking.

Chain-of-Thought (CoT): Guides the model to think step-by-step for logic and math problems.
Tree-of-Thoughts (ToT): Enables exploratory reasoning. The agent explores multiple solution paths, evaluates them, and backtracks if necessary, solving complex decision-making problems.
💾 4. Persistent Vector Memory
A long-term memory layer that allows the agent to "remember" past interactions and stored documents across different sessions, reducing redundant computations.

🚀 Quick Start
Install Ollama

🏗️ Architecture
The system is built around a modular pipeline designed to minimize computational overhead:

<img width="1918" height="747" alt="image" src="https://github.com/user-attachments/assets/e5005997-1468-4948-81cd-3c05e79c6432" />

🤝 Contributing
Contributions are welcome! Areas for improvement:

📄 License
MIT License - feel free to use this for your own local AI projects.
