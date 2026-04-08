import time
from typing import Dict, Callable

class SimpleLAM:
    """
    A robust Large Action Model using the ReAct Pattern.
    """
    
    def __init__(self, llmclient):
        self.llm = llmclient
        
        # Register available tools
        self.tools: Dict[str, Callable] = {
            "calculator": self.tool_calculator,
            "search": self.tool_search_mock, 
            "final_answer": self.tool_final_answer
        }
        
        # Standard ReAct prompt
        self.system_prompt = """
You are a helpful AI assistant. Answer the user's question using the available tools: calculator, search, final_answer.

You must follow this format exactly:
Thought: [your reasoning]
Action: [tool_name] [arguments]

Example:
Thought: I need to calculate 5 plus 5.
Action: calculator 5 + 5
"""
        
    def run(self, user_query: str):
        print(f"\n{'='*40}")
        print(f"USER QUERY: {user_query}")
        print(f"{'='*40}\n")

        current_prompt = f"Question: {user_query}\n"
        max_steps = 5
        max_retries = 2 
        
        for step in range(1, max_steps + 1):
            
            # 1. MERGE PROMPTS
            # We combine system instructions and history into one string.
            # This fixes issues where the client ignores the 'system' argument.
            full_input = f"{self.system_prompt}\n\n{current_prompt}"
            
            # DEBUG: Uncomment the line below to see what is sent to the AI
            # print(f"[DEBUG] Sending to LLM:\n{full_input}\n---")

            # 2. GENERATE RESPONSE
            response = ""
            for attempt in range(max_retries):
                try:
                    raw_output = self.llm.generate(full_input)
                    
                    # 3. HANDLE STREAMING
                    # Ollama often returns a generator (stream). If we don't iterate, it looks empty.
                    if hasattr(raw_output, '__iter__') and not isinstance(raw_output, str):
                        print("[System] Detected streaming response. Aggregating...")
                        response = ""
                        for chunk in raw_output:
                            # Handle different streaming formats (dict or string)
                            if isinstance(chunk, dict):
                                response += chunk.get('response', '')
                            else:
                                response += str(chunk)
                    else:
                        # It's already a string
                        response = raw_output

                except Exception as e:
                    print(f"[System] Exception during generation: {e}")
                    response = ""
                
                # Check if valid
                if response and response.strip() and not response.strip().startswith("[Error"):
                    break 
                else:
                    print(f"[System] Empty/Invalid response (Attempt {attempt + 1}). Retrying...")
                    time.sleep(1) # Wait before retry
            
            # Final Validation
            if not response or not response.strip():
                print("[System] Failed to get response after retries. Check your Ollama connection.")
                break
            if response.strip().startswith("[Error"):
                print(f"[System] API Error: {response}")
                break

            print(f"[Step {step}] LLM Output:\n{response}\n")
            
            # Parse response
            action_name, action_input, thought = self._parse_response(response)
            
            # Execute logic
            if action_name == "final_answer":
                print(f"\n[FINAL ANSWER]: {action_input}")
                break
            elif action_name in self.tools:
                observation = self.tools[action_name](action_input)
                print(f"[OBSERVATION]: {observation}\n")
                
                # Update history
                current_prompt += f"Thought: {thought}\nAction: {action_name} {action_input}\nObservation: {observation}\n"
            else:
                print(f"[System] Unknown action '{action_name}'. Stopping.")
                break

    def _parse_response(self, text: str) -> tuple:
        """Extracts Thought and Action from the LLM text."""
        thought = ""
        action_name = ""
        action_input = ""

        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if not line: continue
            
            lower_line = line.lower()
            
            if lower_line.startswith("thought:"):
                # Handle "Thought: ... Action: ..." on same line
                if "action:" in lower_line:
                    parts = line.split("Action:", 1)
                    thought = parts[0].split(":", 1)[1].strip()
                    action_part = parts[1].strip()
                    a_parts = action_part.split(maxsplit=1)
                    action_name = a_parts[0].lower()
                    action_input = a_parts[1] if len(a_parts) > 1 else ""
                else:
                    thought = line.split(":", 1)[1].strip()
                    
            elif lower_line.startswith("action:"):
                action_full = line.split(":", 1)[1].strip()
                parts = action_full.split(maxsplit=1)
                action_name = parts[0].lower()
                action_input = parts[1] if len(parts) > 1 else ""
        
        return action_name, action_input, thought

    # --- TOOL DEFINITIONS ---
    
    def tool_calculator(self, expression: str) -> str:
        try:
            expression = expression.replace('"', '').replace("'", "")
            allowed_chars = set("0123456789+-*/(). ")
            if not all(c in allowed_chars for c in expression):
                return "Error: Invalid characters."
            return str(eval(expression))
        except Exception as e:
            return f"Error: {e}"

    def tool_search_mock(self, query: str) -> str:
        query_lower = query.lower()
        if "india" in query_lower:
            return "The capital of India is New Delhi."
        elif "france" in query_lower:
            return "The capital of France is Paris."
        else:
            return f"No info found for '{query}'."

    def tool_final_answer(self, answer: str) -> str:
        return answer
