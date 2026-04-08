import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Optional
from dataclasses import dataclass, field

@dataclass
class ReasoningStep:
    step_number: int
    description: str
    reasoning: str
    conclusion: str
    metadata: Dict = field(default_factory=dict)

class ChainOfThoughtReasoner:
    def __init__(self, llm_client, max_retries: int = 2):
        self.llm_client = llm_client
        self.max_retries = max_retries
        self.reasoning_history: List[ReasoningStep] = []

    def _generate_step(self, step_num: int, prompt: str, description: str) -> ReasoningStep:
        response = ""
        attempt = 0
        while attempt <= self.max_retries:
            response = self.llm_client.generate(prompt)
            if response and not response.startswith("[Error:"):
                break
            attempt += 1
            time.sleep(2)

        is_error = not response or response.startswith("[Error:")
        
        # FIX: Improved extraction with a safety fallback
        conclusion = self._extract_conclusion(response)
        
        return ReasoningStep(
            step_number=step_num,
            description=description,
            reasoning=response if response else "No response from LLM.",
            conclusion="Error" if is_error else conclusion,
            metadata={"is_error": is_error}
        )

    def _extract_conclusion(self, text: str) -> str:
        if not text or "[Error:" in text:
            return "No conclusion found."
            
        # Look for specific markers
        match = re.search(r"(?:Result|Conclusion|Answer|Final Answer):\s*(.*)", text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        # Fallback: If no marker is found, take the last non-empty line
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        if lines:
            # If the last line is very short (like "6"), it's likely the answer.
            # If it's long, it's still better than "No conclusion found".
            return lines[-1]
            
        return "No conclusion found."

    def reason(self, problem: str) -> List[ReasoningStep]:
        self.reasoning_history = []

        # --- PARALLEL EXECUTION OF STEP 1 & 2 ---
        # We tell the model EXACTLY how to end the response.
        prompt_1 = f"Problem: {problem}\nSolve it directly. End with 'Result: <answer>'"
        prompt_2 = f"Problem: {problem}\nSolve it using an alternative mental check. End with 'Result: <answer>'"

        print("🚀 Running Methods A & B in parallel...")
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_1 = executor.submit(self._generate_step, 1, prompt_1, "Method A (Direct)")
            future_2 = executor.submit(self._generate_step, 2, prompt_2, "Method B (Alternative)")
            step1 = future_1.result()
            step2 = future_2.result()

        self.reasoning_history.extend([step1, step2])

        # --- STEP 3: VALIDATION ---
        # We pass the conclusions explicitly into the prompt.
        prompt_3 = f"""Compare these two steps for the problem: "{problem}"
Step 1 Result: {step1.conclusion}
Step 2 Result: {step2.conclusion}

Analysis: Are they consistent? If not, which one is right? 
Result: <State the confirmed correct answer here>"""
        
        step3 = self._generate_step(3, prompt_3, "Validation")
        self.reasoning_history.append(step3)

        # --- STEP 4: FINAL OUTPUT ---
        prompt_4 = f"""Based on the validated result: {step3.conclusion}
Provide only the final numerical or short answer for: {problem}
Result:"""
        
        step4 = self._generate_step(4, prompt_4, "Final Output")
        self.reasoning_history.append(step4)

        return self.reasoning_history

    def explain_reasoning(self) -> str:
        explanation = "\n" + "="*60 + "\nFINAL CHAIN OF THOUGHT REPORT\n" + "="*60 + "\n"
        for s in self.reasoning_history:
            explanation += f"📍 STEP {s.step_number}: {s.description}\n"
            explanation += f"   Reasoning: {s.reasoning[:150]}...\n" # Truncated for readability
            explanation += f"   ✅ Conclusion: {s.conclusion}\n"
            explanation += "-"*60 + "\n"
        
        final_ans = self.reasoning_history[-1].conclusion
        explanation += f"🎯 FINAL ANSWER: {final_ans}\n"
        explanation += "="*60
        return explanation
