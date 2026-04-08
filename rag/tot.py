import re
import time
import heapq
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Optional, Tuple, Callable
from dataclasses import dataclass, field

@dataclass
class Thought:
    content: str
    parent: Optional['Thought'] = None
    children: List['Thought'] = field(default_factory=list)
    value: float = 0.0
    depth: int = 0
    is_solution: bool = False

    def get_full_context(self) -> str:
        """Reconstructs the reasoning path from the root to this thought."""
        path = []
        current = self
        while current:
            path.append(current.content)
            current = current.parent
        return "\n".join(reversed(path))

class TreeOfThoughtsReasoner:
    def __init__(self, llm_client, max_depth: int = 3, branch_factor: int = 3):
        self.llm_client = llm_client
        self.max_depth = max_depth
        self.branch_factor = branch_factor # Number of thoughts to try at each level
        self.root: Optional[Thought] = None

    def _generate_thoughts(self, context: str) -> List[str]:
        """Ask LLM to generate multiple possible next steps."""
        prompt = f"""Current reasoning state:
{context}

Given the state above, what are the next {self.branch_factor} logical steps or possible solutions?
Provide them as a numbered list.
Result:"""
        response = self.llm_client.generate(prompt)
        # Extract lines that look like list items
        thoughts = re.findall(r"^\d+\.\s*(.*)", response, re.MULTILINE)
        return thoughts[:self.branch_factor] if thoughts else [response]

    def _evaluate_thought(self, context: str) -> float:
        """Ask LLM to score a path's promise from 0.0 to 1.0."""
        prompt = f"""Evaluate the following reasoning path for correctness and promise:
{context}

Score this path between 0.0 (impossible/wrong) and 1.0 (highly promising/correct).
Provide ONLY the numerical score.
Result:"""
        response = self.llm_client.generate(prompt)
        match = re.search(r"(\d\.\d+|\d)", response)
        try:
            return float(match.group(1)) if match else 0.5
        except:
            return 0.5

    def solve(self, problem: str) -> str:
        """Explores the thought tree using a Breadth-First Search (BFS) / Beam Search style."""
        self.root = Thought(content=f"Problem: {problem}", depth=0, value=1.0)
        current_level = [self.root]

        print(f"🌳 Starting Tree-of-Thoughts search for: {problem[:50]}...")

        for depth in range(self.max_depth):
            next_level_candidates = []
            
            for thought in current_level:
                # Generate new thoughts for each leaf in the current beam
                raw_thoughts = self._generate_thoughts(thought.get_full_context())
                
                # Evaluate each new thought in parallel to save time
                with ThreadPoolExecutor(max_workers=self.branch_factor) as executor:
                    futures = {executor.submit(self._evaluate_thought, thought.get_full_context() + "\n" + t): t for t in raw_thoughts}
                    
                    for future in futures:
                        content = futures[future]
                        score = future.result()
                        new_thought = Thought(content=content, parent=thought, depth=depth+1, value=score)
                        thought.children.append(new_thought)
                        next_level_candidates.append(new_thought)

            # Pruning: Keep only the top 'branch_factor' thoughts for the next depth
            next_level_candidates.sort(key=lambda x: x.value, reverse=True)
            current_level = next_level_candidates[:self.branch_factor]
            
            print(f"📍 Depth {depth+1} complete. Best path score: {current_level[0].value if current_level else 0}")

        # Final Answer Extraction from the best path
        best_final_thought = current_level[0] if current_level else self.root
        return self._get_final_synthesis(problem, best_final_thought.get_full_context())

    def _get_final_synthesis(self, problem: str, best_path: str) -> str:
        prompt = f"""Based on the following successful reasoning paths:
{best_path}

What is the final definitive answer to: {problem}
Result:"""
        return self.llm_client.generate(prompt)