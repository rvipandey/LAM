# vectormemory/vector.py
import numpy as np
import re
import heapq
import hashlib
from typing import List, Dict, Optional
from dataclasses import dataclass, field

# =============================================================================
# VECTOR MEMORY (RAG)
# =============================================================================

class VectorMemory:
    """
    Simple in-memory vector store.
    Uses deterministic hashing for embedding (Fast, no API calls required for this part).
    """
    def __init__(self, dim=768):
        self.dim = dim
        self.vectors: Dict[str, np.ndarray] = {}
        self.metadata: Dict[str, str] = {}

    def _embed(self, text: str) -> np.ndarray:
        """Deterministic mock embedding using hash."""
        h = int(hashlib.md5(text.encode()).hexdigest(), 16)
        seed = h % (2**32 - 1)
        rng = np.random.RandomState(seed)
        v = rng.randn(self.dim).astype(np.float32)
        return v / np.linalg.norm(v)

    def add(self, doc_id: str, text: str):
        self.vectors[doc_id] = self._embed(text)
        self.metadata[doc_id] = text

    def retrieve(self, query: str, k: int = 3) -> List[str]:
        if not self.vectors: return []
        q_vec = self._embed(query)
        scores = [
            (doc_id, float(np.dot(q_vec, vec)))
            for doc_id, vec in self.vectors.items()
        ]
        scores.sort(key=lambda x: x[1], reverse=True)
        return [self.metadata[doc_id] for doc_id, _ in scores[:k]]

# =============================================================================
# TREE OF THOUGHTS (Speed Optimized)
# =============================================================================

@dataclass
class Thought:
    content: str
    parent: Optional['Thought'] = None
    value: float = 0.0

    def get_path(self) -> str:
        path = []
        curr = self
        while curr:
            path.append(curr.content)
            curr = curr.parent
        return "\n".join(reversed(path))

class TreeOfThoughtsReasoner:
    """
    Optimized ToT Reasoner.
    Reduces API calls by asking LLM to Generate AND Evaluate in one prompt.
    """
    def __init__(self, llm_client, max_depth: int = 2, branch_factor: int = 2):
        self.llm = llm_client
        self.max_depth = max_depth
        self.branch = branch_factor

    def _generate_and_eval(self, context: str) -> List[tuple]:
        """
        Generates thoughts AND their scores in a single API call.
        Returns: List of (thought_text, score)
        """
        # Ultra-strict prompt for speed and parsing reliability
        prompt = f"""Current Context:
{context}

Propose {self.branch} distinct next steps to solve the problem.
For each step, rate its promise (0.0 to 1.0).
Format strictly as:
1. [Step Description] [Score]
2. [Step Description] [Score]"""

        res = self.llm.generate(prompt)
        if not res: return []

        results = []
        lines = res.split('\n')
        for line in lines:
            line = line.strip()
            if not line or not line[0].isdigit(): continue
            
            # Try to extract content and score
            # Regex looks for text at the end: [0.0 - 1.0]
            match = re.search(r"(.+?)\s+(\d\.\d+|\d)$", line)
            if match:
                content = match.group(1).strip()
                # Remove leading numbering like "1." or "1)"
                content = re.sub(r"^\d+[\.)]\s*", "", content)
                score = float(match.group(2))
                results.append((content, score))
        
        # Fallback if parsing fails (keep the branch alive)
        if len(results) < self.branch:
            for i in range(len(results), self.branch):
                results.append((f"Step {i+1}", 0.5))
                
        return results

    def solve(self, problem: str) -> str:
        root = Thought(content=problem)
        # Start with the root node
        frontier = [root]

        print(f"   [ToT] Starting Search (Depth: {self.max_depth}, Branch: {self.branch})")

        for depth in range(self.max_depth):
            if not frontier: break
            
            next_level = []
            
            for node in frontier:
                # 1. Generate steps AND scores in ONE call
                steps_scores = self._generate_and_eval(node.get_path())
                
                for step_text, score in steps_scores:
                    new_node = Thought(content=step_text, parent=node, value=score)
                    next_level.append(new_node)
                    print(f"   [ToT] Depth {depth+1}: {step_text[:40]}... (Score: {score})")

            # 2. Prune: Keep only top 'branch' nodes
            frontier = heapq.nlargest(self.branch, next_level, key=lambda x: x.value)
            
            if not frontier:
                break

        # Get the best path found
        best_node = frontier[0] if frontier else root
        best_path = best_node.get_path()
        
        print(f"   [ToT] Best path selected. Synthesizing final answer...")

        # 3. Final Synthesis (One final call)
        syn_prompt = f"Based on these reasoning steps, provide a concise final answer:\n{best_path}"
        final_answer = self.llm.generate(syn_prompt)
        
        return final_answer if final_answer else best_path
