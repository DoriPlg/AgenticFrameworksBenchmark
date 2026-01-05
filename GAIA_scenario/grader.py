import json
from typing import Dict, Optional
from litellm import completion


class AnswerGrader:
    """Grades agent answers using an LLM to handle formatting variations."""
    
    def __init__(self, model: str = "gpt-4"):
        self.model = model
        
    def grade_answer(self, agent_answer: Optional[str], correct_answer: str, question: str) -> Dict:
        """
        Grade a single answer using an LLM.
        
        Returns:
            Dict with keys: is_correct (bool), confidence (str), reasoning (str)
        """
        if agent_answer is None:
            return {
                "is_correct": False,
                "confidence": "certain",
                "reasoning": "No answer provided by agent"
            }
        
        prompt = f"""You are grading an AI agent's answer to a question.

Question: {question}

Correct Answer: {correct_answer}

Agent's Answer: {agent_answer}

Determine if the agent's answer is correct. Consider:
- Semantic equivalence (different phrasings of the same answer)
- Numerical equivalence (with reasonable rounding)
- Formatting variations

Respond in JSON format with:
- is_correct: boolean
- confidence: "certain", "high", "medium", "low"
- reasoning: brief explanation

JSON response:"""

        try:
            response = completion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            
            result_text = response.choices[0].message.content.strip()
            # Extract JSON from markdown code blocks if present
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0].strip()
                
            result = json.loads(result_text)
            return result
            
        except Exception as e:
            return {
                "is_correct": False,
                "confidence": "uncertain",
                "reasoning": f"Grading error: {str(e)}"
            }
