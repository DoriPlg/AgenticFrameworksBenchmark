import json
from typing import Dict, Optional
from langchain_openai import ChatOpenAI


class AnswerGrader:
    """Grades agent answers using an LLM to handle formatting variations."""
    
    def __init__(self, model: dict):
        self.llm = ChatOpenAI(
            model=model['model'],
            base_url=model['base_url'],
            api_key=model['api_key'],
            temperature= 0.01,
        )
        
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
            response = self.llm.invoke([{"role": "user", "content": prompt}])
            result_text = response.content
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

    def access_preformance(self, questions: Dict, grading_summary: Dict) -> str:
        """Assess overall performance based on grading results and the answers given by the agent."""
        answers = [question["agent_answer"] if question["agent_answer"] is not None else "No Answer"  for question in questions]
        prompt = f"""
Go over the list of answers an agent gave on questions, along with the grading result for the entire preformance.
Give a literal summary of the agent's performance, including strengths, weaknesses and recurring pitfalls.
The list of answers:
    {",\n".join(answers)}
The grading summary:
    {json.dumps(grading_summary, indent=2)}
Provide your assessment in a concise one or two sentence paragraph, take into account that the maximal accuracy achieved was about 0.5"
        """
        response = self.llm.invoke([{"role": "user", "content": prompt}])
        return response.content