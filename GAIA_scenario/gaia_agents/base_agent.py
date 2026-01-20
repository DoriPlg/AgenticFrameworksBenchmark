"""Base abstract class for all agent implementations."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional, List


@dataclass
class AgentResponse:
    """Standardized response from any agent."""
    answer: str
    execution_time: float
    reasoning: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class BaseAgent(ABC):
    """Abstract base class that all agent implementations must inherit from."""
    
    def __init__(self, model_config: Dict[str, Any], verbose: bool = False):
        """
        Initialize the agent.
        
        Args:
            model_config: Dictionary containing model configuration (model, base_url, api_key, etc.)
            verbose: Whether to enable verbose output
        """
        self.model_config = model_config
        self.verbose = verbose
        self._name = None
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the agent's name (e.g., 'CrewAI-GPT4')."""
        pass
    
    @abstractmethod
    def run(self, question: str, file_paths: Optional[List[str]] = None) -> AgentResponse:
        """
        Execute the agent on a given question.
        
        Args:
            question: The question to answer
            file_paths: Optional list of file paths that may be needed to answer the question
            
        Returns:
            AgentResponse with answer, execution time, and optional metadata
        """
        pass
