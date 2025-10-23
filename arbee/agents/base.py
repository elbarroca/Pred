"""
Base Agent class for POLYSEER
Provides LangChain integration, state management, and common functionality
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Type, TypeVar
from pydantic import BaseModel
import json
import logging
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnablePassthrough
from config.settings import Settings

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

T = TypeVar('T', bound=BaseModel)


class BaseAgent(ABC):
    """
    Abstract base class for all POLYSEER agents

    Features:
    - LangChain integration with Claude 3.5 Sonnet
    - Pydantic output validation
    - Error handling and retry logic
    - Structured logging
    - Token usage tracking
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.1,
        max_tokens: int = 4096,
        max_retries: int = 3) -> None:
        self.settings = settings or Settings()
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries

        # Initialize logger
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize LLM
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",  # Use GPT-4o-mini as requested
            temperature=temperature,
            max_tokens=max_tokens,
            openai_api_key=self.settings.OPENAI_API_KEY
        )

        # Track usage statistics
        self.stats = {
            'total_calls': 0,
            'successful_calls': 0,
            'failed_calls': 0,
            'total_tokens': 0,
            'errors': []
        }

        self.logger.info(f"{self.__class__.__name__} initialized with model {model_name}")

    @abstractmethod
    def get_system_prompt(self) -> str:
        """
        Return the system prompt for this agent
        Must be implemented by each agent
        """
        pass

    @abstractmethod
    def get_output_schema(self) -> Type[BaseModel]:
        """
        Return the Pydantic schema for this agent's output
        Must be implemented by each agent
        """
        pass

    def create_prompt_template(
        self,
        system_prompt: Optional[str] = None,
        few_shot_examples: Optional[str] = None
    ) -> ChatPromptTemplate:
        """
        Create a LangChain prompt template

        Args:
            system_prompt: Override default system prompt
            few_shot_examples: Optional few-shot examples to include

        Returns:
            ChatPromptTemplate ready for use
        """
        system_msg = system_prompt or self.get_system_prompt()

        # Add few-shot examples if provided (without escaping to keep them readable)
        if few_shot_examples:
            system_msg = f"{system_msg}\n\n## Examples:\n{few_shot_examples}"

        # Simple instruction without schema to avoid template variable issues
        system_msg += "\n\nYou must respond with valid JSON. The response will be validated against the required schema."

        # Create template with human prompt if available
        human_prompt = ""
        template_vars = ["input"]  # Default variable name

        if hasattr(self, 'get_human_prompt'):
            human_prompt = self.get_human_prompt()
            # Extract template variables from the human prompt
            import re
            vars_in_prompt = re.findall(r'\{([^}]+)\}', human_prompt)
            if vars_in_prompt:
                template_vars = vars_in_prompt

        if human_prompt:
            template_str = system_msg + "\n\n" + human_prompt + "\n\nAssistant:"
        else:
            template_str = system_msg + "\n\nHuman: {input}\n\nAssistant:"

        messages = [
            SystemMessagePromptTemplate.from_template(template_str)
        ]

        return ChatPromptTemplate.from_messages(messages)

    def create_chain(self, prompt_template: ChatPromptTemplate):
        """
        Create a LangChain LCEL chain with Pydantic output parsing

        Args:
            prompt_template: The prompt template to use

        Returns:
            Runnable chain: prompt | llm | parser
        """
        output_schema = self.get_output_schema()
        parser = PydanticOutputParser(pydantic_object=output_schema)

        chain = prompt_template | self.llm | parser

        return chain

    async def invoke(
        self,
        input_data: Dict[str, Any],
        prompt_override: Optional[ChatPromptTemplate] = None
    ) -> BaseModel:
        """
        Invoke the agent with input data

        Args:
            input_data: Dictionary of input variables for the prompt
            prompt_override: Optional custom prompt template

        Returns:
            Validated Pydantic model output

        Raises:
            Exception: If all retry attempts fail
        """
        self.stats['total_calls'] += 1

        # Create prompt and chain
        prompt = prompt_override or self.create_prompt_template()
        chain = self.create_chain(prompt)

        # Retry logic
        last_error = None
        for attempt in range(self.max_retries):
            try:
                self.logger.info(
                    f"Invoking {self.__class__.__name__} (attempt {attempt + 1}/{self.max_retries})"
                )

                # Execute chain
                start_time = datetime.now()
                result = await chain.ainvoke(input_data)
                duration = (datetime.now() - start_time).total_seconds()

                # Success
                self.stats['successful_calls'] += 1
                self.logger.info(
                    f"{self.__class__.__name__} completed successfully in {duration:.2f}s"
                )

                return result

            except Exception as e:
                last_error = e
                self.logger.warning(
                    f"{self.__class__.__name__} attempt {attempt + 1} failed: {str(e)}"
                )

                # Record error
                self.stats['errors'].append({
                    'timestamp': datetime.now().isoformat(),
                    'attempt': attempt + 1,
                    'error': str(e)
                })

                # Don't retry on validation errors (bad schema)
                if "validation" in str(e).lower():
                    self.logger.error(f"Validation error - not retrying: {e}")
                    break

        # All retries failed
        self.stats['failed_calls'] += 1
        self.logger.error(
            f"{self.__class__.__name__} failed after {self.max_retries} attempts"
        )
        raise last_error

    def get_stats(self) -> Dict[str, Any]:
        """Return usage statistics"""
        return {
            **self.stats,
            'success_rate': (
                self.stats['successful_calls'] / self.stats['total_calls']
                if self.stats['total_calls'] > 0 else 0.0
            )
        }

    def reset_stats(self):
        """Reset statistics counters"""
        self.stats = {
            'total_calls': 0,
            'successful_calls': 0,
            'failed_calls': 0,
            'total_tokens': 0,
            'errors': []
        }
        self.logger.info(f"{self.__class__.__name__} stats reset")
