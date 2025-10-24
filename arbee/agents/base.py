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
        max_retries: int = 3,
        max_reflection_iterations: int = 3) -> None:
        self.settings = settings or Settings()
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.max_reflection_iterations = max_reflection_iterations

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
            'errors': [],
            'reflection_iterations': []
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

    def validate_output(self, output: BaseModel) -> tuple[bool, Optional[str]]:
        """
        Validate agent output and provide feedback for reflection

        Args:
            output: The generated output to validate

        Returns:
            Tuple of (is_valid, feedback_message)
            - is_valid: True if output passes validation
            - feedback_message: None if valid, otherwise specific feedback for correction

        Note: Override this method in subclasses for custom validation logic
        """
        # Default validation: check that output is not None and is correct type
        if output is None:
            return False, "Output is None"

        expected_type = self.get_output_schema()
        if not isinstance(output, expected_type):
            return False, f"Output type mismatch: expected {expected_type}, got {type(output)}"

        return True, None

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
        if hasattr(self, 'get_human_prompt'):
            human_prompt = self.get_human_prompt()
            # Create separate system and human message templates
            # This allows proper variable parsing in each template
            messages = [
                SystemMessagePromptTemplate.from_template(system_msg),
                HumanMessagePromptTemplate.from_template(human_prompt)
            ]
        else:
            # Fallback to simple input variable
            messages = [
                SystemMessagePromptTemplate.from_template(system_msg),
                HumanMessagePromptTemplate.from_template("{input}")
            ]

        return ChatPromptTemplate.from_messages(messages)

    def create_chain(self, prompt_template: ChatPromptTemplate):
        """
        Create a LangChain LCEL chain with structured output enforcement

        Args:
            prompt_template: The prompt template to use

        Returns:
            Runnable chain: prompt | llm_with_structured_output
        """
        output_schema = self.get_output_schema()

        # Use OpenAI's structured output feature for guaranteed schema compliance
        # This ensures JSON output matches the Pydantic schema exactly
        llm_with_structured = self.llm.with_structured_output(
            output_schema,
            method="json_schema",  # Use strict JSON schema mode
            include_raw=False  # Return parsed object directly
        )

        chain = prompt_template | llm_with_structured

        return chain

    async def invoke(
        self,
        input_data: Dict[str, Any],
        prompt_override: Optional[ChatPromptTemplate] = None,
        enable_reflection: bool = True
    ) -> BaseModel:
        """
        Invoke the agent with input data, optionally with reflection loop

        Args:
            input_data: Dictionary of input variables for the prompt
            prompt_override: Optional custom prompt template
            enable_reflection: Whether to use reflection/validation loop

        Returns:
            Validated Pydantic model output

        Raises:
            Exception: If all retry attempts fail
        """
        self.stats['total_calls'] += 1

        # Create prompt and chain
        prompt = prompt_override or self.create_prompt_template()
        chain = self.create_chain(prompt)

        # Reflection loop (if enabled)
        if enable_reflection:
            return await self._invoke_with_reflection(chain, input_data)
        else:
            return await self._invoke_simple(chain, input_data)

    async def _invoke_simple(
        self,
        chain,
        input_data: Dict[str, Any]
    ) -> BaseModel:
        """Simple invoke without reflection (legacy behavior)"""
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

    async def _invoke_with_reflection(
        self,
        chain,
        input_data: Dict[str, Any]
    ) -> BaseModel:
        """
        Invoke with reflection loop: generate → validate → regenerate if needed

        This implements the Reflection Agent pattern from LangChain best practices
        """
        last_error = None
        best_output = None

        for iteration in range(self.max_reflection_iterations):
            try:
                self.logger.info(
                    f"🔄 {self.__class__.__name__} reflection iteration {iteration + 1}/{self.max_reflection_iterations}"
                )

                # Generate output
                start_time = datetime.now()
                result = await chain.ainvoke(input_data)
                duration = (datetime.now() - start_time).total_seconds()

                # Validate output
                is_valid, feedback = self.validate_output(result)

                if is_valid:
                    # Success! Output passed validation
                    self.stats['successful_calls'] += 1
                    self.stats['reflection_iterations'].append(iteration + 1)
                    self.logger.info(
                        f"✅ {self.__class__.__name__} completed successfully "
                        f"in {duration:.2f}s (after {iteration + 1} reflection iterations)"
                    )
                    return result
                else:
                    # Output failed validation, add feedback for next iteration
                    self.logger.warning(
                        f"⚠️  Reflection iteration {iteration + 1} failed validation: {feedback}"
                    )
                    best_output = result  # Keep best attempt

                    # Add feedback to prompt for next iteration
                    if 'reflection_feedback' not in input_data:
                        input_data['reflection_feedback'] = []
                    input_data['reflection_feedback'].append({
                        'iteration': iteration + 1,
                        'issue': feedback,
                        'instruction': 'Please correct the issues mentioned and regenerate the output.'
                    })

            except Exception as e:
                last_error = e
                self.logger.warning(
                    f"Reflection iteration {iteration + 1} raised exception: {str(e)}"
                )

                # Record error
                self.stats['errors'].append({
                    'timestamp': datetime.now().isoformat(),
                    'reflection_iteration': iteration + 1,
                    'error': str(e)
                })

        # All reflection iterations exhausted
        if best_output is not None:
            self.logger.warning(
                f"⚠️  {self.__class__.__name__} exhausted reflection iterations, "
                f"returning best attempt (may not pass validation)"
            )
            self.stats['successful_calls'] += 1
            return best_output

        # Complete failure
        self.stats['failed_calls'] += 1
        self.logger.error(
            f"❌ {self.__class__.__name__} failed after {self.max_reflection_iterations} reflection iterations"
        )
        if last_error:
            raise last_error
        else:
            raise ValueError("All reflection iterations failed validation with no valid output produced")

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
