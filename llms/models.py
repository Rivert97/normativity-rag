"""Module to define multiple types of models provided by HuggingFace."""

from abc import ABC, abstractmethod

import dotenv
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

dotenv.load_dotenv()

# It's needed to run in the RTX4000
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)

class LLMModel(ABC):
    """Abstract class to define methods for models."""

    @abstractmethod
    def query(self, query:str) -> str:
        """Query an answer based on a question."""

    @abstractmethod
    def query_with_documents(self, query: str, documents:list[dict[str, str]]) -> str:
        """Query an answer based on a question and some documents passed as context."""

class Qwen(LLMModel):
    """Class to load Qwen models."""

    def __init__(self):
        self.name = "Qwen/Qwen3-0.6B"

        self.tokenizer = AutoTokenizer.from_pretrained(self.name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.name,
            torch_dtype='auto',
            device_map='auto'
        )

    def query(self, query:str):
        """Query an answer based on a question."""
        messages = [
            {"role": "user", "content": query}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True # Switches between thinking and non-thinking modes. Default is True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        # Conduct text completion
        generated_tokens = self.model.generate(
            **model_inputs,
            max_new_tokens=32768
        )
        output_ids = generated_tokens[0][len(model_inputs.input_ids[0]):].tolist()

        # parsing thinking content
        try:
            # rindex finding 151668 (</think>)
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0

        #thinking_content =
        #    self.tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

        return content

    def query_with_documents(self, query, documents):
        """Query an answer based on a question and some documents passed as context."""

class LLama(LLMModel):
    """Class to load Meta Llama model."""

    def __init__(self):
        self.pipeline = transformers.pipeline(
            "text-generation",
            model="meta-llama/Meta-Llama-3-8B",
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto"
        )

    def query(self, query:str) -> str:
        """Query an answer based on a question."""
        return self.pipeline(query)

    def query_with_documents(self, query, documents):
        """Query an answer based on a question and some documents passed as context."""
