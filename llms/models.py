"""Module to define multiple types of models provided by HuggingFace."""

from abc import ABC, abstractmethod
import sys

import dotenv
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
from transformers import BitsAndBytesConfig, Gemma3ForConditionalGeneration, Gemma3ForCausalLM
from transformers import Llama4ForConditionalGeneration
import torch

dotenv.load_dotenv()

# It's needed to run in the RTX4000
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)

class HFModel(ABC):
    """Abstract class to define methods for models downloaded from huggingface."""

    @abstractmethod
    def __init__(self, sub_version:str=''):
        """Constructor of the model."""

    @abstractmethod
    def query(self, query:str) -> str:
        """Query an answer based on a question."""

    @abstractmethod
    def query_with_documents(self, query: str, documents:list[dict[str, str]]) -> str:
        """Query an answer based on a question and some documents passed as context."""

class LlamaBuilder:

    @classmethod
    def build_from_variant(cls, variant:str) -> HFModel:
        main_version = variant.split('-')[0].split('.')[0]

        return getattr(sys.modules[__name__], f'Llama{main_version}')(variant)

class GemmaBuilder:

    @classmethod
    def build_from_variant(cls, variant:str='') -> HFModel:
        version = variant.split('-')[0].replace('.', '_')
        sub_version = '-'.join(variant.split('-')[1:])

        return getattr(sys.modules[__name__], f'Gemma{version}')(sub_version)

class QwenBuilder:

    @classmethod
    def build_from_variant(cls, variant:str='') -> HFModel:
        version = variant.split('-')[0].replace('.', '_')
        sub_version = '-'.join(variant.split('-')[1:])

        return getattr(sys.modules[__name__], f'Qwen{version}')(sub_version)

class MistralBuilder:

    @classmethod
    def build_from_variant(cls, variant:str='') -> HFModel:
        return Mistral(variant)

class Llama3(HFModel):
    """Class to load Meta Llama 3.1 and 3.2 model and its variants."""

    def __init__(self, sub_version:str=''):
        self.model_id = "meta-llama/Llama-3.2-1B" if sub_version == '' else f'meta-llama/Llama-{sub_version}'

        tokenizer = AutoTokenizer.from_pretrained(self.model_id, legacy=False)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=self.model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            tokenizer=tokenizer,
            device_map="auto"
        )

    def query(self, query:str) -> str:
        """Query an answer based on a question."""
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "Eres un experto en resolver preguntas."}]
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": query}]
            }
        ]

        result = self.pipeline(messages)

        return result

    def query_with_documents(self, query, documents):
        """Query an answer based on a question and some documents passed as context."""

class Gemma3(HFModel):

    def __init__(self, sub_version:str=''):
        self.model_id = "google/gemma-3-4b-it" if sub_version == '' else f"google/gemma-3-{sub_version}"

        self.multimodal = not(sub_version == '1b-it' or sub_version.endswith('-gguf'))
        self.processor = None
        self.tokenizer = None

        if self.model_id.endswith('-gguf'):
            gguf_file = self.model_id.split('/')[-1].replace('-gguf', '.gguf').replace('-qat', '')
            quantization_config = None
        else:
            gguf_file = None
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        if self.multimodal:
            self.model = Gemma3ForConditionalGeneration.from_pretrained(
                self.model_id, device_map="auto"
            ).eval()
            self.processor = AutoProcessor.from_pretrained(self.model_id, use_fast=True)
        else:
            self.model = Gemma3ForCausalLM.from_pretrained(
                self.model_id, quantization_config=quantization_config, gguf_file=gguf_file
            ).eval()
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, gguf_file=gguf_file)

    def query(self, query:str) -> str:
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "Eres un experto en resolver preguntas."}]
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": query}]
            }
        ]

        if self.multimodal:
            response = self.__process_multimodal(messages)
        else:
            response = self.__process_text(messages)

        return response

    def query_with_documents(self, query, documents):
        """Query an answer based on a question and some documents passed as context."""

    def __process_multimodal(self, messages:list[dict]):
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt"
        ).to(self.model.device, dtype=torch.bfloat16)

        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = self.model.generate(**inputs, max_new_tokens=100, do_sample=False)
            generation = generation[0][input_len:]

        decoded = self.processor.decode(generation, skip_special_tokens=True)

        return decoded

    def __process_text(self, messages:list[dict]):
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)#.to(torch.bfloat16)

        with torch.inference_mode():
            outputs = self.model.generate(**inputs, max_new_tokens=64)

        decoded = self.tokenizer.batch_decode(outputs)

        return decoded

class Qwen3(HFModel):
    """Class to load Qwen models."""

    def __init__(self, sub_version:str=''):
        self.model_id = "Qwen/Qwen3-0.6B" if sub_version == '' else f"Qwen/Qwen3-{sub_version}"

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype='auto',
            device_map='auto'
        )

    def query(self, query:str):
        """Query an answer based on a question."""
        messages = [
            {
                "role": "system",
                "content": "Eres un experto en resolver preguntas."
            },
            {
                "role": "user",
                "content": query
            }
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

class Mistral(HFModel):
    """Class to load Mistral AI models."""

    def __init__(self, sub_version:str=''):
        self.model_id = "mistralai/Mistral-7B-Instruct-v0.3" if sub_version == '' else f"mistralai/Mistral-{sub_version}"

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map='auto'
        )

    def query(self, query:str):
        messages = [
            {
                "role": "system",
                "content": "Eres un experto en resolver preguntas."
            },
            {
                "role": "user",
                "content": query
            }
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_tokens = self.model.generate(
            **model_inputs,
            max_new_tokens=64
        )

        result = self.tokenizer.decode(generated_tokens[0], skip_special_tokens=True).strip("\n")

        return result

    def query_with_documents(self, query, documents):
        return super().query_with_documents(query, documents)