"""Module to define multiple types of models provided by HuggingFace."""

from abc import ABC, abstractmethod
import sys

import dotenv
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
from transformers import BitsAndBytesConfig, Gemma3ForConditionalGeneration, Gemma3ForCausalLM
import torch

from .data import Document

dotenv.load_dotenv()

# It's needed to run in the RTX4000
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)

class HFModel(ABC):
    """Abstract class to define methods for models downloaded from huggingface."""

    @abstractmethod
    def query(self, query:str) -> str:
        """Query an answer based on a question."""

    @abstractmethod
    def query_with_documents(self, query: str, documents:list[Document]) -> str:
        """Query an answer based on a question and some documents passed as context."""

class Model:
    """Base class for all the models."""

    def build_multimodal_messages(self, query:str) -> list[dict[str:str|list]]:
        """Build the structure of the messages to send to the model when they support
        multimodal messages."""
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

        return messages

    def build_text_messages(self, query:str) -> list[dict[str:str]]:
        """Build the structure of the messages to send to the model when they support
        only text."""
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

        return messages

    def build_multimodal_messages_with_context(self,
                                               query:str,
                                               documents:list[Document]) -> list[dict]:
        """Build the structure of the messages passing the context documents when the model
        supports multimodal messages."""
        documents_context = self.__format_documents(documents)

        messages = [
            {
                "role": "system",
                "content": [{
                    "type": "text",
                    "text": "Responde las siguientes preguntas utilizando únicamente los "\
                            f"fragmentos de la normativa siguiente:\n\n{documents_context}"
                }]
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": query}]
            }
        ]

        return messages

    def build_text_messages_with_context(self,
                                         query:str,
                                         documents:list[Document]) -> list[dict[str:str|list]]:
        """Build the structure of the messages passing the context documents when the model
        supports text only messages."""
        documents_context = self.__format_documents(documents)

        messages = [
            {
                "role": "system",
                "content": "Responde las siguientes preguntas utilizando únicamente los "\
                           f"fragmentos de la normativa siguiente:\n\n{documents_context}"
            },
            {
                "role": "user",
                "content": query
            }
        ]

        return messages

    def __format_documents(self, documents:list[Document]) -> str:
        docs_str = ''
        for doc in documents:
            docs_str += f"{doc.get_reference()}:\n\n{doc.get_metadata()['title']}\n\n"\
                        f"{doc.get_content()}\n\n-----------------------------\n\n"

        return docs_str

class LlamaBuilder:
    """Factory method for Llama classes."""

    @classmethod
    def build_from_variant(cls, variant:str) -> HFModel:
        """Return an object of the corresponding Llama class depending on version."""
        main_version = variant.split('-')[0].split('.')[0]

        return getattr(sys.modules[__name__], f'Llama{main_version}')(variant)

class GemmaBuilder:
    """Factory method for Gemma classes."""

    @classmethod
    def build_from_variant(cls, variant:str='') -> HFModel:
        """Return an object of the corresponding Gemma class depending on version."""
        version = variant.split('-')[0].replace('.', '_')
        sub_version = '-'.join(variant.split('-')[1:])

        return getattr(sys.modules[__name__], f'Gemma{version}')(sub_version)

class QwenBuilder:
    """Factory method for Qwen classes."""

    @classmethod
    def build_from_variant(cls, variant:str='') -> HFModel:
        """Return an object of the corresponding Qwen class depending on version."""
        version = variant.split('-')[0].replace('.', '_')
        sub_version = '-'.join(variant.split('-')[1:])

        return getattr(sys.modules[__name__], f'Qwen{version}')(sub_version)

class MistralBuilder:
    """Factory method for Mistral classes."""

    @classmethod
    def build_from_variant(cls, variant:str='') -> HFModel:
        """Return an object of the corresponding Mistral class depending on version."""
        return Mistral(variant)

class Llama3(HFModel, Model):
    """Class to load Meta Llama 3.1 and 3.2 model and its variants."""

    def __init__(self, sub_version:str=''):
        if sub_version == '':
            self.model_id = "meta-llama/Llama-3.2-1B"
        else:
            self.model_id = f'meta-llama/Llama-{sub_version}'

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
        messages = self.build_multimodal_messages(query)

        result = self.pipeline(messages)

        return result

    def query_with_documents(self, query:str, documents:list[Document]):
        """Query an answer based on a question and some documents passed as context."""
        messages = self.build_multimodal_messages_with_context(query, documents)
        response = self.pipeline(messages)

        return response

class Gemma3(HFModel, Model):
    """Class to load Gemma3 model and its variants."""

    def __init__(self, sub_version:str=''):
        if sub_version == '':
            self.model_id = "google/gemma-3-4b-it"
        else:
            self.model_id = f"google/gemma-3-{sub_version}"

        self.multimodal = not(sub_version == '1b-it' or sub_version.endswith('-gguf'))
        self.processor = None
        self.tokenizer = None

        if self.model_id.endswith('-gguf'):
            gguf_file = self.model_id.rsplit('/', maxsplit=1)[-1].replace('-gguf',
                                                                          '.gguf').replace('-qat',
                                                                                           '')
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
        """Query an answer based on a question."""
        messages = self.build_multimodal_messages(query)

        if self.multimodal:
            response = self.__process_multimodal(messages)
        else:
            response = self.__process_text(messages)

        return response

    def query_with_documents(self, query:str, documents:list[Document]):
        """Query an answer based on a question and some documents passed as context."""
        messages = self.build_multimodal_messages_with_context(query, documents)
        if self.multimodal:
            response = self.__process_multimodal(messages)
        else:
            response = self.__process_text(messages)

        return response

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

class Qwen3(HFModel, Model):
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
        messages = self.build_text_messages(query)
        response = self.__get_response_from_model(messages)

        return response

    def query_with_documents(self, query:str, documents:list[Document]):
        """Query an answer based on a question and some documents passed as context."""
        messages = self.build_text_messages_with_context(query, documents)
        response = self.__get_response_from_model(messages)

        return response

    def __get_response_from_model(self, messages:list) -> str:
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

class Mistral(HFModel, Model):
    """Class to load Mistral AI models."""

    def __init__(self, sub_version:str=''):
        if sub_version == '':
            self.model_id = "mistralai/Mistral-7B-Instruct-v0.3"
        else:
            self.model_id = f"mistralai/Mistral-{sub_version}"

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map='auto'
        )

    def query(self, query:str):
        """Query an answer based on a question."""
        messages = self.build_text_messages(query)
        response = self.__get_response_from_model(messages)

        return response

    def query_with_documents(self, query:str, documents:list[Document]):
        """Query an answer based on a question and some documents passed as context."""
        messages = self.build_text_messages_with_context(query, documents)
        response = self.__get_response_from_model(messages)

        return response

    def __get_response_from_model(self, messages:list) -> str:
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
