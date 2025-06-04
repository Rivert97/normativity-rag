"""Module to define multiple types of models provided by HuggingFace."""

from abc import abstractmethod
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

class Model:
    """Base class for all the models."""

    def __init__(self, multimodal:bool=False):
        self.multimodal = multimodal

        self.messages = self.__init_messages()

    @abstractmethod
    def query(self, query:str) -> str:
        """Query an answer based on a question."""

    @abstractmethod
    def query_with_documents(self, query: str, documents:list[Document]) -> str:
        """Query an answer based on a question and some documents passed as context."""

    def add_message(self, query:str):
        """Add a new message to be sent to the model."""
        if self.multimodal:
            self.__add_multimodal_message(query)
        else:
            self.__add_text_message(query)

    def add_message_with_context(self, query:str, documents:list[Document]):
        """Add a new message to be sent to the model including documents to be used as context."""
        if self.multimodal:
            self.__add_multimodal_message_with_context(query, documents)
        else:
            self.__add_text_message_with_context(query, documents)

    def add_response(self, response:str):
        """Add response to the history of conversation."""
        new_message = {
            "role": "assistant",
            "content": response,
        }
        self.messages.append(new_message)

    def __init_messages(self) -> list[dict[str:str|dict]]:
        if self.multimodal:
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "Eres un experto en resolver preguntas."}],
                },
            ]
        else:
            messages = [
                {
                    "role": "system",
                    "content": "Eres un experto en resolver preguntas.",
                },
            ]

        return messages

    def __add_multimodal_message(self, query:str):
        new_message = {
            "role": "user",
            "content": [{"type": "text", "text": query}]
        }
        self.messages.append(new_message)

    def __add_text_message(self, query:str):
        new_message = {
            "role": "user",
            "content": query
        }
        self.messages.append(new_message)

    def __add_multimodal_message_with_context(self, query:str, documents:list[Document]):
        documents_context = self.__format_documents(documents)

        new_messages = [
            {
                "role": "system",
                "content": [{
                    "type": "text",
                    "text": "Responde la siguiente pregunta utilizando únicamente los "\
                            f"fragmentos de la normativa siguiente:\n\n{documents_context}"
                }]
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": query}]
            }
        ]
        self.messages.extend(new_messages)

    def __add_text_message_with_context(self, query:str, documents:list[Document]):
        documents_context = self.__format_documents(documents)

        new_messages = [
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
        self.messages.extend(new_messages)

    def __format_documents(self, documents:list[Document]) -> str:
        docs_str = ''
        for doc in documents:
            docs_str += f"{doc.get_reference()}:\n\n{doc.get_metadata()['title']}\n\n"\
                        f"{doc.get_content()}\n\n-----------------------------\n\n"

        return docs_str

class LlamaBuilder:
    """Factory method for Llama classes."""

    @classmethod
    def build_from_variant(cls, variant:str) -> Model:
        """Return an object of the corresponding Llama class depending on version."""
        main_version = variant.split('-')[0].split('.')[0]

        return getattr(sys.modules[__name__], f'Llama{main_version}')(variant)

class GemmaBuilder:
    """Factory method for Gemma classes."""

    @classmethod
    def build_from_variant(cls, variant:str='') -> Model:
        """Return an object of the corresponding Gemma class depending on version."""
        version = variant.split('-')[0].replace('.', '_')
        sub_version = '-'.join(variant.split('-')[1:])

        return getattr(sys.modules[__name__], f'Gemma{version}')(sub_version)

class QwenBuilder:
    """Factory method for Qwen classes."""

    @classmethod
    def build_from_variant(cls, variant:str='') -> Model:
        """Return an object of the corresponding Qwen class depending on version."""
        version = variant.split('-')[0].replace('.', '_')
        sub_version = '-'.join(variant.split('-')[1:])

        return getattr(sys.modules[__name__], f'Qwen{version}')(sub_version)

class MistralBuilder:
    """Factory method for Mistral classes."""

    @classmethod
    def build_from_variant(cls, variant:str='') -> Model:
        """Return an object of the corresponding Mistral class depending on version."""
        return Mistral(variant)

class Llama3(Model):
    """Class to load Meta Llama 3.1 and 3.2 model and its variants."""

    def __init__(self, sub_version:str=''):
        super().__init__(multimodal=True)

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
        self.add_message(query)

        result = self.pipeline(self.messages)

        return result

    def query_with_documents(self, query:str, documents:list[Document]):
        """Query an answer based on a question and some documents passed as context."""
        self.add_message_with_context(query, documents)
        response = self.pipeline(self.messages)

        return response

class Gemma3(Model):
    """Class to load Gemma3 model and its variants."""

    def __init__(self, sub_version:str=''):
        multimodal = not(sub_version == '1b-it' or sub_version.endswith('-gguf'))
        super().__init__(multimodal=multimodal)

        if sub_version == '':
            self.model_id = "google/gemma-3-4b-it"
        else:
            self.model_id = f"google/gemma-3-{sub_version}"

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
        self.add_message(query)

        if self.multimodal:
            response = self.__process_multimodal(self.messages)
        else:
            response = self.__process_text(self.messages)

        return response

    def query_with_documents(self, query:str, documents:list[Document]):
        """Query an answer based on a question and some documents passed as context."""
        self.add_message_with_context(query, documents)
        if self.multimodal:
            response = self.__process_multimodal(self.messages)
        else:
            response = self.__process_text(self.messages)

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

class Qwen3(Model):
    """Class to load Qwen models."""

    def __init__(self, sub_version:str=''):
        super().__init__(multimodal=False)

        self.model_id = "Qwen/Qwen3-0.6B" if sub_version == '' else f"Qwen/Qwen3-{sub_version}"

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype='auto',
            device_map='auto'
        )

    def query(self, query:str):
        """Query an answer based on a question."""
        self.add_message(query)
        response = self.__get_response_from_model(self.messages)

        return response

    def query_with_documents(self, query:str, documents:list[Document]):
        """Query an answer based on a question and some documents passed as context."""
        self.add_message_with_context(query, documents)
        response = self.__get_response_from_model(self.messages)

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

class Mistral(Model):
    """Class to load Mistral AI models."""

    def __init__(self, sub_version:str=''):
        super().__init__(multimodal=False)

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
        self.add_message(query)
        response = self.__get_response_from_model(self.messages)

        return response

    def query_with_documents(self, query:str, documents:list[Document]):
        """Query an answer based on a question and some documents passed as context."""
        self.add_message_with_context(query, documents)
        response = self.__get_response_from_model(self.messages)

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
