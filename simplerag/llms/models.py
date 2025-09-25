"""Module to define multiple types of models provided by HuggingFace."""

from abc import ABC, abstractmethod
from enum import Enum
import os

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
from transformers import BitsAndBytesConfig, Gemma3ForConditionalGeneration, Gemma3ForCausalLM
import torch
from llama_cpp import Llama

from .data import Document

# It's needed to run in the RTX4000
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)

class Model(ABC):
    """Base class for all the models."""

    def __init__(self, multimodal:bool=False):
        self.multimodal = multimodal

        self.messages = self.__get_init_messages()

    def query(self, query:str, add_to_history:bool=True):
        """Query an answer based on a question."""
        messages = self.str_to_message(query)

        response = self.get_response_from_model(messages)

        if add_to_history:
            self.messages += messages + self.response_to_message(response)

        return response

    def query_with_documents(self, query:str, documents:list[Document], add_to_history:bool=True):
        """Query an answer based on a question and some documents passed as context."""
        messages = self.str_to_message_with_context(query, documents)

        response = self.get_response_from_model(messages)

        if add_to_history:
            self.messages += self.str_to_message(query) + self.response_to_message(response)

        return response

    def query_with_conversation(self, messages:list[dict[str, str]]) -> str:
        """Query an answer based on a full conversation."""
        response = self.get_response_from_model(messages)

        return self.response_to_message(response)

    def query_with_conversation_and_documents(self, messages:list[dict[str, str]],
                                              documents:list[Document]) -> str:
        """
        Query an answer based on a full conversation and some documents passed as context to
        the last question.
        """
        last_query = messages[-1]['content']
        messages = messages[:-1] + self.str_to_message_with_context(last_query, documents)
        response = self.get_response_from_model(messages)

        return self.response_to_message(response)

    def str_to_message(self, query:str):
        """Add a new message to be sent to the model."""
        if self.multimodal:
            message = self.__get_multimodal_message(query)
        else:
            message = self.__get_text_message(query)

        return [message]

    def str_to_message_with_context(self, query:str, documents:list[Document]):
        """Add a new message to be sent to the model including documents to be used as context."""
        if self.multimodal:
            messages = self.__get_multimodal_message_with_context(query, documents)
        else:
            messages = self.__get_text_message_with_context(query, documents)

        return messages

    def response_to_message(self, response:str):
        """Add response to the history of conversation."""
        response = {
            "role": "assistant",
            "content": response,
        }

        return [response]

    @abstractmethod
    def get_response_from_model(self, messages: list[dict[str, str]]) -> str:
        """Calls the model inference method and returns an answer."""

    def __get_init_messages(self) -> list[dict[str:str|dict]]:
        instruction = 'Eres un asistente que ayuda a responder preguntas.'
        if self.multimodal:
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": instruction}],
                },
            ]
        else:
            messages = [
                {
                    "role": "system",
                    "content": instruction,
                },
            ]

        return messages

    def __get_multimodal_message(self, query:str):
        new_message = {
            "role": "user",
            "content": [{"type": "text", "text": query}]
        }

        return new_message

    def __get_text_message(self, query:str):
        new_message = {
            "role": "user",
            "content": query
        }

        return new_message

    def __get_multimodal_message_with_context(self, query:str, documents:list[Document]):
        documents_context = self.__format_documents(documents)

        new_messages = [
            {
                "role": "system",
                "content": [{
                    "type": "text",
                    "text": "Responde la siguiente pregunta. Utiliza únicamente los "\
                            f"fragmentos de documentos siguientes:\n\n{documents_context}"
                }]
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": query}]
            }
        ]

        return new_messages

    def __get_text_message_with_context(self, query:str, documents:list[Document]):
        documents_context = self.__format_documents(documents)

        new_messages = [
            {
                "role": "system",
                "content": "Responde la siguiente pregunta. Utiliza únicamente los "\
                           f"fragmentos de documentos siguientes:\n\n{documents_context}"
            },
            {
                "role": "user",
                "content": query
            }
        ]

        return new_messages

    def __format_documents(self, documents:list[Document]) -> str:
        docs_str = ''
        for doc in documents:
            docs_str += f"{doc.get_reference()}:\n\n{doc.get_metadata()['title']}\n\n"\
                        f"{doc.get_content()}\n\n"

        return docs_str

class ModelBuilder:
    """Factory class to create the different types of models"""

    @classmethod
    def get_from_id(cls, model_id:str) -> Model:
        """Return an object of the corresponding class depending of the type of model."""
        if model_id == '':
            return None

        id_parts = model_id.split('/')
        full_name = id_parts[-1]

        full_name_parts = full_name.split('-')
        name = full_name_parts[0].upper()
        try:
            return Models[name].value(model_id)
        except KeyError:
            return None

    @classmethod
    def get_from_gguf_file(cls, gguf_file:str) -> Model:
        """Return an object of the corresponding class to use a GGUF model"""
        if not os.path.exists(gguf_file):
            return None

        return GGUFModel(gguf_file)

class Llama3(Model):
    """Class to load Meta Llama 3.1 and 3.2 model."""

    def __init__(self, model_id:str):
        super().__init__(multimodal=True)

        self.model_id = model_id

        tokenizer = AutoTokenizer.from_pretrained(self.model_id, legacy=False)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=self.model_id,
            torch_dtype=torch.bfloat16,
            tokenizer=tokenizer,
            device_map="auto"
        )

    def get_response_from_model(self, messages:list[dict[str, str]]) -> str:
        all_messages = self.messages + messages
        output = self.pipeline(all_messages, max_new_tokens=1024)

        response = output[0].get('generated_text')[-1].get('content', '')

        return response

class Gemma(Model):
    """Class to load Gemma3 model."""

    def __init__(self, model_id: str):
        multimodal = not(model_id.endswith('1b-it') or model_id.endswith('-gguf'))
        super().__init__(multimodal=multimodal)

        self.model_id = model_id
        self.processor = None
        self.tokenizer = None

        if self.model_id.endswith('-gguf'):
            gguf_file = self.model_id.rsplit('/', maxsplit=1)[-1].replace('-gguf',
                                                                          '.gguf').replace('-qat',
                                                                                           '')
            quantization_config = None
            device_map=None # Can't run qantized models in GPU for now
        else:
            gguf_file = None
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            device_map='auto'

        if self.multimodal:
            self.model = Gemma3ForConditionalGeneration.from_pretrained(
                self.model_id, device_map=device_map
            ).eval()
            self.processor = AutoProcessor.from_pretrained(self.model_id, use_fast=True)
        else:
            self.model = Gemma3ForCausalLM.from_pretrained(
                self.model_id, quantization_config=quantization_config, gguf_file=gguf_file,
                device_map=device_map
            ).eval()
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, gguf_file=gguf_file)

    def get_response_from_model(self, messages):
        if self.multimodal:
            response = self.__process_multimodal(messages)
        else:
            response = self.__process_text(messages)

        return response

    def __process_multimodal(self, messages:list[dict]):
        all_messages = self.messages + messages
        inputs = self.processor.apply_chat_template(
            all_messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt"
        ).to(self.model.device, dtype=torch.bfloat16)

        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = self.model.generate(**inputs, max_new_tokens=1024, do_sample=False)
            generation = generation[0][input_len:]

        decoded = self.processor.decode(generation, skip_special_tokens=True)

        return decoded

    def __process_text(self, messages:list[dict]):
        all_messages = self.messages + messages
        inputs = self.tokenizer.apply_chat_template(
            all_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.inference_mode():
            outputs = self.model.generate(**inputs, max_new_tokens=1024)

        decoded = self.tokenizer.batch_decode(outputs)
        response = decoded[0]

        bos = response.find('<bos>')
        bos_pos = bos + 5 if bos > -1 else 0
        eos = response.find('<eos>')
        eos_pos = eos if eos > -1 else None
        response = response[bos_pos:eos_pos]

        sot = response.find('<start_of_turn>model\n')
        sot_pos = sot + 21 if sot > -1 else 0
        eot = response.find('<end_of_turn>', sot_pos)
        eot_pos = eot if eot > -1 else None
        response = response[sot_pos:eot_pos]

        return response

class Qwen3(Model):
    """Class to load Qwen models."""

    def __init__(self, model_id:str, thinking:bool=False):
        super().__init__(multimodal=False)
        self.model_id = model_id

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype='auto',
            device_map='auto',
        )

        self.thinking = thinking

    def get_response_from_model(self, messages:list[dict[str, str]]) -> str:
        all_messages = self.messages + messages
        text = self.tokenizer.apply_chat_template(
            all_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.thinking, # Switches between thinking and non-thinking modes
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        print("Tokens input:", len(model_inputs['input_ids'][0]))

        # Conduct text completion
        generated_tokens = self.model.generate(
            **model_inputs,
            max_new_tokens=1024
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
        print("Tokens output:", len(output_ids[index:]))
        content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

        return content

class Mistral(Model):
    """Class to load Mistral AI models."""

    def __init__(self, model_id:str):
        super().__init__(multimodal=False)

        self.model_id = model_id

        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=self.model_id,
            torch_dtype=torch.bfloat16,
            tokenizer=tokenizer,
            device_map="auto"
        )

    def get_response_from_model(self, messages:list[dict[str, str]]) -> str:
        all_messages = self.messages + messages
        output = self.pipeline(all_messages, max_new_tokens=1024)

        response = output[0].get('generated_text')[-1].get('content', '')

        return response

class GPT(Model):
    """Class to load GPT-OSS models."""

    def __init__(self, model_id:str):
        super().__init__(multimodal=False)

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map="auto"
        )

    def get_response_from_model(self, messages:list[dict[str, str]]) -> str:
        all_messages = self.messages + messages

        inputs = self.tokenizer.apply_chat_template(
            all_messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.7
        )

        content = self.tokenizer.decode(outputs[0])

        return content

class GGUFModel(Model):
    """Class to load models with GGUF format."""

    def __init__(self, ggu_file:str, thinking:bool=False):
        super().__init__(multimodal=False)
        self.thinking = thinking

        self.model = Llama(
            model_path=ggu_file,
            embedding=False,
            n_gpu_layers=-1,
            n_ctx=8192,
            verbose=False,
        )

    def get_response_from_model(self, messages):
        all_messages = self.messages + messages
        if not self.thinking:
            all_messages = [
                {
                    'role': m['role'],
                    'content': f"/no_think {m['content']}"
                } for m in all_messages
            ]

        response = self.model.create_chat_completion(
            messages=all_messages,
            max_tokens=1024
        )
        response_str = response['choices'][0]['message']['content']
        index = response_str.find('</think>\n\n')
        if index >= 0:
            response_str = response_str[index + 10:]

        print("Tokens input:", response['usage']['prompt_tokens'])
        print("Tokens output:", response['usage']['completion_tokens'])

        return response_str

class Models(Enum):
    """Different types of models that are available."""
    QWEN3 = Qwen3
    GEMMA = Gemma
    LLAMA = Llama3
    MISTRAL = Mistral
