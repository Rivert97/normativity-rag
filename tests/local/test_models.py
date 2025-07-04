"""Module to add unit tests for LLM models.

These tests require a medium GPU to run and may take too long.
"""
import unittest

#pylint: disable=wrong-import-position, import-error

from llms.models import QwenBuilder, GemmaBuilder, LlamaBuilder, MistralBuilder
from llms.rag import RAGQueryConfig

#pylint: enable=wrong-import-position, import-error

class TestModels(unittest.TestCase):
    """Class for unittesting the models available.

    Only methods to test smaller and one of the big models are available to avoid
    too long tests.
    """
    def setUp(self):
        self.query_config = RAGQueryConfig(
            collection='',
            num_docs=5,
            max_distance=1.0,
            add_to_history=False
        )

    def test_qwen_3_6m(self):
        """Test for Qwen3-0.6B model. Smallest model for Qwen."""
        model = QwenBuilder.build_from_variant("3-0.6B")
        response = model.query("Qué es un LLM?", self.query_config)

        self.assertIsInstance(response, str, "Model response is not a string")
        self.assertTrue(len(response) > 0, "Model response is empty")

    def test_gemma_3_1b(self):
        """Test for Gemma-3-1b-it model. Smallest model for Gemma."""
        model = GemmaBuilder.build_from_variant("3-1b-it")
        response = model.query("Qué es un LLM?", self.query_config)

        self.assertIsInstance(response, str, "Model response is not a string")
        self.assertTrue(len(response) > 0, "Model response is empty")

    def test_gemma_3_4b(self):
        """Test for Gemma-3-4b-it model. 2nd bigger model for Gemma."""
        model = GemmaBuilder.build_from_variant("3-4b-it")
        response = model.query("Qué es un LLM?", self.query_config)

        self.assertIsInstance(response, str, "Model response is not a string")
        self.assertTrue(len(response) > 0, "Model response is empty")

    def test_gemma_3_1b_qat(self):
        """Test for Gemma-3-1b-it-qat-q4_0 model. Smallest qantized model for Gemma."""
        model = GemmaBuilder.build_from_variant("3-1b-it-qat-q4_0-gguf")
        response = model.query("Qué es un LLM?", self.query_config)

        self.assertIsInstance(response, str, "Model response is not a string")
        self.assertTrue(len(response) > 0, "Model response is empty")

    def test_llama_32_1b(self):
        """Test for Llama-3.2-1B-Instruct model. Smallest model for Llama."""
        model = LlamaBuilder.build_from_variant("3.2-1B-Instruct")
        response = model.query("Qué es un LLM?", self.query_config)

        self.assertIsInstance(response, str, "Model response is not a string")
        self.assertTrue(len(response) > 0, "Model response is empty")

    def test_llama_31_8b(self):
        """Test for Llama-3.1-8B-Instruct model. Biggest model for Llama."""
        model = LlamaBuilder.build_from_variant("3.1-8B-Instruct")
        response = model.query("Qué es un LLM?", self.query_config)

        self.assertIsInstance(response, str, "Model response is not a string")
        self.assertTrue(len(response) > 0, "Model response is empty")

    def test_mistral_7b(self):
        """Test for Mistral-7b-instruct-v0.3 model."""
        model = MistralBuilder.build_from_variant("7b-instruct-v0.3")
        response = model.query("Qué es un LLM?", self.query_config)

        self.assertIsInstance(response, str, "Model response is not a string")
        self.assertTrue(len(response) > 0, "Model response is empty")

if __name__ == '__main__':
    unittest.main()
