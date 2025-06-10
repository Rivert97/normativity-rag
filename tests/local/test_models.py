import unittest

from llms.models import QwenBuilder, GemmaBuilder, LlamaBuilder

class TestModels(unittest.TestCase):

    def test_qwen3_6M(self):
        model = QwenBuilder.build_from_variant("3-0.6B")
        response = model.query("Qué es un LLM?")

        self.assertIsInstance(response, str, "Model response is not a string")
        self.assertTrue(len(response) > 0, "Model response is empty")

    def test_gemma3_1B(self):
        model = GemmaBuilder.build_from_variant("3-1b-it")
        response = model.query("Qué es un LLM?")

        self.assertIsInstance(response, str, "Model response is not a string")
        self.assertTrue(len(response) > 0, "Model response is empty")

    def test_gemma3_4B(self):
        model = GemmaBuilder.build_from_variant("3-4b-it")
        response = model.query("Qué es un LLM?")

        self.assertIsInstance(response, str, "Model response is not a string")
        self.assertTrue(len(response) > 0, "Model response is empty")

    def test_gemma3_1B_qat(self):
        model = GemmaBuilder.build_from_variant("1b-it-qat-q4_0")
        response = model.query("Qué es un LLM?")

        self.assertIsInstance(response, str, "Model response is not a string")
        self.assertTrue(len(response) > 0, "Model response is empty")

    def test_llama32_1B(self):
        model = LlamaBuilder.build_from_variant("3.2-1B-Instruct")
        response = model.query("Qué es un LLM?")

        self.assertIsInstance(response, str, "Model response is not a string")
        self.assertTrue(len(response) > 0, "Model response is empty")

    def test_llama31_8B(self):
        model = LlamaBuilder.build_from_variant("3.1-8B-Instruct")
        response = model.query("Qué es un LLM?")

        self.assertIsInstance(response, str, "Model response is not a string")
        self.assertTrue(len(response) > 0, "Model response is empty")

if __name__ == '__main__':
    unittest.main()