import unittest
from unittest.mock import patch

from app.agent import ResearchAgent


class ResearchAgentLangChainTests(unittest.TestCase):
    def test_langchain_mode_falls_back_to_existing_workflow(self) -> None:
        agent = ResearchAgent(use_langchain=True)

        with patch.object(agent, "_llm_complete", side_effect=lambda prompt: "1. Review the topic"):
            result = agent.run(
                "Test question",
                use_web_search=False,
                use_pdf_search=False,
                use_memory=False,
            )

        self.assertEqual(result["question"], "Test question")
        self.assertEqual(result["plan"], ["Review the topic"])
        self.assertTrue(agent.use_langchain)


if __name__ == "__main__":
    unittest.main()
