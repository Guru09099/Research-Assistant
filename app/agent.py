from typing import Any, Dict, List

import openai

from .config import settings
from .schemas import ToolResult
from .tools import query_web, search_memory, search_pdf


class ResearchAgent:
    def __init__(self, use_langchain: bool = False) -> None:
        self.use_langchain = use_langchain
        self._langchain_chain = None
        openai.api_key = settings.openai_api_key or settings.gemini_api_key
        self._initialize_langchain()

    def _initialize_langchain(self) -> None:
        if not self.use_langchain:
            return

        try:
            from langchain_core.output_parsers import StrOutputParser
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_openai import ChatOpenAI
        except ImportError:
            self._langchain_chain = None
            return

        if not settings.openai_api_key:
            self._langchain_chain = None
            return

        llm = ChatOpenAI(
            model=settings.llm_model,
            api_key=settings.openai_api_key,
            temperature=0.3,
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an autonomous research assistant. Break tasks into research steps, choose tools, and summarize findings.",
                ),
                ("human", "{prompt}"),
            ]
        )
        self._langchain_chain = prompt | llm | StrOutputParser()

    def _llm_complete(self, prompt: str) -> str:
        if not settings.openai_api_key and not settings.gemini_api_key:
            raise RuntimeError("OPENAI_API_KEY or GEMINI_API_KEY is required for the LLM agent.")

        if self._langchain_chain is not None:
            return self._langchain_chain.invoke({"prompt": prompt})

        response = openai.ChatCompletion.create(
            model=settings.llm_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an autonomous research assistant. Break tasks into "
                        "research steps, choose tools, and summarize findings."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=800,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()

    def _decompose(self, query: str) -> List[str]:
        prompt = (
            "Decompose the research goal into a concise plan of 3-5 subquestions or steps.\n\n"
            f"Research goal: {query}\n\n"
            "Return each step as a numbered item."
        )
        result = self._llm_complete(prompt)
        steps = [line.strip(" .") for line in result.splitlines() if line.strip()]
        return steps[:5] if steps else [query]

    def _select_tools(
        self,
        step: str,
        use_web_search: bool,
        use_pdf_search: bool,
        use_memory: bool,
    ) -> List[str]:
        tools = []
        if use_memory:
            tools.append("memory")
        if use_pdf_search:
            tools.append("pdf")
        if use_web_search:
            tools.append("web")
        return tools

    def _collect_tool_results(self, step: str, tools: List[str]) -> List[ToolResult]:
        results: List[ToolResult] = []
        for tool in tools:
            if tool == "web":
                result = query_web(step, top_k=settings.max_search_results)
            elif tool == "pdf":
                result = search_pdf(step, top_k=settings.max_search_results)
            elif tool == "memory":
                result = search_memory(step)
            else:
                continue
            results.append(ToolResult(**result))
        return results

    def _reflect(self, question: str, plan: List[str], results: List[ToolResult]) -> str:
        summary_parts = []
        for result in results:
            summary_parts.append(
                "Source: {source}\n"
                "Summary: {summary}\n"
                "Details: {details}".format(
                    source=result.source,
                    summary=result.summary,
                    details=result.details,
                )
            )
        summary_text = "\n\n".join(summary_parts)
        plan_lines = [f"- {step}" for step in plan]
        plan_text = "\n".join(plan_lines)
        prompt = (
            "You are a research analyst. The question is: {question}\n"
            "The execution plan was:\n{plan_text}\n\n"
            "Tool results:\n{summary_text}\n\n"
            "Create a structured research report with an executive summary, "
            "key findings, and next steps."
        ).format(question=question, plan_text=plan_text, summary_text=summary_text)
        return self._llm_complete(prompt)

    def run(
        self,
        query: str,
        use_web_search: bool = True,
        use_pdf_search: bool = True,
        use_memory: bool = True,
    ) -> Dict[str, Any]:
        plan = self._decompose(query)
        results: List[ToolResult] = []
        for step in plan:
            tools = self._select_tools(step, use_web_search, use_pdf_search, use_memory)
            results.extend(self._collect_tool_results(step, tools))
        final_report = self._reflect(query, plan, results)
        return {
            "question": query,
            "plan": plan,
            "results": [result.dict() for result in results],
            "final_report": final_report,
        }
