from typing import Any, Dict, List

import openai
from openai import OpenAI

from .config import settings
from .schemas import ToolResult
from .tools import build_langchain_tools, query_web, search_memory, search_pdf


class ResearchAgent:
    def __init__(self, use_langchain: bool = False) -> None:
        self.use_langchain = use_langchain
        self._langchain_chain = None
        self._langchain_executor = None
        self._langchain_tools = []
        openai.api_key = settings.openai_api_key or settings.gemini_api_key
        self._initialize_langchain()

    def _initialize_langchain(self) -> None:
        if not self.use_langchain:
            return

        try:
            from langchain.agents import AgentExecutor, create_tool_calling_agent
            from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
            from langchain_openai import ChatOpenAI
        except ImportError:
            self._langchain_chain = None
            self._langchain_executor = None
            self._langchain_tools = []
            return

        if not (settings.openai_api_key or settings.gemini_api_key):
            self._langchain_chain = None
            self._langchain_executor = None
            self._langchain_tools = []
            return

        llm = ChatOpenAI(
            model=settings.llm_model,
            api_key=settings.openai_api_key or settings.gemini_api_key,
            temperature=0.2,
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an autonomous research assistant. Break tasks into "
                    "research steps, use tools when helpful, and return a "
                    "structured research report.",
                ),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

        tools = build_langchain_tools(
            use_web_search=True,
            use_pdf_search=True,
            use_memory=True,
            top_k=settings.max_search_results,
        )
        if not tools:
            self._langchain_executor = None
            self._langchain_tools = []
            return

        try:
            agent = create_tool_calling_agent(llm, tools, prompt)
            self._langchain_executor = AgentExecutor(
                agent=agent,
                tools=tools,
                verbose=False,
                handle_parsing_errors=True,
            )
            self._langchain_tools = tools
        except Exception:
            self._langchain_executor = None
            self._langchain_tools = []

    def _llm_complete(self, prompt: str) -> str:
        if not settings.openai_api_key and not settings.gemini_api_key:
            raise RuntimeError("OPENAI_API_KEY or GEMINI_API_KEY is required for the LLM agent.")

        if self._langchain_chain is not None:
            return self._langchain_chain.invoke({"prompt": prompt})

        api_key = settings.openai_api_key or settings.gemini_api_key
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY or GEMINI_API_KEY is required for the LLM agent.")

        try:
            client = OpenAI(api_key=api_key)
            response = client.responses.create(
                model=settings.llm_model,
                input=[
                    {
                        "role": "system",
                        "content": (
                            "You are an autonomous research assistant. Break tasks into "
                            "research steps, choose tools, and summarize findings."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                max_output_tokens=800,
                temperature=0.3,
            )
            output_text = getattr(response, "output_text", None)
            if output_text:
                return output_text.strip()
            if getattr(response, "output", None):
                for item in response.output:
                    if getattr(item, "type", None) == "message":
                        parts = []
                        for content in getattr(item, "content", []) or []:
                            if getattr(content, "text", None):
                                parts.append(content.text)
                        if parts:
                            return "\n".join(parts).strip()
        except Exception:
            raise RuntimeError("Unable to reach the configured LLM backend.")

        return ""

    def _decompose(self, query: str) -> List[str]:
        if self._langchain_executor is not None:
            return [query]

        prompt = (
            "Decompose the research goal into a concise plan of 3-5 subquestions or steps.\n\n"
            f"Research goal: {query}\n\n"
            "Return each step as a numbered item."
        )
        try:
            result = self._llm_complete(prompt)
        except RuntimeError:
            return [query]
        steps = []
        for line in result.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith(("1.", "2.", "3.", "4.", "5.")):
                stripped = stripped.split(".", 1)[1].strip()
            steps.append(stripped.strip(" ."))
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
        for tool_name in tools:
            if tool_name == "web":
                result = query_web(step, top_k=settings.max_search_results)
            elif tool_name == "pdf":
                result = search_pdf(step, top_k=settings.max_search_results)
            elif tool_name == "memory":
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

    def _run_with_langchain(self, query: str, plan: List[str], results: List[ToolResult]) -> str:
        if self._langchain_executor is None:
            return self._reflect(query, plan, results)

        try:
            evidence = []
            for result in results:
                evidence.append(
                    f"Source: {result.source}\nSummary: {result.summary}\nDetails: {result.details}"
                )
            payload = {
                "input": (
                    f"Research question: {query}\n\n"
                    f"Suggested plan:\n{'\n'.join(f'- {step}' for step in plan)}\n\n"
                    "Collected evidence:\n"
                    f"{'\n\n'.join(evidence) if evidence else 'No tool evidence available.'}"
                )
            }
            langchain_result = self._langchain_executor.invoke(payload)
            return str(langchain_result.get("output", langchain_result.get("result", "")))
        except Exception:
            return self._reflect(query, plan, results)

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
        final_report = self._run_with_langchain(query, plan, results)
        return {
            "question": query,
            "plan": plan,
            "results": [result.dict() for result in results],
            "final_report": final_report,
        }
