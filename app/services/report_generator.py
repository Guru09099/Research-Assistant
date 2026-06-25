from typing import List


def generate_report(question: str, steps: List[str], results: List[dict]) -> str:
    report_lines = [
        "# Research Report\n",
        f"**Question:** {question}\n",
        "## Research Plan\n",
    ]
    for idx, step in enumerate(steps, start=1):
        report_lines.append(f"{idx}. {step}\n")

    report_lines.append("## Findings\n")
    for result in results:
        report_lines.append(f"### {result.get('source', 'tool')}\n")
        report_lines.append(f"**Summary:** {result.get('summary')}\n")
        report_lines.append(f"**Details:**\n{result.get('details')}\n")

    report_lines.append("## Conclusion\n")
    report_lines.append(
        "Research findings were aggregated from available web search results, "
        "PDF knowledge stores, and memory evidence.\n"
    )
    return "\n".join(report_lines)
