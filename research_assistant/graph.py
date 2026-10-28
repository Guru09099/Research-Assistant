"""LangGraph Workflow Definition for the Autonomous Research Assistant.

LangGraph Core Concepts Explained:
----------------------------------
1. StateGraph:
   The blueprint for our agent workflow. It requires a State definition (ResearchState)
   so it knows what data is passed between nodes.

2. Nodes:
   The workers of the graph. Each node is a function that receives the current state
   and returns updates.

3. Edges:
   The connections between nodes that define the execution order.
   - START: Special LangGraph entrypoint. When graph.invoke() is called, execution starts here.
   - END: Special LangGraph exitpoint. When execution reaches END, the graph stops and returns the final state.

4. compile():
   Transforms the StateGraph blueprint into a runnable object with .invoke(), .stream(), etc.
"""

from langgraph.graph import END, START, StateGraph

from research_assistant.nodes import generate_report, plan_queries, search_web
from research_assistant.state import ResearchState


def create_research_graph():
    """Builds and compiles the 3-node LangGraph workflow.

    Linear Workflow:
    START -> plan_queries -> search_web -> generate_report -> END
    """
    # 1. Initialize the StateGraph with our TypedDict schema
    builder = StateGraph(ResearchState)

    # 2. Register the three nodes
    builder.add_node("plan_queries", plan_queries)
    builder.add_node("search_web", search_web)
    builder.add_node("generate_report", generate_report)

    # 3. Connect the nodes in sequence with edges
    # START -> plan_queries
    builder.add_edge(START, "plan_queries")
    # plan_queries -> search_web
    builder.add_edge("plan_queries", "search_web")
    # search_web -> generate_report
    builder.add_edge("search_web", "generate_report")
    # generate_report -> END
    builder.add_edge("generate_report", END)

    # 4. Compile the graph into an executable runnable
    return builder.compile()


# Create a single compiled instance ready to run
research_graph = create_research_graph()

