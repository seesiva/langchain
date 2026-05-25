"""Multi-agent city brief team - graph entry point for LangGraph Studio.

Exposes a top-level `app` symbol that `langgraph.json` points at.
The dev server attaches its own checkpointer at runtime, so we do not
attach one here.
"""

from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, Field
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent


WEATHER_DATA = {
    "new york": "72F, Sunny",
    "london": "58F, Cloudy",
    "tokyo": "68F, Rainy",
}

POPULATION_DATA = {
    "new york": "8.3 million",
    "london": "8.9 million",
    "tokyo": "13.9 million",
}


@tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return WEATHER_DATA.get(city.lower(), f"No weather data for {city}")


@tool
def get_population(city: str) -> str:
    """Get population of a city."""
    return POPULATION_DATA.get(city.lower(), f"No population data for {city}")


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    next: str


class Route(BaseModel):
    """Supervisor's structured routing decision."""

    next: Literal["researcher", "writer", "FINISH"] = Field(
        description="Who acts next. FINISH only when the user's request is fully delivered."
    )


_supervisor_prompt = (
    "You are a supervisor managing a team of workers: researcher, writer. "
    "The researcher uses tools to look up factual data. "
    "The writer drafts a short paragraph using facts already in the conversation. "
    "Given the conversation so far, decide who should act next. "
    "Respond with FINISH only when the user's original request has been fully delivered."
)

_supervisor_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(Route)


def supervisor(state: State) -> dict:
    messages = [SystemMessage(content=_supervisor_prompt)] + state["messages"]
    decision = _supervisor_llm.invoke(messages)
    return {"next": decision.next}


_researcher_agent = create_react_agent(
    ChatOpenAI(model="gpt-4o-mini", temperature=0),
    tools=[get_weather, get_population],
    prompt=(
        "You are a researcher. Use the tools to gather the facts requested. "
        "Report the facts plainly. Do not write a paragraph. Do not editorialize."
    ),
)


def researcher(state: State) -> dict:
    result = _researcher_agent.invoke({"messages": state["messages"]})
    last = result["messages"][-1]
    return {"messages": [HumanMessage(content=last.content, name="researcher")]}


_writer_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.4)
_writer_prompt = (
    "You are a writer. Using only the facts already in the conversation, "
    "draft a single short paragraph (2-3 sentences) that answers the user's original request. "
    "Do not invent facts. Do not ask follow-up questions."
)


def writer(state: State) -> dict:
    messages = [SystemMessage(content=_writer_prompt)] + state["messages"]
    result = _writer_llm.invoke(messages)
    return {"messages": [HumanMessage(content=result.content, name="writer")]}


def _route_from_supervisor(state: State) -> str:
    decision = state["next"]
    if decision == "FINISH":
        return END
    return decision


_workflow = StateGraph(State)
_workflow.add_node("supervisor", supervisor)
_workflow.add_node("researcher", researcher)
_workflow.add_node("writer", writer)

_workflow.add_edge(START, "supervisor")
_workflow.add_conditional_edges(
    "supervisor",
    _route_from_supervisor,
    {"researcher": "researcher", "writer": "writer", END: END},
)
_workflow.add_edge("researcher", "supervisor")
_workflow.add_edge("writer", "supervisor")

app = _workflow.compile()
