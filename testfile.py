from typing import List, TypedDict, Dict, Any
import re
import json
from pymongo import MongoClient, errors   # <-- MongoDB only

from langchain_core.messages import (
    HumanMessage, AIMessage, ToolMessage, BaseMessage, SystemMessage
)
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END

from config import get_client, OPENAI_MODEL

# ---------- OpenAI client ----------
client = get_client()

# ---------- MongoDB Client ----------
try:
    mongo_client = MongoClient("mongodb://localhost:27017", serverSelectionTimeoutMS=2000)
    mongo_client.admin.command("ping")  # test connection
    orders_coll = mongo_client["shop"]["orders"]
    print("✅ Connected to MongoDB (shop.orders)")
except errors.ServerSelectionTimeoutError:
    raise RuntimeError("❌ Could not connect to MongoDB at localhost:27017. Please ensure MongoDB is running.")

# ---------- Tools ----------
@tool
def doc_tool(query: str = "") -> str:
    """Answer music-related questions using the LLM only."""
    q = (query or "").strip()
    if not q:
        return "Please provide a music-related question."

    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "You are MusicQueryAgent. Only answer questions about music, musicians, or instruments."},
            {"role": "user", "content": q}
        ],
        temperature=0.7,
    )
    return resp.choices[0].message.content


@tool
def tracking_tool(customer_id: str = "") -> str:
    """Return a JSON-style summary of all orders for the given customer_id."""
    cid = (customer_id or "").strip().upper()
    if not cid:
        return "Please provide a customer_id (e.g., 'CUST123')."

    docs = list(orders_coll.find({"customer_id": cid}))
    if not docs:
        return json.dumps({"customer_id": cid, "order_count": 0, "orders": []}, indent=2)

    orders = []
    for d in docs:
        orders.append({
            "order_id": d.get("order_id"),
            "status": d.get("order_status"),
            "items": d.get("items", []),   # <-- items field comes directly from Mongo
            "purchased": str(d.get("order_purchase_timestamp")),
            "approved": str(d.get("order_approved_at")),
            "handed_to_carrier": str(d.get("order_delivered_carrier_date")),
            "delivered_to_customer": str(d.get("order_delivered_customer_date")),
            "estimated_delivery": str(d.get("order_estimated_delivery_date")),
        })

    summary = {"customer_id": cid, "order_count": len(orders), "orders": orders}
    return json.dumps(summary, indent=2)

# ---------- Tool Schemas (for OpenAI) ----------
OPENAI_TOOLS: Dict[str, Dict[str, Any]] = {
    "tracking_tool": {
        "type": "function",
        "function": {
            "name": "tracking_tool",
            "description": tracking_tool.__doc__ or "",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_id": {"type": "string", "description": "Customer ID like 'CUST123'."}
                },
                "required": []
            },
        },
    },
    "doc_tool": {
        "type": "function",
        "function": {
            "name": "doc_tool",
            "description": doc_tool.__doc__ or "",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Music-related query."}
                },
                "required": ["query"],
            },
        },
    },
}

# ---------- Shared State ----------
class AgentState(TypedDict):
    messages: List[BaseMessage]

# ---------- OpenAI Call Helper ----------
def call_openai(messages: List[BaseMessage], tools: List[Dict[str, Any]]):
    """Call OpenAI with messages and tool definitions. Handles tool responses properly."""
    raw_messages: List[Dict[str, Any]] = []

    for m in messages:
        if isinstance(m, SystemMessage):
            raw_messages.append({"role": "system", "content": m.content})
        elif isinstance(m, HumanMessage):
            raw_messages.append({"role": "user", "content": m.content})
        elif isinstance(m, AIMessage):
            entry: Dict[str, Any] = {"role": "assistant", "content": m.content or ""}
            if getattr(m, "tool_calls", None):
                entry["tool_calls"] = [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc.get("args", {}))
                        },
                    }
                    for tc in m.tool_calls
                ]
            raw_messages.append(entry)
        elif isinstance(m, ToolMessage):
            raw_messages.append({
                "role": "tool",
                "tool_call_id": getattr(m, "tool_call_id", None),
                "content": m.content or "",
            })

    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=raw_messages,
        temperature=0,
        tools=tools,
        tool_choice="auto",
    )

    choice = resp.choices[0].message
    tool_calls = []
    if getattr(choice, "tool_calls", None):
        for tc in choice.tool_calls:
            args = {}
            try:
                args = json.loads(tc.function.arguments or "{}")
            except Exception:
                args = {}
            tool_calls.append({
                "id": tc.id,
                "name": tc.function.name,
                "args": args,
            })

    return AIMessage(content=choice.content or "", tool_calls=tool_calls)

# ---------- Agents ----------
ORDER_SYS_PROMPT = (
    "You are OrderTrackingAgent. "
    "If you need order info, call 'tracking_tool' with the provided customer_id. "
    "If the customer_id is missing, ask for it. "
    "Use the tool result (JSON with order_count and orders) to answer follow-up questions "
    "without calling the tool again unless a new customer_id is provided."
)

def build_order_tracking_agent():
    def model_node(state: AgentState):
        ai = call_openai(
            [SystemMessage(content=ORDER_SYS_PROMPT)] + state["messages"],
            tools=[OPENAI_TOOLS["tracking_tool"]],
        )
        return {"messages": state["messages"] + [ai]}

    def tool_node(state: AgentState):
        msgs = state["messages"]
        if not msgs:
            return {"messages": msgs}
        ai = msgs[-1]
        results: List[ToolMessage] = []
        for call in (getattr(ai, "tool_calls", []) or []):
            if call.get("name") == "tracking_tool":
                res = tracking_tool.invoke({"customer_id": call.get("args", {}).get("customer_id", "")})
                results.append(ToolMessage(content=str(res), tool_call_id=call["id"]))
        return {"messages": msgs + results}

    def needs_tool(state: AgentState):
        if not state["messages"]:
            return "end"
        last = state["messages"][-1]
        return "tools" if isinstance(last, AIMessage) and getattr(last, "tool_calls", None) else "end"

    g = StateGraph(AgentState)
    g.add_node("model", model_node)
    g.add_node("tools", tool_node)
    g.add_edge(START, "model")
    g.add_conditional_edges("model", needs_tool, {"tools": "tools", "end": END})
    g.add_edge("tools", "model")
    return g.compile()

GENERAL_SYS_PROMPT = (
    "You are GeneralQueryAgent. "
    "Answer only using the 'doc_tool'. "
    "Do not answer anything else than music-related. "
    "Always call 'doc_tool' with the music-related question."
)

def _latest_user_text(messages: List[BaseMessage]) -> str:
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            return m.content or ""
    return ""

def build_general_query_agent():
    def model_node(state: AgentState):
        ai = call_openai(
            [SystemMessage(content=GENERAL_SYS_PROMPT)] + state["messages"],
            tools=[OPENAI_TOOLS["doc_tool"]],
        )
        return {"messages": state["messages"] + [ai]}

    def tool_node(state: AgentState):
        msgs = state["messages"]
        if not msgs:
            return {"messages": msgs}
        ai = msgs[-1]
        results: List[ToolMessage] = []
        for call in (getattr(ai, "tool_calls", []) or []):
            if call.get("name") == "doc_tool":
                q = call.get("args", {}).get("query", "").strip()
                if not q:
                    q = _latest_user_text(msgs).strip()
                res = doc_tool.invoke({"query": q})
                results.append(ToolMessage(content=str(res), tool_call_id=call["id"]))
        return {"messages": msgs + results}

    def needs_tool(state: AgentState):
        if not state["messages"]:
            return "end"
        last = state["messages"][-1]
        return "tools" if isinstance(last, AIMessage) and getattr(last, "tool_calls", None) else "end"

    g = StateGraph(AgentState)
    g.add_node("model", model_node)
    g.add_node("tools", tool_node)
    g.add_edge(START, "model")
    g.add_conditional_edges("model", needs_tool, {"tools": "tools", "end": END})
    g.add_edge("tools", "model")
    return g.compile()

# ---------- Supervisor ----------
order_agent = build_order_tracking_agent()
general_agent = build_general_query_agent()
CUST_PATTERN = re.compile(r"\bCUST[0-9A-Z_-]+\b", flags=re.I)

def supervisor_router(state: AgentState) -> str:
    """Route: orders -> order_agent, everything else -> general_agent."""
    recent = state["messages"][-8:]
    combined = " ".join([(m.content or "") for m in recent if hasattr(m, "content")])

    if CUST_PATTERN.search(combined):
        return "order_agent"

    for m in reversed(recent):
        if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
            if any(tc.get("name") == "tracking_tool" for tc in m.tool_calls):
                return "order_agent"
            break

    return "general_agent"

g = StateGraph(AgentState)
g.add_node("supervisor", lambda s: s)
g.add_node("order_agent", order_agent)
g.add_node("general_agent", general_agent)
g.add_edge(START, "supervisor")
g.add_conditional_edges("supervisor", supervisor_router, {
    "order_agent": "order_agent",
    "general_agent": "general_agent",
})
g.add_edge("order_agent", END)
g.add_edge("general_agent", END)
supervisor_agent = g.compile()

# ---------- Graph Visualization ----------
def _save_graph_images():
    try:
        supervisor_agent.get_graph().draw_png("supervisor_graph.png")
        order_agent.get_graph().draw_png("order_agent_graph.png")
        general_agent.get_graph().draw_png("general_agent_graph.png")
        print("Saved graph PNGs: supervisor_graph.png, order_agent_graph.png, general_agent_graph.png")
    except Exception as e:
        with open("supervisor_graph.mmd", "w", encoding="utf-8") as f:
            f.write(supervisor_agent.get_graph().draw_mermaid())
        with open("order_agent_graph.mmd", "w", encoding="utf-8") as f:
            f.write(order_agent.get_graph().draw_mermaid())
        with open("general_agent_graph.mmd", "w", encoding="utf-8") as f:
            f.write(general_agent.get_graph().draw_mermaid())
        print("Graphviz not available; wrote Mermaid files instead: *.mmd")
        print(f"(error was: {e})")

# ---------- REPL ----------
if __name__ == "__main__":
    _save_graph_images()
    print("Supervisor Agent ready! Type 'exit' to quit.\n")

    history: List[BaseMessage] = []

    while True:
        user_msg = input("You: ")
        if user_msg.lower() in {"exit", "quit", "q"}:
            print("Goodbye 👋")
            break

        history.append(HumanMessage(content=user_msg))
        out = supervisor_agent.invoke({"messages": history})
        latest = out["messages"][-1]
        print("Agent:", getattr(latest, "content", "") or "", "\n")
        history = out["messages"]
