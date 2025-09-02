from typing import List, TypedDict, Dict, Any
import re
import json
import pandas as pd

from langchain_core.messages import (
    HumanMessage, AIMessage, ToolMessage, BaseMessage, SystemMessage
)
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END

from config import get_client, OPENAI_MODEL

# ---------- OpenAI client ----------
client = get_client()

# ---------- Order Data ----------
orders_df = pd.read_csv("sample_orders.csv")

# ---------- Instrument Data ----------
instrument_df = pd.read_csv("instruments.csv")

# ---------- Tools ----------
@tool
def doc_tool(query: str = "") -> str:
    """Look up information about musical instruments from instruments.csv."""
    if not query:
        return "Please provide a query about an instrument."

    results = instrument_df[
        instrument_df["name"].str.contains(query, case=False, na=False)
        | instrument_df["description"].str.contains(query, case=False, na=False)
    ]

    if results.empty:
        return f"No instrument found matching '{query}'."

    out = []
    for _, r in results.iterrows():
        out.append(
            f"**{r['name']}**\n"
            f"- Type: {r['type']}\n"
            f"- Origin: {r['origin']}\n"
            f"- Description: {r['description']}"
        )
    return "\n\n".join(out)


@tool
def tracking_tool(customer_id: str = "") -> str:
    """Return a JSON-style summary of all orders for the given customer_id."""
    if not customer_id:
        return "Please provide a customer_id (e.g., 'CUST123')."

    rows = orders_df.loc[orders_df["customer_id"].astype(str).str.upper() == customer_id.upper()]
    if rows.empty:
        return f"No orders found for customer '{customer_id}'."

    orders = []
    for _, r in rows.iterrows():
        orders.append({
            "order_id": r["order_id"],
            "status": r["order_status"],
            "purchased": str(r["order_purchase_timestamp"]),
            "approved": str(r["order_approved_at"]),
            "handed_to_carrier": str(r["order_delivered_carrier_date"]),
            "delivered_to_customer": str(r["order_delivered_customer_date"]),
            "estimated_delivery": str(r["order_estimated_delivery_date"]),
        })

    summary = {
        "customer_id": customer_id,
        "order_count": len(orders),
        "orders": orders
    }
    return json.dumps(summary, indent=2)


# ---------- Tool Schemas ----------
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
                    "query": {"type": "string", "description": "Instrument name or keyword."}
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
    raw_messages = []

    for m in messages:
        if isinstance(m, SystemMessage):
            raw_messages.append({"role": "system", "content": m.content})
        elif isinstance(m, HumanMessage):
            raw_messages.append({"role": "user", "content": m.content})
        elif isinstance(m, AIMessage):
            entry = {"role": "assistant", "content": m.content}
            if getattr(m, "tool_calls", None):
                entry["tool_calls"] = [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {"name": tc["name"], "arguments": json.dumps(tc["args"])},
                    }
                    for tc in m.tool_calls
                ]
            raw_messages.append(entry)
        elif isinstance(m, ToolMessage):
            raw_messages.append({
                "role": "tool",
                "tool_call_id": getattr(m, "tool_call_id", None),
                "content": m.content,
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
    if choice.tool_calls:
        for tc in choice.tool_calls:
            args = {}
            try:
                args = json.loads(tc.function.arguments or "{}")
            except Exception:
                pass
            tool_calls.append({
                "id": tc.id,
                "name": tc.function.name,
                "args": args,
            })

    return AIMessage(content=choice.content or "", tool_calls=tool_calls)


# ---------- Agents ----------
ORDER_SYS_PROMPT = (
    "You are OrderTrackingAgent. "
    "Always call 'tracking_tool' exactly once with the customer_id if present. "
    "If missing, call it empty to request it. "
    "Use the tool result (JSON with order_count and orders) to answer user questions. "
    "When users ask follow-up questions (like delivery date, status, count), "
    "refer back to the tool result instead of asking again."
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
        ai = msgs[-1]
        results = []
        for call in getattr(ai, "tool_calls", []) or []:
            if call["name"] == "tracking_tool":
                res = tracking_tool.invoke({"customer_id": call.get("args", {}).get("customer_id", "")})
                results.append(ToolMessage(content=str(res), tool_call_id=call["id"]))
        return {"messages": msgs + results}

    def needs_tool(state: AgentState):
        last = state["messages"][-1]
        return "tools" if isinstance(last, AIMessage) and getattr(last, "tool_calls", None) else END

    g = StateGraph(AgentState)
    g.add_node("model", model_node)
    g.add_node("tools", tool_node)
    g.add_edge(START, "model")
    g.add_conditional_edges("model", needs_tool, {"tools": "tools", END: END})
    g.add_edge("tools", "model")
    return g.compile()


GENERAL_SYS_PROMPT = (
    "You are GeneralQueryAgent. "
    "Answer only using the 'doc_tool' which fetches from instruments.csv. "
    "Do not answer directly or use outside knowledge. "
    "Always call 'doc_tool' once with the instrument name or query."
)

def build_general_query_agent():
    def model_node(state: AgentState):
        ai = call_openai(
            [SystemMessage(content=GENERAL_SYS_PROMPT)] + state["messages"],
            tools=[OPENAI_TOOLS["doc_tool"]],
        )
        return {"messages": state["messages"] + [ai]}

    def tool_node(state: AgentState):
        msgs = state["messages"]
        ai = msgs[-1]
        results = []
        for call in getattr(ai, "tool_calls", []) or []:
            if call["name"] == "doc_tool":
                q = call.get("args", {}).get("query", msgs[-2].content if len(msgs) > 1 else "")
                res = doc_tool.invoke({"query": q})
                results.append(ToolMessage(content=res, tool_call_id=call["id"]))
        return {"messages": msgs + results}

    def needs_tool(state: AgentState):
        last = state["messages"][-1]
        return "tools" if isinstance(last, AIMessage) and getattr(last, "tool_calls", None) else END

    g = StateGraph(AgentState)
    g.add_node("model", model_node)
    g.add_node("tools", tool_node)
    g.add_edge(START, "model")
    g.add_conditional_edges("model", needs_tool, {"tools": "tools", END: END})
    g.add_edge("tools", "model")
    return g.compile()


# ---------- Supervisor ----------
order_agent = build_order_tracking_agent()
general_agent = build_general_query_agent()
CUST_PATTERN = re.compile(r"\bCUST[0-9A-Z_-]+\b", flags=re.I)

def supervisor_router(state: AgentState) -> str:
    """Route: orders -> order_agent, everything else -> general_agent."""
    recent_msgs = state["messages"][-8:]
    combined = " ".join([m.content or "" for m in recent_msgs])

    last_ai = next((m for m in reversed(recent_msgs) if isinstance(m, AIMessage)), None)
    last_tool = next((m for m in reversed(recent_msgs) if isinstance(m, ToolMessage)), None)

    if CUST_PATTERN.search(combined):
        return "order_agent"
    if last_ai and getattr(last_ai, "tool_calls", None):
        for tc in last_ai.tool_calls:
            if tc.get("name") == "tracking_tool":
                return "order_agent"
    if last_tool and "Order" in last_tool.content:
        return "order_agent"
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

# ---------- REPL ----------
if __name__ == "__main__":
    print("Supervisor Agent ready! Type 'exit' to quit.\n")

    history = []

    while True:
        user_msg = input("You: ")
        if user_msg.lower() in {"exit", "quit", "q"}:
            print("Goodbye 👋")
            break

        history.append(HumanMessage(content=user_msg))
        out = supervisor_agent.invoke({"messages": history})
        print("Agent:", out["messages"][-1].content, "\n")
        history = out["messages"]
