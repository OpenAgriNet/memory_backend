"""Build cutoff-safe reply fixtures from a recorded Amul agent's FIRST request."""
import json


def messages_from_recorded(raw):
    result = []
    for item in raw:
        role = item.get("role")
        if role not in ("system", "user", "assistant", "tool"):
            continue
        if isinstance(item.get("content"), str):
            result.append({"role": role, "content": item["content"]})
            continue
        parts = item.get("parts") or []
        text = "\n".join(str(p.get("content") or "") for p in parts if p.get("type") == "text")
        # Chat conversation history contains rendered user/assistant text. A request
        # already containing current tool calls is not a clean first-request fixture.
        if any(p.get("type") not in ("text",) for p in parts):
            raise ValueError("Non-text history in initial request needs explicit conversion")
        if text:
            result.append({"role": role, "content": text})
    return result


def prepare(trace, query):
    observations = [o for o in trace.get("observations", []) if isinstance(o, dict)]
    candidates = sorted([o for o in observations if isinstance(o.get("input"), dict)
                         and len(o["input"].get("tools") or []) >= 3
                         and isinstance(o["input"].get("messages"), list)],
                        key=lambda o: o.get("startTime") or "")
    if not candidates:
        raise ValueError("No complete Amul agent request recorded")
    first = candidates[0]
    messages = messages_from_recorded(first["input"]["messages"])
    if not messages or messages[-1]["role"] != "user":
        raise ValueError("Earliest agent request is not a pre-answer user request")
    # Earlier generations could be a different nested agent; require Amul identity.
    if not any("SarlaBen" in m["content"] or "Sarlaben" in m["content"]
               for m in messages if m["role"] == "system"):
        raise ValueError("Recorded system prompt is not the Amul agent")
    tools = []
    for tool in first["input"]["tools"]:
        fn = tool.get("function") or tool
        if fn.get("name", "").startswith(("memory_", "search_memories", "read_memory", "list_memories")):
            continue
        tools.append({"type": "function", "function": {k: fn[k] for k in
                     ("name", "description", "parameters") if k in fn}})
    fixtures = {}
    for o in sorted(observations, key=lambda o: o.get("startTime") or ""):
        if o.get("type") == "TOOL":
            fixtures.setdefault(o["name"], []).append({"input": o.get("input"), "output": o.get("output")})
    translated = query
    for o in observations:
        if o.get("name") == "query_pretranslation":
            value = o.get("output")
            if isinstance(value, str) and value.strip():
                translated = value
    return {"messages": messages, "tools": tools, "fixtures": fixtures, "query_en": translated,
            "first_request_time": first.get("startTime"), "historical_answer": trace.get("output")}
