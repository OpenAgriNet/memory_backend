"""Paired replay of frozen Amul requests with actual memory tools; no live actions."""
import asyncio
import copy
import csv
import json
import os
from pathlib import Path
import sys
import time
import types

import pydantic_ai
from pydantic_ai import Tool
from openai import AsyncOpenAI
import test_bot_tools as fixture
real_deps = fixture._load("agents.deps", "agents/deps.py")
WORK = Path(os.getenv("AMUL_BACKTEST_WORK", "/backtest"))
config = json.loads((WORK / "config.json").read_text())
client = AsyncOpenAI(base_url=config["vllm"], api_key="unused", timeout=120, max_retries=2)
functions = {fn.__name__: fn for fn in (fixture.memory_tools.search_memories,
             fixture.memory_tools.list_memories, fixture.memory_tools.read_memory)}
memory_schemas = []
for name, fn in functions.items():
    tool = Tool(fn, takes_ctx=True, require_parameter_descriptions=True)
    memory_schemas.append({"type": "function", "function": {"name": name,
        "description": tool.description, "parameters": tool.function_schema.json_schema}})
RESULTS = WORK / "results"
RESULTS.mkdir(mode=0o700, exist_ok=True)
FIELDS = ["case_id", "farmer", "cutoff", "query_ts", "query", "query_en",
          "answer_without_memory", "answer_with_memory", "memory_improvement_score", "review_reason",
          "review_confidence", "without_memory_seconds", "with_memory_seconds",
          "memory_tool_calls", "memory_context_chars", "replay_limitations", "generation_error"]


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    path.chmod(0o600)


def export_csv():
    path = WORK / "memory_comparison.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=FIELDS)
        writer.writeheader()
        for result in sorted(RESULTS.glob("*.json")):
            item = json.loads(result.read_text())
            off, on = item.get("off", {}), item.get("on", {})
            writer.writerow({**{k: item.get(k, "") for k in FIELDS[:6]},
                "answer_without_memory": off.get("answer", ""), "answer_with_memory": on.get("answer", ""),
                "memory_improvement_score": item.get("review", {}).get("score", ""),
                "review_reason": item.get("review", {}).get("reason", ""),
                "review_confidence": item.get("review", {}).get("confidence", ""),
                "without_memory_seconds": round(off.get("seconds", 0), 3),
                "with_memory_seconds": round(on.get("seconds", 0), 3),
                "memory_tool_calls": on.get("memory_calls", 0),
                "memory_context_chars": len(on.get("memory_context", "")),
                "replay_limitations": "; ".join(sorted(set(off.get("limitations", []) + on.get("limitations", [])))),
                "generation_error": off.get("error", "") or on.get("error", "")})
    path.chmod(0o600)


def frozen_tool(case, name, args):
    fixtures = case["fixtures"].get(name) or []
    if not fixtures:
        return {"available": False, "reason": "No recorded tool result is available for this question."}, "missing_fixture:" + name
    exact = next((f for f in fixtures if f["input"] == args), None)
    if case.get("synthetic") and name == "get_farmer_milk_collection_details" and exact is None:
        return {"available": False, "reason": "No synthetic milk-record fixture covers the requested date range."}, "missing_fixture:" + name
    selected = exact or fixtures[0]
    return selected["output"], (None if exact else "frozen_same_question_result:" + name)


async def run_arm(case, enabled, max_tokens=1800):
    started = time.monotonic()
    deps = real_deps.FarmerContext(query=case["query_en"], mobile=case["farmer_id"],
                                  persona="farmer", use_translation_pipeline=True)
    ctx = types.SimpleNamespace(deps=deps)
    messages = copy.deepcopy(case["messages"])
    tools = copy.deepcopy(case["tools"])
    limitations, calls = [], []
    if enabled:
        deps.memory_context = await fixture.memory_svc.fetch_memory_context(deps.mobile, case["query_en"])
        if deps.memory_context:
            messages[0]["content"] += "\n\n" + real_deps.MEMORY_INSTRUCTIONS
            messages[-1]["content"] = ("Earlier conversational evidence:\n" + deps.memory_context +
                                         "\n\n" + messages[-1]["content"])
        tools += memory_schemas
    output = {"memory_context": deps.memory_context, "limitations": limitations,
              "calls": calls, "initial_messages": copy.deepcopy(messages), "max_tokens": max_tokens}
    try:
        for step in range(10):
            response = await client.chat.completions.create(model=config["model"], messages=messages,
                tools=tools, tool_choice="auto", temperature=0, max_tokens=max_tokens, parallel_tool_calls=False)
            message = response.choices[0].message
            messages.append(message.model_dump(exclude_none=True))
            if not message.tool_calls:
                if response.choices[0].finish_reason == "length":
                    raise RuntimeError("Reply reached the completion limit; not accepted as a complete answer")
                output["answer"] = message.content or ""
                break
            for call in message.tool_calls:
                name = call.function.name
                args = json.loads(call.function.arguments)
                if name in functions and enabled:
                    try:
                        result = await functions[name](ctx, **args)
                    except (TypeError, ValueError) as exc:
                        result = "Invalid memory arguments: " + str(exc)
                else:
                    result, limitation = frozen_tool(case, name, args)
                    if limitation:
                        limitations.append(limitation)
                text = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
                calls.append({"name": name, "args": args, "result": result})
                messages.append({"role": "tool", "tool_call_id": call.id, "content": text})
        else:
            raise RuntimeError("Agent request limit reached without an answer")
    except Exception as exc:
        output["error"] = str(exc)
    output.update(seconds=time.monotonic()-started, memory_calls=deps.memory_tool_calls,
                  memory_chars=deps.memory_tool_chars, messages=messages)
    return output


async def main():
    ids = json.loads((WORK / "case-index.json").read_text())
    limit = int(os.getenv("AMUL_BACKTEST_LIMIT", "0"))
    if limit:
        ids = ids[:limit]
    semaphore = asyncio.Semaphore(3)
    async def pair(case_id):
        async with semaphore:
            path = RESULTS / (case_id + ".json")
            max_tokens = 1800
            if path.exists():
                previous = json.loads(path.read_text())
                if all(previous.get(arm, {}).get("answer") and not previous[arm].get("error") for arm in ("off", "on")):
                    return
                # Preserve failed attempts and give both arms the same larger output
                # allowance when either answer exceeded the original replay limit.
                archive = WORK / "failed-attempts"
                archive.mkdir(exist_ok=True)
                (archive / f"{case_id}-{time.time_ns()}.json").write_text(path.read_text())
                if any("completion limit" in previous.get(arm, {}).get("error", "") for arm in ("off", "on")):
                    max_tokens = 7200
            case = json.loads((WORK / "cases" / (case_id + ".json")).read_text())
            row = {k: case[k] for k in FIELDS[:6]}
            # Alternate order to reduce warm-cache/order effects; model/prompt fixtures match.
            arms = [False, True] if int(case_id[-2:]) % 2 else [True, False]
            for enabled in arms:
                row["on" if enabled else "off"] = await run_arm(case, enabled, max_tokens)
                save(path, row)
            export_csv()
            print(json.dumps({"case": case_id, "off_error": row["off"].get("error"),
                              "on_error": row["on"].get("error"), "memory_calls": row["on"]["memory_calls"]}), flush=True)
    await asyncio.gather(*(pair(case_id) for case_id in ids))
    export_csv()
    print(f"Answers exported to {WORK / 'memory_comparison.csv'}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
