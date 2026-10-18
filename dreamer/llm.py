"""Provider-agnostic LLM calls for the dreamer.

Any OpenAI-compatible endpoint works, so a local vLLM and OpenRouter are the same
code path — only base_url/model/key differ. That matters here because this runs
entirely in the background: nothing stops us pointing the extraction step at a
stronger model than the one serving live chat (per the compute-budget principle in
memory-overview.md, off-turn work has no latency budget to respect).

  DREAMER_PROVIDER=vllm        (default)  -> DREAMER_BASE_URL, DREAMER_MODEL
  DREAMER_PROVIDER=openrouter            -> key read from oan_creds/.openrouter_api

JSON parsing is tolerant of markdown fences and leading prose, and structured
output is requested where the backend supports it — both lessons taken from
/amulpfsdata/gautam/llm-judge/judge.py, which hit exactly these problems.
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "service"))
if __package__:
    from service.runtime_config import required
else:
    from runtime_config import required

from openai import OpenAI

PROVIDER = os.environ.get("DREAMER_PROVIDER", "vllm").lower()
OPENROUTER_KEY_FILE = os.environ.get(
    "OPENROUTER_KEY_FILE", "/amulpfsdata/gautam/oan-brain/oan_creds/.openrouter_api"
)


def _openrouter_key():
    with open(OPENROUTER_KEY_FILE) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            return line.split("=", 1)[-1].strip().strip('"').strip("'") if "=" in line else line
    raise RuntimeError("no key found in OPENROUTER_KEY_FILE")


if PROVIDER == "openrouter":
    BASE_URL = required("DREAMER_BASE_URL")
    # The SAME model the local vLLM serves, deliberately: every measurement in this
    # project was made against gemma-4-31b-it, so keeping the id identical means a
    # run through OpenRouter stays comparable with a run against the local box.
    MODEL = os.environ.get("DREAMER_MODEL", "google/gemma-4-31b-it")
    API_KEY = _openrouter_key()
    SUPPORTS_VLLM_STRUCTURED = False
else:
    BASE_URL = required("DREAMER_BASE_URL")
    MODEL = os.environ.get("DREAMER_MODEL", "gemma-4-31b-it")
    API_KEY = "EMPTY"
    SUPPORTS_VLLM_STRUCTURED = True

client = OpenAI(base_url=BASE_URL, api_key=API_KEY, timeout=600.0, max_retries=2)


def describe() -> str:
    return f"{PROVIDER}:{MODEL} @ {BASE_URL}"


def _parse(raw: str):
    """Parse JSON out of a model reply, tolerating fences and stray prose."""
    for cand in (raw, raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```")):
        try:
            return json.loads(cand.strip())
        except json.JSONDecodeError:
            pass
    i, j = raw.find("{"), raw.rfind("}")
    if i != -1 and j > i:
        try:
            return json.loads(raw[i:j + 1])
        except json.JSONDecodeError:
            pass
    raise ValueError(f"no parseable JSON in model output: {raw[:200]}")


def _shape_ok(obj, schema):
    """Does the parsed object actually have the keys the caller asked for?

    Needed because only the local vLLM path enforces the schema. Everywhere else,
    `response_format: json_object` guarantees VALID json and says nothing about its
    KEYS — and a model handed a prompt with headings will happily mirror those
    headings instead. Observed on OpenRouter: a condense call returned
    {"situation","status","history","account"} rather than {"expanded"}, which parsed
    perfectly and meant nothing. Valid and useless is the worst failure shape, because
    nothing raises.
    """
    if not isinstance(obj, dict):
        return False, "the top level must be a JSON object"
    missing = [k for k in (schema.get("required") or []) if k not in obj]
    if missing:
        return False, "missing required key(s): " + ", ".join(repr(k) for k in missing)

    # The right key holding the wrong kind of value fails just as silently as a
    # missing key, and further downstream. Observed: {"expanded": 0.0} from a call
    # asking for a string, which passed a presence check and then blew up on
    # .strip() several frames away from the cause.
    expected = {"string": str, "integer": int, "number": (int, float),
                "boolean": bool, "array": list, "object": dict}
    props = schema.get("properties") or {}
    for k in obj:
        want = (props.get(k) or {}).get("type")
        py = expected.get(want)
        if not py:
            continue
        val = obj[k]
        # bool is an int in Python, but a boolean where a number is wanted is a bug.
        if want in ("integer", "number") and isinstance(val, bool):
            return False, f"key {k!r} must be a {want}, not a boolean"
        if not isinstance(val, py):
            return False, (f"key {k!r} must be a {want}, got "
                           f"{type(val).__name__} ({val!r:.40})")
        rule = props.get(k) or {}
        if "enum" in rule and val not in rule["enum"]:
            return False, f"key {k!r} must be one of {rule['enum']}"
        if want == "object":
            valid, reason = _shape_ok(val, rule)
            if not valid:
                return False, f"{k}: {reason}"
        if want == "array" and rule.get("items"):
            for index, item in enumerate(val):
                valid, reason = _shape_ok({"item": item}, {
                    "type": "object", "properties": {"item": rule["items"]},
                    "required": ["item"]})
                if not valid:
                    return False, f"{k}[{index}]: {reason}"
    if schema.get("additionalProperties") is False and any(k not in props for k in obj):
        return False, "unexpected key(s): " + ", ".join(k for k in obj if k not in props)
    return True, ""


def _required_hint(schema):
    req = schema.get("required") or []
    props = schema.get("properties") or {}
    if not req:
        return ""
    parts = []
    for k in req:
        t = (props.get(k) or {}).get("type", "value")
        parts.append(f'"{k}": <{t}>')
    return ("\n\nReturn a JSON object with exactly these top-level keys: "
            + ", ".join(f'"{k}"' for k in req)
            + ". Shape: {" + ", ".join(parts) + "}. Use these key names verbatim — do "
            "NOT invent key names from the headings in the text above.")


def ask_json(system: str, user: str, schema: dict, max_tokens=2000, attempts=3) -> dict:
    """One structured call. Retries on output that is unparseable OR the wrong shape.

    Two different failures, both real:

    - **Unparseable.** llm-judge documented this on the local vLLM build: a small
      share of calls fall into a whitespace loop inside the JSON and hit the token
      limit before closing the object. Changing the prompt is what breaks it;
      resampling alone does not.
    - **Valid but wrong-shaped.** Only the vLLM path enforces the schema. Elsewhere
      the model can return immaculate JSON with keys it made up, which parses and then
      silently does nothing. See _shape_ok.

    The required keys are always stated in the prompt now, rather than relied upon
    from schema enforcement that only one provider actually applies.
    """
    nudge = ("\n\nReturn the JSON compactly on a single line, no spaces or newlines "
             "between tokens.")
    user = user + _required_hint(schema)
    last = None
    for attempt in range(attempts):
        kwargs = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user + (nudge if attempt else "")},
            ],
            "temperature": 0.0 if attempt == 0 else 0.3,
            "max_tokens": max_tokens + (600 * attempt),
        }
        if SUPPORTS_VLLM_STRUCTURED:
            kwargs["extra_body"] = {"structured_outputs": {"json": schema}}
        else:
            kwargs["response_format"] = {"type": "json_object"}
        r = client.chat.completions.create(**kwargs)
        raw = r.choices[0].message.content or ""
        try:
            parsed = _parse(raw)
        except ValueError as e:
            last = e
            continue
        ok, why = _shape_ok(parsed, schema)
        if ok:
            return parsed
        last = ValueError(f"{why}; got keys {list(parsed) if isinstance(parsed, dict) else type(parsed).__name__}")
        # Say plainly what was wrong: a bare resample tends to reproduce the same
        # invented key names.
        user = user + f"\n\nYour previous answer was rejected: {why}."
    raise RuntimeError(f"model never returned usable JSON: {last}")


def chat(messages, tools=None, tool_choice="auto", max_tokens=1200, temperature=0.0):
    """One turn of a tool-using conversation. Returns the raw assistant message.

    Verified against the local vLLM build serving gemma-4-31b-it: native tool calling
    works, and it emits SEVERAL tool calls in one message when the calls are
    independent — which is what the agent loop relies on to fan out.
    """
    kwargs = {
        "model": MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = tool_choice
    r = client.chat.completions.create(**kwargs)
    return r.choices[0].message
