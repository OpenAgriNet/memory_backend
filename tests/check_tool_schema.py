"""Validate registration using the actual bot image's Pydantic AI dependency."""
import pydantic_ai  # Prevent the offline test loader from replacing the installed SDK.
from pydantic_ai import Tool
import test_bot_tools as fixture

for fn in (fixture.memory_tools.search_memories, fixture.memory_tools.list_memories, fixture.memory_tools.read_memory):
    tool = Tool(fn, takes_ctx=True, docstring_format="auto", require_parameter_descriptions=True)
    schema = tool.function_schema.json_schema
    assert "ctx" not in schema.get("properties", {})
    assert "farmer_id" not in schema.get("properties", {})
    expected = {"search_memories": {"query", "memory_ref"},
                "list_memories": {"state", "filters", "cursor"},
                "read_memory": {"memory_ref", "chunk_ids", "include_history"}}
    assert set(schema.get("properties", {})) == expected[fn.__name__]
    print(fn.__name__, sorted(schema.get("properties", {})))
