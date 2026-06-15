# Contributing

## Running tests

```bash
make install   # install dependencies including dev extras
make test      # runs all 15 unit tests (no API key needed)
```

## Adding a new agent node

1. Create `agents/your_node.py` following the pattern of any existing agent:
   - Import `GEMINI_MODEL` from `tools/config.py`
   - Use `parse_json_response(response, "your_node")` from `tools/llm_utils.py`
   - Accept `CVState` and return a partial `CVState` dict

2. Add the new field(s) to `CVState` in `graph/state.py`

3. Register the node and wire its edges in `graph/graph.py`

4. Add a test file `tests/test_your_node.py` — mock `agents.your_node._chain` and assert the returned state shape

## Code style

- No comments unless the reason is non-obvious
- Type hints required on all node functions
- All LLM responses must go through `parse_json_response` — never call `json.loads` directly in agent code
