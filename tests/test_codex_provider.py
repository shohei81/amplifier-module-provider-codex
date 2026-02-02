import asyncio
import json
import logging
import pytest

from amplifier_core.message_models import ChatRequest
from amplifier_core.message_models import Message
from amplifier_core.message_models import ToolCallBlock

from amplifier_module_provider_codex import CodexProvider


class FakeHooks:
    def __init__(self):
        self.events: list[tuple[str, dict]] = []

    async def emit(self, name: str, payload: dict) -> None:
        self.events.append((name, payload))


class FakeCoordinator:
    def __init__(self):
        self.hooks = FakeHooks()


class FakeStdin:
    def __init__(self):
        self.buffer = b""

    def write(self, data: bytes) -> None:
        self.buffer += data

    async def drain(self) -> None:
        return None

    def close(self) -> None:
        return None

    async def wait_closed(self) -> None:
        return None


class FakeStream:
    def __init__(self, lines: list[bytes]):
        self._lines = lines
        self._idx = 0

    async def readline(self) -> bytes:
        if self._idx >= len(self._lines):
            return b""
        line = self._lines[self._idx]
        self._idx += 1
        return line

    async def read(self) -> bytes:
        return b""


class FakeProcess:
    def __init__(self, lines: list[dict], returncode: int = 0):
        self.stdin = FakeStdin()
        encoded = [json.dumps(line).encode("utf-8") + b"\n" for line in lines]
        self.stdout = FakeStream(encoded)
        self.stderr = FakeStream([])
        self.returncode = returncode

    async def wait(self) -> int:
        return self.returncode


class HangingStream:
    async def readline(self) -> bytes:
        await asyncio.sleep(10)
        return b""

    async def read(self) -> bytes:
        await asyncio.sleep(10)
        return b""


class HangingProcess:
    def __init__(self):
        self.stdin = FakeStdin()
        self.stdout = HangingStream()
        self.stderr = HangingStream()
        self.returncode = None
        self.terminated = False
        self.killed = False

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = -15

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9

    async def wait(self) -> int:
        if self.returncode is not None:
            return self.returncode
        await asyncio.sleep(10)
        return self.returncode or 0


def _make_subprocess_stub(lines: list[dict]):
    async def _stub(*_args, **_kwargs):
        return FakeProcess(lines)

    return _stub


def test_codex_basic_response(monkeypatch):
    provider = CodexProvider(config={"skip_git_repo_check": True})

    monkeypatch.setattr("shutil.which", lambda _cmd: "/usr/bin/codex")
    lines = [
        {"type": "thread.started", "thread_id": "thread_123"},
        {
            "type": "item.completed",
            "item": {
                "type": "message",
                "content": [{"type": "output_text", "text": "Hello Codex"}],
            },
        },
        {"type": "turn.completed", "usage": {"input_tokens": 10, "output_tokens": 5}},
    ]
    monkeypatch.setattr(
        asyncio, "create_subprocess_exec", _make_subprocess_stub(lines)
    )

    request = ChatRequest(messages=[Message(role="user", content="Hi")])
    response = asyncio.run(provider.complete(request))

    assert response.text == "Hello Codex"
    assert response.metadata.get("codex:session_id") == "thread_123"
    assert response.usage.input_tokens == 10
    assert response.usage.output_tokens == 5


def test_codex_tool_call_from_item(monkeypatch):
    provider = CodexProvider(config={"skip_git_repo_check": True})

    monkeypatch.setattr("shutil.which", lambda _cmd: "/usr/bin/codex")
    lines = [
        {
            "type": "item.completed",
            "item": {
                "type": "tool_call",
                "id": "call_1",
                "name": "search",
                "arguments": {"q": "test"},
            },
        },
        {"type": "turn.completed", "usage": {"input_tokens": 1, "output_tokens": 1}},
    ]
    monkeypatch.setattr(
        asyncio, "create_subprocess_exec", _make_subprocess_stub(lines)
    )

    request = ChatRequest(
        messages=[Message(role="user", content="Hi")],
        tools=[{"name": "search", "description": "", "parameters": {}}],
    )
    response = asyncio.run(provider.complete(request))

    assert response.tool_calls
    assert response.tool_calls[0].name == "search"
    assert response.tool_calls[0].arguments == {"q": "test"}


def test_codex_tool_call_from_item_parses_string_arguments(monkeypatch):
    provider = CodexProvider(config={"skip_git_repo_check": True})

    monkeypatch.setattr("shutil.which", lambda _cmd: "/usr/bin/codex")
    lines = [
        {
            "type": "item.completed",
            "item": {
                "type": "tool_call",
                "id": "call_1",
                "name": "search",
                "arguments": "{\"q\": \"test\"}",
            },
        },
        {"type": "turn.completed", "usage": {"input_tokens": 1, "output_tokens": 1}},
    ]
    monkeypatch.setattr(
        asyncio, "create_subprocess_exec", _make_subprocess_stub(lines)
    )

    request = ChatRequest(
        messages=[Message(role="user", content="Hi")],
        tools=[{"name": "search", "description": "", "parameters": {}}],
    )
    response = asyncio.run(provider.complete(request))

    assert response.tool_calls
    assert response.tool_calls[0].arguments == {"q": "test"}


def test_codex_tool_call_from_message_content_block(monkeypatch):
    provider = CodexProvider(config={"skip_git_repo_check": True})

    monkeypatch.setattr("shutil.which", lambda _cmd: "/usr/bin/codex")
    lines = [
        {
            "type": "item.completed",
            "item": {
                "type": "message",
                "content": [
                    {"type": "output_text", "text": "Calling tool"},
                    {
                        "type": "tool_call",
                        "id": "call_1",
                        "name": "search",
                        "arguments": {"q": "test"},
                    },
                ],
            },
        },
        {"type": "turn.completed", "usage": {"input_tokens": 1, "output_tokens": 1}},
    ]
    monkeypatch.setattr(
        asyncio, "create_subprocess_exec", _make_subprocess_stub(lines)
    )

    request = ChatRequest(
        messages=[Message(role="user", content="Hi")],
        tools=[{"name": "search", "description": "", "parameters": {}}],
    )
    response = asyncio.run(provider.complete(request))

    assert response.tool_calls
    assert response.tool_calls[0].id == "call_1"
    assert response.tool_calls[0].name == "search"
    assert response.tool_calls[0].arguments == {"q": "test"}


def test_codex_function_call_item_type_supported(monkeypatch):
    provider = CodexProvider(config={"skip_git_repo_check": True})

    monkeypatch.setattr("shutil.which", lambda _cmd: "/usr/bin/codex")
    lines = [
        {
            "type": "item.completed",
            "item": {
                "type": "function_call",
                "call_id": "call_1",
                "name": "search",
                "arguments": "{\"q\": \"test\"}",
            },
        },
        {"type": "turn.completed", "usage": {"input_tokens": 1, "output_tokens": 1}},
    ]
    monkeypatch.setattr(
        asyncio, "create_subprocess_exec", _make_subprocess_stub(lines)
    )

    request = ChatRequest(
        messages=[Message(role="user", content="Hi")],
        tools=[{"name": "search", "description": "", "parameters": {}}],
    )
    response = asyncio.run(provider.complete(request))

    assert response.tool_calls
    assert response.tool_calls[0].id == "call_1"
    assert response.tool_calls[0].name == "search"
    assert response.tool_calls[0].arguments == {"q": "test"}


def test_codex_tool_call_from_item_filters_invalid(monkeypatch):
    provider = CodexProvider(config={"skip_git_repo_check": True})

    monkeypatch.setattr("shutil.which", lambda _cmd: "/usr/bin/codex")
    lines = [
        {
            "type": "item.completed",
            "item": {
                "type": "tool_call",
                "id": "call_1",
                "name": "invalid",
                "arguments": {"q": "test"},
            },
        },
        {"type": "turn.completed", "usage": {"input_tokens": 1, "output_tokens": 1}},
    ]
    monkeypatch.setattr(
        asyncio, "create_subprocess_exec", _make_subprocess_stub(lines)
    )

    request = ChatRequest(
        messages=[Message(role="user", content="Hi")],
        tools=[{"name": "search", "description": "", "parameters": {}}],
    )
    response = asyncio.run(provider.complete(request))

    assert not response.tool_calls
    assert provider._filtered_tool_calls
    assert provider._filtered_tool_calls[0]["name"] == "invalid"


def test_codex_parses_response_output_item_done(monkeypatch):
    provider = CodexProvider(config={"skip_git_repo_check": True})

    monkeypatch.setattr("shutil.which", lambda _cmd: "/usr/bin/codex")
    lines = [
        {
            "type": "response.output_item.done",
            "item": {
                "type": "message",
                "content": [{"type": "output_text", "text": "Hello from response event"}],
            },
        },
        {
            "type": "response.completed",
            "response": {"usage": {"input_tokens": 12, "output_tokens": 3}},
        },
    ]
    monkeypatch.setattr(
        asyncio, "create_subprocess_exec", _make_subprocess_stub(lines)
    )

    request = ChatRequest(messages=[Message(role="user", content="Hi")])
    response = asyncio.run(provider.complete(request))

    assert response.text == "Hello from response event"
    assert response.usage.input_tokens == 12
    assert response.usage.output_tokens == 3


def test_codex_parses_function_call_from_response_output_item_done(monkeypatch):
    provider = CodexProvider(config={"skip_git_repo_check": True})

    monkeypatch.setattr("shutil.which", lambda _cmd: "/usr/bin/codex")
    lines = [
        {
            "type": "response.output_item.done",
            "item": {
                "type": "function_call",
                "call_id": "call_1",
                "name": "search",
                "arguments": "{\"q\": \"test\"}",
            },
        },
        {
            "type": "response.completed",
            "response": {"usage": {"input_tokens": 5, "output_tokens": 1}},
        },
    ]
    monkeypatch.setattr(
        asyncio, "create_subprocess_exec", _make_subprocess_stub(lines)
    )

    request = ChatRequest(
        messages=[Message(role="user", content="Hi")],
        tools=[{"name": "search", "description": "", "parameters": {}}],
    )
    response = asyncio.run(provider.complete(request))

    assert response.tool_calls
    assert response.tool_calls[0].id == "call_1"
    assert response.tool_calls[0].name == "search"
    assert response.tool_calls[0].arguments == {"q": "test"}


def test_codex_falls_back_to_response_output_text_delta(monkeypatch):
    provider = CodexProvider(config={"skip_git_repo_check": True})

    monkeypatch.setattr("shutil.which", lambda _cmd: "/usr/bin/codex")
    lines = [
        {"type": "response.output_text.delta", "delta": "Hello"},
        {"type": "response.output_text.delta", "delta": " world"},
        {
            "type": "response.completed",
            "response": {"usage": {"input_tokens": 4, "output_tokens": 2}},
        },
    ]
    monkeypatch.setattr(
        asyncio, "create_subprocess_exec", _make_subprocess_stub(lines)
    )

    request = ChatRequest(messages=[Message(role="user", content="Hi")])
    response = asyncio.run(provider.complete(request))

    assert response.text == "Hello world"
    assert response.usage.input_tokens == 4
    assert response.usage.output_tokens == 2


def test_codex_tool_calls_blocked_without_tools(monkeypatch):
    provider = CodexProvider(config={"skip_git_repo_check": True})

    monkeypatch.setattr("shutil.which", lambda _cmd: "/usr/bin/codex")
    lines = [
        {
            "type": "item.completed",
            "item": {
                "type": "tool_call",
                "id": "call_1",
                "name": "search",
                "arguments": {"q": "test"},
            },
        },
        {"type": "turn.completed", "usage": {"input_tokens": 1, "output_tokens": 1}},
    ]
    monkeypatch.setattr(
        asyncio, "create_subprocess_exec", _make_subprocess_stub(lines)
    )

    request = ChatRequest(messages=[Message(role="user", content="Hi")])
    response = asyncio.run(provider.complete(request))

    assert not response.tool_calls
    assert provider._filtered_tool_calls
    assert provider._filtered_tool_calls[0]["name"] == "search"


def test_codex_ignores_mcp_tool_calls(monkeypatch):
    provider = CodexProvider(config={"skip_git_repo_check": True})

    monkeypatch.setattr("shutil.which", lambda _cmd: "/usr/bin/codex")
    lines = [
        {
            "type": "item.completed",
            "item": {
                "type": "mcp_tool_call",
                "id": "mcp_1",
                "name": "web_search",
                "arguments": {"query": "test"},
            },
        },
        {"type": "turn.completed", "usage": {"input_tokens": 1, "output_tokens": 1}},
    ]
    monkeypatch.setattr(
        asyncio, "create_subprocess_exec", _make_subprocess_stub(lines)
    )

    request = ChatRequest(
        messages=[Message(role="user", content="Hi")],
        tools=[{"name": "search", "description": "", "parameters": {}}],
    )
    response = asyncio.run(provider.complete(request))

    assert not response.tool_calls


def test_codex_tool_call_from_text_filters_invalid():
    provider = CodexProvider(config={})
    provider._valid_tool_names = {"allowed"}
    provider._tools_enabled = True

    text = (
        "Here is a tool call:\n"
        "<tool_use>{\"tool\":\"invalid\",\"id\":\"call_1\",\"input\":{}}</tool_use>"
    )
    calls = provider._extract_tool_calls_from_text(text)

    assert calls == []
    assert provider._filtered_tool_calls
    assert provider._filtered_tool_calls[0]["name"] == "invalid"


def test_codex_dedupes_filtered_tool_calls_before_injection(monkeypatch):
    """Test that duplicate filtered tool calls are deduplicated.

    Note: The provider now works on a copy of request.messages to avoid
    mutating the caller's data (P1-2 fix). This test verifies:
    1. The original request.messages is NOT mutated
    2. The deduplication logic still processes filtered calls correctly
    """
    provider = CodexProvider(config={"skip_git_repo_check": True})
    provider._filtered_tool_calls = [
        {"id": "call_dupe", "name": "invalid", "arguments": {}},
        {"id": "call_dupe", "name": "invalid", "arguments": {}},
    ]

    monkeypatch.setattr("shutil.which", lambda _cmd: "/usr/bin/codex")
    lines = [
        {
            "type": "item.completed",
            "item": {
                "type": "message",
                "content": [{"type": "output_text", "text": "ok"}],
            },
        },
        {"type": "turn.completed", "usage": {"input_tokens": 1, "output_tokens": 1}},
    ]
    monkeypatch.setattr(
        asyncio, "create_subprocess_exec", _make_subprocess_stub(lines)
    )

    request = ChatRequest(messages=[Message(role="user", content="Hi")])
    original_messages_len = len(request.messages)
    asyncio.run(provider.complete(request))

    # P1-2 fix: request.messages should NOT be mutated
    assert len(request.messages) == original_messages_len
    rejected_in_request = [
        msg
        for msg in request.messages
        if msg.role == "tool" and msg.content.startswith("[SYSTEM NOTICE: Tool call rejected]")
    ]
    assert len(rejected_in_request) == 0, "request.messages should not be mutated"


def test_codex_tool_call_from_markdown_block():
    """Test extracting tool calls that are wrapped in markdown code blocks."""
    provider = CodexProvider(config={})
    provider._valid_tool_names = {"allowed"}
    provider._tools_enabled = True
    
    # Test lowercase json
    text1 = """
    <tool_use>
    ```json
    {
        "tool": "allowed",
        "id": "call_1",
        "input": {"a": 1}
    }
    ```
    </tool_use>
    """
    calls1 = provider._extract_tool_calls_from_text(text1)
    assert len(calls1) == 1
    assert calls1[0]["name"] == "allowed"
    assert calls1[0]["arguments"] == {"a": 1}

    # Test uppercase JSON
    text2 = """
    <tool_use>
    ```JSON
    {
        "tool": "allowed",
        "id": "call_2",
        "input": {"b": 2}
    }
    ```
    </tool_use>
    """
    calls2 = provider._extract_tool_calls_from_text(text2)
    assert len(calls2) == 1
    assert calls2[0]["id"] == "call_2"
    
    # Test whitespace around trailing fence
    text3 = """
    <tool_use>
    ```
    {
        "tool": "allowed",
        "id": "call_3",
        "input": {"c": 3}
    }
      ```   
    </tool_use>
    """
    calls3 = provider._extract_tool_calls_from_text(text3)
    assert len(calls3) == 1
    assert calls3[0]["id"] == "call_3"


def test_codex_repairs_missing_tool_results(monkeypatch):
    provider = CodexProvider(config={"skip_git_repo_check": True})
    fake_coordinator = FakeCoordinator()
    provider.coordinator = fake_coordinator

    monkeypatch.setattr("shutil.which", lambda _cmd: "/usr/bin/codex")
    lines = [
        {
            "type": "item.completed",
            "item": {
                "type": "message",
                "content": [{"type": "output_text", "text": "Done"}],
            },
        },
        {"type": "turn.completed", "usage": {"input_tokens": 1, "output_tokens": 1}},
    ]
    monkeypatch.setattr(
        asyncio, "create_subprocess_exec", _make_subprocess_stub(lines)
    )

    messages = [
        Message(
            role="assistant",
            content=[ToolCallBlock(id="call_1", name="do", input={"x": 1})],
        ),
        Message(role="user", content="continue"),
    ]
    request = ChatRequest(messages=messages)

    asyncio.run(provider.complete(request))

    repair_events = [
        e for e in fake_coordinator.hooks.events if e[0] == "provider:tool_sequence_repaired"
    ]
    assert len(repair_events) == 1
    assert repair_events[0][1]["repair_count"] == 1


def test_codex_repairs_missing_tool_results_from_dict_block(monkeypatch):
    provider = CodexProvider(config={"skip_git_repo_check": True})
    fake_coordinator = FakeCoordinator()
    provider.coordinator = fake_coordinator

    monkeypatch.setattr("shutil.which", lambda _cmd: "/usr/bin/codex")
    lines = [
        {
            "type": "item.completed",
            "item": {
                "type": "message",
                "content": [{"type": "output_text", "text": "Done"}],
            },
        },
        {"type": "turn.completed", "usage": {"input_tokens": 1, "output_tokens": 1}},
    ]
    monkeypatch.setattr(
        asyncio, "create_subprocess_exec", _make_subprocess_stub(lines)
    )

    messages = [
        Message(
            role="assistant",
            content=[{"type": "tool_call", "id": "call_1", "name": "do", "input": {"x": 1}}],
        ),
        Message(role="user", content="continue"),
    ]
    request = ChatRequest(messages=messages)

    asyncio.run(provider.complete(request))

    repair_events = [
        e for e in fake_coordinator.hooks.events if e[0] == "provider:tool_sequence_repaired"
    ]
    assert len(repair_events) == 1
    assert repair_events[0][1]["repair_count"] == 1


def test_codex_clears_repaired_tool_ids_each_request(monkeypatch):
    """Test that _repaired_tool_ids is cleared at the start of each request (P2-1 fix).

    This prevents unbounded memory growth in long sessions.
    """
    provider = CodexProvider(config={"skip_git_repo_check": True})
    # Simulate IDs from a previous request
    provider._repaired_tool_ids = {"old_call_1", "old_call_2", "old_call_3"}

    monkeypatch.setattr("shutil.which", lambda _cmd: "/usr/bin/codex")
    lines = [
        {
            "type": "item.completed",
            "item": {
                "type": "message",
                "content": [{"type": "output_text", "text": "ok"}],
            },
        },
        {"type": "turn.completed", "usage": {"input_tokens": 1, "output_tokens": 1}},
    ]
    monkeypatch.setattr(
        asyncio, "create_subprocess_exec", _make_subprocess_stub(lines)
    )

    request = ChatRequest(messages=[Message(role="user", content="Hi")])
    asyncio.run(provider.complete(request))

    # The old IDs should have been cleared
    assert "old_call_1" not in provider._repaired_tool_ids
    assert "old_call_2" not in provider._repaired_tool_ids
    assert "old_call_3" not in provider._repaired_tool_ids


def test_codex_does_not_repair_tool_results_without_call_id(monkeypatch):
    provider = CodexProvider(config={"skip_git_repo_check": True})
    fake_coordinator = FakeCoordinator()
    provider.coordinator = fake_coordinator

    monkeypatch.setattr("shutil.which", lambda _cmd: "/usr/bin/codex")
    lines = [
        {
            "type": "item.completed",
            "item": {
                "type": "message",
                "content": [{"type": "output_text", "text": "Done"}],
            },
        },
        {"type": "turn.completed", "usage": {"input_tokens": 1, "output_tokens": 1}},
    ]
    monkeypatch.setattr(
        asyncio, "create_subprocess_exec", _make_subprocess_stub(lines)
    )

    messages = [
        Message(
            role="assistant",
            content=[ToolCallBlock(id="call_1", name="do", input={"x": 1})],
        ),
        Message(role="tool", content="ok", name="do"),
        Message(role="user", content="continue"),
    ]
    request = ChatRequest(messages=messages)

    asyncio.run(provider.complete(request))

    repair_events = [
        e for e in fake_coordinator.hooks.events if e[0] == "provider:tool_sequence_repaired"
    ]
    assert repair_events == []


def test_codex_build_command_includes_permission_flags():
    provider = CodexProvider(
        config={
            "profile": "dev",
            "sandbox": "workspace-write",
            "full_auto": True,
            "skip_git_repo_check": True,
            "search": True,
            "ask_for_approval": "on-failure",
            "network_access": True,
            "add_dir": ["/tmp/extra", "/var/data"],
        }
    )

    cmd = provider._build_command("/usr/bin/codex", "gpt-5.2-codex", None)

    assert cmd == [
        "/usr/bin/codex",
        "--profile",
        "dev",
        "--sandbox",
        "workspace-write",
        "--full-auto",
        "--ask-for-approval",
        "on-failure",
        "--search",
        "--add-dir",
        "/tmp/extra",
        "--add-dir",
        "/var/data",
        "exec",
        "--json",
        "--model",
        "gpt-5.2-codex",
        "--config",
        "sandbox_workspace_write.network_access=true",
        "--skip-git-repo-check",
        "-",
    ]


def test_codex_build_command_includes_flags_with_resume():
    provider = CodexProvider(
        config={
            "search": True,
            "ask_for_approval": "never",
            "network_access": False,
            "add_dir": "/tmp/dir",
            "skip_git_repo_check": False,
            "sandbox": "workspace-write",
        }
    )

    cmd = provider._build_command("/usr/bin/codex", "gpt-5.2-codex", "thread_1")

    assert cmd == [
        "/usr/bin/codex",
        "--sandbox",
        "workspace-write",
        "--ask-for-approval",
        "never",
        "--search",
        "--add-dir",
        "/tmp/dir",
        "exec",
        "resume",
        "thread_1",
        "--json",
        "--model",
        "gpt-5.2-codex",
        "--config",
        "sandbox_workspace_write.network_access=false",
        "-",
    ]


def test_codex_resume_does_not_receive_global_only_flags():
    provider = CodexProvider(
        config={
            "profile": "dev",
            "sandbox": "workspace-write",
            "search": True,
            "ask_for_approval": "never",
            "add_dir": ["/tmp/extra"],
        }
    )

    cmd = provider._build_command("/usr/bin/codex", "gpt-5.2-codex", "thread_1")
    resume_idx = cmd.index("resume")
    after_resume = cmd[resume_idx + 1 :]

    assert "--profile" not in after_resume
    assert "--sandbox" not in after_resume
    assert "--search" not in after_resume
    assert "--ask-for-approval" not in after_resume
    assert "--add-dir" not in after_resume


def test_codex_places_global_flags_before_exec():
    provider = CodexProvider(
        config={
            "profile": "dev",
            "sandbox": "workspace-write",
            "full_auto": True,
            "search": True,
            "ask_for_approval": "never",
            "add_dir": ["/tmp/a"],
        }
    )

    cmd = provider._build_command("/usr/bin/codex", "gpt-5.2-codex", None)
    exec_idx = cmd.index("exec")
    before_exec = cmd[:exec_idx]
    after_exec = cmd[exec_idx + 1 :]

    assert "--search" in before_exec
    assert "--ask-for-approval" in before_exec
    assert "--profile" in before_exec
    assert "--sandbox" in before_exec
    assert "--full-auto" in before_exec
    assert "--add-dir" in before_exec
    assert "--search" not in after_exec
    assert "--ask-for-approval" not in after_exec
    assert "--profile" not in after_exec
    assert "--sandbox" not in after_exec
    assert "--full-auto" not in after_exec
    assert "--add-dir" not in after_exec


def test_codex_warns_on_ask_for_approval_on_request_without_full_auto(caplog):
    with caplog.at_level(logging.WARNING):
        CodexProvider(config={"ask_for_approval": "on-request"})

    assert "ask_for_approval=on-request may block" in caplog.text


def test_codex_warns_on_danger_full_access_sandbox(caplog):
    with caplog.at_level(logging.WARNING):
        CodexProvider(config={"sandbox": "danger-full-access"})

    assert "sandbox=danger-full-access is unsafe" in caplog.text


def test_codex_warns_on_invalid_ask_for_approval_value(caplog):
    with caplog.at_level(logging.WARNING):
        provider = CodexProvider(config={"ask_for_approval": "sometimes"})

    assert provider.ask_for_approval is None
    assert "Invalid ask_for_approval config" in caplog.text


def test_codex_warns_on_non_string_ask_for_approval(caplog):
    with caplog.at_level(logging.WARNING):
        provider = CodexProvider(config={"ask_for_approval": 123})

    assert provider.ask_for_approval is None
    assert "Invalid ask_for_approval config" in caplog.text


def test_codex_warns_on_network_access_with_non_workspace_sandbox(caplog):
    provider = CodexProvider(
        config={
            "sandbox": "read-only",
            "network_access": True,
        }
    )

    with caplog.at_level(logging.WARNING):
        cmd = provider._build_command("/usr/bin/codex", "gpt-5.2-codex", None)

    assert "Ignoring network_access" in caplog.text
    assert all(
        "sandbox_workspace_write.network_access" not in arg for arg in cmd
    )


def test_codex_warns_on_invalid_network_access_type(caplog):
    provider = CodexProvider(config={"network_access": "yes"})

    with caplog.at_level(logging.WARNING):
        cmd = provider._build_command("/usr/bin/codex", "gpt-5.2-codex", None)

    assert "Invalid network_access config" in caplog.text
    assert all(
        "sandbox_workspace_write.network_access" not in arg for arg in cmd
    )


def test_codex_times_out_and_terminates_process(monkeypatch):
    provider = CodexProvider(config={"skip_git_repo_check": True, "timeout": 0.01})

    process_holder = {}

    async def _stub(*_args, **_kwargs):
        proc = HangingProcess()
        process_holder["proc"] = proc
        return proc

    monkeypatch.setattr("shutil.which", lambda _cmd: "/usr/bin/codex")
    monkeypatch.setattr(asyncio, "create_subprocess_exec", _stub)

    request = ChatRequest(messages=[Message(role="user", content="Hi")])

    with pytest.raises(TimeoutError):
        asyncio.run(provider.complete(request))

    proc = process_holder["proc"]
    assert proc.terminated or proc.killed


def test_codex_builds_command_with_reasoning_effort():
    provider = CodexProvider(config={"reasoning_effort": "low"})

    effort = provider._resolve_reasoning_effort(
        ChatRequest(
            messages=[Message(role="user", content="Hi")],
            metadata={"reasoning_effort": "high"},
        ),
        model="gpt-5.2-codex",
    )

    cmd = provider._build_command(
        cli_path="/usr/bin/codex",
        model="gpt-5.2-codex",
        session_id=None,
        reasoning_effort=effort,
    )

    assert "--config" in cmd
    idx = cmd.index("--config")
    assert cmd[idx + 1] == 'model_reasoning_effort="high"'


def test_codex_ignores_reasoning_effort_not_supported_by_model():
    provider = CodexProvider(config={"reasoning_effort": "none"})

    effort = provider._resolve_reasoning_effort(
        ChatRequest(messages=[Message(role="user", content="Hi")]),
        model="gpt-5-codex",
    )

    assert effort is None


def test_codex_allows_none_for_gpt_5_2_models():
    provider = CodexProvider()

    effort = provider._resolve_reasoning_effort(
        ChatRequest(
            messages=[Message(role="user", content="Hi")],
            metadata={"reasoning_effort": "none"},
        ),
        model="gpt-5.2-codex",
    )

    assert effort == "none"


def test_codex_reasoning_effort_config_precedence_and_fallback():
    provider = CodexProvider(
        config={"reasoning_effort": "HIGH", "model_reasoning_effort": "low"}
    )
    effort = provider._resolve_reasoning_effort(
        ChatRequest(messages=[Message(role="user", content="Hi")]),
        model="gpt-5.2-codex",
    )
    assert effort == "high"

    fallback = CodexProvider(config={"model_reasoning_effort": "LOW"})
    effort = fallback._resolve_reasoning_effort(
        ChatRequest(messages=[Message(role="user", content="Hi")]),
        model="gpt-5.2-codex",
    )
    assert effort == "low"


def test_codex_reasoning_effort_hyphenated_keys_and_case_insensitivity():
    provider = CodexProvider()

    effort = provider._resolve_reasoning_effort(
        ChatRequest(messages=[Message(role="user", content="Hi")]),
        model="gpt-5.2-codex",
        **{"reasoning-effort": "HIGH"},
    )
    assert effort == "high"

    effort = provider._resolve_reasoning_effort(
        ChatRequest(
            messages=[Message(role="user", content="Hi")],
            metadata={"reasoning-effort": "LoW"},
        ),
        model="gpt-5.2-codex",
    )
    assert effort == "low"


def test_codex_reasoning_effort_gpt_5_1_validation():
    provider = CodexProvider()

    effort = provider._resolve_reasoning_effort(
        ChatRequest(
            messages=[Message(role="user", content="Hi")],
            metadata={"reasoning_effort": "none"},
        ),
        model="gpt-5.1-codex",
    )
    assert effort == "none"

    effort = provider._resolve_reasoning_effort(
        ChatRequest(
            messages=[Message(role="user", content="Hi")],
            metadata={"reasoning_effort": "xhigh"},
        ),
        model="gpt-5.1-codex",
    )
    assert effort is None


def test_codex_reasoning_effort_invalid_value_logs_warning(caplog):
    provider = CodexProvider()

    with caplog.at_level(logging.WARNING):
        effort = provider._resolve_reasoning_effort(
            ChatRequest(
                messages=[Message(role="user", content="Hi")],
                metadata={"reasoning_effort": "super"},
            ),
            model="gpt-5.2-codex",
        )

    assert effort is None
    assert "Ignoring invalid reasoning_effort" in caplog.text


def test_codex_reasoning_effort_unknown_model_passthrough_logs_warning(caplog):
    provider = CodexProvider()

    with caplog.at_level(logging.WARNING):
        effort = provider._resolve_reasoning_effort(
            ChatRequest(
                messages=[Message(role="user", content="Hi")],
                metadata={"reasoning_effort": "high"},
            ),
            model="custom-model",
        )

    assert effort == "high"
    assert "Passing through reasoning_effort" in caplog.text


def test_codex_adjusts_minimal_reasoning_when_search_enabled(caplog):
    provider = CodexProvider(config={"search": True})

    with caplog.at_level(logging.WARNING):
        adjusted = provider._adjust_reasoning_effort_for_search(
            model="gpt-5-codex", reasoning_effort="minimal"
        )

    assert adjusted == "low"
    assert "search=true is incompatible with reasoning_effort=minimal" in caplog.text


def test_codex_keeps_reasoning_effort_when_search_disabled():
    provider = CodexProvider(config={"search": False})

    adjusted = provider._adjust_reasoning_effort_for_search(
        model="gpt-5-codex", reasoning_effort="minimal"
    )

    assert adjusted == "minimal"


def test_codex_builds_resume_command_with_reasoning_effort():
    provider = CodexProvider()

    cmd = provider._build_command(
        cli_path="/usr/bin/codex",
        model="gpt-5.2-codex",
        session_id="thread_123",
        reasoning_effort="high",
    )

    assert "resume" in cmd
    assert "--config" in cmd
