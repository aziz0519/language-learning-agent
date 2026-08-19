"""Unit tests for main.py.

These tests focus exclusively on the code introduced/changed by the PR:
- `build_graph`: now wires up the state graph nodes/edges and returns the
  compiled graph instead of `None`.
- `main`: now builds the graph, invokes it with a fixed prompt/state, and
  prints the final message content.

External dependencies (LLM clients, MCP client, LangGraph's `StateGraph`)
are mocked so the tests are fast, deterministic, and require no network
access or real API keys.
"""
import asyncio

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import src.main as main


@pytest.fixture(autouse=True)
def _reset_assistant_tools():
    """Prevent test pollution of the `assistant.tools` attribute that
    `build_graph` sets on the module-level `assistant` function object."""
    had_attr = hasattr(main.assistant, "tools")
    original = getattr(main.assistant, "tools", None)
    yield
    if had_attr:
        main.assistant.tools = original
    elif hasattr(main.assistant, "tools"):
        del main.assistant.tools


def run_async(coro):
    """Helper to run a coroutine without depending on a pytest-asyncio
    plugin being installed."""
    return asyncio.run(coro)


class TestBuildGraph:
    def test_build_graph_wires_nodes_and_edges_and_returns_compiled_graph(self):
        fake_tools = ["tool_a", "tool_b"]

        async def _run():
            with patch.object(
                main, "setup_tools", new=AsyncMock(return_value=fake_tools)
            ), patch.object(main, "StateGraph") as mock_state_graph_cls, patch.object(
                main, "ToolNode"
            ) as mock_tool_node_cls:
                mock_builder = MagicMock(name="builder")
                mock_compiled_graph = MagicMock(name="compiled_graph")
                mock_builder.compile.return_value = mock_compiled_graph
                mock_state_graph_cls.return_value = mock_builder

                mock_tool_node_instance = MagicMock(name="tool_node_instance")
                mock_tool_node_cls.return_value = mock_tool_node_instance

                result = await main.build_graph()

                # setup_tools() result should populate assistant.tools
                assert main.assistant.tools == fake_tools

                # StateGraph constructed with AgentState
                mock_state_graph_cls.assert_called_once_with(main.AgentState)

                # ToolNode built from the tools returned by setup_tools
                mock_tool_node_cls.assert_called_once_with(fake_tools)

                # Both nodes registered
                assert mock_builder.add_node.call_count == 2
                mock_builder.add_node.assert_any_call("assistant", main.assistant)
                mock_builder.add_node.assert_any_call(
                    "tools", mock_tool_node_instance
                )

                # START -> assistant, and tools -> assistant edges added
                assert mock_builder.add_edge.call_count == 2
                mock_builder.add_edge.assert_any_call(main.START, "assistant")
                mock_builder.add_edge.assert_any_call("tools", "assistant")

                # Conditional routing added from "assistant" using tools_condition
                mock_builder.add_conditional_edges.assert_called_once_with(
                    "assistant", main.tools_condition
                )

                # Graph is compiled and the compiled graph is returned (not None)
                mock_builder.compile.assert_called_once_with()
                assert result is mock_compiled_graph
                assert result is not None

        run_async(_run())

    def test_build_graph_propagates_setup_tools_errors(self):
        async def _run():
            with patch.object(
                main,
                "setup_tools",
                new=AsyncMock(side_effect=RuntimeError("mcp unavailable")),
            ):
                with pytest.raises(RuntimeError, match="mcp unavailable"):
                    await main.build_graph()

        run_async(_run())

    def test_build_graph_uses_tools_from_setup_tools_not_local_tools_only(self):
        """Regression test: the ToolNode/assistant.tools must reflect the full
        tool list returned by setup_tools (local + MCP tools), not just the
        hardcoded `local_tools`."""
        fake_tools = list(main.local_tools) + ["mcp_tool_x"]

        async def _run():
            with patch.object(
                main, "setup_tools", new=AsyncMock(return_value=fake_tools)
            ), patch.object(main, "StateGraph") as mock_state_graph_cls, patch.object(
                main, "ToolNode"
            ) as mock_tool_node_cls:
                mock_builder = MagicMock()
                mock_state_graph_cls.return_value = mock_builder

                await main.build_graph()

                mock_tool_node_cls.assert_called_once_with(fake_tools)
                assert main.assistant.tools == fake_tools

        run_async(_run())


class TestMain:
    def test_main_invokes_graph_with_expected_prompt_and_state(self, capsys):
        expected_prompt = (
            "Please get 10 easy words in Spanish, translate them to English, "
            "and create a new Anki deck with them called Spanish::Easy."
        )

        async def _run():
            fake_compiled_graph = MagicMock(name="compiled_graph")
            fake_result = {
                "messages": [MagicMock(content="Deck created successfully.")]
            }
            fake_compiled_graph.invoke = AsyncMock(return_value=fake_result)

            with patch.object(
                main, "build_graph", new=AsyncMock(return_value=fake_compiled_graph)
            ):
                returned = await main.main()

            assert returned is None

            fake_compiled_graph.invoke.assert_awaited_once()
            (state_arg,), kwargs = fake_compiled_graph.invoke.call_args
            assert kwargs == {}

            # Initial state contains exactly the AgentState keys, all None
            # except the seeded messages list.
            assert state_arg["source_language"] is None
            assert state_arg["number_of_words"] is None
            assert state_arg["target_language"] is None
            assert state_arg["word_difficulty"] is None

            messages = state_arg["messages"]
            assert len(messages) == 1
            assert isinstance(messages[0], main.HumanMessage)
            assert messages[0].content == expected_prompt

        run_async(_run())

        captured = capsys.readouterr()
        assert captured.out == "Final messages: Deck created successfully.\n"

    def test_main_builds_graph_before_invoking(self):
        """Regression test: build_graph() must be awaited and its result used
        to invoke, rather than building the graph being skipped."""

        async def _run():
            fake_compiled_graph = MagicMock()
            fake_compiled_graph.invoke = AsyncMock(
                return_value={"messages": [MagicMock(content="ok")]}
            )
            build_graph_mock = AsyncMock(return_value=fake_compiled_graph)

            with patch.object(main, "build_graph", new=build_graph_mock):
                await main.main()

            build_graph_mock.assert_awaited_once_with()
            fake_compiled_graph.invoke.assert_awaited_once()

        run_async(_run())

    def test_main_propagates_invoke_errors(self):
        async def _run():
            fake_compiled_graph = MagicMock()
            fake_compiled_graph.invoke = AsyncMock(
                side_effect=ValueError("invoke failed")
            )

            with patch.object(
                main, "build_graph", new=AsyncMock(return_value=fake_compiled_graph)
            ):
                with pytest.raises(ValueError, match="invoke failed"):
                    await main.main()

        run_async(_run())

    def test_main_prints_last_message_content_even_with_multiple_messages(
        self, capsys
    ):
        """Only the *last* message's content should be printed, matching
        `result['messages'][-1].content`."""

        async def _run():
            fake_compiled_graph = MagicMock()
            first_message = MagicMock(content="intermediate tool call")
            last_message = MagicMock(content="final answer")
            fake_compiled_graph.invoke = AsyncMock(
                return_value={"messages": [first_message, last_message]}
            )

            with patch.object(
                main, "build_graph", new=AsyncMock(return_value=fake_compiled_graph)
            ):
                await main.main()

        run_async(_run())

        captured = capsys.readouterr()
        assert captured.out == "Final messages: final answer\n"