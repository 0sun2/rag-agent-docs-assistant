"""Agent CLI — ReAct 에이전트 스모크 테스트 & 데모.

사용:
    uv run python -m src.agent.cli "How do I create a custom tool in LangChain?"

현재 바인딩된 tool: docs_search (Phase 3 프로덕션 구성 재사용).
다른 tool (code_generate / web_search / error_analyze) 은 다음 단계에서 추가 예정.
"""

from __future__ import annotations

import argparse
import logging

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.agent.graph.agent import build_agent_graph


def main() -> None:
    parser = argparse.ArgumentParser(description="ReAct docs assistant agent")
    parser.add_argument("question", help="User question")
    parser.add_argument("-v", "--verbose", action="store_true", help="Trace intermediate steps")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    graph = build_agent_graph()

    initial_state = {
        "messages": [HumanMessage(content=args.question)],
        "tool_results": [],
        "current_query": args.question,
        "iteration_count": 0,
    }

    print(f"\n❓ {args.question}\n")
    final_state = graph.invoke(initial_state)

    if args.verbose:
        print("─── Trace ───")
        for msg in final_state["messages"]:
            if isinstance(msg, HumanMessage):
                print(f"[user] {msg.content}")
            elif isinstance(msg, AIMessage):
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        print(f"[agent→tool] {tc['name']}({tc['args']})")
                if msg.content:
                    print(f"[agent] {msg.content[:300]}")
            elif isinstance(msg, ToolMessage):
                preview = str(msg.content)[:200].replace("\n", " ")
                print(f"[tool:{msg.name}] {preview}...")
        print("─────────────\n")

    # 최종 답변은 마지막 AIMessage (tool_calls 없는 것)
    answer = next(
        (
            m.content
            for m in reversed(final_state["messages"])
            if isinstance(m, AIMessage) and not m.tool_calls
        ),
        "(no final answer)",
    )
    print(f"🤖 {answer}\n")
    print(f"(iterations: {final_state['iteration_count']})")


if __name__ == "__main__":
    main()
