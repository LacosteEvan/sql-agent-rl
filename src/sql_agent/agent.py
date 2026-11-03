"""Agent loop: the model explores the database, reads errors, corrects itself, then returns its final query.

The protocol is deliberately textual (<sql>…</sql> to explore, <answer>…</answer> to conclude) rather than
tool calling: a small open model learns it quickly, and the very same conversations serve as evaluation
traces, SFT data and GRPO prompts.
"""
from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

from .data import Example, schema_prompt
from .llm import LLM, Completion
from .sandbox import ExecResult, execute, format_observation

SYSTEM_PROMPT = """You are an expert data analyst answering questions about a SQLite database.

In each reply, think briefly, then output exactly one of:
<sql>QUERY</sql>       runs QUERY to explore the data; you will see the first rows of the result or the error.
<answer>QUERY</answer> your final SQL query. It is executed: if it fails, you will see the error and can fix it.

Rules:
- SQLite dialect, read-only, a single statement per tag.
- The final query must return exactly the columns the question asks for, nothing more.
- You have at most {max_turns} replies."""

_TAG = re.compile(r"<(sql|answer)>(.*?)(?:</\1>|$)", re.DOTALL | re.IGNORECASE)
_FENCE = re.compile(r"^```(?:sql)?\s*|\s*```$", re.IGNORECASE)


@dataclass
class Action:
    kind: str  # "sql", "answer" or "invalid"
    query: str = ""
    text: str = ""  # the model's reply, cut after the first closing tag


def parse_action(text: str) -> Action:
    """Read the reply's first tag and ignore the rest (models sometimes hallucinate the query result)."""
    m = _TAG.search(text)
    if not m:
        return Action(kind="invalid", text=text)
    query = _FENCE.sub("", m.group(2).strip()).strip().rstrip(";").strip()
    kind = m.group(1).lower()
    return Action(kind=kind if query else "invalid", query=query, text=text[: m.end()])


@dataclass
class Episode:
    example_id: str
    messages: list[dict]
    final_sql: str | None = None
    answered: bool = False  # ended with an <answer> that executed (otherwise: fallback)
    first_sql: str | None = None  # first query attempted: measures what self-correction brings
    n_turns: int = 0
    n_exec_errors: int = 0
    n_format_errors: int = 0
    usage: list[Completion] = field(default_factory=list)


def initial_messages(ex: Example, max_turns: int, sample_rows: int = 3) -> list[dict]:
    schema = schema_prompt(ex.db_path, sample_rows)
    evidence = f"\nEvidence: {ex.evidence}" if ex.evidence else ""
    return [
        {"role": "system", "content": SYSTEM_PROMPT.format(max_turns=max_turns)},
        {"role": "user", "content": f"Database schema:\n{schema}\n\nQuestion: {ex.question}{evidence}"},
    ]


def feedback_message(res: ExecResult | None, remaining: int, after_answer: bool = False) -> dict:
    """Message sent back to the model after a reply. Shared with training-data generation, so that the
    model trains on exactly what it will see at evaluation time."""
    if res is None:
        body = "FORMAT ERROR: reply with exactly one <sql>…</sql> or <answer>…</answer> block."
    elif after_answer:
        body = f"Your answer failed:\n<result>\n{format_observation(res)}\n</result>"
    else:
        body = f"<result>\n{format_observation(res)}\n</result>"
    hint = "This is your last reply: give your <answer>." if remaining == 1 else f"{remaining} replies left."
    return {"role": "user", "content": f"{body}\n{hint}"}


def run_episode(llm: LLM, ex: Example, max_turns: int = 4, sample_rows: int = 3, timeout_s: float = 5.0,
                on_message: Callable[[dict], None] | None = None) -> Episode:
    """Play one question. `on_message` sees each new message as it happens (used by the live demo)."""
    ep = Episode(example_id=ex.id, messages=initial_messages(ex, max_turns, sample_rows))

    def add(message: dict) -> None:
        ep.messages.append(message)
        if on_message:
            on_message(message)

    last_valid_sql = None
    for turn in range(1, max_turns + 1):
        completion = llm.chat(ep.messages)
        ep.usage.append(completion)
        ep.n_turns = turn
        action = parse_action(completion.text)
        add({"role": "assistant", "content": action.text})
        remaining = max_turns - turn

        if action.kind == "invalid":
            ep.n_format_errors += 1
            res = None
        else:
            ep.first_sql = ep.first_sql or action.query
            res = execute(ex.db_path, action.query, timeout_s=timeout_s, max_rows=1000)
            if res.ok:
                last_valid_sql = action.query
                if action.kind == "answer":
                    ep.final_sql, ep.answered = action.query, True
                    return ep
            else:
                ep.n_exec_errors += 1
        if remaining == 0:
            break
        add(feedback_message(res, remaining, after_answer=action.kind == "answer"))

    ep.final_sql = last_valid_sql
    return ep
