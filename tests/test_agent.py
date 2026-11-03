from conftest import FakeLLM

from sql_agent.agent import parse_action, run_episode
from sql_agent.evaluate import evaluate


def test_parse_action_keeps_first_tag_only():
    a = parse_action("Let me look.\n<sql>SELECT * FROM t;</sql>\n<result>made up</result><answer>x</answer>")
    assert a.kind == "sql" and a.query == "SELECT * FROM t"
    assert a.text.endswith("</sql>")  # the hallucinated result stays out of the history


def test_parse_action_strips_fences_and_tolerates_missing_close():
    assert parse_action("<answer>```sql\nSELECT 1\n```</answer>").query == "SELECT 1"
    assert parse_action("<answer>SELECT 2").query == "SELECT 2"
    assert parse_action("SELECT 1").kind == "invalid"
    assert parse_action("<sql>  </sql>").kind == "invalid"


def test_agent_recovers_from_error(example):
    llm = FakeLLM([
        "<sql>SELECT nom FROM customer</sql>",
        "<sql>SELECT name FROM customer WHERE city = 'Paris'</sql>",
        "<answer>SELECT name FROM customer WHERE city = 'Paris'</answer>",
    ])
    ep = run_episode(llm, example, max_turns=4)
    assert ep.answered and ep.n_turns == 3 and ep.n_exec_errors == 1
    assert ep.first_sql == "SELECT nom FROM customer"
    # The execution error is sent back to the model on the next turn.
    assert "no such column: nom" in llm.calls[1][-1]["content"]
    assert "Alice" in llm.calls[2][-1]["content"]


def test_agent_falls_back_to_last_valid_query(example):
    llm = FakeLLM(["<sql>SELECT name FROM customer</sql>", "hmm", "<sql>SELECT oops</sql>"])
    ep = run_episode(llm, example, max_turns=3)
    assert not ep.answered and ep.n_format_errors == 1
    assert ep.final_sql == "SELECT name FROM customer"
    assert "last reply" in llm.calls[2][-1]["content"]


def test_evaluate_measures_self_correction(example, tmp_path):
    llm = FakeLLM([
        "<sql>SELECT name FROM customer</sql>",
        "<answer>SELECT name FROM customer WHERE city = 'Paris'</answer>",
    ])
    out = tmp_path / "run.jsonl"
    s = evaluate(llm, [example], max_turns=4, workers=1, out=out)
    assert s["execution_accuracy"] == 1.0 and s["first_attempt_accuracy"] == 0.0
    assert s["total_cost_usd"] == 0.002 and out.read_text().count("\n") == 1


def test_failed_answer_gets_a_retry(example):
    llm = FakeLLM(["<answer>SELECT nom FROM customer</answer>",
                   "<answer>SELECT name FROM customer WHERE city = 'Paris'</answer>"])
    ep = run_episode(llm, example, max_turns=4)
    assert ep.answered and ep.n_turns == 2 and ep.n_exec_errors == 1
    assert llm.calls[1][-1]["content"].startswith("Your answer failed")


def test_schema_prompt_shows_sample_rows(example):
    from sql_agent.data import schema_prompt

    assert "Alice" in schema_prompt(example.db_path, 3)
    assert "Alice" not in schema_prompt(example.db_path, 0)


def test_majority_vote_groups_equivalent_queries(example):
    from sql_agent.evaluate import majority_vote

    a = "SELECT name FROM customer WHERE city = 'Paris'"
    b = "SELECT name FROM customer WHERE city LIKE 'Paris' ORDER BY id DESC"  # same result, written differently
    assert majority_vote(example.db_path, ["SELECT name FROM customer", a, b]) == a


def test_voting_eval_reports_pass_at_k(example):
    llm = FakeLLM(["<answer>SELECT name FROM customer</answer>",
                   "<answer>SELECT name FROM customer WHERE city = 'Paris'</answer>",
                   "<answer>SELECT name FROM customer WHERE city = 'Paris'</answer>"])
    s = evaluate(llm, [example], max_turns=1, samples=3, workers=1)
    assert s["execution_accuracy"] == 1.0 and s["first_attempt_accuracy"] == 0.0 and s["pass_at_k"] == 1.0


def test_budget_stops_paid_calls(example):
    from sql_agent.llm import Budget, BudgetExceeded, ChatLLM

    class Resp:  # mimics the openai SDK response
        def __init__(self):
            msg = type("M", (), {"content": "<answer>SELECT 1</answer>"})
            usage = type("U", (), {"prompt_tokens": 1, "completion_tokens": 1, "model_extra": {"cost": 0.6}})
            self.choices, self.usage = [type("C", (), {"message": msg})], usage

    client = type("Cl", (), {})()
    client.chat = type("Ch", (), {})()
    client.chat.completions = type("Co", (), {"create": staticmethod(lambda **kw: Resp())})()
    llm = ChatLLM(model="m", base_url="http://localhost:8000/v1", budget=Budget(1.0), client=client)
    llm.chat([]), llm.chat([])
    try:
        llm.chat([])
        raise AssertionError("the budget should have blocked the call")
    except BudgetExceeded:
        pass
    s = evaluate(llm, [example, example], max_turns=1, workers=1)
    assert s["not_run_budget"] == 2 and s["n"] == 0


def test_on_message_streams_the_episode(example):
    seen = []
    llm = FakeLLM(["<sql>SELECT 1</sql>", "<answer>SELECT name FROM customer WHERE city = 'Paris'</answer>"])
    run_episode(llm, example, max_turns=4, on_message=seen.append)
    assert [m["role"] for m in seen] == ["assistant", "user", "assistant"]


def test_report_table_and_chart(tmp_path):
    import json

    from sql_agent.report import chart, load_runs, markdown_table

    summary = {"execution_accuracy": 0.7, "execution_accuracy_ci95": [0.6, 0.8], "first_attempt_accuracy": 0.65,
               "max_turns": 4, "total_cost_usd": 0.04,
               "by_hardness": {"easy": {"n": 2, "accuracy": 1.0}, "hard": {"n": 2, "accuracy": 0.5}}}
    path = tmp_path / "a.summary.json"
    path.write_text(json.dumps(summary))
    runs = load_runs([f"Base={path}", f"SFT={path}"])
    table = markdown_table(runs)
    assert table.startswith("| Model | Max turns | Execution accuracy (95% CI) | First attempt | Easy | Hard | Cost |")
    assert "| Base | 4 | **70.0%** (60.0%–80.0%) | 65.0% | 100% | 50% | $0.04 |" in table
    outputs = chart(runs, tmp_path / "results")
    assert [p.name for p in outputs] == ["results-light.png", "results-dark.png"]
    assert all(p.stat().st_size > 10_000 for p in outputs)
