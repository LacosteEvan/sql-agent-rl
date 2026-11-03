import random

from sql_agent.data import resolve_db
from sql_agent.hardness import hardness
from sql_agent.rl import correctness_reward, format_reward
from sql_agent.stats import bootstrap_ci, paired_bootstrap
from sql_agent.synth import build_examples, clean_sql, corrupt, rows_from_run


def _sql(**over):
    base = {"select": [False, [[0, [0, [0, 1, False], None]]]], "from": {"table_units": [["table_unit", 0]],
            "conds": []}, "where": [], "groupBy": [], "having": [], "orderBy": [], "limit": None,
            "intersect": None, "union": None, "except": None}
    return base | over


def test_hardness_levels():
    assert hardness(_sql()) == "easy"
    two_cols = _sql(select=[False, [[0, [0, [0, 1, False], None]], [0, [0, [0, 2, False], None]]]])
    assert hardness(two_cols | {"where": [[False, 2, [0, [0, 1, False], None], 1.0, None]]}) == "medium"
    assert hardness(_sql(intersect=_sql())) == "hard"
    assert hardness(_sql(intersect=_sql(), union=_sql())) == "extra"


def test_corrupt_produces_a_real_error(db):
    broken, res = corrupt("SELECT name FROM customer WHERE city = 'Paris'", db, random.Random(0))
    assert not res.ok and ("no such column" in res.error or "no such table" in res.error)
    assert "'Paris'" in broken  # string literals are never modified


def test_build_examples_direct_and_correction(example):
    rows = build_examples([example], correction_rate=1.0)
    kinds = [r["kind"] for r in rows]
    assert kinds == ["direct", "correction"]
    fix = rows[1]
    assert fix["prompt"][-1]["content"].startswith("Your answer failed")
    # The broken query is never a training target: it only appears in the prompt.
    assert fix["completion"][0]["content"] == f"<answer>{example.gold_sql}</answer>"
    assert resolve_db(fix["db_path"]) == example.db_path.resolve()


def test_clean_sql():
    assert clean_sql("SELECT T1.name ,  T2.age FROM t") == "SELECT T1.name, T2.age FROM t"


def test_rows_from_run_skips_failed_turns():
    rec = {"correct": True, "gold_sql": "g", "db_path": "d", "example_id": "x", "messages": [
        {"role": "system", "content": "s"}, {"role": "user", "content": "q"},
        {"role": "assistant", "content": "<sql>bad</sql>"},
        {"role": "user", "content": "<result>\nERROR: x\n</result>"},
        {"role": "assistant", "content": "<answer>good</answer>"}]}
    rows = rows_from_run([rec, rec | {"correct": False}])
    assert [r["completion"][0]["content"] for r in rows] == ["<answer>good</answer>"]
    assert len(rows[0]["prompt"]) == 4


def test_grpo_rewards(example):
    good = [{"role": "assistant", "content": f"<answer>{example.gold_sql}</answer>"}]
    explore = [{"role": "assistant", "content": f"<sql>{example.gold_sql}</sql>"}]
    junk = [{"role": "assistant", "content": "no idea"}]
    kw = {"gold_sql": [example.gold_sql] * 3, "db_path": [str(example.db_path)] * 3}
    assert correctness_reward([good, explore, junk], **kw) == [1.0, 0.0, 0.0]
    assert format_reward([good, explore, junk]) == [0.1, 0.1, 0.0]


def test_bootstrap():
    lo, hi = bootstrap_ci([1.0] * 70 + [0.0] * 30)
    assert lo < 0.7 < hi and hi - lo < 0.2
    res = paired_bootstrap([0.0] * 50 + [1.0] * 50, [1.0] * 100)
    assert res["delta"] == 0.5 and res["p_not_better"] == 0.0
