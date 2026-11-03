import pytest

from sql_agent.reward import GoldError, score
from sql_agent.sandbox import execute, format_observation


def test_execute_returns_rows(db):
    res = execute(db, "SELECT name FROM customer WHERE age = 25 ORDER BY id")
    assert res.ok and res.columns == ["name"] and res.rows == [("Bob",), ("David",)]


def test_writes_are_rejected(db):
    res = execute(db, "DELETE FROM customer")
    assert not res.ok and "readonly" in res.error
    assert execute(db, "SELECT count(*) FROM customer").rows == [(4,)]


def test_multiple_statements_and_attach_are_rejected(db, tmp_path):
    assert not execute(db, "SELECT 1; SELECT 2").ok
    assert not execute(db, f"ATTACH DATABASE '{tmp_path / 'x.db'}' AS x").ok


def test_runaway_query_times_out(db):
    res = execute(db, "WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM n) SELECT count(*) FROM n",
                  timeout_s=0.2)
    assert not res.ok and res.error.startswith("Timeout")


def test_row_cap_marks_truncation(db):
    res = execute(db, "SELECT * FROM customer", max_rows=2)
    assert res.truncated and len(res.rows) == 2
    assert format_observation(res).startswith("2+ rows")


def test_syntax_error_is_reported_not_raised(db):
    res = execute(db, "SELEC name FROM customer")
    assert not res.ok and "syntax error" in res.error
    assert format_observation(res).startswith("ERROR:")


GOLD = "SELECT name, age FROM customer WHERE city = 'Paris'"


@pytest.mark.parametrize("pred, correct", [
    (GOLD, True),
    ("SELECT age, name FROM customer WHERE city = 'Paris'", True),       # columns swapped
    ("SELECT name, age FROM customer WHERE city = 'Paris' ORDER BY age DESC", True),  # any row order
    ("SELECT name FROM customer WHERE city = 'Paris'", False),            # missing column
    ("SELECT name, age, id FROM customer WHERE city = 'Paris'", False),   # extra column
    ("SELECT name, age FROM customer", False),                            # wrong filter
])
def test_score_unordered(db, pred, correct):
    assert score(pred, GOLD, db).correct is correct


def test_order_matters_when_gold_orders(db):
    gold = "SELECT name FROM customer ORDER BY age, id"
    assert score("SELECT name FROM customer ORDER BY age, id", gold, db).correct
    assert not score("SELECT name FROM customer ORDER BY age DESC", gold, db).correct


def test_int_and_float_results_are_equal(db):
    assert score("SELECT 30.0", "SELECT 30", db).correct
    assert score("SELECT sum(amount) FROM orders", "SELECT 37.75", db).correct


def test_failing_prediction_and_gold(db):
    v = score("SELECT nope FROM customer", GOLD, db)
    assert not v.correct and not v.executed and "no such column" in v.error
    assert not score(None, GOLD, db).correct
    with pytest.raises(GoldError):
        score(GOLD, "SELECT broken FROM nowhere", db)


def test_bird_metric_compares_row_sets(db):
    gold = "SELECT name FROM customer WHERE age = 25 ORDER BY id"
    # BIRD ignores row order and duplicates, even when the gold query is ordered...
    assert score("SELECT name FROM customer WHERE age = 25 ORDER BY id DESC", gold, db, metric="bird").correct
    assert score("SELECT name FROM customer WHERE age = 25 UNION ALL SELECT 'Bob'", gold, db, metric="bird").correct
    # ...but keeps column order, unlike Spider.
    two = "SELECT name, age FROM customer WHERE city = 'Paris'"
    swapped = "SELECT age, name FROM customer WHERE city = 'Paris'"
    assert score(swapped, two, db).correct and not score(swapped, two, db, metric="bird").correct


def test_evidence_is_shown_to_the_model(example):
    from dataclasses import replace

    from sql_agent.agent import initial_messages

    hinted = replace(example, evidence="Paris refers to city = 'Paris'")
    assert initial_messages(hinted, 4)[-1]["content"].endswith("Evidence: Paris refers to city = 'Paris'")
    assert "Evidence" not in initial_messages(example, 4)[-1]["content"]
