import sqlite3

import pytest

from sql_agent.data import Example
from sql_agent.llm import Completion


class FakeLLM:
    """Replays scripted replies and records the conversations it receives."""

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.calls: list[list[dict]] = []

    def chat(self, messages):
        self.calls.append([dict(m) for m in messages])
        return Completion(text=self.replies.pop(0), prompt_tokens=10, completion_tokens=5, cost_usd=0.001)


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "shop.sqlite"
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE customer (id INTEGER PRIMARY KEY, name TEXT, city TEXT, age INTEGER);
        INSERT INTO customer VALUES (1, 'Alice', 'Paris', 30), (2, 'Bob', 'Lyon', 25),
                                    (3, 'Chloé', 'Paris', 41), (4, 'David', NULL, 25);
        CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER, amount REAL);
        INSERT INTO orders VALUES (1, 1, 10.5), (2, 1, 20.0), (3, 3, 7.25);
    """)
    conn.close()
    return path


@pytest.fixture
def example(db):
    return Example(id="t-0", question="Which customers live in Paris?",
                   gold_sql="SELECT name FROM customer WHERE city = 'Paris'", db_id="shop", db_path=db)
