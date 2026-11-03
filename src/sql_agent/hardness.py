"""Spider difficulty level (easy / medium / hard / extra), computed from the parsed query (`sql` in the Spider
JSON files). Faithfully mirrors `eval_hardness` from the official script, approximations included, so that
per-level numbers are comparable with published ones."""
from __future__ import annotations

_LIKE = 9  # index of "like" in WHERE_OPS in the official script


def _has_agg(unit) -> bool:
    return unit[0] != 0


def _count_agg(units) -> int:
    return sum(1 for u in units if _has_agg(u))


def _conds(sql: dict) -> list:
    return sql["from"]["conds"][::2] + sql["where"][::2] + sql["having"][::2]


def _nested(sql: dict) -> list[dict]:
    nested = [v for c in _conds(sql) for v in (c[3], c[4]) if isinstance(v, dict)]
    return nested + [sql[k] for k in ("intersect", "except", "union") if sql[k] is not None]


def _component1(sql: dict) -> int:
    count = sum(bool(sql[k]) for k in ("where", "groupBy", "orderBy"))
    count += sql["limit"] is not None
    if sql["from"]["table_units"]:
        count += len(sql["from"]["table_units"]) - 1  # joins
    logic = sql["from"]["conds"][1::2] + sql["where"][1::2] + sql["having"][1::2]
    count += sum(1 for t in logic if t == "or")
    count += sum(1 for c in _conds(sql) if c[1] == _LIKE)
    return count


def _others(sql: dict) -> int:
    agg = _count_agg(sql["select"][1]) + _count_agg(sql["where"][::2]) + _count_agg(sql["groupBy"])
    if sql["orderBy"]:
        units = sql["orderBy"][1]
        agg += _count_agg([u[1] for u in units if u[1]] + [u[2] for u in units if u[2]])
    agg += _count_agg(sql["having"])
    return (agg > 1) + (len(sql["select"][1]) > 1) + (len(sql["where"]) > 1) + (len(sql["groupBy"]) > 1)


def hardness(sql: dict) -> str:
    c1, c2, o = _component1(sql), len(_nested(sql)), _others(sql)
    if c1 <= 1 and o == 0 and c2 == 0:
        return "easy"
    if (o <= 2 and c1 <= 1 and c2 == 0) or (c1 <= 2 and o < 2 and c2 == 0):
        return "medium"
    if (o > 2 and c1 <= 2 and c2 == 0) or (2 < c1 <= 3 and o <= 2 and c2 == 0) or (c1 <= 1 and o == 0 and c2 <= 1):
        return "hard"
    return "extra"
