"""补评分派（最大分派）纯算法测试。

用独立的 Dinic 最大流做可行性/最优性 oracle，逐边贪心校验
「评委 → 项目 → 选手」固定顺序下的唯一规范方案（含并列结果稳定性）。
"""
import os
import random
import sys
import unittest
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.assignment import (  # noqa: E402
    AssignmentValidationError,
    normalize_capacities,
    normalize_target,
    plan_supplement_assignments,
)


def recs(rows):
    return [{"contestant": c, "judge": j, "item": i} for c, j, i in rows]


class _OracleEdge:
    __slots__ = ("to", "rev", "cap")

    def __init__(self, to, rev, cap):
        self.to, self.rev, self.cap = to, rev, cap


class _OracleNet:
    """与被测代码相互独立的 Dinic 实现（用于做可行性 oracle）。"""

    def __init__(self, n):
        self.g = [[] for _ in range(n)]

    def add(self, u, v, c):
        fwd = _OracleEdge(v, len(self.g[v]), c)
        self.g[u].append(fwd)
        rev = _OracleEdge(u, len(self.g[u]) - 1, 0)
        self.g[v].append(rev)
        fwd.rev = len(self.g[v]) - 1

    def max_flow(self, S, T):
        total = 0
        while True:
            level = [-1] * len(self.g)
            level[S] = 0
            q = deque([S])
            while q:
                x = q.popleft()
                for e in self.g[x]:
                    if e.cap > 0 and level[e.to] < 0:
                        level[e.to] = level[x] + 1
                        q.append(e.to)
            if level[T] < 0:
                break
            it = [0] * len(self.g)

            def dfs(x, pushed):
                if x == T:
                    return pushed
                while it[x] < len(self.g[x]):
                    e = self.g[x][it[x]]
                    if e.cap > 0 and level[e.to] == level[x] + 1:
                        d = dfs(e.to, min(pushed, e.cap))
                        if d:
                            e.cap -= d
                            self.g[e.to][e.rev].cap += d
                            return d
                    it[x] += 1
                return 0

            while True:
                f = dfs(S, 10**18)
                if not f:
                    break
                total += f
        return total


def _structure(records, target):
    jo = list(dict.fromkeys(j for _, j, _ in records))
    io = list(dict.fromkeys(i for _, _, i in records))
    co = list(dict.fromkeys(c for c, _, _ in records))
    item_judges = {}
    have = {}
    for c, j, i in records:
        item_judges.setdefault(i, set()).add(j)
        have.setdefault((c, i), set()).add(j)
    units = sorted(have, key=lambda k: (io.index(k[1]), co.index(k[0])))
    needs = {u: max(0, target - len(have[u])) for u in units}
    edges = []
    for (c, i) in units:
        for j in jo:
            if j in item_judges[i] and j not in have[(c, i)]:
                edges.append((j, c, i))
    edges.sort(key=lambda e: (jo.index(e[0]), io.index(e[2]), co.index(e[1])))
    return jo, units, needs, edges


def max_with_forced(records, target, caps, forced=frozenset()):
    """固定 forced 边后，剩余网络还能达到的最大分派总数；非法 forced 返回 -1。"""
    jo, units, needs, edges = _structure(records, target)
    jload, uload = {}, {}
    for j, c, i in forced:
        jload[j] = jload.get(j, 0) + 1
        uload[(c, i)] = uload.get((c, i), 0) + 1
    if any(jload.get(j, 0) > caps.get(j, 0) for j in jload):
        return -1
    if any(uload.get(u, 0) > needs[u] for u in uload):
        return -1
    S = 0
    jn = {j: 1 + k for k, j in enumerate(jo)}
    un = {u: 1 + len(jo) + k for k, u in enumerate(units)}
    T = 1 + len(jo) + len(units)
    net = _OracleNet(T + 1)
    for j in jo:
        net.add(S, jn[j], caps.get(j, 0) - jload.get(j, 0))
    for u in units:
        net.add(un[u], T, needs[u] - uload.get(u, 0))
    for e in edges:
        if e in forced:
            continue
        j, c, i = e
        net.add(jn[j], un[(c, i)], 1)
    return sum(jload.values()) + net.max_flow(S, T)


def canonical_plan(records, target, caps):
    """规范贪心：按固定边序，保留「仍可扩展到最大总数」的边。"""
    _, _, _, edges = _structure(records, target)
    m_star = max_with_forced(records, target, caps)
    chosen = set()
    for e in edges:
        if max_with_forced(records, target, caps, frozenset(chosen | {e})) >= m_star:
            chosen.add(e)
    return m_star, chosen


def _random_instance(rng):
    judges = [f"J{k}" for k in range(rng.randint(2, 6))]
    items = [f"I{k}" for k in range(rng.randint(1, 4))]
    records = []
    for i in items:
        js = rng.sample(judges, rng.randint(2, len(judges)))
        contestants = [f"C{k}" for k in range(rng.randint(1, 7))]
        for c in contestants:
            for j in js:
                if rng.random() < rng.uniform(0.3, 0.95):
                    records.append((c, j, i))
    target = rng.randint(1, len(judges) + 1)
    caps = {j: rng.randint(0, 5) for j in judges}
    return records, target, caps


class TestSupplementAssignment(unittest.TestCase):
    def test_basic_gap_and_capacity(self):
        # 项目 I1 评委 A/B/C（C 评过 W）。X 有 A、Y 有 B、W 有 C；目标 3，容量各 1。
        r = recs([("X", "A", "I1"), ("Y", "B", "I1"), ("W", "C", "I1")])
        res = plan_supplement_assignments(r, 3, {"A": 1, "B": 1, "C": 1})
        edges = [(a["judge"], a["contestant"]) for a in res["assignments"]]
        self.assertEqual(res["summary"]["total_need"], 6)
        self.assertEqual(res["summary"]["assigned_total"], 3)
        self.assertEqual(res["summary"]["unmet_total"], 3)
        # 规范顺序：A 最早可接 Y；B 最早可接 X；C 最早可接 X。
        self.assertEqual(edges, [("A", "Y"), ("B", "X"), ("C", "X")])
        by = {(u["contestant"]): u for u in res["units"]}
        self.assertEqual(by["X"]["unmet"], 0)
        self.assertEqual(by["Y"]["unmet"], 1)
        self.assertEqual(by["W"]["unmet"], 2)
        self.assertEqual(by["X"]["eligible_judges"], ["B", "C"])

    def test_judge_must_have_scored_item(self):
        # D 从未评过 I1，即使有容量也不能接 I1 的任何单元。
        r = recs([("X", "A", "I1"), ("Y", "B", "I1")])
        res = plan_supplement_assignments(r, 3, {"A": 9, "B": 9, "D": 9})
        # I1 评委只有 A/B，X、Y 各缺 2 但候选评委只有对方 1 人 -> 至多 2 单。
        self.assertEqual(res["summary"]["assigned_total"], 2)
        self.assertTrue(all(a["judge"] in ("A", "B") for a in res["assignments"]))

    def test_cross_item_eligibility(self):
        # 评委 A 只评过 I2，不能接 I1 的单元；评委 B 只评过 I1，不能接 I2。
        r = recs([("X", "A", "I2"), ("Y", "B", "I1")])
        res = plan_supplement_assignments(r, 2, {"A": 5, "B": 5})
        self.assertEqual(res["summary"]["assigned_total"], 0)
        for u in res["units"]:
            self.assertEqual(u["eligible_judges"], [])

    def test_target_deducts_existing(self):
        # 单元已有 2 名评委、目标 2 -> 缺口 0，不产生任务也不计未满足。
        r = recs([("X", "A", "I1"), ("X", "B", "I1"), ("Y", "A", "I1")])
        res = plan_supplement_assignments(r, 2, {"A": 9, "B": 9})
        x = next(u for u in res["units"] if u["contestant"] == "X")
        y = next(u for u in res["units"] if u["contestant"] == "Y")
        self.assertEqual(x["need"], 0)
        self.assertEqual(x["assigned"], 0)
        self.assertEqual(y["need"], 1)
        self.assertEqual(res["summary"]["already_complete"], 1)
        self.assertEqual(res["summary"]["units_with_need"], 1)
        self.assertEqual([(a["judge"], a["contestant"]) for a in res["assignments"]],
                         [("B", "Y")])

    def test_zero_capacity(self):
        r = recs([("X", "A", "I1"), ("Y", "B", "I1")])
        res = plan_supplement_assignments(r, 2, {"A": 0, "B": 0})
        self.assertEqual(res["summary"]["assigned_total"], 0)
        self.assertEqual(res["summary"]["unmet_total"], 2)
        for j in res["judges"]:
            self.assertEqual(j["assigned_count"], 0)
            self.assertEqual(j["remaining"], 0)

    def test_empty_records(self):
        res = plan_supplement_assignments([], 2, {})
        self.assertEqual(res["units"], [])
        self.assertEqual(res["assignments"], [])
        self.assertEqual(res["summary"]["assigned_total"], 0)

    def test_stable_output_order(self):
        r = recs([("Z", "B", "I2"), ("A", "A", "I1"), ("M", "A", "I2"),
                  ("Z", "A", "I2"), ("A", "B", "I1")])
        res = plan_supplement_assignments(r, 2, {"A": 5, "B": 5})
        # assignments 按 评委 → 项目 → 选手 固定顺序。
        keys = [(a["judge"], a["item"], a["contestant"]) for a in res["assignments"]]
        self.assertEqual(keys, sorted(keys))
        # 重复调用结果完全一致（并列方案稳定性）。
        res2 = plan_supplement_assignments(r, 2, {"A": 5, "B": 5})
        self.assertEqual(res["assignments"], res2["assignments"])

    def test_constraints_hold_and_maximal_against_oracle(self):
        rng = random.Random(20240924)
        for _ in range(300):
            records, target, caps = _random_instance(rng)
            if not records:
                continue
            res = plan_supplement_assignments(recs(records), target, caps)
            # 容量约束。
            used = {j["judge"]: j["assigned_count"] for j in res["judges"]}
            for j, n in used.items():
                self.assertLessEqual(n, caps[j])
            # 每单元分派不超过缺口，且分派评委确实评过该项目、未评过该单元。
            have = {}
            item_judges = {}
            for c, j, i in records:
                have.setdefault((c, i), set()).add(j)
                item_judges.setdefault(i, set()).add(j)
            per_unit = {}
            for a in res["assignments"]:
                self.assertIn(a["judge"], item_judges[a["item"]])
                self.assertNotIn(a["judge"], have[(a["contestant"], a["item"])])
                per_unit[(a["contestant"], a["item"])] = \
                    per_unit.get((a["contestant"], a["item"]), 0) + 1
            for u in res["units"]:
                self.assertLessEqual(
                    per_unit.get((u["contestant"], u["item"]), 0), u["need"])
            # 最优性 + 规范方案与独立 oracle 完全一致。
            m_star, chosen = canonical_plan(records, target, caps)
            got = {(a["judge"], a["contestant"], a["item"]) for a in res["assignments"]}
            self.assertEqual(res["summary"]["assigned_total"], m_star)
            self.assertEqual(res["summary"]["max_possible"], m_star)
            self.assertEqual(got, chosen)

    def test_validation_errors(self):
        with self.assertRaises(AssignmentValidationError):
            normalize_target(0)
        with self.assertRaises(AssignmentValidationError):
            normalize_target(True)  # bool 不算整数
        with self.assertRaises(AssignmentValidationError):
            normalize_target("3")
        judges = ["A", "B"]
        with self.assertRaises(AssignmentValidationError):
            normalize_capacities([{"judge": "A", "capacity": 1}], judges)  # 缺 B
        with self.assertRaises(AssignmentValidationError):
            normalize_capacities([{"judge": "X", "capacity": 1}], judges)  # 评委不存在
        with self.assertRaises(AssignmentValidationError):
            normalize_capacities(
                [{"judge": "A", "capacity": 1}, {"judge": "A", "capacity": 2}],
                judges,
            )  # 重复
        with self.assertRaises(AssignmentValidationError):
            normalize_capacities(
                [{"judge": "A", "capacity": -1}, {"judge": "B", "capacity": 0}],
                judges,
            )
        with self.assertRaises(AssignmentValidationError):
            normalize_capacities("not-a-list", judges)
        # 合法输入通过。
        self.assertEqual(
            normalize_capacities(
                [{"judge": "B", "capacity": 3}, {"judge": "A", "capacity": 0}], judges
            ),
            {"B": 3, "A": 0},
        )


if __name__ == "__main__":
    unittest.main()
