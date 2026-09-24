"""补评分派：在评委容量与每单元缺口约束下求最大分派（纯只读计算）。

需求单元 = 当前会话中**已有评分**的「选手 × 项目」组合：

- 用户为所有单元统一设定「目标评委人数」，扣除该单元已有评分后得到缺口；
- 每位评委分别设定「本轮可接任务数」容量；
- 只有**已给该项目打过分、但尚未给该单元打分**的评委才能接该单元的单。

这是一个带容量的二部图最大基数匹配问题（源点→评委（容量）→单元（单位边）
→汇点（缺口）），用最大流求分派任务总数最大的方案。最大最优方案往往不止一
个，为在并列结果中稳定输出唯一方案，先求出任意一个最大流，再按
「评委 → 项目 → 选手」的固定顺序做 matroid 式贪心强制：对每条候选边，若
在已强制接受的边之外仍能保持最大总数则接受，否则拒绝；可行性由残余网络中
是否存在增广圈（等价于单元→评委的残余有向路）判定，因此结果与初始最大流
的选取无关，完全确定。

整个模块是纯函数，不读不写数据库，也不产生任何评分或裁决。
"""
from __future__ import annotations

from collections import deque
from typing import Any, Mapping, Sequence


class AssignmentValidationError(ValueError):
    """补评分派参数不合法（返回 400）。"""


# ---------------- Dinic 最大流（内部使用） ----------------


class _Edge:
    __slots__ = ("to", "rev", "cap")

    def __init__(self, to: int, rev: int, cap: int) -> None:
        self.to = to
        self.rev = rev
        self.cap = cap


class _Dinic:
    def __init__(self, n: int) -> None:
        self.g: list[list[_Edge]] = [[] for _ in range(n)]
        self.level = [-1] * n
        self.it = [0] * n

    def add_edge(self, u: int, v: int, cap: int) -> _Edge:
        # 先追加正向边：反向边的 rev 必须指向正向边在 g[u] 中的下标。
        fwd = _Edge(v, len(self.g[v]), cap)
        self.g[u].append(fwd)
        rev = _Edge(u, len(self.g[u]) - 1, 0)
        self.g[v].append(rev)
        fwd.rev = len(self.g[v]) - 1
        return fwd
    def _bfs(self, s: int, t: int) -> bool:
        n = len(self.g)
        self.level = [-1] * n
        self.level[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            for e in self.g[u]:
                if e.cap > 0 and self.level[e.to] < 0:
                    self.level[e.to] = self.level[u] + 1
                    q.append(e.to)
        return self.level[t] >= 0

    def _dfs(self, u: int, t: int, pushed: int) -> int:
        if u == t:
            return pushed
        while self.it[u] < len(self.g[u]):
            e = self.g[u][self.it[u]]
            if e.cap > 0 and self.level[e.to] == self.level[u] + 1:
                d = self._dfs(e.to, t, min(pushed, e.cap))
                if d > 0:
                    e.cap -= d
                    self.g[e.to][e.rev].cap += d
                    return d
            self.it[u] += 1
        return 0

    def max_flow(self, s: int, t: int) -> int:
        flow = 0
        inf = 10**18
        while self._bfs(s, t):
            self.it = [0] * len(self.g)
            while True:
                f = self._dfs(s, t, inf)
                if f == 0:
                    break
                flow += f
        return flow


# ---------------- 输入校验 / 规范化 ----------------


def normalize_target(target: Any) -> int:
    if isinstance(target, bool) or not isinstance(target, int):
        raise AssignmentValidationError("每单元目标评委人数必须是整数。")
    if target < 1:
        raise AssignmentValidationError("每单元目标评委人数至少为 1。")
    if target > 1_000_000:
        raise AssignmentValidationError("每单元目标评委人数不能超过 1000000。")
    return target


def normalize_capacities(
    raw: Any, judges: Sequence[str]
) -> dict[str, int]:
    """校验评委容量表：评委必须存在、不可重复，容量为非负整数。"""
    if not isinstance(raw, list):
        raise AssignmentValidationError("capacities 必须是数组。")
    capacities: dict[str, int] = {}
    for i, row in enumerate(raw, start=1):
        where = f"第 {i} 位评委容量"
        if not isinstance(row, Mapping):
            raise AssignmentValidationError(f"{where}必须是对象。")
        judge = row.get("judge")
        if not isinstance(judge, str) or not judge.strip():
            raise AssignmentValidationError(f"{where}的评委名不能为空。")
        judge = judge.strip()
        if judge not in set(judges):
            raise AssignmentValidationError(
                f"{where}引用的评委「{judge}」不存在，可选：{', '.join(judges) or '(无)'}。"
            )
        if judge in capacities:
            raise AssignmentValidationError(f"评委「{judge}」的容量被重复设置。")
        cap = row.get("capacity")
        if isinstance(cap, bool) or not isinstance(cap, int):
            raise AssignmentValidationError(f"{where}的可接任务数必须是非负整数。")
        if not 0 <= cap <= 1_000_000:
            raise AssignmentValidationError(f"{where}的可接任务数需在 0–1000000 之间。")
        capacities[judge] = cap
    missing = [j for j in judges if j not in capacities]
    if missing:
        raise AssignmentValidationError(
            "以下评委未设置本轮可接任务数：" + "、".join(missing) + "。"
        )
    return capacities


# ---------------- 分派主逻辑 ----------------


def plan_supplement_assignments(
    records: Sequence[Mapping[str, Any]],
    target_judges: int,
    judge_capacities: Mapping[str, int],
) -> dict[str, Any]:
    """求补评分任务的最大分派方案（纯函数，结果确定）。

    records: 当前会话评分记录（contestant/judge/item），顺序即首次出现顺序。
    target_judges: 每个单元统一的目标评委人数（≥1）。
    judge_capacities: 每位评委本轮可接任务数（非负整数）。

    返回单元缺口、逐条分派与汇总；不落库、不产生评分。
    """
    # 首次出现顺序（与 analyze 保持一致，保证固定的评委/项目/选手顺序）。
    judge_order: list[str] = list(dict.fromkeys(r["judge"] for r in records))
    item_order: list[str] = list(dict.fromkeys(r["item"] for r in records))
    contestant_order: list[str] = list(dict.fromkeys(r["contestant"] for r in records))
    jidx = {j: i for i, j in enumerate(judge_order)}
    iidx = {it: i for i, it in enumerate(item_order)}
    cidx = {c: i for i, c in enumerate(contestant_order)}

    # 每个项目下打过过分的评委（按评委首次出现顺序）；
    # 以及每个单元（选手×项目）已有评分的评委集合。
    item_judges: dict[str, set[str]] = {it: set() for it in item_order}
    existing: dict[tuple[str, str], set[str]] = {}
    for r in records:
        item_judges[r["item"]].add(r["judge"])
        existing.setdefault((r["contestant"], r["item"]), set()).add(r["judge"])

    # 单元按 项目顺序 → 选手顺序 固定排列。
    unit_keys = sorted(existing.keys(), key=lambda k: (iidx[k[1]], cidx[k[0]]))
    units: list[dict[str, Any]] = []
    for contestant, item in unit_keys:
        have = existing[(contestant, item)]
        need = max(0, target_judges - len(have))
        # 只有「评过该项目但没评过该单元」的评委能接单。
        candidates = [
            j for j in judge_order if j in item_judges[item] and j not in have
        ]
        units.append(
            {
                "item": item,
                "contestant": contestant,
                "existing_count": len(have),
                "target": target_judges,
                "need": need,
                "assigned": 0,
                "unmet": need,
                "candidates": candidates,
            }
        )

    n_judges = len(judge_order)
    n_units = len(units)
    # 节点编号：0=源点，[1,n_judges]=评委，之后=单元，末位=汇点。
    s_node = 0
    j_node = lambda j: 1 + j  # noqa: E731
    u_node = lambda u: 1 + n_judges + u  # noqa: E731
    t_node = 1 + n_judges + n_units

    dinic = _Dinic(t_node + 1)
    source_edges = {
        j: dinic.add_edge(s_node, j_node(j), int(judge_capacities[name]))
        for j, name in enumerate(judge_order)
    }
    sink_edges = [dinic.add_edge(u_node(u), t_node, units[u]["need"]) for u in range(n_units)]

    # 候选边（评委→单元），容量 1；记录残余边对象以便读取初始流。
    edge_refs: dict[tuple[int, int], _Edge] = {}
    for u, unit in enumerate(units):
        for j in (jidx[j] for j in unit["candidates"]):
            edge_refs[(j, u)] = dinic.add_edge(j_node(j), u_node(u), 1)

    max_total = dinic.max_flow(s_node, t_node) if n_judges and n_units else 0

    # 当前各弧流量（初始取任意最大流；强制阶段会规范成唯一方案）。
    flow_sj = {
        j: int(judge_capacities[name]) - source_edges[j].cap
        for j, name in enumerate(judge_order)
    }
    flow_ut = {u: units[u]["need"] - sink_edges[u].cap for u in range(n_units)}
    flow_ju = {(j, u): 1 - edge_refs[(j, u)].cap for (j, u) in edge_refs}

    # 强制阶段：按 评委 → 项目 → 选手 的顺序逐条决定候选边是否入选。
    # locked_forward: 拒绝边/已接受边的正向残余弧（评委→单元）禁用；
    # locked_reverse: 已接受边的反向残余弧（单元→评委）禁用（禁止再被改道让出）。
    locked_forward: set[tuple[int, int]] = set()
    locked_reverse: set[tuple[int, int]] = set()

    # 维护二部图上的残余邻接集合，使每次可行性 BFS 为标准 O(V+E)：
    # fwd[j]：flow=0 且未锁正向弧的单元；rev[u]：flow=1 且未锁反向弧的评委。
    fwd: dict[int, set[int]] = {j: set() for j in range(n_judges)}
    rev: dict[int, set[int]] = {u: set() for u in range(n_units)}
    for (j, u), fl in flow_ju.items():
        if fl:
            rev[u].add(j)
        else:
            fwd[j].add(u)

    def _set_flow_0(j: int, u: int) -> None:
        """候选弧 1→0（沿反向弧改道）。"""
        flow_ju[(j, u)] = 0
        rev[u].discard(j)
        if (j, u) not in locked_forward:
            fwd[j].add(u)

    def _set_flow_1(j: int, u: int) -> None:
        """候选弧 0→1（沿正向弧改道）。"""
        flow_ju[(j, u)] = 1
        fwd[j].discard(u)
        if (j, u) not in locked_reverse:
            rev[u].add(j)

    def residual_neighbors(node: int, forbid_judge: int | None = None) -> list[int]:
        """按节点编号升序返回残余有向邻居（确定性 BFS）。

        forbid_judge：可行性判定时的目标评委——不能经源点弧进入该评委，
        因为经源点到达并不释放该评委的一个容量槽（只有经别的单元的
        反向候选弧进入才对应「把该评委从别的单元改道过来」）。
        """
        nbrs: list[int] = []
        if node == s_node:
            nbrs += [
                j_node(j)
                for j in range(n_judges)
                if j != forbid_judge
                and flow_sj[j] < int(judge_capacities[judge_order[j]])
            ]
        elif node == t_node:
            nbrs += [u_node(u) for u in range(n_units) if flow_ut[u] > 0]
        elif 1 <= node <= n_judges:
            j = node - 1
            if flow_sj[j] > 0:
                nbrs.append(s_node)  # 源点弧的反向残余
            nbrs += (u_node(u) for u in sorted(fwd[j]))
        else:
            u = node - 1 - n_judges
            # 经反向候选弧到达任意评委（含目标评委）都是合法的：
            # 到达目标评委正表示从别的单元那里释放出了他的一个容量槽。
            nbrs += (j_node(j) for j in sorted(rev[u]))
            if flow_ut[u] < units[u]["need"]:
                nbrs.append(t_node)  # 单元→汇点正向残余
        return nbrs

    def find_path(u_start: int, j_target: int) -> list[int] | None:
        """残余网络中从单元 u_start 到评委 j_target 的合法改道路（避开锁定弧）。

        沿找到的路再叠加目标边 (j_target→u_start) 构成残余圈，因此路中至多
        能包含源点 s 或汇点 t 中的一个（都经过就成了增广路，会把流量提升到
        超过最大基数）：

        - 只经 s：从别的评委处借来一个容量槽（s→j' 正向残余）；
        - 只经 t：把某个单元的汇点槽位让过来（u'→t 正向残余）；
        - 都不经过：纯二部图改道。

        此外目标评委只能经别的单元的反向候选弧到达（不能经源点直接进入，
        否则没有释放任何容量），故源点弧不允许通向目标评委。
        """
        start, goal = u_node(u_start), j_node(j_target)
        # state: 0=尚未经过 s/t；1=已过 s；2=已过 t。两个都经过即非法。
        prev = {(start, 0): None}
        q = deque([(start, 0)])
        found_state: int | None = None
        while q:
            node, state = q.popleft()
            if node == goal:
                found_state = state
                break
            for nxt in residual_neighbors(node, forbid_judge=j_target):
                nxt_state = state
                if nxt == s_node or node == s_node:
                    if state == 2:
                        continue  # 已经过 t，不能再过 s
                    nxt_state = 1
                if nxt == t_node or node == t_node:
                    if state == 1:
                        continue  # 已经过 s，不能再过 t
                    nxt_state = 2
                key = (nxt, nxt_state)
                if key not in prev:
                    prev[key] = (node, state)
                    q.append(key)
        if found_state is None:
            return None
        path: list[int] = []
        key = (goal, found_state)
        while key is not None:
            path.append(key[0])
            key = prev[key]
        path.reverse()
        return path

    def toggle_arc(u: int, v: int) -> None:
        """沿残余路 u→v 推 1 单位流量（可能是正向弧或反向弧）。"""
        if u == s_node:
            flow_sj[v - 1] += 1
        elif v == s_node:
            flow_sj[u - 1] -= 1
        elif u == t_node:
            flow_ut[v - 1 - n_judges] -= 1
        elif v == t_node:
            flow_ut[u - 1 - n_judges] += 1
        elif u < v:  # 评委→单元 正向候选弧
            _set_flow_1(u - 1, v - 1 - n_judges)
        else:        # 单元→评委 反向候选弧
            _set_flow_0(v - 1, u - 1 - n_judges)

    ordered_edges = sorted(
        edge_refs.keys(), key=lambda ju: (ju[0], iidx[units[ju[1]]["item"]], cidx[units[ju[1]]["contestant"]])
    )
    for j, u in ordered_edges:
        if flow_ju[(j, u)] == 1:
            # 已在当前最大流中：直接强制保留，禁用其正反残余弧。
            locked_forward.add((j, u))
            locked_reverse.add((j, u))
            fwd[j].discard(u)
            rev[u].discard(j)
            continue
        path = find_path(u, j)
        if path is None:
            # 在不牺牲已强制任务的前提下无法纳入此边 → 永久拒绝。
            locked_forward.add((j, u))
            fwd[j].discard(u)
            continue
        # 沿残余圈改道，再把该边流量置 1 并锁定，总数保持最大值。
        for a, b in zip(path, path[1:]):
            toggle_arc(a, b)
        flow_ju[(j, u)] = 1
        locked_forward.add((j, u))
        locked_reverse.add((j, u))
        fwd[j].discard(u)
        rev[u].discard(j)

    assigned_edges = sorted(
        (ju for ju in edge_refs if flow_ju[ju] == 1),
        key=lambda ju: (ju[0], iidx[units[ju[1]]["item"]], cidx[units[ju[1]]["contestant"]]),
    )
    assignments = [
        {
            "judge": judge_order[j],
            "item": units[u]["item"],
            "contestant": units[u]["contestant"],
        }
        for j, u in assigned_edges
    ]
    for j, u in assigned_edges:
        units[u]["assigned"] += 1
    for unit in units:
        unit["unmet"] = unit["need"] - unit["assigned"]

    assigned_per_judge = {j: 0 for j in range(n_judges)}
    for j, _ in assigned_edges:
        assigned_per_judge[j] += 1

    need_units = [u for u in units if u["need"] > 0]
    total_need = sum(u["need"] for u in units)
    unmet_total = sum(u["unmet"] for u in units)
    summary = {
        "target_judges": target_judges,
        "units": len(units),
        "units_with_need": len(need_units),
        "already_complete": len(units) - len(need_units),
        "total_need": total_need,
        "assigned_total": len(assignments),
        "unmet_total": unmet_total,
        "max_possible": max_total,
        "fully_satisfied": sum(1 for u in need_units if u["unmet"] == 0),
        "partially_satisfied": sum(1 for u in need_units if 0 < u["assigned"] < u["need"]),
        "unsatisfied": sum(1 for u in need_units if u["assigned"] == 0),
        "candidate_edges": len(edge_refs),
        "capacity_total": sum(int(v) for v in judge_capacities.values()),
    }

    return {
        "judges": [
            {
                "judge": name,
                "capacity": int(judge_capacities[name]),
                "assigned_count": assigned_per_judge[j],
                "remaining": int(judge_capacities[name]) - assigned_per_judge[j],
            }
            for j, name in enumerate(judge_order)
        ],
        "units": [
            {
                "item": u["item"],
                "contestant": u["contestant"],
                "existing_count": u["existing_count"],
                "target": u["target"],
                "need": u["need"],
                "assigned": u["assigned"],
                "unmet": u["unmet"],
                "eligible_judges": u["candidates"],
            }
            for u in units
        ],
        "assignments": assignments,
        "summary": summary,
    }
