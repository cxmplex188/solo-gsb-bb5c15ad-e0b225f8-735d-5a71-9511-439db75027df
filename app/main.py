"""FastAPI application: multi-judge score calibration."""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import database as db
from .assignment import (
    AssignmentValidationError,
    normalize_capacities,
    normalize_target,
    plan_supplement_assignments,
)
from .csv_io import CSVParseError, build_audit_csv, parse_scores
from .merge import (
    STATUS_LABELS as MERGE_STATUS_LABELS,
    MergeValidationError,
    basis_matches,
    build_merge_plan,
    current_basis,
    normalize_incoming_records,
)
from .rules import (
    RuleValidationError,
    evaluate_rules,
    normalize_rules,
    score_rows_from_analysis,
)
from .scoring import (
    SCOPES,
    SCOPE_JUDGE,
    SCOPE_LABELS,
    analyze,
    compare_contestants,
    stability_experiment,
)

SAMPLE_PATH = os.environ.get(
    "CALIB_SAMPLE_PATH",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "samples", "sample_scores.csv"),
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    yield


app = FastAPI(title="多评委评分校准", version="1.0.0", lifespan=lifespan)


def _build_analysis(conn, session_id: str, scope: str) -> dict[str, Any]:
    records = db.load_scores(conn, session_id)
    decisions = db.load_decisions(conn, session_id)
    result = analyze(records, decisions, scope=scope)
    result["batch_rules"] = db.load_rules(conn, session_id)
    return result


def _resolve_scope(conn, session_id: str, override: str | None) -> str:
    if override is not None:
        if override not in SCOPES:
            raise HTTPException(status_code=400, detail=f"不支持的校准范围: {override}")
        return override
    stored = db.get_scope(conn, session_id)
    return stored or SCOPE_JUDGE


class ScopeBody(BaseModel):
    scope: Literal["judge", "judge_item"]


class DecisionBody(BaseModel):
    use_raw: bool
    reason: str = Field(default="", max_length=2000)


class StabilityBody(BaseModel):
    exclude_type: Literal["judge", "item"]
    exclude_name: str = Field(min_length=1, max_length=200)


class JudgeCapacityBody(BaseModel):
    judge: str = Field(min_length=1, max_length=200)
    capacity: int = Field(ge=0, le=1_000_000)


class SupplementBody(BaseModel):
    # 所有「选手 × 项目」单元统一的目标评委人数（扣除已有评分后得到缺口）。
    target_judges: int = Field(ge=1, le=1_000_000)
    # 每位评委本轮可接任务数，需覆盖会话中出现过的全部评委。
    capacities: list[JudgeCapacityBody] = Field(min_length=1)


class RuleBody(BaseModel):
    name: str = Field(default="", max_length=200)
    judge: str | None = None
    item: str | None = None
    raw_min: float | None = None
    raw_max: float | None = None
    cal_min: float | None = None
    cal_max: float | None = None
    use_raw: bool = False
    reason_template: str = Field(default="", max_length=2000)
    priority: int = Field(default=100, ge=0, le=1_000_000)


class RuleSetBody(BaseModel):
    rules: list[RuleBody] = Field(default_factory=list)


class MergeConfirmBody(BaseModel):
    # 客户端原样回传预览时解析出的记录与基线快照；
    # 服务端重新校验并核对基线，避免「过期预览」覆盖新数据。
    records: list[dict[str, Any]] = Field(min_length=1)
    basis: list[dict[str, Any]] = Field(min_length=0)
    # 仅在会话尚不存在（向空会话合并）时用于创建会话；已有会话保持原校准范围。
    scope: Literal["judge", "judge_item"] | None = None


def _normalize_session_rules(body: RuleSetBody, analysis: dict[str, Any]) -> list[dict[str, Any]]:
    """Validate/normalize submitted rules against the session's judge/item names."""
    try:
        rules = normalize_rules([r.model_dump() for r in body.rules])
    except RuleValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    judges = set(analysis["judges"])
    items = set(analysis["items"])
    for i, rule in enumerate(rules, start=1):
        if rule["judge"] is not None and rule["judge"] not in judges:
            raise HTTPException(
                status_code=400,
                detail=f"第 {i} 条规则引用的评委「{rule['judge']}」不存在，可选：{', '.join(analysis['judges']) or '(无)'}。",
            )
        if rule["item"] is not None and rule["item"] not in items:
            raise HTTPException(
                status_code=400,
                detail=f"第 {i} 条规则引用的项目「{rule['item']}」不存在，可选：{', '.join(analysis['items']) or '(无)'}。",
            )
    return rules


def _rule_preview(analysis: dict[str, Any], rules: list[dict[str, Any]]) -> dict[str, Any]:
    """Run read-only rule matching on top of an analysis result."""
    rows = score_rows_from_analysis(analysis)
    decisions = {
        s["id"]: (s["use_raw"], s["reason"])
        for block in analysis["contestants"]
        for ib in block["items"]
        for s in ib["scores"]
    }
    preview = evaluate_rules(rows, rules, decisions)
    preview["scope"] = analysis["scope"]
    preview["scope_label"] = analysis["scope_label"]
    return preview


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/sessions/{session_id}/analysis")
def get_analysis(session_id: str, scope: str | None = None) -> dict[str, Any]:
    with db.get_conn() as conn:
        effective_scope = _resolve_scope(conn, session_id, scope)
        return _build_analysis(conn, session_id, effective_scope)


@app.post("/api/sessions/{session_id}/scope")
def update_scope(session_id: str, body: ScopeBody) -> dict[str, Any]:
    with db.get_conn() as conn:
        db.ensure_session(conn, session_id, body.scope)
        db.set_scope(conn, session_id, body.scope)
        return _build_analysis(conn, session_id, body.scope)


@app.post("/api/sessions/{session_id}/upload")
async def upload_scores(
    session_id: str,
    file: UploadFile = File(...),
    scope: str = SCOPE_JUDGE,
) -> dict[str, Any]:
    if scope not in SCOPES:
        raise HTTPException(status_code=400, detail=f"不支持的校准范围: {scope}")
    raw = await file.read()
    try:
        records = parse_scores(raw)
    except CSVParseError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    with db.get_conn() as conn:
        db.ensure_session(conn, session_id, scope)
        db.replace_scores(conn, session_id, records)
        return _build_analysis(conn, session_id, scope)


@app.post("/api/sessions/{session_id}/sample")
def load_sample(session_id: str, scope: str = SCOPE_JUDGE) -> dict[str, Any]:
    if scope not in SCOPES:
        raise HTTPException(status_code=400, detail=f"不支持的校准范围: {scope}")
    try:
        with open(SAMPLE_PATH, "rb") as fh:
            records = parse_scores(fh.read())
    except OSError:
        raise HTTPException(status_code=404, detail="示例数据文件不可用。")
    except CSVParseError as exc:
        raise HTTPException(status_code=500, detail=f"示例数据格式错误: {exc}")
    with db.get_conn() as conn:
        db.ensure_session(conn, session_id, scope)
        db.replace_scores(conn, session_id, records)
        return _build_analysis(conn, session_id, scope)


@app.post("/api/sessions/{session_id}/merge/preview")
async def merge_preview(
    session_id: str,
    file: UploadFile = File(...),
    scope: str | None = None,
) -> dict[str, Any]:
    """合并导入的只读预览：解析上传文件并与当前会话数据逐条比对。

    按「选手+评委+项目」识别 新增/未变/分数变更/仅存在于当前会话 四类记录，
    逐行给出新旧分数与受影响的旧裁决。不落库、不创建会话、不改变任何裁决。
    """
    raw = await file.read()
    try:
        records = parse_scores(raw)
    except CSVParseError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    with db.get_conn() as conn:
        stored_scope = db.get_scope(conn, session_id)
        if scope is not None:
            if scope not in SCOPES:
                raise HTTPException(status_code=400, detail=f"不支持的校准范围: {scope}")
            effective_scope = scope
        else:
            effective_scope = stored_scope or SCOPE_JUDGE
        current = db.load_scores(conn, session_id)
        decisions = db.load_decisions(conn, session_id)
        plan = build_merge_plan(current, records, decisions)
    return {
        "scope": effective_scope,
        "scope_label": SCOPE_LABELS[effective_scope],
        "session_existed": stored_scope is not None,
        "status_labels": MERGE_STATUS_LABELS,
        "rows": plan["rows"],
        "counts": plan["counts"],
        "total_current": plan["total_current"],
        "total_incoming": plan["total_incoming"],
        "will_modify": plan["will_modify"],
        "decisions_cleared": plan["decisions_cleared"],
        "records": records,
        "basis": current_basis(current),
    }


@app.post("/api/sessions/{session_id}/merge/confirm")
def merge_confirm(session_id: str, body: MergeConfirmBody) -> dict[str, Any]:
    """确认合并：在同一个数据库事务中落库预览结果。

    分数变更/删除记录的旧裁决在事务内自动清除；未变记录及其裁决、已保存的
    批处理规则均不触碰。若预览后会话数据被其他操作改动（基线不匹配），
    返回 409 要求重新预览，绝不覆盖。
    """
    try:
        records = normalize_incoming_records(body.records)
    except MergeValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if not all(
        isinstance(item, dict)
        and isinstance(item.get("contestant"), str)
        and isinstance(item.get("judge"), str)
        and isinstance(item.get("item"), str)
        and isinstance(item.get("score"), (int, float))
        and not isinstance(item.get("score"), bool)
        for item in body.basis
    ):
        raise HTTPException(status_code=400, detail="basis 基线快照格式不正确，请重新预览。")
    with db.get_conn() as conn:
        current = db.load_scores(conn, session_id)
        decisions = db.load_decisions(conn, session_id)
        stored_scope = db.get_scope(conn, session_id)
        if not basis_matches(body.basis, current):
            raise HTTPException(
                status_code=409,
                detail="预览后当前会话数据已变化，请重新预览后再确认合并。",
            )
        plan = build_merge_plan(current, records, decisions)
        # 全部变更随本事务一起提交；任何失败整体回滚。
        if stored_scope is None:
            db.ensure_session(conn, session_id, body.scope or SCOPE_JUDGE)
        result = db.merge_scores(conn, session_id, plan["rows"])
        scope = _resolve_scope(conn, session_id, None)
        analysis = _build_analysis(conn, session_id, scope)
    analysis["merge_applied"] = {
        "counts": result["counts"],
        "decisions_cleared": result["decisions_cleared"],
        "rules_touched": 0,
    }
    return analysis


@app.patch("/api/sessions/{session_id}/scores/{score_id}/decision")
def patch_decision(session_id: str, score_id: int, body: DecisionBody) -> dict[str, Any]:
    with db.get_conn() as conn:
        try:
            db.upsert_decision(conn, session_id, score_id, body.use_raw, body.reason)
        except db.NotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        scope = _resolve_scope(conn, session_id, None)
        return _build_analysis(conn, session_id, scope)


@app.post("/api/sessions/{session_id}/decisions/reset")
def reset_decisions(session_id: str) -> dict[str, Any]:
    with db.get_conn() as conn:
        if db.get_scope(conn, session_id) is None:
            raise HTTPException(status_code=404, detail="会话不存在。")
        removed = db.reset_decisions(conn, session_id)
        scope = _resolve_scope(conn, session_id, None)
        analysis = _build_analysis(conn, session_id, scope)
        analysis["reset_count"] = removed
        return analysis


@app.get("/api/sessions/{session_id}/rules")
def get_rules(session_id: str) -> dict[str, Any]:
    with db.get_conn() as conn:
        return {"rules": db.load_rules(conn, session_id)}


@app.put("/api/sessions/{session_id}/rules")
def save_rules(session_id: str, body: RuleSetBody) -> dict[str, Any]:
    """Persist a rule set without applying it to any scores."""
    with db.get_conn() as conn:
        if db.get_scope(conn, session_id) is None:
            raise HTTPException(status_code=404, detail="会话不存在，请先上传数据。")
        scope = _resolve_scope(conn, session_id, None)
        analysis = _build_analysis(conn, session_id, scope)
        if analysis["counts"]["scores"] == 0:
            raise HTTPException(status_code=400, detail="当前会话没有评分数据，无法保存规则。")
        rules = _normalize_session_rules(body, analysis)
        saved = db.replace_rules(conn, session_id, rules)
    return {"saved": saved, "rules": rules}


@app.post("/api/sessions/{session_id}/rules/preview")
def preview_rules(session_id: str, body: RuleSetBody) -> dict[str, Any]:
    """Read-only dry run: hit/miss/conflict counts and priority resolution.

    Never persists rules and never modifies decisions.
    """
    with db.get_conn() as conn:
        if db.get_scope(conn, session_id) is None:
            raise HTTPException(status_code=404, detail="会话不存在，请先上传数据。")
        scope = _resolve_scope(conn, session_id, None)
        analysis = _build_analysis(conn, session_id, scope)
        if analysis["counts"]["scores"] == 0:
            raise HTTPException(status_code=400, detail="当前会话没有评分数据，无法预览规则。")
        rules = _normalize_session_rules(body, analysis)
        preview = _rule_preview(analysis, rules)
    preview["rules_submitted"] = len(rules)
    preview["rules_saved"] = len(analysis["batch_rules"])
    return preview


@app.post("/api/sessions/{session_id}/rules/apply")
def apply_rules(session_id: str, body: RuleSetBody) -> dict[str, Any]:
    """Persist rules and write all adjudications in a single transaction."""
    with db.get_conn() as conn:
        if db.get_scope(conn, session_id) is None:
            raise HTTPException(status_code=404, detail="会话不存在，请先上传数据。")
        scope = _resolve_scope(conn, session_id, None)
        pre_analysis = _build_analysis(conn, session_id, scope)
        if pre_analysis["counts"]["scores"] == 0:
            raise HTTPException(status_code=400, detail="当前会话没有评分数据，无法应用规则。")
        rules = _normalize_session_rules(body, pre_analysis)
        preview = _rule_preview(pre_analysis, rules)
        # Everything below commits together; any failure rolls the whole batch back.
        db.replace_rules(conn, session_id, rules)
        written = db.apply_batch_decisions(conn, session_id, preview["assignments"])
        analysis = _build_analysis(conn, session_id, scope)
    analysis["rules_saved"] = len(rules)
    analysis["written"] = written
    analysis["matched"] = preview["matched"]
    analysis["unmatched"] = preview["unmatched"]
    analysis["conflicted"] = preview["conflicted"]
    analysis["changed"] = preview["would_change"]
    analysis["rule_preview"] = {
        k: preview[k]
        for k in ("total_scores", "matched", "unmatched", "conflicted", "would_change", "rules")
    }
    return analysis


@app.post("/api/sessions/{session_id}/stability")
def stability_check(session_id: str, body: StabilityBody) -> dict[str, Any]:
    """Temporary what-if run: exclude one judge/item and diff ranks/totals.

    Pure read-only computation — no scores, decisions or session state are
    modified. The client discards the result when returning to baseline.
    """
    with db.get_conn() as conn:
        if db.get_scope(conn, session_id) is None:
            raise HTTPException(status_code=404, detail="会话不存在，请先上传数据。")
        scope = _resolve_scope(conn, session_id, None)
        records = db.load_scores(conn, session_id)
        decisions = db.load_decisions(conn, session_id)
    if not records:
        raise HTTPException(status_code=404, detail="当前会话没有评分数据，无法进行实验。")
    try:
        return stability_experiment(
            records, decisions, scope, body.exclude_type, body.exclude_name.strip()
        )
    except LookupError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/sessions/{session_id}/supplement/preview")
def supplement_preview(session_id: str, body: SupplementBody) -> dict[str, Any]:
    """补评分派：只读预览最大任务分派方案。

    需求单元为当前会话中已有评分的「选手 × 项目」组合；每单元统一目标评委
    人数（扣除已有评分后为缺口），各评委分别限制本轮容量，只有评过该项目
    但尚未评该单元的评委可以接单。在这些约束下求分派任务总数最大的方案，
    并按固定的评委、项目、选手顺序稳定输出并列结果。

    纯只读计算：不写入评分、不产生裁决，也不改变任何会话状态。
    """
    try:
        target = normalize_target(body.target_judges)
    except AssignmentValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    with db.get_conn() as conn:
        if db.get_scope(conn, session_id) is None:
            raise HTTPException(status_code=404, detail="会话不存在，请先上传数据。")
        records = db.load_scores(conn, session_id)
        judges = list(dict.fromkeys(r["judge"] for r in records))
    if not records:
        raise HTTPException(status_code=404, detail="当前会话没有评分数据，无法分派补评任务。")
    try:
        capacities = normalize_capacities(
            [c.model_dump() for c in body.capacities], judges
        )
        return plan_supplement_assignments(records, target, capacities)
    except AssignmentValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/api/sessions/{session_id}/compare")
def compare_two_contestants(session_id: str, a: str, b: str) -> dict[str, Any]:
    """Read-only head-to-head comparison, recomputed from current data.

    The comparison is derived from the same analysis pipeline on every call,
    so it always reflects the current calibration scope and manual
    decisions. Nothing is persisted and no scores are duplicated.
    """
    with db.get_conn() as conn:
        if db.get_scope(conn, session_id) is None:
            raise HTTPException(status_code=404, detail="会话不存在，请先上传数据。")
        scope = _resolve_scope(conn, session_id, None)
        analysis = _build_analysis(conn, session_id, scope)
    if analysis["counts"]["scores"] == 0:
        raise HTTPException(status_code=404, detail="当前会话没有评分数据，无法对比。")
    try:
        return compare_contestants(analysis, a, b)
    except (LookupError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.delete("/api/sessions/{session_id}")
def clear_session(session_id: str) -> dict[str, Any]:
    with db.get_conn() as conn:
        existed = db.delete_session(conn, session_id)
    return {"cleared": existed}


@app.get("/api/sessions/{session_id}/export")
def export_audit(session_id: str) -> Response:
    with db.get_conn() as conn:
        if db.get_scope(conn, session_id) is None:
            raise HTTPException(status_code=404, detail="会话不存在，请先上传数据。")
        scope = _resolve_scope(conn, session_id, None)
        analysis = _build_analysis(conn, session_id, scope)
    if analysis["counts"]["scores"] == 0:
        raise HTTPException(status_code=400, detail="当前会话没有可导出的数据。")
    payload = build_audit_csv(analysis)
    headers = {"Content-Disposition": 'attachment; filename="calibration_audit.csv"'}
    return Response(content=payload, media_type="text/csv; charset=utf-8", headers=headers)


# Frontend (served last so /api routes take precedence).
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))
