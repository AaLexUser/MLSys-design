#!/usr/bin/env python3
"""Offline MCPVerse snapshot analysis. Python 3.10+, standard library only."""

import argparse
import csv
import hashlib
import html
import json
import math
import re
import statistics
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_DATA = HERE / "data"
# Study design supplied by the project, not inferred from successful runs.
STORIES = {
    "hf_dataset_inspect": ["Q59", "Q60", "Q61", "Q62", "Q63", "Q64", "Q149", "Q150", "Q151"],
    "paper_fact_lookup": ["Q81", "Q154", "Q155", "Q163", "Q167", "Q192", "Q193"],
    "files_find_read": ["Q87", "Q89", "Q91", "Q92", "Q131", "Q132", "Q173", "Q174", "Q175"],
    "files_prepare": ["Q88", "Q170", "Q171", "Q172", "Q176"],
    "git_repo_state": ["Q93", "Q94", "Q95", "Q133"],
    "table_data_lookup": ["Q21", "Q22", "Q128"],
    "stock_quote_compare": [
        "Q194",
        "Q200",
        "Q205",
        "Q206",
        "Q226",
        "Q291",
        "Q292",
        "Q293",
        "Q294",
        "Q295",
        "Q296",
        "Q297",
        "Q304",
    ],
    "app_store_research": ["Q251", "Q252", "Q253", "Q275", "Q276", "Q277"],
    "horoscope": ["Q190", "Q214", "Q224"],
}
EXCLUSIONS = {
    "code": "python_execution_outside_mcp",
    "amap-maps": "unavailable_service_or_credentials",
    "12306-mcp": "unavailable_service_or_credentials",
    "variflight": "unavailable_service_or_credentials",
    "3rd_party_mcp_server_shuidi": "unavailable_service_or_credentials",
    "investor": "package_not_executable",
    "datagov": "upstream_unavailable",
}
# Three known reference defects. Q121 removed the task from coverage entirely.
# Q200 stays covered by its story but is excluded from scoring.
# Q162 was never covered and sits in the outside-stories remainder.
DEFECTS = {
    "Q200": "question asks for August 2025 trading days, reference is for April 2025",
    "Q121": "reference hour pillar is wrong for the stated birth time",
    "Q162": "reference astronomy image does not match the stated date",
}
STORY_LABELS = {
    "hf_dataset_inspect": "Поиск и сборка датасетов",
    "paper_fact_lookup": "Работа со статьями",
    "files_find_read": "Поиск и чтение файлов",
    "files_prepare": "Подготовка файлов",
    "git_repo_state": "Работа с кодом репозитория",
    "table_data_lookup": "Табличные отчёты",
    "stock_quote_compare": "Запросы и сравнение котировок",
    "app_store_research": "Исследование приложений",
    "horoscope": "Вопросы гороскопов",
}

NOT_SCORED = {"Q200"}
REMOVED_FROM_COVERAGE = {"Q121"}
MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}
DATE_RE = re.compile(
    r"\b(January|February|March|April|May|June|July|August|September|"
    r"October|November|December)\s+(\d{1,2}),?\s+(\d{4})\b",
    re.IGNORECASE,
)
WORD_RE = re.compile(r"[a-z0-9]+")
COLORS = ["#3477aa", "#d49632", "#8b579f", "#319172"]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path):
    text = path.read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def servers(row):
    return {part.strip() for part in row["MCP"].split("+") if part.strip()}


def tools(row):
    # Unique names in the annotation, not runtime calls or steps.
    parts = [re.sub(r"\s*\*\s*\d+\s*$", "", p.strip()) for p in row["tool"].split("+")]
    return {re.sub(r"^[\w-]+[./](?=\w)", "", p) for p in parts if p}


def answer_kind(row):
    answer = row["answer"].strip()
    if row["eval_method"] == "eval_script":
        return "Проверка файла"
    if row["source_kind"] == "sensitive":
        return "Вычисляемый эталон"
    if re.fullmatch(r"-?[\d.,]+", answer):
        return "Число"
    if len(answer) < 60 and "," in answer:
        return "Короткий список"
    if len(answer) < 80:
        return "Короткий текст"
    return "Длинный текст"


def word_bag(text):
    return set(WORD_RE.findall(text.lower()))


def numeric_summary(values):
    values = sorted(values)
    if not values:
        return {"n": 0}

    def percentile(p):
        # Linear interpolation, equivalent to quantile type 7.
        i = (len(values) - 1) * p
        a = int(i)
        b = min(a + 1, len(values) - 1)
        return round(values[a] + (values[b] - values[a]) * (i - a), 4)

    return {
        "n": len(values),
        "min": values[0],
        "median": statistics.median(values),
        "p90": percentile(0.9),
        "max": values[-1],
    }


def exact_mcnemar(improved, regressed):
    n = improved + regressed
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(min(improved, regressed) + 1)) / 2**n)


def check_financial_dates(tasks, qids):
    """Dates inside financial questions: weekend dates and answer months that differ."""
    found = []
    for q in sorted(qids, key=lambda x: int(x[1:])):
        row = tasks[q]
        dates = [
            date(int(y), MONTHS[m.lower()], int(d)) for m, d, y in DATE_RE.findall(row["question"])
        ]
        if not dates:
            continue
        weekend = [str(x) for x in dates if x.weekday() >= 5]
        q_months = {x.month for x in dates}
        a_months = {MONTHS[m.lower()] for m, _, _ in DATE_RE.findall(row["answer"])}
        foreign = sorted(a_months - q_months) if a_months else []
        if weekend or foreign:
            found.append(
                {"question_id": q, "weekend_dates": weekend, "foreign_answer_months": foreign}
            )
    return found


def chart(path, title, rows, legend, unit=None, note=None):
    """Accessible stacked horizontal bars, all labels are XML escaped."""
    width, left, right, row_height = 1050, 330, 100, 39
    maximum = max((sum(values) for _, values in rows), default=1) or 1

    lx, ly, row_y = 20, 54, 79
    legend_items = []
    for i, name in enumerate(legend):
        w = max(150, len(name) * 8 + 40)
        if lx + w > width - 20 and lx > 20:
            lx, ly, row_y = 20, ly + 18, row_y + 18
        legend_items.append((i, lx, ly, name))
        lx += w
    extra = 34 if (unit or note) else 0
    height = row_y + row_height * len(rows) + 21 + extra

    def label(x, y, value, **attrs):
        extra = " ".join(f'{k.replace("_", "-")}="{v}"' for k, v in attrs.items())
        return f'<text x="{x}" y="{y}" {extra}>{html.escape(str(value))}</text>'

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title">',
        f'<title id="title">{html.escape(title)}</title>',
        '<rect width="100%" height="100%" fill="white"/>',
        '<g font-family="Arial, sans-serif" font-size="14" fill="#243447">',
        label(20, 26, title, font_size="19", font_weight="bold"),
    ]
    for i, x, y, name in legend_items:
        out.extend(
            [
                f'<rect x="{x}" y="{y - 11}" width="12" height="12" fill="{COLORS[i]}"/>',
                label(x + 18, y, name),
            ]
        )
    for i, (name, values) in enumerate(rows):
        y = row_y + i * row_height
        out.append(label(left - 14, y + 17, name, text_anchor="end"))
        x = float(left)
        for j, v in enumerate(values):
            bar_width = (width - left - right) * v / maximum
            if v:
                rect = f'<rect x="{x:.3f}" y="{y}" width="{bar_width:.3f}" height="24"'
                out.append(rect + f' fill="{COLORS[j]}"/>')
                if bar_width > 22:
                    cx = round(x + bar_width / 2, 3)
                    out.append(label(cx, y + 17, v, text_anchor="middle", fill="white"))
            x += bar_width
        out.append(label(round(x + 8, 3), y + 17, sum(values)))
    fy = row_y + row_height * len(rows) + 14
    if note:
        out.append(label(20, fy, note, font_size="12", fill="#57606a"))
        fy += 16
    if unit:
        out.append(label(20, fy, unit, font_size="12", fill="#57606a"))
    out.extend(["</g>", "</svg>"])
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def write_json(path, value):
    text = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path.write_text(text, encoding="utf-8")


def analyze(data, output):
    output.mkdir(parents=True, exist_ok=True)
    diagrams = output.parent / "diagrams"
    diagrams.mkdir(parents=True, exist_ok=True)
    rows = []
    source_quality = {}
    for kind in ("invariant", "sensitive"):
        path = data / f"mcpverse_time_{kind}_v1.1.csv"
        with path.open(encoding="utf-8-sig", newline="") as stream:
            part = list(csv.DictReader(stream))
        source_quality[kind] = {
            "rows": len(part),
            "missing_by_field": {key: sum(not row[key].strip() for row in part) for key in part[0]},
            "exact_duplicate_rows": len(part) - len({json.dumps(r, sort_keys=True) for r in part}),
        }
        rows.extend(dict(row, source_kind=kind) for row in part)
    ids = Counter(row["question_id"] for row in rows)
    assert not any(n > 1 for n in ids.values()), "Duplicate task IDs"
    tasks = {row["question_id"]: row for row in rows}
    story_of = {q: story for story, qs in STORIES.items() for q in qs}
    assert sum(map(len, STORIES.values())) == len(story_of), "Overlapping stories"
    assert set(story_of) <= set(tasks), "Unknown story task"
    assert all(r["complexity"] in {"L1", "L2", "L3"} for r in rows)
    assert all(r["eval_method"] in {"eval_script", "llm_as_a_judge"} for r in rows)
    assert all(bool(r["answer"].strip()) == (r["source_kind"] == "invariant") for r in rows)
    assert not (set(story_of) & REMOVED_FROM_COVERAGE)

    cohort = []
    for q, row in tasks.items():
        reasons = [EXCLUSIONS[s] for s in sorted(servers(row) & set(EXCLUSIONS))]
        if reasons:
            status = "excluded"
        elif q in REMOVED_FROM_COVERAGE:
            status = "coverage_removed"
        elif q not in story_of:
            status = "outside_stories"
        elif q in NOT_SCORED:
            status = "covered_not_scored"
        else:
            status = "scored"
        cohort.append(
            {
                "question_id": q,
                "source_kind": row["source_kind"],
                "servers": sorted(servers(row)),
                "story": story_of.get(q),
                "status": status,
                "policy_reasons": reasons,
                "defect": DEFECTS.get(q),
            }
        )
    runnable = {r["question_id"] for r in cohort if r["status"] != "excluded"}
    story_ids = set(story_of)
    scored = story_ids - NOT_SCORED
    assert story_ids <= runnable
    assert (len(tasks), len(runnable), len(story_ids), len(scored)) == (163, 77, 59, 58)

    full = read_json(data / "tool_full.json")["mcpServers"]
    config = read_json(data / "opencode.json")
    configured = set(config["mcp"]["servers"])
    smoke = read_json(data / "smoke.json")
    connected = {name for name, item in smoke["servers"].items() if item["opencode"] == "connected"}
    needed = set().union(*(servers(tasks[q]) for q in story_ids))
    all_needed = set().union(*(servers(row) for row in rows)) - {"code"}
    connected_story = needed & connected
    tool_count = sum(smoke["servers"][s].get("tools") or 0 for s in connected)
    story_tool_count = sum(smoke["servers"][s].get("tools") or 0 for s in connected_story)
    named_tools = set().union(*(tools(tasks[q]) for q in story_ids))

    results = []
    result_sources = {}
    for name in ("baseline-glm-results.jsonl", "baseline-glm-sens-results.jsonl"):
        records = read_jsonl(data / name)
        assert len({r["question_id"] for r in records}) == len(records)
        assert all(r["question_id"] in tasks for r in records)
        result_sources[name] = {
            "n": len(records),
            "status": dict(Counter(r["status"] for r in records)),
        }
        results.extend(records)
    assert len({r["question_id"] for r in results}) == len(results)
    ok = {r["question_id"]: r for r in results if r["status"] == "ok"}
    metrics = read_json(data / "baseline-metrics.json")
    assert set(metrics) == set(ok), "Metrics and completed results differ"
    assert all(metrics[q]["answer"] == ok[q]["answer"] for q in metrics)
    assert all(abs(metrics[q]["seconds"] - ok[q]["seconds"]) < 0.01 for q in metrics)

    def behavior(q):
        m = metrics[q]
        if sum(m["mcp_calls"].values()):
            return "mcp"
        if sum(m["native"].values()):
            return "native_without_mcp"
        if m["searches"]:
            return "search_only"
        return "no_recorded_calls"

    def called_servers(q):
        return {name.split(".")[0] for name, count in metrics[q]["mcp_calls"].items() if count}

    per_story = {}
    for story, qs in STORIES.items():
        per_story[story] = {
            "task_ids": qs,
            "n": len(qs),
            "scored": len(set(qs) & scored),
            "sensitive": sum(tasks[q]["source_kind"] == "sensitive" for q in qs),
            "complexity": dict(Counter(tasks[q]["complexity"] for q in qs)),
            "servers": sorted(set().union(*(servers(tasks[q]) for q in qs))),
            "completed": len(set(qs) & set(ok)),
            "behavior": dict(Counter(behavior(q) for q in qs if q in metrics)),
        }

    score_sources = {}
    for name in ("smoke-glm-scores.jsonl", "baseline-glm-sens-scores.jsonl"):
        records = read_jsonl(data / name)
        assert len({r["question_id"] for r in records}) == len(records)
        valid = [r for r in records if type(r["score"]) in (int, float) and r["score"] in (0, 1)]
        valid_story = [r for r in valid if r["question_id"] in scored]
        score_sources[name] = {
            "n": len(records),
            "valid": len(valid),
            "pass": sum(r["score"] for r in valid),
            "fail": sum(r["score"] == 0 for r in valid),
            "null_ids": [r["question_id"] for r in records if r["score"] is None],
            "methods": dict(Counter(r["method"] for r in records)),
            "valid_scored_story": len(valid_story),
            "pass_scored_story": sum(r["score"] for r in valid_story),
            "reference_dates": sorted({r["ref_date"] for r in valid if r.get("ref_date")}),
            "provenance_has_session_id": all("session_id" in r for r in records),
        }

    # What the snapshot itself says about the two weekday references.
    calendar_checks = []
    for rec in read_jsonl(data / "baseline-glm-sens-scores.jsonl"):
        if rec["question_id"] in {"Q306", "Q307"}:
            today = date.fromisoformat(rec["ref_date"])
            monday = today - timedelta(days=today.weekday() + 7)
            expected = monday + timedelta(days=6 if rec["question_id"] == "Q306" else 0)
            calendar_checks.append(
                {
                    "question_id": rec["question_id"],
                    "expected_date": str(expected),
                    "reference_matches": str(expected) in rec["ref_answer"],
                }
            )

    excluded_by_server = Counter()
    for row in rows:
        blocked = sorted(servers(row) & set(EXCLUSIONS))
        if blocked:
            excluded_by_server[blocked[0]] += 1

    # Servers called in completed runs beyond the story annotation.
    annotated_by_task = {q: servers(tasks[q]) for q in story_ids}
    outside_calls = []
    probe_contradicts = []
    for q in sorted(story_ids, key=lambda x: int(x[1:])):
        if q not in metrics:
            continue
        used = called_servers(q)
        extra = sorted(used - annotated_by_task[q])
        if extra:
            outside_calls.append(
                {"question_id": q, "annotated": sorted(annotated_by_task[q]), "extra_called": extra}
            )
        for s in used:
            if s in smoke["servers"] and smoke["servers"][s]["opencode"] != "connected":
                probe_contradicts.append({"question_id": q, "server": s})

    # Text quality of questions inside stories.
    near_pairs = []
    for story, qs in STORIES.items():
        bags = {q: word_bag(tasks[q]["question"]) for q in qs}
        for i, left in enumerate(qs):
            for right in qs[i + 1 :]:
                union = len(bags[left] | bags[right])
                score = len(bags[left] & bags[right]) / union if union else 0
                if score >= 0.55:
                    near_pairs.append(
                        {"story": story, "a": left, "b": right, "jaccard": round(score, 3)}
                    )
    near_pairs.sort(key=lambda item: -item["jaccard"])
    mixed_quotes = [
        q
        for q in story_ids
        if re.search(r"`[^'`\n]*'", tasks[q]["question"])
        or re.search(r"'[^'`\n]*`", tasks[q]["question"])
    ]
    stray_paren = [
        q
        for q in story_ids
        if re.search(r"\)\s*\.\s*$", tasks[q]["question"])
        and tasks[q]["question"].count("(") < tasks[q]["question"].count(")")
    ]

    baseline = {}
    for scope, scope_ids in [
        ("all_completed", set(ok)),
        ("stories_completed", set(ok) & story_ids),
        ("scored_stories_completed", set(ok) & scored),
    ]:
        baseline[scope] = {
            "n": len(scope_ids),
            "behavior": dict(Counter(behavior(q) for q in scope_ids)),
            "seconds": numeric_summary([metrics[q]["seconds"] for q in scope_ids]),
            "input_tokens": numeric_summary([metrics[q]["tin"] for q in scope_ids]),
            "output_tokens": numeric_summary([metrics[q]["tout"] for q in scope_ids]),
            "reasoning_tokens": numeric_summary([metrics[q]["treas"] for q in scope_ids]),
        }
    no_mcp = (q for q in story_ids & set(metrics) if behavior(q) != "mcp")
    baseline["story_ids_without_mcp"] = sorted(no_mcp)
    baseline["scored_story_ids_without_completed_run"] = sorted(scored - set(ok))
    baseline["result_sources"] = result_sources
    baseline["scores_by_source"] = score_sources
    with_result = {r["question_id"] for r in results}
    baseline["source_tasks_without_any_result"] = sorted(set(tasks) - with_result)
    baseline["calls_beyond_annotation"] = outside_calls
    baseline["called_servers_failed_in_probe"] = probe_contradicts

    stats = {
        "cohorts": {
            "source": len(tasks),
            "invariant": source_quality["invariant"]["rows"],
            "sensitive": source_quality["sensitive"]["rows"],
            "runnable_by_policy": len(runnable),
            "stories": len(story_ids),
            "scored": len(scored),
            "outside_stories": len(runnable - story_ids - REMOVED_FROM_COVERAGE),
            "excluded_by_server": dict(excluded_by_server),
            "defects": DEFECTS,
            "removed_from_coverage": sorted(REMOVED_FROM_COVERAGE),
            "covered_but_not_scored": sorted(NOT_SCORED),
        },
        "quality": {
            "sources": source_quality,
            "duplicate_ids": [],
            "unknown_complexities": [],
            "weekday_references_in_snapshot": calendar_checks,
            "financial_date_checks": check_financial_dates(
                tasks, {q for q in story_ids if story_of.get(q) == "stock_quote_compare"}
            ),
            "near_duplicate_questions": near_pairs,
            "mixed_quote_questions": sorted(mixed_quotes),
            "stray_paren_questions": sorted(stray_paren),
            "missing_required_runtime_artifacts": [
                "tasks-ru.jsonl",
                "reference evaluator source",
                "eval_script source",
                "test_data fixtures",
            ],
        },
        "distributions": {
            name: {
                "n": len(qs),
                "complexity": dict(Counter(tasks[q]["complexity"] for q in qs)),
                "evaluation": dict(Counter(tasks[q]["eval_method"] for q in qs)),
                "source_kind": dict(Counter(tasks[q]["source_kind"] for q in qs)),
                "answer_kind_heuristic": dict(Counter(answer_kind(tasks[q]) for q in qs)),
                "unique_annotated_tools_per_task": dict(Counter(len(tools(tasks[q])) for q in qs)),
            }
            for name, qs in [("runnable", runnable), ("stories", story_ids), ("scored", scored)]
        },
        "per_story": per_story,
        "environment": {
            "configured": len(configured),
            "connected": len(connected),
            "required_by_source": len(all_needed),
            "required_by_stories": len(needed),
            "connected_story_servers": len(connected_story),
            "disconnected_story_servers": sorted(needed - connected),
            "config_server_sets_equal": set(full) == configured,
            "connected_tools": tool_count,
            "connected_story_tools": story_tool_count,
            "named_story_tools": len(named_tools),
            "non_story_server_tools": tool_count - story_tool_count,
            "connected_without_tool_inventory": sorted(
                s for s in connected if smoke["servers"][s].get("tools") is None
            ),
            "schema_bytes_connected": sum(
                smoke["servers"][s].get("schema_bytes") or 0 for s in connected
            ),
            "schema_bytes_story": sum(
                smoke["servers"][s].get("schema_bytes") or 0 for s in connected_story
            ),
            "snapshot_permissions_are_allow_all": config.get("permissions")
            == [{"action": "*", "resource": "*", "effect": "allow"}],
        },
        "baseline": baseline,
        "design_sensitivity_not_observed_results": {
            "n_scored_tasks": len(scored),
            "n_story_clusters": len(STORIES),
            "one_task_pp": 100 / len(scored),
            "planned_runs": len(scored) * 3 * 3,
            "illustrations_single_binary_run": [
                {
                    "improved": a,
                    "regressed": b,
                    "difference_pp": 100 * (a - b) / len(scored),
                    "mcnemar_exact_two_sided_p": exact_mcnemar(a, b),
                }
                for a, b in [(9, 0), (25, 16)]
            ],
        },
    }
    manifest = {
        "files": [
            dict(
                name=f.name,
                bytes=f.stat().st_size,
                sha256=hashlib.sha256(f.read_bytes()).hexdigest(),
            )
            for f in sorted(data.iterdir())
            if f.is_file()
        ],
        "analysis_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "selection_policy": EXCLUSIONS,
        "story_map": STORIES,
        "quantiles": "type 7 linear interpolation",
        "network_used": False,
    }
    write_json(output / "eda-stats.json", stats)
    write_json(output / "input-manifest.json", manifest)
    write_json(output / "cohort.json", cohort)
    chart(
        diagrams / "funnel.svg",
        "Отбор задач для сравнения навыков",
        [
            ("Исходные задачи", [len(tasks)]),
            ("Доступны для выполнения", [len(runnable)]),
            ("Входят в девять историй", [len(story_ids)]),
            ("Включены в расчёт метрик", [len(scored)]),
        ],
        ["Число задач"],
    )
    chart(
        diagrams / "stories.svg",
        "Заявленная сложность задач в девяти историях",
        [
            (STORY_LABELS[s], [v["complexity"].get(level, 0) for level in ("L1", "L2", "L3")])
            for s, v in per_story.items()
        ],
        ["Простые", "Средние", "Сложные"],
        unit="Число задач",
    )
    chart(
        diagrams / "baseline.svg",
        "Использование инструментов в предварительном прогоне: 59 задач",
        [
            (
                STORY_LABELS[s],
                [
                    v["behavior"].get("mcp", 0),
                    v["behavior"].get("native_without_mcp", 0),
                    v["behavior"].get("search_only", 0) + v["behavior"].get("no_recorded_calls", 0),
                    v["n"] - v["completed"],
                ],
            )
            for s, v in per_story.items()
        ],
        [
            "Были вызовы подключённых источников",
            "Только встроенные инструменты среды",
            "Только поиск инструментов либо вызовы не записаны",
            "Нет завершённого результата",
        ],
        unit="Число задач",
    )
    chart(
        diagrams / "servers.svg",
        "Источники данных и инструментов: настройка и подключение",
        [
            ("Указаны в настройках ассистента", [len(configured)]),
            ("Успешно подключены при проверке", [len(connected)]),
            ("Нужны для задач девяти историй", [len(needed)]),
            ("Успешно подключены из необходимых историям", [len(connected_story)]),
        ],
        ["Число источников"],
        note="Успешное подключение не гарантирует выполнение запросов",
    )
    scored_dist = stats["distributions"]["scored"]
    answer_labels = {
        "Проверка файла": "Проверка состояния файла",
        "Вычисляемый эталон": "Эталон вычисляется в день прогона",
    }
    chart(
        diagrams / "answer-types.svg",
        "Эталоны и способы проверки 58 оцениваемых задач",
        [
            (answer_labels.get(k, k), [v])
            for k, v in sorted(scored_dist["answer_kind_heuristic"].items(), key=lambda x: -x[1])
        ],
        ["Число задач"],
    )
    score_labels = {
        "smoke-glm-scores": "Пробная часть прогона",
        "baseline-glm-sens-scores": "Задачи, чувствительные ко времени",
    }
    chart(
        diagrams / "score-coverage.svg",
        "Наличие оценок и результаты предварительного прогона",
        [
            (
                score_labels.get(name.removesuffix(".jsonl"), name),
                [v["pass"], v["fail"], len(v["null_ids"])],
            )
            for name, v in score_sources.items()
        ],
        ["Правильный ответ", "Неверный ответ", "Нет оценки: не удалось вычислить эталон"],
        unit="Число задач",
    )
    summary = {
        "cohorts": stats["cohorts"],
        "baseline_stories": baseline["stories_completed"],
        "score_sources": score_sources,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=HERE)
    args = parser.parse_args()
    same_dir = args.data.resolve() == args.output.resolve()
    output_in_data = args.data.resolve() in args.output.resolve().parents
    if same_dir or output_in_data:
        parser.error("Output must be outside the source data directory")
    analyze(args.data.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
