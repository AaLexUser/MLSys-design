"""EDA of the MCPVerse tasks selected for the SkillFactory benchmark.

Reads the MCPVerse task files, fixtures and reference templates, the selection constants
of the SkillFactory runner and its refusal tasks, checks them against docs/eda/inputs.json,
and writes stats.json,
selection.jsonl and figures/*.png next to this file. Neither the runner nor MCPVerse code is
executed: the runner and the templates are parsed with ast, fixtures are only read.

    uv run --group eda python docs/eda/analyze.py \
        --mcpverse PATH_TO_MCPVERSE_CHECKOUT --runner PATH_TO_benchmarks/mcpverse
"""

import argparse
import ast
import csv
import datetime as dt
import hashlib
import itertools
import json
import re
import sqlite3
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import openpyxl

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"

# Reference palette (dataviz skill, light mode).
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#b9b8b2"
INK, INK2 = "#0b0b0b", "#52514e"

STORY_RU = {
    "hf_dataset_inspect": "Датасеты Hugging Face",
    "paper_fact_lookup": "Факты из статей",
    "files_find_read": "Поиск и чтение файлов",
    "files_prepare": "Подготовка файлов",
    "git_repo_state": "Состояние git-репозитория",
    "table_data_lookup": "Таблицы и база данных",
    "stock_quote_compare": "Котировки акций",
    "app_store_research": "Приложения App Store",
    "bazi_reading": "Китайский гороскоп (бацзы)",
}
# Where the data behind a server lives. Decides whether a reference can drift.
LOCAL_SERVERS = {"filesystem", "git", "sqlite", "excel"}  # fixtures in the MCPVerse repo
CALC_SERVERS = {"Bazi"}  # deterministic calendar computation, no external data
MARKET_SERVERS = {"yahoo-finance", "kospi-kosdaq", "investor"}
REFUSAL_RE = re.compile(
    r"\b(cannot|can't|unable|not available|no access|permission denied|refuse)\b", re.I
)
CJK_RE = re.compile(r"[　-〿一-鿿＀-￯]")
DATE_RE = re.compile(
    r"\b(January|February|March|April|May|June|July|August|September|October|"
    r"November|December) (\d{1,2}), (\d{4})"
)
STEMS, BRANCHES = "甲乙丙丁戊己庚辛壬癸", "子丑寅卯辰巳午未申酉戌亥"
SHORT_REF = 30  # characters; a reference this short can be checked without a judge
REFUSAL_RU = {
    "no_data": "Отказ: нет данных",
    "no_access": "Отказ: нет доступа",
    "out_of_domain": "Отказ: вне домена",
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dir_digest(path):
    """SHA-256 over sorted relative paths and contents: any added, removed or edited file counts."""
    h = hashlib.sha256()
    for f in sorted(p for p in path.rglob("*") if p.is_file()):
        h.update(f.relative_to(path).as_posix().encode() + b"\0" + f.read_bytes() + b"\0")
    return h.hexdigest()


def verify_inputs(roots, manifest):
    """Stop if any input differs from the manifest: numbers below depend on exact files."""
    for key, root in roots.items():
        spec = manifest[key]
        for rel, digest in spec["files"].items():
            got = sha256(root / rel)
            if got != digest:
                sys.exit(f"{key}/{rel}: sha256 {got} != manifest {digest}")
        for rel, digest in spec.get("dirs", {}).items():
            got = dir_digest(root / rel)
            if got != digest:
                sys.exit(f"{key}/{rel}/: sha256 {got} != manifest {digest}")
        head = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True
        ).stdout.strip()
        if head and head != spec["commit"]:
            print(f"warning: {key} HEAD {head[:7]} != manifest {spec['commit'][:7]}")


def runner_constants(path):
    """Literal constants of runner.py plus story names taken from the STORY_QIDS comments."""
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    names = {"REQUIRED_KEYS", "CN_ONLY_SERVERS", "BROKEN_SERVERS", "DEFECT_QIDS", "STORY_QIDS"}
    consts, story_span = {}, None
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in names:
                value = node.value
                if isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute):
                    value = value.func.value  # "Q1,Q2".split(",") -> keep the string
                consts[name] = ast.literal_eval(value)
                if name == "STORY_QIDS":
                    story_span = (node.lineno, node.end_lineno)
    if story_span is None:
        sys.exit("STORY_QIDS not found in runner.py")
    stories, pending = {}, []
    for line in src.splitlines()[story_span[0] - 1 : story_span[1]]:
        code, _, comment = line.partition("#")
        pending += re.findall(r"Q\d+", code)
        if comment.strip():
            for qid in pending:
                stories[qid] = comment.strip()
            pending = []
    order = consts["STORY_QIDS"].split(",")
    if sorted(order) != sorted(stories):
        sys.exit("story comments do not cover STORY_QIDS")
    consts["STORY_OF"] = stories
    return consts


def task_servers(row):
    return {m.strip() for m in re.split(r"[+;]", row["MCP"] or "") if m.strip()}


def parse_tools(raw):
    """'yahoo-finance/get_x * 2 + get_y' -> {'get_x': 2, 'get_y': 1}."""
    calls = {}
    for part in raw.split("+"):
        part = part.strip()
        if not part:
            continue
        mult = re.search(r"\*\s*(\d+)\s*$", part)
        name = re.sub(r"\*\s*\d+\s*$", "", part).strip().split("/")[-1]
        calls[name] = calls.get(name, 0) + (int(mult.group(1)) if mult else 1)
    return calls


def load_tasks(mcpverse):
    rows = []
    for name in ("mcpverse_time_invariant_v1.1.csv", "mcpverse_time_sensitive_v1.1.csv"):
        with open(mcpverse / "mcpverse" / "data" / name, newline="", encoding="utf-8") as f:
            rows += list(csv.DictReader(f))
    return rows


def funnel(rows, c):
    """Apply the selection in the order the author describes it, one reason per task."""
    keyed = set(c["REQUIRED_KEYS"])
    steps = [
        ("unavailable", "сервис из Китая или нужен ключ, которого нет"),
        ("broken", "портал открытых данных не отвечает"),
        ("code", "нужно исполнять код"),
        ("defect", "эталон противоречит вопросу или неверен"),
        ("outside", "не складываются в истории бенчмарка"),
    ]
    hits = Counter()
    for r in rows:
        qid, need = r["question_id"], task_servers(r) - {"code"}
        flags = {
            "unavailable": bool(need & (c["CN_ONLY_SERVERS"] | keyed)),
            "broken": bool(need & c["BROKEN_SERVERS"]),
            "code": not need,
            "defect": qid in c["DEFECT_QIDS"],
            "outside": qid not in c["STORY_OF"],
        }
        hits[sum(flags[k] for k in ("unavailable", "broken", "code", "defect"))] += 1
        r["status"] = next((k for k, _ in steps if flags[k]), "scored")
        r["story"] = c["STORY_OF"].get(qid)
    left, out = len(rows), [("Все задачи MCPVerse v1.1", len(rows), None)]
    for key, why in steps:
        n = sum(r["status"] == key for r in rows)
        left -= n
        out.append((why, left, n))
    keyed_only = sum(
        bool(task_servers(r) & keyed) and not task_servers(r) & c["CN_ONLY_SERVERS"] for r in rows
    )
    return out, {"tasks_hit_by_several_filters": hits[2] + hits[3], "keyed_not_cn": keyed_only}


def reference_kind(r):
    if r["time-sensitive"] == "Yes":
        return "computed"
    if r["eval_method"] == "eval_script":
        return "file"
    return "text"


def reference_basis(r):
    """Where the right answer comes from. Decides whether and how a reference can go stale."""
    if r["eval_method"] == "eval_script":
        return "script"
    if r["time-sensitive"] == "Yes":
        return "run_day"
    if task_servers(r) <= LOCAL_SERVERS | CALC_SERVERS:
        return "local"
    return "external"


def jaccard_matrix(texts):
    toks = [set(re.findall(r"\w+", t.lower())) for t in texts]
    n = len(toks)
    m = np.eye(n)
    for i, j in itertools.combinations(range(n), 2):
        m[i, j] = m[j, i] = len(toks[i] & toks[j]) / len(toks[i] | toks[j])
    return m


def families(m, thr):
    parent = list(range(len(m)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in zip(*np.where(np.triu(m, 1) >= thr), strict=True):
        parent[find(i)] = find(j)
    return len({find(i) for i in range(len(m))})


def hour_pillar(day_stem, hour):
    """Hour pillar by the five-rats rule: the day stem fixes the stem of the rat hour."""
    branch = ((hour + 1) // 2) % 12
    return STEMS[(STEMS.index(day_stem) % 5 * 2 + branch) % 10] + BRANCHES[branch]


def defect_checks(rows):
    """Turn the three known reference defects into rules that run on every task."""
    out = {"weekend_market_dates": [], "bazi_hour_mismatch": [], "julian_day_dates": {}}
    for r in rows:
        q, a = r["question"], r["answer"]
        if task_servers(r) & MARKET_SERVERS:
            for mon, day, year in DATE_RE.findall(q):
                d = dt.datetime.strptime(f"{mon} {day} {year}", "%B %d %Y").date()
                if d.weekday() >= 5:
                    out["weekend_market_dates"].append(f"{r['question_id']} {d.isoformat()}")
            q_dates, a_dates = set(DATE_RE.findall(q)), set(DATE_RE.findall(a))
            if q_dates and a_dates and not a_dates & q_dates:
                out.setdefault("answer_dates_not_in_question", []).append(r["question_id"])
        pillars = re.fullmatch(r"(?:[一-鿿]{2}\s+){2}([一-鿿]{2})\s+([一-鿿]{2})", a.strip())
        clock = re.search(r"\b(\d{1,2}):(\d{2})", q)
        if pillars and clock:
            day, ref_hour = pillars.groups()
            hour = int(clock.group(1))
            if hour_pillar(day[0], hour) != ref_hour:
                rec = {
                    "qid": r["question_id"],
                    "stated_time": clock.group(0),
                    "pillar_for_stated_time": hour_pillar(day[0], hour),
                    "reference_pillar": ref_hour,
                }
                born = DATE_RE.search(q)
                if born and "UTC+8" in q:
                    d = dt.datetime.strptime(" ".join(born.groups()), "%B %d %Y")
                    local = d.replace(hour=hour, minute=int(clock.group(2)))
                    offset = local.replace(tzinfo=ZoneInfo("Asia/Shanghai")).utcoffset()
                    shift = (offset or dt.timedelta(hours=8)) - dt.timedelta(hours=8)
                    rec["shanghai_utc_offset_hours"] = (offset or shift).total_seconds() / 3600
                    rec["pillar_if_time_is_shanghai_local"] = hour_pillar(
                        day[0], (local - shift).hour
                    )
                out["bazi_hour_mismatch"].append(rec)
        jd = re.search(r"Julian Day (\d+(?:\.\d+)?)", q)
        if jd:
            date = dt.date(1970, 1, 1) + dt.timedelta(days=float(jd.group(1)) - 2440587.5)
            out["julian_day_dates"][r["question_id"]] = {"date": date.isoformat(), "ref": a}
    return out


def translation_checks(tasks, ru):
    issues = []
    for r in tasks:
        tr = ru.get(r["question_id"])
        if not tr or not tr["question_ru"].strip():
            issues.append(f"{r['question_id']}: no translation")
            continue
        if tr["question_en"] != r["question"]:
            issues.append(f"{r['question_id']}: English text differs from MCPVerse")
        en, ruq = tr["question_en"], tr["question_ru"]
        if Counter(re.findall(r"\d+(?:\.\d+)?", en)) != Counter(re.findall(r"\d+(?:\.\d+)?", ruq)):
            issues.append(f"{r['question_id']}: numbers differ")
        for path in re.findall(r"\./[\w./{}-]*\w", en):
            if path not in ruq:
                issues.append(f"{r['question_id']}: path {path} lost")
    return issues


def baseline_skill_checks(runner, tasks):
    tools_by_story = defaultdict(set)
    for r in tasks:
        tools_by_story[r["story"]] |= set(parse_tools(r["tool"]))
    out = {}
    for path in sorted((runner / "skills-baseline-v1").glob("*/SKILL.md")):
        story = path.parent.name.removesuffix("-v1")
        text = path.read_text(encoding="utf-8")
        out[story] = {
            "words": len(text.split()),
            "says_no_environment_contract": bool(
                re.search(r"контракт окружения не передан|привязк\w*[^.\n]*отсутствует", text, re.I)
            ),
            "benchmark_tool_names_mentioned": sorted(
                t for t in tools_by_story[story] if re.search(rf"\b{re.escape(t)}\b", text)
            ),
        }
    return out


def server_pinning(tool_json, servers):
    out = {}
    for name in sorted(servers):
        spec = tool_json[name]
        args = spec.get("args", [])
        if spec.get("command") == "uv" and "--directory" in args:
            out[name] = "vendored"
            continue
        pkg = next((a for a in args if not a.startswith("-") and a not in {"tool", "run"}), "")
        out[name] = "pinned" if re.search(r"@\d|==", pkg) else f"unpinned ({pkg})"
    return out


def structure_checks(rows, tool_json):
    nums = [int(r["question_id"][1:]) for r in rows]
    used = set().union(*(task_servers(r) for r in rows)) - {"code"}
    empty = {
        col: [r["question_id"] for r in rows if not (r[col] or "").strip()]
        for col in ("question", "MCP", "tool", "complexity", "answer", "eval_method")
    }
    return {
        "rows": len(rows),
        "unique_ids": len(set(nums)),
        "id_range": [min(nums), max(nums)],
        "empty_cells": {k: len(v) for k, v in empty.items()},
        "empty_answers_all_run_day": all(
            r["time-sensitive"] == "Yes" for r in rows if r["question_id"] in empty["answer"]
        ),
        "servers_used": len(used),
        "servers_configured": len(tool_json),
        "servers_missing_in_config": sorted(used - set(tool_json)),
    }


def template_checks(rows, templates, selected):
    """Static check of run-day templates: same question, same server as the task. Not executed."""
    sens = [r for r in rows if r["time-sensitive"] == "Yes"]
    by_q = {" ".join(r["question"].split()): r["question_id"] for r in sens}
    out = {"total": len(sens), "missing": [], "label_differs": {}, "other_server": []}
    out["selected_same_server"] = 0
    for r in sens:
        path = templates / f"t{r['get_answer_id']}.py"
        if not path.exists():
            out["missing"].append(r["question_id"])
            continue
        src = path.read_text(encoding="utf-8")
        label = next(
            (
                " ".join(ast.literal_eval(n.value).split())
                for n in ast.parse(src).body
                if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "question"
            ),
            None,
        )
        if label != " ".join(r["question"].split()):
            ref = re.search(r'reference\s*=\s*"([^"]*)"', src)
            out["label_differs"][r["question_id"]] = {
                "label_is_question_of": by_q.get(label),
                "reference_text": ref.group(1) if ref else None,
            }
        called = set(re.findall(r'"mcp":\s*"([^"]+)"', src))
        if called and called <= task_servers(r):
            out["selected_same_server"] += r["question_id"] in selected
        else:
            out["other_server"].append(r["question_id"])
    return out


def recompute_local_refs(mcpverse, by_id):
    """Recompute references of local-fixture tasks from the files, without any server.

    Each check compares the literal reading of the question with the reference. Where the
    folder holds more than the question assumes, the other readings are kept as well.
    """
    td = mcpverse / "mcpverse" / "test_data"
    ans = {q: r["answer"].strip() for q, r in by_id.items()}
    ex, tree = td / "filesystem" / "example", td / "filesystem" / "tree_example"
    trunc = td / "txt_truncated"
    top = sorted(trunc.glob("*.txt"))
    deep = sorted(trunc.rglob("*.txt"))
    deepest = max((f for f in tree.rglob("*") if f.is_file()), key=lambda f: len(f.parts))
    e_files = {f.name: f.read_text(encoding="utf-8").strip() for f in sorted(ex.glob("e*"))}

    def nature(files, ascii_bounds=True):
        pat = re.compile(r"\bNature\b", re.A if ascii_bounds else 0)
        return sum(len(pat.findall(f.read_text(encoding="utf-8"))) for f in files)

    def preview(files):
        text = "\n---\n".join(f.read_text(encoding="utf-8")[:100] for f in files)
        return len(text.encode("utf-8"))

    def biggest(files):
        return max(files, key=lambda f: f.stat().st_size).name

    def reports(files):
        return [f.name for f in files if "年报" in f.read_text(encoding="utf-8")]

    db = sqlite3.connect(f"file:{td / 'california_schools.db'}?mode=ro", uri=True)
    tables = sorted(x[0] for x in db.execute("SELECT name FROM sqlite_master WHERE type='table'"))
    best = db.execute("SELECT sname FROM satscores ORDER BY AvgScrMath DESC LIMIT 1").fetchone()[0]
    db.close()
    wb = openpyxl.load_workbook(td / "excel" / "excel_test.xlsx", read_only=True)
    five = {
        ws.title: [c for c in next(ws.iter_rows(values_only=True)) if c is not None]
        for ws in wb.worksheets
        if ws.max_column == 5
    }
    wb.close()
    checks = {
        "Q87": (e_files["e1.txt"] in ans["Q87"], {}),
        "Q89": (ans["Q89"] == str(sum(1 for _ in ex.iterdir())), {}),
        "Q91": (
            next(tree.rglob("t1-1-1.txt")).relative_to(td).as_posix() in ans["Q91"],
            {},
        ),
        "Q92": (ans["Q92"].startswith(str((ex / "e1.txt").stat().st_size)), {}),
        "Q131": (all(k in ans["Q131"] and v in ans["Q131"] for k, v in e_files.items()), {}),
        "Q132": (deepest.read_text(encoding="utf-8").strip() in ans["Q132"], {}),
        "Q172": (ans["Q172"] == str(len(b"Test Data")), {}),
        "Q173": (
            reports(top) == [ans["Q173"]],
            {"literal": reports(top), "with_subfolders": reports(deep)},
        ),
        "Q174": (
            biggest(top) == ans["Q174"],
            {"literal": biggest(top), "with_subfolders": biggest(deep)},
        ),
        "Q175": (
            ans["Q175"] == str(nature(top)),
            {
                "literal": nature(top),
                "with_subfolders": nature(deep),
                "unicode_word_bounds": nature(top, False),
            },
        ),
        "Q176": (
            ans["Q176"].split()[0] == str(preview(top)),
            {"literal": preview(top), "with_subfolders": preview(deep)},
        ),
        "Q21": (ans["Q21"] == best, {}),
        "Q22": (all(t in ans["Q22"] for t in tables), {}),
        "Q128": (
            len(five) == 1
            and all(str(x) in ans["Q128"] for x in [*five, *next(iter(five.values()))]),
            {},
        ),
    }
    tracked = subprocess.run(
        ["git", "-C", str(mcpverse), "ls-files", "--", str(trunc.relative_to(mcpverse))],
        capture_output=True,
        text=True,
    ).stdout.split()
    extra = [f.relative_to(trunc).as_posix() for f in deep if f.parent != trunc]
    return {
        "recomputed": len(checks),
        "match": sum(ok for ok, _ in checks.values()),
        "mismatch": sorted(q for q, (ok, _) in checks.items() if not ok),
        "ambiguous": {
            q: alt
            for q, (_, alt) in checks.items()
            if len({json.dumps(v) for v in alt.values()}) > 1
        },
        "txt_truncated_top_files": len(top),
        "txt_truncated_extra_files": extra,
        "extra_files_in_commit": [
            x for x in extra if f"{trunc.relative_to(mcpverse).as_posix()}/{x}" in tracked
        ],
        "read_calls_in_tool_column": {q: parse_tools(by_id[q]["tool"]) for q in ("Q173", "Q174")},
    }


def unused_fixtures(mcpverse, rows, tool_json):
    """Top-level fixture files that no task question and no server config points to."""
    td = mcpverse / "mcpverse" / "test_data"
    cfg = json.dumps(tool_json)
    return sorted(
        f.name
        for f in td.iterdir()
        if f.is_file() and f.name not in cfg and not any(f.name in r["question"] for r in rows)
    )


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GREY)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.set_axisbelow(True)


def save(fig, name):
    fig.savefig(FIG / name, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_funnel(steps):
    left = [s[1] for s in steps]
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    style(ax)
    y = np.arange(len(steps))[::-1]
    ax.barh(y, left, color=[GREY] + [BLUE] * (len(steps) - 2) + [AQUA], height=0.62)
    for yi, (lab, n, drop) in zip(y, steps, strict=True):
        txt = f"{n}" if drop is None else f"{n}   (−{drop})"
        ax.text(n + 2, yi, txt, va="center", fontsize=9, color=INK)
        ax.text(
            -3,
            yi,
            lab if drop is None else f"− {lab}",
            va="center",
            ha="right",
            fontsize=9,
            color=INK,
        )
    ax.set_yticks([])
    ax.set_xlim(0, 185)
    ax.set_xlabel("Осталось задач после шага", color=INK2, fontsize=9)
    ax.grid(axis="x", color="#ecebe7", linewidth=0.7)
    save(fig, "funnel.png")


def plot_stacked(tasks, key, cats, colors, legend, name, xlabel):
    size = Counter(t["story"] for t in tasks)  # insertion order breaks ties stably
    stories = sorted(size, key=lambda s: -size[s])
    fig, ax = plt.subplots(figsize=(8.5, 3.9))
    style(ax)
    y = np.arange(len(stories))[::-1]
    left = np.zeros(len(stories))
    for cat, col, lab in zip(cats, colors, legend, strict=True):
        vals = np.array([sum(t["story"] == s and key(t) == cat for t in tasks) for s in stories])
        ax.barh(
            y, vals, left=left, color=col, height=0.62, label=lab, edgecolor="white", linewidth=1.5
        )
        left += vals
    for yi, total in zip(y, left, strict=True):
        ax.text(total + 0.2, yi, f"{int(total)}", va="center", fontsize=9, color=INK)
    ax.set_yticks(y, [STORY_RU[s] for s in stories], fontsize=9, color=INK)
    ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    ax.grid(axis="x", color="#ecebe7", linewidth=0.7)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right", labelcolor=INK)
    save(fig, name)


def plot_similarity(m, tasks):
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    im = ax.imshow(m, cmap="Blues", vmin=0, vmax=1)
    bounds, start, ticks = [], 0, []
    stories = list(dict.fromkeys(t["story"] for t in tasks))
    for s in stories:
        n = sum(t["story"] == s for t in tasks)
        ticks.append(start + n / 2 - 0.5)
        start += n
        bounds.append(start - 0.5)
    for b in bounds[:-1]:
        ax.axhline(b, color=ORANGE, linewidth=0.8)
        ax.axvline(b, color=ORANGE, linewidth=0.8)
    ax.set_xticks(ticks, [STORY_RU[s] for s in stories], rotation=60, ha="right", fontsize=8)
    ax.set_yticks(ticks, [STORY_RU[s] for s in stories], fontsize=8)
    ax.tick_params(length=0, colors=INK)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("Сходство формулировок (доля общих слов)", color=INK2, fontsize=8.5)
    cb.ax.tick_params(labelsize=8, colors=INK2)
    for side in ax.spines.values():
        side.set_visible(False)
    save(fig, "similarity.png")


def plot_coverage(have):
    labels = list(have)
    need = [20, 10, 10, 10]
    fig, ax = plt.subplots(figsize=(8.5, 2.9))
    style(ax)
    y = np.arange(len(labels))[::-1]
    ax.barh(y + 0.18, need, height=0.34, color=GREY, label="Требование раздела 2.1")
    ax.barh(y - 0.18, list(have.values()), height=0.34, color=BLUE, label="Есть в бенчмарке")
    for yi, v in zip(y, have.values(), strict=True):
        ax.text(v + 0.6, yi - 0.18, str(v), va="center", fontsize=9, color=INK)
    ax.set_yticks(y, labels, fontsize=9, color=INK)
    ax.set_xlabel("Задач", color=INK2, fontsize=9)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right", labelcolor=INK)
    ax.grid(axis="x", color="#ecebe7", linewidth=0.7)
    save(fig, "requirements.png")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--mcpverse", type=Path, required=True)
    p.add_argument("--runner", type=Path, required=True)
    a = p.parse_args()
    manifest = json.loads((HERE / "inputs.json").read_text(encoding="utf-8"))
    verify_inputs({"mcpverse": a.mcpverse, "runner": a.runner}, manifest)
    FIG.mkdir(exist_ok=True)

    c = runner_constants(a.runner / "runner.py")
    rows = load_tasks(a.mcpverse)
    steps, overlap = funnel(rows, c)
    by_id = {r["question_id"]: r for r in rows}
    order = c["STORY_QIDS"].split(",")
    tasks = [by_id[q] for q in order if by_id[q]["status"] == "scored"]
    ru = {}
    for line in (a.runner / "tasks-ru.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            ru[rec["question_id"]] = rec
    tool_json = json.loads((a.mcpverse / "mcpverse" / "tool_full.json").read_text())["mcpServers"]
    refusal = [
        json.loads(line)
        for line in (a.runner / "tasks-refusal.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    bad = [
        t["id"]
        for t in refusal
        if t["category"] not in REFUSAL_RU
        or (t["source_qid"] is not None and t["source_qid"] not in by_id)
        or (t["source_qid"] is None and t["story"] is not None)
        or t["expected_outcome"] != "refusal"
        or (t["story"] is not None and t["story"] != by_id[t["source_qid"]]["story"])
    ]
    if bad:
        sys.exit(f"malformed refusal tasks {bad}")

    with open(HERE / "selection.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(
                json.dumps(
                    {
                        k: r[k]
                        for k in (
                            "question_id",
                            "MCP",
                            "complexity",
                            "eval_method",
                            "time-sensitive",
                            "status",
                            "story",
                        )
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    per_story = {}
    for s in dict.fromkeys(t["story"] for t in tasks):
        ts = [t for t in tasks if t["story"] == s]
        calls = [sum(parse_tools(t["tool"]).values()) for t in ts]
        per_story[s] = {
            "tasks": len(ts),
            "servers": sorted(set().union(*(task_servers(t) for t in ts))),
            "distinct_tools": len(set().union(*(parse_tools(t["tool"]) for t in ts))),
            "complexity": dict(sorted(Counter(t["complexity"] for t in ts).items())),
            "reference": dict(Counter(reference_kind(t) for t in ts)),
            "reference_basis": dict(Counter(reference_basis(t) for t in ts)),
            "tasks_with_2plus_calls": sum(x >= 2 for x in calls),
        }

    text_refs = [t for t in tasks if reference_kind(t) == "text"]
    lengths = [len(t["answer"].strip()) for t in text_refs]
    sim = jaccard_matrix([t["question"] for t in tasks])
    same = np.array([[x["story"] == y["story"] for y in tasks] for x in tasks])
    iu = np.triu_indices(len(tasks), 1)
    pairs_hi = [(sim[i, j], same[i, j]) for i, j in zip(*iu, strict=True) if sim[i, j] >= 0.5]
    narrow = [r for r in rows if r["status"] == "outside"]
    code_rows = [r for r in rows if r["status"] == "code"]
    code_ext = Counter(
        (re.findall(r"\.(xlsx|docx|pdf|pptx|md|txt|json)\b", r["question"]) or ["none"])[0]
        for r in code_rows
    )
    templates = a.mcpverse / "mcpverse" / "templates"
    computed = [t for t in tasks if reference_kind(t) == "computed"]
    raw_tools = [t["tool"] for t in tasks]

    refusal_by_cat = Counter(t["category"] for t in refusal)
    have = {"Задачи с эталоном": len(tasks)} | {
        ru: refusal_by_cat[cat] for cat, ru in REFUSAL_RU.items()
    }
    refusal_refs = [t["question_id"] for t in tasks if REFUSAL_RE.search(t["answer"])]
    if refusal_refs:
        sys.exit(f"unexpected refusal references {refusal_refs}: update the coverage figure")

    sizes = [v["tasks"] for v in per_story.values()]
    # Outside tasks: a server with 3+ of them is a candidate tuning story, the rest are single
    # tasks. A story with swapped references keeps too few tasks and is not used at all.
    by_server = Counter(r["MCP"] for r in narrow)
    template_info = template_checks(rows, templates, {t["question_id"] for t in tasks})
    swapped = sorted(q for q in template_info["label_differs"] if by_id[q]["status"] == "outside")
    swapped_servers = {by_id[q]["MCP"] for q in swapped}
    story_servers = {sv: n for sv, n in by_server.items() if n >= 3}
    tuning_servers = {sv: n for sv, n in story_servers.items() if sv not in swapped_servers}
    # Smallest per-story spread (SD of the gain) at which a gain equal to the threshold keeps
    # the lower bound of the two-sided 90% bootstrap interval above zero (normal approximation).
    z90, threshold_pp = 1.645, 15
    # 95% interval of Cohen's kappa at kappa 0.8 and chance agreement 0.5 on n labelled answers.
    kappa, p_e, n_judge = 0.8, 0.5, 100
    p_o = kappa * (1 - p_e) + p_e
    selected_servers = set().union(*(task_servers(t) for t in tasks))

    stats = {
        "funnel": [{"step": s, "left": n, "dropped": d} for s, n, d in steps],
        "funnel_overlap": overlap,
        "story_qids_listed": len(order),
        "story_qids_not_scored": [q for q in order if by_id[q]["status"] != "scored"],
        "outside_story_by_server": dict(Counter(r["MCP"] for r in narrow)),
        "outside_story_qids": [r["question_id"] for r in narrow],
        "scored_tasks": len(tasks),
        "per_story": per_story,
        "complexity": dict(sorted(Counter(t["complexity"] for t in tasks).items())),
        "complexity_by_status": {
            st: dict(sorted(Counter(r["complexity"] for r in rows if r["status"] == st).items()))
            for st in dict.fromkeys(r["status"] for r in rows)
        },
        "reference_kind": dict(Counter(reference_kind(t) for t in tasks)),
        "reference_basis": dict(Counter(reference_basis(t) for t in tasks)),
        "story_sizes": sizes,
        "effective_stories": round(sum(sizes) ** 2 / sum(x * x for x in sizes), 2),
        "one_task_pp": round(100 / len(tasks), 2),
        "story_server": {s: v["servers"] for s, v in per_story.items()},
        "servers_shared_by_stories": {
            sv: [s for s, v in per_story.items() if sv in v["servers"]]
            for sv in sorted(selected_servers)
            if sum(sv in v["servers"] for v in per_story.values()) > 1
        },
        "multi_server_tasks": [t["question_id"] for t in tasks if len(task_servers(t)) > 1],
        "writes_files_but_judged_by_text": [
            t["question_id"]
            for t in tasks
            if "./outputs/" in t["question"] and t["eval_method"] != "eval_script"
        ],
        "computed_refs_with_template": sum(
            (templates / f"t{t['get_answer_id']}.py").exists() for t in computed
        ),
        "templates": template_info,
        "structure": structure_checks(rows, tool_json),
        "local_reference_recompute": recompute_local_refs(a.mcpverse, by_id),
        "unused_fixtures": unused_fixtures(a.mcpverse, rows, tool_json),
        "computed_refs_relative_to_run_date": [t["question_id"] for t in computed],
        "text_reference_count": len(text_refs),
        "text_reference_short": sum(x <= SHORT_REF for x in lengths),
        "text_reference_length_median": float(np.median(lengths)),
        "text_reference_length_max": max(lengths),
        "cjk_in_question": sum(bool(CJK_RE.search(t["question"])) for t in tasks),
        "cjk_in_reference": sum(bool(CJK_RE.search(t["answer"])) for t in tasks),
        "tool_label_formats": {
            "with_server_prefix": sum("/" in x for x in raw_tools),
            "with_multiplier": sum("*" in x for x in raw_tools),
            "tasks_2plus_expected_calls": sum(sum(parse_tools(x).values()) >= 2 for x in raw_tools),
        },
        "near_duplicates_jaccard_ge_0.5": {
            "pairs": len(pairs_hi),
            "within_story": int(sum(s for _, s in pairs_hi)),
            "max_cross_story": round(float(sim[~same].max()), 2),
            "families": families(sim, 0.5),
        },
        "defect_checks": defect_checks(rows),
        "q162_title_checked_manually": {
            "apod_2020_06_01": "The Lively Center of the Lagoon Nebula",
            "how": "redirect of apod.nasa.gov/apod/ap200601.html, 2026-11-14",
        },
        "translation_issues": translation_checks(tasks, ru),
        "translations_total": len(ru),
        "baseline_skills": baseline_skill_checks(a.runner, tasks),
        "server_pinning": server_pinning(tool_json, selected_servers),
        "selected_servers_env": {
            sv: sorted(tool_json[sv].get("env", {}))
            for sv in sorted(selected_servers)
            if tool_json[sv].get("env")
        },
        "code_tasks_by_file_type": dict(code_ext),
        "refusal_references": refusal_refs,
        "runs_per_benchmark_pass": len(tasks) * 3 * 3,
        "refusal_tasks": len(refusal),
        "refusal_by_category": {cat: refusal_by_cat[cat] for cat in REFUSAL_RU},
        "refusal_by_story": dict(Counter(t["story"] or "outside" for t in refusal)),
        "refusal_by_source_status": dict(
            Counter(by_id[t["source_qid"]]["status"] if t["source_qid"] else "new" for t in refusal)
        ),
        "runs_refusal": len(refusal) * 3 * 3,
        "runs_outside_story_control": len(narrow) * 3 * 3,
        "runs_per_test_pass": (len(tasks) + len(refusal)) * 3 * 3,
        "tuning_from_outside": {
            "stories": tuning_servers,
            "single_tasks": len(narrow) - sum(story_servers.values()),
            "excluded_swapped_reference": swapped,
            "stories_not_used": {
                sv: n for sv, n in story_servers.items() if sv not in tuning_servers
            },
            "tasks_used": sum(tuning_servers.values()),
        },
        "gain_detectable": {
            "threshold_pp": threshold_pp,
            "max_story_spread_pp": {
                n: round(threshold_pp * n**0.5 / z90, 1) for n in (len(per_story), 18)
            },
        },
        "judge_check": {
            "answers": n_judge,
            "kappa": kappa,
            "chance_agreement": p_e,
            "kappa_ci95_halfwidth": round(1.96 * (p_o * (1 - p_o) / n_judge) ** 0.5 / (1 - p_e), 3),
        },
    }
    (HERE / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )

    plot_funnel(steps)
    plot_stacked(
        tasks,
        lambda t: t["complexity"],
        ["L1", "L2", "L3"],
        [AQUA, BLUE, ORANGE],
        ["L1", "L2", "L3"],
        "complexity.png",
        "Задач (уровень сложности в разметке MCPVerse)",
    )
    plot_stacked(
        tasks,
        reference_basis,
        ["local", "external", "run_day", "script"],
        [AQUA, BLUE, ORANGE, GREY],
        [
            "из данных набора или расчёта",
            "записан один раз с внешнего сервиса",
            "считается в день прогона",
            "файл проверяет программа",
        ],
        "references.png",
        "Задач (откуда берётся эталон)",
    )
    plot_similarity(sim, tasks)
    plot_coverage(have)
    print(json.dumps({k: stats[k] for k in ("funnel", "scored_tasks")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
