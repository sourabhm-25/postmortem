"""
tools/devops_tools.py
======================
Postmortem agent tools — real GitHub integration, no mock data.

Philosophy:
  - Every tool hits a real API (GitHub, or your log source)
  - Tools are hypothesis-driven: each returns structured data
    the LLM uses to decide what to investigate next
  - Toolbox covers ALL incident types, not just deploy regressions
  - @tool decorator + Pydantic schemas = zero manual JSON schema writing

NEW in this version (4 tools added):
  - search_repo_for_class   → find a file by class name in the repo tree
  - read_file_from_github   → read actual source code at a specific commit
  - get_commit_file_diff    → line-by-line diff for one file in a commit
  - analyze_code_for_bug    → static analysis: finds exact bug line from error+code
  - create_github_issue     → opens a real GitHub issue with full root cause report
"""

import os
import re
import json
import base64
import requests
from datetime import datetime, timezone, timedelta
from collections import Counter
from langchain_core.tools import tool
from pydantic import BaseModel, Field


# ── Step counter (terminal trace) ─────────────────────────────────────────────
_step = 0

def _trace(tool_name: str, summary: str) -> None:
    global _step
    _step += 1
    print(f"  [{_step}] {tool_name.ljust(32)} → {summary}")

def reset_step_counter() -> None:
    global _step
    _step = 0


# ── GitHub helpers ────────────────────────────────────────────────────────────

def _gh(path: str, accept: str = "application/vnd.github.v3+json") -> requests.Response:
    """Authenticated GitHub GET. Raises on 4xx/5xx."""
    token = os.environ.get("GITHUB_TOKEN", "")
    if not token:
        raise EnvironmentError("GITHUB_TOKEN is not set.")
    resp = requests.get(
        f"https://api.github.com{path}",
        headers={"Authorization": f"token {token}", "Accept": accept},
        timeout=15,
    )
    resp.raise_for_status()
    return resp

def _gh_post(path: str, payload: dict) -> requests.Response:
    """Authenticated GitHub POST."""
    token = os.environ.get("GITHUB_TOKEN", "")
    resp = requests.post(
        f"https://api.github.com{path}",
        headers={
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
        },
        json=payload,
        timeout=15,
    )
    resp.raise_for_status()
    return resp


# ═══════════════════════════════════════════════════════════════════
# PYDANTIC INPUT SCHEMAS
# ═══════════════════════════════════════════════════════════════════

class LogInput(BaseModel):
    log_text: str = Field(description="Raw log text from your service. Paste the full log block.")

class SpikeInput(BaseModel):
    log_text: str = Field(description="Raw log text to analyze for error spike timing.")

class CommitsInput(BaseModel):
    repo:  str = Field(description="GitHub repo in 'owner/name' format, e.g. 'sourabhm-25/payment-service'")
    hours: int = Field(default=12, description="How many hours back to look for commits. Default 12.")

class PRDiffInput(BaseModel):
    repo:      str = Field(description="GitHub repo in 'owner/name' format")
    pr_number: int = Field(description="Pull request number to fetch the diff for")

class BlameInput(BaseModel):
    repo:     str = Field(description="GitHub repo in 'owner/name' format")
    filepath: str = Field(description="File path relative to repo root, e.g. 'src/services/PaymentService.java'")
    line:     int = Field(description="Line number to find who last changed")

class CorrelateInput(BaseModel):
    spike_time: str = Field(description="Timestamp when errors started spiking, e.g. '2025-04-11 02:31'")
    commits:    str = Field(description="JSON string returned by get_recent_commits")

class PatternInput(BaseModel):
    log_text: str = Field(description="Raw log text for deep pattern analysis")

class IssuesInput(BaseModel):
    repo:  str = Field(description="GitHub repo in 'owner/name' format")
    query: str = Field(default="", description="Search term to filter issues, e.g. 'NullPointerException' or '500 error'")

class RunbookInput(BaseModel):
    incident_title:   str = Field(description="Short title, e.g. 'POST /api/refunds 500 errors'")
    incident_type:    str = Field(description="DEPLOY_REGRESSION | INFRASTRUCTURE | CONFIGURATION | DEPENDENCY_FAILURE | DATA_CORRUPTION | SECURITY | UNKNOWN")
    root_cause:       str = Field(description="Exact root cause with file, line, and reason where applicable")
    affected_service: str = Field(description="Which service or endpoint was affected")
    timeline:         str = Field(description="Key timestamps: when it started, when detected, when resolved")
    fix_applied:      str = Field(description="What was done or needs to be done to resolve")
    prevention:       str = Field(description="Concrete steps to prevent recurrence")
    ruled_out:        str = Field(default="", description="What was investigated and ruled out")

# ── NEW schemas ───────────────────────────────────────────────────────────────

class SearchClassInput(BaseModel):
    repo:      str = Field(description="GitHub repo in 'owner/name' format")
    classname: str = Field(description="Class or file name to find, e.g. 'PaymentService' or 'JwtTokenValidator'")

class ReadFileInput(BaseModel):
    repo: str = Field(description="GitHub repo in 'owner/name' format")
    path: str = Field(description="File path in repo, e.g. 'src/security/JwtTokenValidator.java'")
    ref:  str = Field(default="main", description="Branch or commit SHA to read the file at. Default 'main'. Pass culprit commit SHA to see broken version.")

class CommitFileDiffInput(BaseModel):
    repo:     str = Field(description="GitHub repo in 'owner/name' format")
    sha:      str = Field(description="Full commit SHA (from get_recent_commits full_sha field)")
    filename: str = Field(description="File path to get the diff for, e.g. 'src/security/JwtTokenValidator.java'")

class BugAnalysisInput(BaseModel):
    error_class:   str = Field(description="Class name from stack trace, e.g. 'JwtTokenValidator'")
    error_method:  str = Field(description="Method name from stack trace, e.g. 'validateAndGetUserId'")
    error_line:    int = Field(default=0, description="Line number from stack trace, e.g. 21. Pass 0 if unknown.")
    error_message: str = Field(description="Full error message, e.g. 'SignatureException: JWT signature does not match'")
    file_content:  str = Field(description="Source file content returned by read_file_from_github")

class CreateIssueInput(BaseModel):
    repo:   str = Field(description="GitHub repo to create the issue in")
    title:  str = Field(description="Issue title — be specific. E.g. 'SignatureException in JwtTokenValidator.java:21 — HS256/RS256 algorithm mismatch'")
    body:   str = Field(description="Full issue body in Markdown — include root cause, exact file+line, fix, prevention, PR reference")
    labels: str = Field(default="bug,incident", description="Comma-separated labels, e.g. 'bug,incident,p0'")


# ═══════════════════════════════════════════════════════════════════
# TOOL 1 — Parse Logs
# ═══════════════════════════════════════════════════════════════════

@tool(args_schema=LogInput)
def parse_logs(log_text: str) -> str:
    """
    Parse raw log text and extract structured error information.

    ALWAYS call this first — before any other tool.
    Identifies: error types, frequency, first occurrence, stack traces,
    affected endpoints, and gives an initial incident classification signal.

    Returns: error count, warning count, top error patterns, first/last
    error timestamps, stack trace snippets, and an initial_classification
    hint (DEPLOY_REGRESSION / INFRASTRUCTURE / DEPENDENCY_FAILURE / etc).
    """
    lines = log_text.strip().split("\n")

    errors, warnings, info_lines = [], [], []
    for line in lines:
        if "ERROR" in line or "FATAL" in line or "CRITICAL" in line:
            errors.append(line)
        elif "WARN" in line or "WARNING" in line:
            warnings.append(line)
        elif "INFO" in line:
            info_lines.append(line)

    patterns = Counter(
        re.sub(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}[^\s]*", "", e).strip()[:120]
        for e in errors
    ).most_common(5)

    stack_traces, current = [], []
    for line in lines:
        stripped = line.strip()
        if (stripped.startswith("at ") or stripped.startswith("caused by") or
                stripped.startswith("Caused by")) and "." in stripped:
            current.append(stripped)
        else:
            if len(current) >= 2:
                stack_traces.append("\n".join(current))
            current = []

    all_errors_text = " ".join(errors).lower()
    if any(k in all_errors_text for k in ["nullpointer", "nullreference", "typeerror", "attributeerror", "keyerror"]):
        hint = "DEPLOY_REGRESSION — null/type error, likely a code change introduced a bug"
    elif any(k in all_errors_text for k in ["out of memory", "oom", "heap space", "memory"]):
        hint = "INFRASTRUCTURE — memory exhaustion"
    elif any(k in all_errors_text for k in ["connection refused", "timeout", "502", "503", "upstream"]):
        hint = "DEPENDENCY_FAILURE — upstream service or database unreachable"
    elif any(k in all_errors_text for k in ["connection pool", "too many connections", "pool exhausted"]):
        hint = "INFRASTRUCTURE — database connection pool exhaustion"
    elif any(k in all_errors_text for k in ["401", "403", "unauthorized", "forbidden", "jwt", "token", "signature"]):
        hint = "SECURITY or CONFIGURATION — auth/JWT failures"
    elif any(k in all_errors_text for k in ["config", "env", "environment", "missing key", "not found"]):
        hint = "CONFIGURATION — missing or wrong config value"
    else:
        hint = "UNKNOWN — need more data"

    deploy_lines = [l for l in info_lines if any(k in l.lower() for k in ["deploy", "release", "started", "v2", "v3"])]

    top_pattern = patterns[0][0] if patterns else "unknown"
    _trace("parse_logs", f"{len(errors)} errors | {top_pattern[:45]}")

    return json.dumps({
        "total_errors":           len(errors),
        "total_warnings":         len(warnings),
        "first_error_at":         errors[0][:50] if errors else None,
        "last_error_at":          errors[-1][:50] if errors else None,
        "top_error_patterns":     [{"pattern": p, "count": c} for p, c in patterns],
        "stack_traces":           stack_traces[:3],
        "deploy_events_in_log":   deploy_lines[:3],
        "initial_classification": hint,
    }, indent=2)


# ═══════════════════════════════════════════════════════════════════
# TOOL 2 — Get Error Spike Time
# ═══════════════════════════════════════════════════════════════════

@tool(args_schema=SpikeInput)
def get_error_spike_time(log_text: str) -> str:
    """
    Find the exact minute when errors started spiking.

    This timestamp is your anchor for everything that follows.
    Use it with correlate_deploy_with_spike to find the culprit deploy.

    Spike = first minute with 3+ errors OR a sudden jump (5x baseline rate).

    Returns: spike_time, errors_per_minute breakdown, peak minute,
    and minutes_to_peak (how fast it escalated).
    """
    counts: dict = {}
    for line in log_text.split("\n"):
        if not any(k in line for k in ["ERROR", "FATAL", "CRITICAL"]):
            continue
        m = re.search(r"(\d{4}-\d{2}-\d{2})[ T](\d{2}:\d{2})", line)
        if m:
            key = f"{m.group(1)} {m.group(2)}"
            counts[key] = counts.get(key, 0) + 1

    if not counts:
        _trace("get_error_spike_time", "no timestamped errors found")
        return json.dumps({"spike_time": None, "message": "No timestamped errors found in log"})

    sorted_minutes = sorted(counts)
    spike_time = next(
        (m for m in sorted_minutes if counts[m] >= 3),
        sorted_minutes[0]
    )

    peak_minute = max(counts, key=counts.get)
    spike_idx   = sorted_minutes.index(spike_time)
    peak_idx    = sorted_minutes.index(peak_minute)

    _trace("get_error_spike_time", f"spike at {spike_time} | peak {counts[peak_minute]} errors/min")

    return json.dumps({
        "spike_time":        spike_time,
        "errors_per_minute": counts,
        "peak_minute":       peak_minute,
        "peak_count":        counts[peak_minute],
        "minutes_to_peak":   peak_idx - spike_idx,
        "onset_pattern":     "sudden" if counts[peak_minute] > 5 * counts.get(sorted_minutes[max(0, spike_idx-1)], 1) else "gradual",
    }, indent=2)


# ═══════════════════════════════════════════════════════════════════
# TOOL 3 — Get Recent Commits (Real GitHub)
# ═══════════════════════════════════════════════════════════════════

@tool(args_schema=CommitsInput)
def get_recent_commits(repo: str, hours: int = 12) -> str:
    """
    Fetch all commits pushed to a GitHub repository in the last N hours.

    Includes: sha, commit message, author email, timestamp, associated PR number,
    and files changed per commit.

    Call this after you know the spike_time. Use hours=24 if nothing shows
    up in the default 12h window.

    Returns JSON list sorted newest-first.
    """
    since = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
    resp = _gh(f"/repos/{repo}/commits?since={since}&per_page=30")
    raw_commits = resp.json()

    if not raw_commits:
        _trace("get_recent_commits", f"0 commits in last {hours}h")
        return json.dumps({"commits": [], "message": f"No commits found in the last {hours} hours"})

    commits = []
    for c in raw_commits:
        sha = c["sha"]
        detail = _gh(f"/repos/{repo}/commits/{sha}")
        files_changed = [f["filename"] for f in detail.json().get("files", [])]

        pr_number = None
        try:
            pr_resp = _gh(
                f"/repos/{repo}/commits/{sha}/pulls",
                accept="application/vnd.github.groot-preview+json"
            )
            if pr_resp.json():
                pr_number = pr_resp.json()[0]["number"]
        except Exception:
            pass

        commits.append({
            "sha":           sha[:7],
            "full_sha":      sha,
            "message":       c["commit"]["message"].split("\n")[0],
            "author":        c["commit"]["author"]["email"],
            "timestamp":     c["commit"]["author"]["date"],
            "pr_number":     pr_number,
            "files_changed": files_changed,
        })

    most_recent = commits[0]
    _trace(
        "get_recent_commits",
        f"{len(commits)} commits | latest: {most_recent['message'][:40]} (PR #{most_recent['pr_number']})"
    )
    return json.dumps(commits, indent=2)


# ═══════════════════════════════════════════════════════════════════
# TOOL 4 — Correlate Deploy With Spike
# ═══════════════════════════════════════════════════════════════════

@tool(args_schema=CorrelateInput)
def correlate_deploy_with_spike(spike_time: str, commits: str) -> str:
    """
    THE CORE TOOL. Find which deployment caused the error spike.

    Scores every commit by: how close it was deployed before the spike,
    and whether it touched high-risk files (service, controller, api, handler,
    middleware, auth, cache, db, queue, worker, router, config, migration).

    Returns: most_likely_culprit with PR number and likelihood score,
    all_candidates ranked, and a plain-English summary.
    """
    try:
        commit_list = json.loads(commits)
        if isinstance(commit_list, dict) and "commits" in commit_list:
            commit_list = commit_list["commits"]
    except Exception:
        return "Error: could not parse commits JSON"

    try:
        spike_dt = datetime.fromisoformat(
            spike_time.strip().replace(" ", "T").replace("Z", "+00:00")
        )
        if spike_dt.tzinfo is None:
            spike_dt = spike_dt.replace(tzinfo=timezone.utc)
    except Exception as e:
        return f"Error parsing spike_time '{spike_time}': {e}"

    HIGH_RISK = [
        "service", "controller", "api", "handler", "middleware",
        "auth", "cache", "db", "database", "queue", "worker",
        "router", "config", "migration", "schema", "model",
        "payment", "refund", "order", "user", "session", "jwt", "token",
    ]

    candidates = []
    for c in commit_list:
        try:
            ts = c.get("timestamp", "")
            commit_dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            if commit_dt.tzinfo is None:
                commit_dt = commit_dt.replace(tzinfo=timezone.utc)
            delta_min = (spike_dt - commit_dt).total_seconds() / 60

            if not (0 < delta_min < 360):
                continue

            score = 100.0
            if 5 <= delta_min <= 60:    score += 60
            elif delta_min < 5:         score += 20
            elif 60 < delta_min <= 120: score += 20
            elif delta_min > 120:       score -= 20

            for f in c.get("files_changed", []):
                if any(k in f.lower() for k in HIGH_RISK):
                    score += 25
                    break

            candidates.append({
                "sha":                  c.get("sha"),
                "full_sha":             c.get("full_sha", ""),
                "pr_number":            c.get("pr_number"),
                "message":              c.get("message"),
                "author":               c.get("author"),
                "deployed_at":          ts[:19],
                "minutes_before_spike": round(delta_min, 1),
                "files_changed":        c.get("files_changed", []),
                "likelihood_score":     round(score, 1),
            })
        except Exception:
            continue

    candidates.sort(key=lambda x: x["likelihood_score"], reverse=True)

    if not candidates:
        _trace("correlate_deploy_with_spike", "no candidates found in window")
        return json.dumps({
            "culprit": None,
            "message": "No commits found in the 0–360 minute window before the spike. "
                       "Consider: config change, infra issue, or longer lookback window."
        })

    top = candidates[0]
    _trace(
        "correlate_deploy_with_spike",
        f"PR #{top['pr_number']} score={int(top['likelihood_score'])} "
        f"| {top['minutes_before_spike']}min before spike ← likely culprit"
    )
    return json.dumps({
        "most_likely_culprit": top,
        "all_candidates":      candidates,
        "total_candidates":    len(candidates),
        "summary": (
            f"PR #{top['pr_number']} ('{top['message']}') deployed "
            f"{top['minutes_before_spike']} minutes before the error spike. "
            f"Likelihood score: {int(top['likelihood_score'])}. "
            f"Inspect this PR's diff to confirm root cause."
        ),
    }, indent=2)


# ═══════════════════════════════════════════════════════════════════
# TOOL 5 — Get PR Diff (Real GitHub)
# ═══════════════════════════════════════════════════════════════════

@tool(args_schema=PRDiffInput)
def get_pr_diff(repo: str, pr_number: int) -> str:
    """
    Fetch the full code diff for a specific pull request.

    Read it carefully to find the EXACT line that introduced the bug.

    Look for:
      - Null/None dereference without null check
      - Missing error handling around new external calls
      - Changed method signatures that callers weren't updated for
      - Algorithm or configuration changes applied inconsistently (e.g. one
        side of a signing/verification pair updated but not the other)
      - Removed validation that was previously protecting downstream code

    Returns the raw unified diff (up to 6000 chars).
    """
    resp = _gh(
        f"/repos/{repo}/pulls/{pr_number}",
        accept="application/vnd.github.v3.diff"
    )
    diff = resp.text

    truncated = len(diff) > 6000
    diff_out  = diff[:6000]
    if truncated:
        diff_out += "\n\n[... diff truncated at 6000 chars ...]"

    _trace("get_pr_diff", f"PR #{pr_number} | {len(diff)} chars diff | {'truncated' if truncated else 'full'}")
    return diff_out


# ═══════════════════════════════════════════════════════════════════
# TOOL 6 — Git Blame (Real GitHub)
# ═══════════════════════════════════════════════════════════════════

@tool(args_schema=BlameInput)
def get_git_blame(repo: str, filepath: str, line: int) -> str:
    """
    Find who last changed a specific line and in which commit.

    USE THIS as a shortcut when the stack trace gives you an exact file
    and line number. Directly finds the commit that last touched the broken line.

    Returns: commit sha, author, timestamp, PR number, commit message.
    """
    resp = _gh(f"/repos/{repo}/commits?path={filepath}&per_page=5")
    commits = resp.json()

    if not commits:
        _trace("get_git_blame", f"no commits found for {filepath}")
        return json.dumps({"error": f"No commits found for file: {filepath}"})

    c = commits[0]
    sha = c["sha"]

    pr_number = None
    try:
        pr_resp = _gh(
            f"/repos/{repo}/commits/{sha}/pulls",
            accept="application/vnd.github.groot-preview+json"
        )
        if pr_resp.json():
            pr_number = pr_resp.json()[0]["number"]
    except Exception:
        pass

    result = {
        "filepath":  filepath,
        "line":      line,
        "sha":       sha[:7],
        "full_sha":  sha,
        "author":    c["commit"]["author"]["email"],
        "timestamp": c["commit"]["author"]["date"],
        "message":   c["commit"]["message"].split("\n")[0],
        "pr_number": pr_number,
        "note":      f"Most recent commit touching this file. Verify it changed line {line} by checking the PR diff.",
    }

    _trace("get_git_blame", f"{filepath}:{line} → last changed by {result['author']} in PR #{pr_number}")
    return json.dumps(result, indent=2)


# ═══════════════════════════════════════════════════════════════════
# TOOL 7 — Analyze Error Patterns
# ═══════════════════════════════════════════════════════════════════

@tool(args_schema=PatternInput)
def analyze_error_patterns(log_text: str) -> str:
    """
    Deep analysis of error patterns — goes beyond simple counting.

    USE THIS when parse_logs gives you errors but the cause is unclear —
    for example when there are multiple error types and you need to find
    which one is primary vs which are downstream effects.

    Returns: onset_pattern, primary_error, secondary_errors (cascading),
    affected_endpoints, pre_error_warnings, and cascade_analysis.
    """
    lines    = log_text.strip().split("\n")
    errors   = [l for l in lines if any(k in l for k in ["ERROR", "FATAL"])]
    warnings = [l for l in lines if "WARN" in l]

    first_error_ts = None
    for e in errors:
        m = re.search(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}", e)
        if m:
            first_error_ts = m.group(0)
            break

    pre_error_warnings = []
    if first_error_ts:
        for w in warnings:
            m = re.search(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}", w)
            if m and m.group(0) < first_error_ts:
                pre_error_warnings.append(w.strip())

    endpoints = re.findall(r"(?:GET|POST|PUT|DELETE|PATCH)\s+(/[^\s]+)", log_text)
    endpoint_counts = Counter(endpoints).most_common(5)

    error_types = Counter()
    for e in errors:
        for pattern in ["NullPointerException", "NullReferenceException", "TypeError",
                        "AttributeError", "KeyError", "TimeoutError", "ConnectionError",
                        "OutOfMemoryError", "StackOverflowError", "HTTP 500", "HTTP 502",
                        "HTTP 503", "pool exhausted", "connection refused",
                        "SignatureException", "JWT", "401", "403"]:
            if pattern.lower() in e.lower():
                error_types[pattern] += 1

    primary   = error_types.most_common(1)[0][0] if error_types else "unknown"
    secondary = [e for e, _ in error_types.most_common()[1:3]]

    time_counts: dict = {}
    for e in errors:
        m = re.search(r"\d{4}-\d{2}-\d{2}[ T](\d{2}:\d{2})", e)
        if m:
            time_counts[m.group(1)] = time_counts.get(m.group(1), 0) + 1

    max_rate = max(time_counts.values()) if time_counts else 0
    onset    = "sudden" if max_rate >= 5 else "gradual"
    cascade  = len(error_types) > 2

    _trace("analyze_error_patterns", f"primary={primary[:30]} | onset={onset} | cascade={cascade}")

    return json.dumps({
        "total_errors":         len(errors),
        "error_type_breakdown": dict(error_types.most_common()),
        "primary_error":        primary,
        "secondary_errors":     secondary,
        "cascade_likely":       cascade,
        "cascade_analysis":     "Multiple error types suggest cascading failure — find the PRIMARY error" if cascade else "Single error type — not a cascade",
        "onset_pattern":        onset,
        "affected_endpoints":   [{"endpoint": ep, "count": c} for ep, c in endpoint_counts],
        "pre_error_warnings":   pre_error_warnings[:5],
        "investigation_hint":   f"Focus on '{primary}' — this is the primary error. {'Other errors are likely downstream effects.' if cascade else ''}",
    }, indent=2)


# ═══════════════════════════════════════════════════════════════════
# TOOL 8 — Check Open Issues (Real GitHub)
# ═══════════════════════════════════════════════════════════════════

@tool(args_schema=IssuesInput)
def get_open_issues(repo: str, query: str = "") -> str:
    """
    Fetch recent open GitHub issues for a repo, optionally filtered by keyword.

    USE THIS to check if this incident is already known/reported, or if
    there are related issues that give more context about the problem.

    Returns: list of recent open issues with title, number, labels, and created_at.
    """
    path = f"/repos/{repo}/issues?state=open&per_page=20&sort=created&direction=desc"
    resp = _gh(path)
    issues = resp.json()

    if query:
        q = query.lower()
        issues = [i for i in issues if q in i["title"].lower() or q in i.get("body", "").lower()]

    result = [
        {
            "number":     i["number"],
            "title":      i["title"],
            "labels":     [l["name"] for l in i.get("labels", [])],
            "created_at": i["created_at"],
            "url":        i["html_url"],
        }
        for i in issues[:10]
        if "pull_request" not in i   # filter out PRs listed as issues
    ]

    _trace("get_open_issues", f"{len(result)} matching issues found")
    return json.dumps(result, indent=2)


# ═══════════════════════════════════════════════════════════════════
# TOOL 9 — Search Repo for Class (NEW)
# ═══════════════════════════════════════════════════════════════════

@tool(args_schema=SearchClassInput)
def search_repo_for_class(repo: str, classname: str) -> str:
    """
    Search the repo for a file matching a class name.

    USE THIS when you see a class in the stack trace (e.g. 'JwtTokenValidator')
    but don't know the exact file path. Returns matching paths so you can call
    read_file_from_github with the correct path.

    Example: stack trace says 'JwtTokenValidator.validateAndGetUserId(JwtTokenValidator.java:21)'
    → search_repo_for_class(repo, 'JwtTokenValidator')
    → finds 'src/security/JwtTokenValidator.java'
    """
    # Try code search API first
    try:
        resp = requests.get(
            "https://api.github.com/search/code",
            headers={
                "Authorization": f"token {os.environ.get('GITHUB_TOKEN', '')}",
                "Accept": "application/vnd.github.v3+json",
            },
            params={"q": f"{classname} repo:{repo}", "per_page": 5},
            timeout=15,
        )
        if resp.status_code == 200:
            items = resp.json().get("items", [])
            if items:
                results = [{"path": i["path"], "url": i["html_url"]} for i in items[:5]]
                _trace("search_repo_for_class", f"found {len(results)} file(s) for '{classname}'")
                return json.dumps({"classname": classname, "matches": results}, indent=2)
    except Exception:
        pass

    # Fallback: walk the git tree
    try:
        resp = _gh(f"/repos/{repo}/git/trees/HEAD?recursive=1")
        tree  = resp.json().get("tree", [])
        lower = classname.lower()
        matches = [
            {"path": item["path"]}
            for item in tree
            if item["type"] == "blob" and lower in item["path"].lower()
            and not item["path"].startswith(".")
        ]
        _trace("search_repo_for_class", f"tree search: {len(matches)} match(es) for '{classname}'")
        if not matches:
            return json.dumps({"message": f"No files found matching '{classname}' in {repo}"})
        return json.dumps({"classname": classname, "matches": matches[:5]}, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


# ═══════════════════════════════════════════════════════════════════
# TOOL 10 — Read File from GitHub (NEW)
# ═══════════════════════════════════════════════════════════════════

@tool(args_schema=ReadFileInput)
def read_file_from_github(repo: str, path: str, ref: str = "main") -> str:
    """
    Read the actual source code of a file from GitHub.

    IMPORTANT: Pass the culprit commit's SHA as ref — not 'main'.
    The bug may already be fixed on main. You need to read the code
    AS IT WAS when the incident happened (at the broken commit).

    Returns the file content with line numbers prepended so you can
    reference exact lines when calling analyze_code_for_bug.
    """
    try:
        token = os.environ.get("GITHUB_TOKEN", "")
        resp  = requests.get(
            f"https://api.github.com/repos/{repo}/contents/{path}",
            headers={"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"},
            params={"ref": ref},
            timeout=15,
        )
        if resp.status_code == 404:
            _trace("read_file_from_github", f"NOT FOUND: {path} @ {ref[:7]}")
            return json.dumps({"error": f"File '{path}' not found at ref '{ref}'. Try search_repo_for_class to find the correct path."})
        resp.raise_for_status()

        content = base64.b64decode(resp.json()["content"]).decode("utf-8", errors="replace")
        lines   = content.split("\n")
        numbered = "\n".join(f"{i+1:4d}  {line}" for i, line in enumerate(lines))

        _trace("read_file_from_github", f"{path} @ {ref[:7]} | {len(lines)} lines")
        return json.dumps({
            "path":        path,
            "ref":         ref,
            "total_lines": len(lines),
            "content":     numbered[:8000],
            "truncated":   len(numbered) > 8000,
        }, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)})


# ═══════════════════════════════════════════════════════════════════
# TOOL 11 — Get Commit File Diff (NEW)
# ═══════════════════════════════════════════════════════════════════

@tool(args_schema=CommitFileDiffInput)
def get_commit_file_diff(repo: str, sha: str, filename: str) -> str:
    """
    Get the exact line-by-line diff for ONE specific file in a commit.

    USE THIS after correlate_deploy_with_spike identifies the culprit commit.
    Shows exactly which lines were added (+) and removed (-) in that file.
    More precise than get_pr_diff when you already know the broken file.

    Lines marked ADDED: are new code. Lines marked REMOVED: are what was there before.
    """
    try:
        resp  = _gh(f"/repos/{repo}/commits/{sha}")
        files = resp.json().get("files", [])

        for f in files:
            if f["filename"] == filename or filename in f["filename"]:
                patch = f.get("patch", "No diff available")
                annotated = "\n".join(
                    f"ADDED  : {l[1:]}" if l.startswith("+") and not l.startswith("+++") else
                    f"REMOVED: {l[1:]}" if l.startswith("-") and not l.startswith("---") else
                    f"CONTEXT: {l}"
                    for l in patch.split("\n")
                )
                _trace("get_commit_file_diff", f"{filename} in {sha[:7]} | {f['additions']}+ {f['deletions']}-")
                return json.dumps({
                    "filename":  f["filename"],
                    "status":    f["status"],
                    "additions": f["additions"],
                    "deletions": f["deletions"],
                    "patch":     annotated,
                }, indent=2)

        all_files = [f["filename"] for f in files]
        _trace("get_commit_file_diff", f"'{filename}' not changed in {sha[:7]}")
        return json.dumps({
            "message":             f"'{filename}' was not changed in commit {sha[:7]}",
            "all_files_in_commit": all_files,
        }, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)})


# ═══════════════════════════════════════════════════════════════════
# TOOL 12 — Analyze Code for Bug (NEW)
# ═══════════════════════════════════════════════════════════════════

@tool(args_schema=BugAnalysisInput)
def analyze_code_for_bug(
    error_class:   str,
    error_method:  str,
    error_line:    int,
    error_message: str,
    file_content:  str,
) -> str:
    """
    Given source code and error details, find the EXACT line causing the bug.

    USE THIS after read_file_from_github. Pass the file content + the error
    details from the stack trace. Performs static analysis to identify the
    exact problematic line, explain WHY it fails, and suggest the fix.

    This answers "which exact line caused the bug" without needing to run the code.
    Covers: NPE, SignatureException (JWT algorithm mismatch), type errors,
    missing null checks, algorithm inconsistencies, and more.
    """
    # Parse numbered content from read_file_from_github
    lines = []
    try:
        data = json.loads(file_content)
        raw  = data.get("content", file_content)
    except Exception:
        raw = file_content

    for line in raw.split("\n"):
        m = re.match(r"^\s*(\d+)\s{2}(.*)", line)
        if m:
            lines.append((int(m.group(1)), m.group(2)))

    # Find the method body
    method_lines = []
    in_method    = False
    brace_depth  = 0

    for lineno, code in lines:
        if not in_method:
            if error_method in code and any(k in code for k in
                    ["void ", "public ", "private ", "protected ", "def ", "func ", "function "]):
                in_method   = True
                brace_depth = 0
        if in_method:
            method_lines.append((lineno, code))
            brace_depth += code.count("{") - code.count("}")
            if brace_depth <= 0 and len(method_lines) > 1:
                break

    # Find exact line from stack trace
    target_line = None
    target_code = ""
    if error_line > 0:
        for lineno, code in lines:
            if lineno == error_line:
                target_line = lineno
                target_code = code.strip()
                break

    # Static analysis patterns
    err_lower = error_message.lower()

    BUG_PATTERNS = [
        # NPE patterns
        (r"\.toUpperCase\(\)",        "NPE",        "Variable is null — .toUpperCase() cannot be called on null"),
        (r"\.toLowerCase\(\)",        "NPE",        "Variable is null — .toLowerCase() cannot be called on null"),
        (r"\.length\(\)",             "NPE",        "Variable is null — .length() cannot be called on null"),
        (r"(\w+)\.(\w+)\(\)\.(\w+)", "NPE",        "Chained call — intermediate result may be null"),
        # JWT / signature patterns
        (r"setSigningKey\(",          "SIGNATURE",  "setSigningKey uses wrong algorithm or key type"),
        (r"SignatureAlgorithm\.",      "SIGNATURE",  "Algorithm changed here — verify the other side matches"),
        (r"parseClaimsJws\(",         "SIGNATURE",  "Token parsing — algorithm must match the signing side"),
        # Auth patterns
        (r"HS256|RS256|HS512",        "ALGORITHM",  "Algorithm reference — check if signing and verification use the same algorithm"),
    ]

    null_risks = []
    for lineno, code in (method_lines if method_lines else lines[:60]):
        for pattern, bug_type, reason in BUG_PATTERNS:
            if re.search(pattern, code):
                is_target = (lineno == error_line)
                null_risks.append({
                    "line":          lineno,
                    "code":          code.strip(),
                    "bug_type":      bug_type,
                    "reason":        reason,
                    "is_error_line": is_target,
                })

    def _explain(code: str, error_msg: str) -> str:
        if "signatureexception" in error_msg.lower() or "jwt" in error_msg.lower():
            if "setSigningKey" in code:
                return ("The signing key type does not match the algorithm. "
                        "If JwtTokenProvider signs with RS256 (RSA private key), "
                        "JwtTokenValidator must verify with RS256 (RSA public key), "
                        "NOT an HS256 shared secret string.")
            if "parseClaimsJws" in code:
                return ("Token verification failed because the algorithm used here "
                        "does not match the algorithm used when the token was signed. "
                        "One side was updated (HS256→RS256) but the other was not.")
        if "nullpointerexception" in error_msg.lower() or "null" in error_msg.lower():
            if ".toUpperCase()" in code:
                var = re.search(r"(\w+)\.toUpperCase\(\)", code)
                v   = var.group(1) if var else "variable"
                return f"'{v}' is null at runtime. A method returned null but no null check was performed before calling .toUpperCase()."
        return f"This line causes: {error_msg}"

    def _fix(code: str, error_msg: str) -> str:
        if "signatureexception" in error_msg.lower() or "jwt" in error_msg.lower():
            return (
                "Update JwtTokenValidator to use the RSA public key for RS256 verification:\n"
                "  // Replace:\n"
                "  .setSigningKey(SECRET)  // HS256 string secret\n"
                "  // With:\n"
                "  .setSigningKey(rsaPublicKey)  // RSA PublicKey object for RS256\n\n"
                "Or revert JwtTokenProvider back to HS256 for consistency.\n"
                "Key rule: signing algorithm and verification algorithm MUST match."
            )
        if "nullpointerexception" in error_msg.lower():
            if ".toUpperCase()" in code:
                var = re.search(r"(\w+)\.toUpperCase\(\)", code)
                v   = var.group(1) if var else "value"
                return (f"Add null check before calling .toUpperCase():\n"
                        f"  String safe = ({v} != null) ? {v}.toUpperCase() : \"UNKNOWN\";\n"
                        f"Or: Objects.requireNonNullElse({v}, \"\").toUpperCase()")
        return "Add appropriate null check or ensure algorithm consistency across signing and verification."

    analysis = {
        "error_summary":   {"class": error_class, "method": error_method, "line": error_line, "message": error_message},
        "method_found":    bool(method_lines),
        "method_code":     [(ln, c) for ln, c in method_lines[:25]],
        "null_risk_lines": null_risks[:8],
        "exact_bug_line":  None,
        "fix_suggestion":  None,
    }

    if target_line:
        analysis["exact_bug_line"] = {
            "line_number": target_line,
            "code":        target_code,
            "explanation": _explain(target_code, error_message),
            "fix":         _fix(target_code, error_message),
        }
        analysis["fix_suggestion"] = _fix(target_code, error_message)
    elif null_risks:
        top = null_risks[0]
        analysis["exact_bug_line"] = {
            "line_number": top["line"],
            "code":        top["code"],
            "explanation": top["reason"],
            "fix":         _fix(top["code"], error_message),
        }

    _trace("analyze_code_for_bug", f"{error_class}.{error_method}() line {error_line} → {'bug found' if analysis['exact_bug_line'] else 'risk lines identified'}")
    return json.dumps(analysis, indent=2)


# ═══════════════════════════════════════════════════════════════════
# TOOL 13 — Create GitHub Issue (NEW)
# ═══════════════════════════════════════════════════════════════════

@tool(args_schema=CreateIssueInput)
def create_github_issue(repo: str, title: str, body: str, labels: str = "bug,incident") -> str:
    """
    Create a real GitHub issue with the incident root cause report.

    Call this as your FINAL action after generate_runbook.
    The issue appears immediately in the repo's Issues tab.

    Write a detailed body with ALL sections:
      ## Root Cause       — exact class, method, line, and why it fails
      ## The Bug          — exact broken code snippet
      ## The Fix          — corrected code snippet
      ## What Changed     — commit SHA, author, what lines were added
      ## Affected Endpoint — HTTP method + path
      ## Prevention       — unit test, code review rule, pattern to avoid

    Labels: 'bug,incident,p0' for P0, 'bug,incident,p1' for P1
    """
    label_list = [l.strip() for l in labels.split(",") if l.strip()]

    # Ensure labels exist
    token = os.environ.get("GITHUB_TOKEN", "")
    h     = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"}
    colors = {"bug": "d73a4a", "incident": "e4e669", "p0": "b60205", "p1": "ff7619", "postmortem": "0075ca"}
    for label in label_list:
        try:
            requests.post(f"https://api.github.com/repos/{repo}/labels", headers=h,
                          json={"name": label, "color": colors.get(label, "ededed")}, timeout=5)
        except Exception:
            pass

    try:
        resp  = _gh_post(f"/repos/{repo}/issues", {"title": title, "body": body, "labels": label_list})
        issue = resp.json()
        _trace("create_github_issue", f"Issue #{issue['number']} created → {issue['html_url']}")
        return json.dumps({
            "success":       True,
            "issue_number":  issue["number"],
            "url":           issue["html_url"],
            "message":       f"Issue #{issue['number']} created: {issue['html_url']}",
        }, indent=2)
    except Exception as e:
        _trace("create_github_issue", f"FAILED: {e}")
        return json.dumps({"success": False, "error": str(e)})


# ═══════════════════════════════════════════════════════════════════
# TOOL 14 — Generate Runbook (Final Action)
# ═══════════════════════════════════════════════════════════════════

@tool(args_schema=RunbookInput)
def generate_runbook(
    incident_title:   str,
    incident_type:    str,
    root_cause:       str,
    affected_service: str,
    timeline:         str,
    fix_applied:      str,
    prevention:       str,
    ruled_out:        str = "",
) -> str:
    """
    Generate and save a structured post-incident runbook as Markdown.

    Call this after you have confirmed root cause. Then call create_github_issue
    to open a real GitHub issue with the findings.

    If root cause is inconclusive after exhausting investigation paths,
    still call this with incident_type=UNKNOWN and document what was ruled out.
    """
    os.makedirs("incidents", exist_ok=True)

    slug     = re.sub(r"[^a-z0-9]+", "-", incident_title.lower())[:45]
    filename = f"incidents/{datetime.now().strftime('%Y-%m-%d-%H%M')}-{slug}.md"

    ruled_out_section = f"\n## What Was Ruled Out\n{ruled_out}\n" if ruled_out.strip() else ""

    content = f"""# Incident Runbook: {incident_title}

**Date:** {datetime.now().strftime("%B %d, %Y at %H:%M UTC")}
**Type:** {incident_type}
**Generated by:** Postmortem — AI DevOps Agent (Gemini + LangGraph)

---

## Root Cause
{root_cause}

## Affected Service
{affected_service}

## Timeline
{timeline}

## Fix Applied
{fix_applied}

## Prevention
{prevention}
{ruled_out_section}
---
*Auto-generated — verify all details before publishing to your team*
"""

    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)

    _trace("generate_runbook", f"saved → {filename}")
    return f"Runbook saved → {filename}\n\n{content}"


# ═══════════════════════════════════════════════════════════════════
# ALL TOOLS — imported by graph.py
# ═══════════════════════════════════════════════════════════════════

ALL_TOOLS = [
    parse_logs,
    get_error_spike_time,
    analyze_error_patterns,
    get_recent_commits,
    correlate_deploy_with_spike,
    get_pr_diff,
    get_git_blame,
    get_open_issues,
    # NEW — code intelligence
    search_repo_for_class,
    read_file_from_github,
    get_commit_file_diff,
    analyze_code_for_bug,
    # NEW — documentation
    create_github_issue,
    # Always last
    generate_runbook,
]