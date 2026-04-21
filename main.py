"""
main.py
=======
Postmortem — Interactive AI Incident Response Agent
Gemini 2.0 Flash + LangGraph

Usage:
  export GEMINI_API_KEY=AIza...
  export GITHUB_TOKEN=ghp_...
  python main.py                    # interactive: pick repo + issue from GitHub
  python main.py --type deploy      # run deploy regression demo directly
  python main.py --type auth        # run JWT auth incident demo directly
  python main.py --type infra       # run OOM infrastructure demo directly
  python main.py --type config      # run config/env var demo directly
  python main.py --repo owner/repo  # interactive but pre-select a repo

What happens (interactive mode):
  1. Fetches all your GitHub repos and lets you pick one
  2. Fetches all open issues from that repo and lets you pick one
  3. Automatically finds the most relevant merged PR for that issue
  4. Runs the full investigation — no hardcoded repo names or PR numbers
"""

import os
import sys
import re
import requests
from datetime import datetime, timezone

# ── Validate env ──────────────────────────────────────────────────────────────
missing = []
if not os.environ.get("GEMINI_API_KEY"):
    missing.append("GEMINI_API_KEY")
if not os.environ.get("GITHUB_TOKEN"):
    missing.append("GITHUB_TOKEN")

if missing:
    print("\n  Error: missing required environment variables:")
    for m in missing:
        print(f"    export {m}=...")
    print("\n  Get them:")
    print("    GEMINI_API_KEY  → https://aistudio.google.com/apikey  (free)")
    print("    GITHUB_TOKEN    → https://github.com/settings/tokens  (repo scope)")
    sys.exit(1)

from core.graph import investigate

# ── GitHub helper ─────────────────────────────────────────────────────────────

HEADERS = {
    "Authorization": f"token {os.environ['GITHUB_TOKEN']}",
    "Accept":        "application/vnd.github.v3+json",
}

def gh_get(path: str, params: dict = None):
    resp = requests.get(
        f"https://api.github.com{path}",
        headers=HEADERS,
        params=params or {},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()


# ── Display helpers ───────────────────────────────────────────────────────────

def divider(char="─", width=62):
    print(f"  {char * width}")

def header(title: str):
    print(f"\n  {'=' * 62}")
    print(f"  {title}")
    print(f"  {'=' * 62}")

def pick(prompt: str, options: list, descriptions: list = None) -> int:
    print()
    for i, opt in enumerate(options):
        idx  = f"  [{i + 1}]"
        desc = f"   {descriptions[i]}" if descriptions and i < len(descriptions) and descriptions[i] else ""
        print(f"{idx:<7} {opt:<45}{desc}")
    print()
    while True:
        try:
            raw = input(f"  {prompt} (1-{len(options)}): ").strip()
            if raw.isdigit() and 1 <= int(raw) <= len(options):
                return int(raw) - 1
            print(f"  Please enter a number between 1 and {len(options)}.")
        except (KeyboardInterrupt, EOFError):
            print("\n  Aborted.")
            sys.exit(0)


# ── Step 1: Repo selection ────────────────────────────────────────────────────

def select_repo() -> str:
    header("POSTMORTEM  |  Gemini 2.0 Flash + LangGraph")
    print("\n  Fetching your GitHub repositories...")

    repos = gh_get("/user/repos", params={
        "type": "all", "sort": "updated", "direction": "desc", "per_page": 100,
    })

    if not repos:
        print("  No repositories found.")
        sys.exit(1)

    repos_with_issues = [r for r in repos if r.get("open_issues_count", 0) > 0]
    display_repos     = repos_with_issues if repos_with_issues else repos

    names = [r["full_name"] for r in display_repos]
    descs = []
    for r in display_repos:
        parts = []
        if r.get("open_issues_count"): parts.append(f"{r['open_issues_count']} issues")
        if r.get("language"):          parts.append(r["language"])
        updated = r.get("updated_at", "")[:10]
        if updated:                    parts.append(f"updated {updated}")
        descs.append("  |  ".join(parts))

    divider()
    print(f"  SELECT REPOSITORY  ({len(display_repos)} repos with open issues)")
    divider()

    idx      = pick("Select repo", names, descs)
    selected = display_repos[idx]["full_name"]
    print(f"\n  Selected: {selected}")
    return selected


# ── Step 2: Issue selection ───────────────────────────────────────────────────

def select_issue(repo: str) -> dict:
    print(f"\n  Fetching open issues for {repo}...")

    issues = gh_get(f"/repos/{repo}/issues", params={
        "state": "open", "sort": "created", "direction": "desc", "per_page": 30,
    })
    issues = [i for i in issues if "pull_request" not in i]

    if not issues:
        print(f"\n  No open issues found in {repo}.")
        print("  Run setup/setup_payment_incident.py or setup/setup_auth_incident.py first.")
        sys.exit(1)

    titles = [f"#{i['number']}  {i['title'][:50]}" for i in issues]
    descs  = [i.get("created_at", "")[:10] for i in issues]

    divider()
    print(f"  SELECT ISSUE  ({len(issues)} open issues in {repo})")
    divider()

    idx      = pick("Select issue to investigate", titles, descs)
    selected = issues[idx]
    print(f"\n  Selected: #{selected['number']} — {selected['title']}")
    return selected


# ── Step 3: Auto-detect relevant PRs ─────────────────────────────────────────

def find_relevant_prs(repo: str, issue: dict) -> list:
    print(f"\n  Scanning merged PRs to find the culprit...")

    issue_number  = issue["number"]
    issue_title   = issue["title"].lower()
    issue_created = issue["created_at"]

    all_prs = gh_get(f"/repos/{repo}/pulls", params={
        "state": "closed", "sort": "updated", "direction": "desc", "per_page": 50,
    })
    merged_prs = [pr for pr in all_prs if pr.get("merged_at")]

    if not merged_prs:
        print("  No merged PRs found.")
        return []

    STOP_WORDS = {
        "with", "from", "that", "this", "have", "been", "error",
        "issue", "problem", "after", "when", "returning", "http",
        "service", "endpoint", "request", "response",
    }

    scored = []
    for pr in merged_prs:
        score   = 0
        reasons = []
        pr_body  = (pr.get("body") or "").lower()
        pr_title = (pr.get("title") or "").lower()
        merged_at = pr.get("merged_at", "")

        if f"#{issue_number}" in pr_body or f"#{issue_number}" in pr_title:
            score += 100
            reasons.append(f"directly references issue #{issue_number}")

        fix_pattern = rf"(?:closes|fixes|resolves|fix|close|resolve)\s*#?{issue_number}\b"
        if re.search(fix_pattern, pr_body) or re.search(fix_pattern, pr_title):
            score += 80
            reasons.append("PR says it fixes this issue")

        try:
            merged_dt = datetime.fromisoformat(merged_at.replace("Z", "+00:00"))
            issue_dt  = datetime.fromisoformat(issue_created.replace("Z", "+00:00"))
            delta_hrs = (issue_dt - merged_dt).total_seconds() / 3600
            if 0 <= delta_hrs <= 6:
                score += 60
                reasons.append(f"merged {delta_hrs:.1f}h before incident reported")
            elif 0 <= delta_hrs <= 24:
                score += 30
                reasons.append(f"merged {delta_hrs:.1f}h before incident reported")
        except Exception:
            pass

        issue_kw = set(re.findall(r"[a-z]{4,}", issue_title)) - STOP_WORDS
        pr_kw    = set(re.findall(r"[a-z]{4,}", pr_title))    - STOP_WORDS
        overlap  = issue_kw & pr_kw
        if overlap:
            score += len(overlap) * 10
            reasons.append(f"keyword match: {', '.join(list(overlap)[:3])}")

        if score > 0:
            scored.append({
                "number":    pr["number"],
                "title":     pr["title"],
                "merged_at": merged_at,
                "score":     score,
                "reasons":   reasons,
                "url":       pr["html_url"],
            })

    scored.sort(key=lambda x: x["score"], reverse=True)

    found = len(scored)
    top   = scored[0] if scored else None
    if top:
        print(f"  Found {found} candidate PR(s). Top match: PR #{top['number']} (score={top['score']})")
    else:
        print("  No matching PRs found — agent will infer from logs.")

    return scored[:5]


# ── Step 4: Confirm PR selection ──────────────────────────────────────────────

def confirm_pr(repo: str, issue: dict, candidates: list) -> int | None:
    divider()
    print("  CANDIDATE PULL REQUESTS  (ranked by relevance to the issue)")
    divider()

    if not candidates:
        print("\n  No matching PRs found. Agent will investigate from logs only.\n")
        return None

    labels = []
    descs  = []
    for c in candidates:
        labels.append(f"PR #{c['number']}  {c['title'][:42]}")
        reason_str = " | ".join(c["reasons"][:2])
        descs.append(f"score={c['score']}  —  {reason_str}")

    labels.append("Let agent decide automatically")
    descs.append("Agent will correlate commits with error spike time")

    labels.append("None of the above — investigate from logs only")
    descs.append("")

    idx = pick("Confirm the culprit PR", labels, descs)

    if idx == len(candidates):      # "Let agent decide"
        return None
    if idx == len(candidates) + 1:  # "None of the above"
        return None

    chosen = candidates[idx]["number"]
    print(f"\n  Confirmed: PR #{chosen} — {candidates[idx]['title']}")
    return chosen


# ── Step 5: Build incident text ───────────────────────────────────────────────

def build_incident(repo: str, issue: dict, pr_number: int | None) -> str:
    title = issue["title"]
    body  = (issue.get("body") or "No details provided.").strip()
    pr_hint = ""
    if pr_number:
        pr_hint = (
            f"\n\nSUSPECTED CULPRIT: PR #{pr_number} — "
            f"investigate its diff to confirm root cause.\n"
        )
    return f"""
INCIDENT REPORT  (GitHub Issue #{issue['number']})
===================================================
Repository : {repo}
Issue      : {title}
{pr_hint}
{body}
"""


# ── Demo incidents ────────────────────────────────────────────────────────────

DEMO_DEPLOY = """
INCIDENT — P0 (Critical)
Reported: 02:35 AM | Repo: sourabhm-25/payment-service

Symptoms:
  - POST /api/refunds returning HTTP 500 errors
  - Error rate: 100% of refund requests failing
  - NullPointerException in logs, stack trace at PaymentService.java:47
  - Errors started at 02:31 AM — 13 minutes after a deploy

Logs:
2025-04-11 02:18:03 INFO  Deploy a3f9c12 completed — PaymentService v2.4.1
2025-04-11 02:29:12 INFO  POST /api/refunds 200 88ms
2025-04-11 02:31:04 ERROR NullPointerException: Cannot invoke String.toUpperCase() on null
  at PaymentService.processRefund(PaymentService.java:47)
  at RefundController.handle(RefundController.java:23)
2025-04-11 02:31:05 ERROR NullPointerException: Cannot invoke String.toUpperCase() on null
2025-04-11 02:31:06 ERROR NullPointerException: Cannot invoke String.toUpperCase() on null
2025-04-11 02:31:09 ERROR NullPointerException: Cannot invoke String.toUpperCase() on null
2025-04-11 02:31:14 WARN  CPU spike: 94%
2025-04-11 02:31:17 ERROR Database connection pool exhausted (max=20, active=20)
2025-04-11 02:31:20 ERROR HTTP 500 POST /api/refunds — NullPointerException
2025-04-11 02:32:01 ERROR HTTP 500 POST /api/refunds — NullPointerException

Investigation needed:
  - Find the exact line that caused the NullPointerException
  - Read the actual source file from GitHub at the broken commit
  - Identify the fix
  - Create a GitHub issue with full root cause analysis
"""

DEMO_AUTH = """
INCIDENT — P0 (Critical)
Reported: 16:45 PM | Repo: sourabhm-25/auth-service

Symptoms:
  - All authenticated endpoints returning 401 Unauthorized
  - New logins work (tokens issued fine) but fail immediately on first use
  - Error: io.jsonwebtoken.SignatureException: JWT signature does not match
  - Started at 16:38 PM — 6 minutes after a "security hardening" PR was merged
  - Every user effectively logged out simultaneously

Logs:
2025-04-13 16:32:01 INFO  Deploy completed — AuthService v3.1.0
2025-04-13 16:32:18 INFO  POST /auth/login          200  44ms
2025-04-13 16:32:19 INFO  GET  /api/dashboard       200  88ms
2025-04-13 16:38:02 ERROR io.jsonwebtoken.SignatureException: JWT signature does not match locally computed signature
  at io.jsonwebtoken.impl.DefaultJwtParser.parse(DefaultJwtParser.java:354)
  at com.example.auth.security.JwtTokenValidator.validateAndGetUserId(JwtTokenValidator.java:21)
  at com.example.auth.security.JwtAuthFilter.doFilterInternal(JwtAuthFilter.java:28)
2025-04-13 16:38:03 ERROR io.jsonwebtoken.SignatureException: JWT signature does not match locally computed signature
2025-04-13 16:38:05 WARN  401 rate crossed threshold: 45% of requests
2025-04-13 16:38:07 ERROR io.jsonwebtoken.SignatureException: JWT signature does not match locally computed signature
2025-04-13 16:38:11 WARN  401 rate crossed threshold: 89% of requests
2025-04-13 16:38:14 ERROR HTTP 401 GET /api/dashboard — Unauthorized
2025-04-13 16:38:15 ERROR HTTP 401 GET /api/orders — Unauthorized
2025-04-13 16:38:16 ERROR HTTP 401 GET /api/profile — Unauthorized
2025-04-13 16:39:45 ERROR HTTP 401 GET /api/orders — Unauthorized

Investigation needed:
  - Find the exact mismatch in JWT signing/verification code
  - Read JwtTokenValidator.java from GitHub at the broken commit
  - Identify which line uses the wrong algorithm/key
  - Create a GitHub issue with full root cause analysis
"""

DEMO_INFRA = """
INCIDENT — P2 (High)
Reported: 11:45 AM | Repo: sourabhm-25/payment-service

Symptoms:
  - payment-service pod restarting every 15 minutes with OOMKilled
  - Memory usage climbs from 2GB to 14GB over ~45 minutes then crashes
  - No new deploys today — last deploy was 3 days ago
  - Error rate is low but latency is high (p99 > 8 seconds)
  - Only happening on 2 of 6 pods

Logs:
2025-04-13 11:00:12 INFO  Service started — heap=2048MB
2025-04-13 11:15:33 WARN  Memory usage: 5.2GB / 16GB
2025-04-13 11:28:44 WARN  Memory usage: 9.8GB / 16GB — approaching limit
2025-04-13 11:38:02 WARN  GC overhead limit exceeded — collection taking >98% of time
2025-04-13 11:41:09 ERROR OutOfMemoryError: Java heap space
  at java.util.Arrays.copyOf(Arrays.java:3236)
  at OrderCacheService.loadAllOrders(OrderCacheService.java:89)
2025-04-13 11:41:10 ERROR OutOfMemoryError: Java heap space
2025-04-13 11:41:11 FATAL Pod restarting — OOMKilled
"""

DEMO_CONFIG = """
INCIDENT — P2 (High)
Reported: 09:15 AM | Repo: sourabhm-25/payment-service

Symptoms:
  - All payment-service instances failing to start after this morning's deployment
  - Error: "Missing required environment variable: STRIPE_SECRET_KEY"
  - Previously working — last night's version was fine
  - Rollback attempted but same error persists (suggesting config issue not code)

Logs:
2025-04-13 09:00:01 INFO  Starting PaymentService v2.5.0
2025-04-13 09:00:02 INFO  Loading configuration from environment
2025-04-13 09:00:02 ERROR Missing required environment variable: STRIPE_SECRET_KEY
2025-04-13 09:00:02 FATAL Service failed to start — configuration error
2025-04-13 09:00:15 INFO  Starting PaymentService v2.5.0  [restart #1]
2025-04-13 09:00:16 ERROR Missing required environment variable: STRIPE_SECRET_KEY
2025-04-13 09:00:16 FATAL Service failed to start — configuration error
"""

DEMOS = {
    "deploy": (DEMO_DEPLOY,  "sourabhm-25/payment-service"),
    "auth":   (DEMO_AUTH,    "sourabhm-25/auth-service"),
    "infra":  (DEMO_INFRA,   "sourabhm-25/payment-service"),
    "config": (DEMO_CONFIG,  "sourabhm-25/payment-service"),
}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = sys.argv[1:]

    # ── Direct demo mode: python main.py --type auth ──────────────────────────
    if "--type" in args:
        idx = args.index("--type")
        if idx + 1 < len(args):
            demo_key = args[idx + 1]
            if demo_key not in DEMOS:
                print(f"  Unknown type '{demo_key}'. Available: {', '.join(DEMOS)}")
                sys.exit(1)
            incident, repo = DEMOS[demo_key]
            print(f"\n  Running demo: {demo_key.upper()}  |  repo: {repo}")
            result = investigate(incident, repo=repo)
            print(f"\n  Done — {result['steps']} LLM calls in {result['elapsed']}s")
            return

    # ── Pre-selected repo: python main.py --repo owner/repo ──────────────────
    pre_repo = None
    if "--repo" in args:
        idx = args.index("--repo")
        if idx + 1 < len(args):
            pre_repo = args[idx + 1]

    # ── Interactive mode ──────────────────────────────────────────────────────
    repo  = pre_repo or select_repo()
    issue = select_issue(repo)

    candidates = find_relevant_prs(repo, issue)
    pr_number  = confirm_pr(repo, issue, candidates)
    incident   = build_incident(repo, issue, pr_number)

    print(f"\n  {'=' * 62}")
    print(f"  READY TO INVESTIGATE")
    print(f"  {'=' * 62}")
    print(f"  Repo  : {repo}")
    print(f"  Issue : #{issue['number']} — {issue['title'][:52]}")
    print(f"  PR    : {'#' + str(pr_number) + ' (confirmed)' if pr_number else 'agent will auto-detect'}")
    print(f"  {'=' * 62}")

    try:
        input("\n  Press Enter to start the investigation  (Ctrl+C to cancel)...")
    except (KeyboardInterrupt, EOFError):
        print("\n  Cancelled.")
        sys.exit(0)

    result = investigate(incident, repo=repo)
    print(f"\n  Done — {result['steps']} LLM calls in {result['elapsed']}s")


if __name__ == "__main__":
    main()