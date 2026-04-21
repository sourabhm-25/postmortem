# Postmortem — AI DevOps Incident Response Agent

> **Gemini 2.0 Flash + LangGraph + GitHub API**
>
> When an alert fires, Postmortem investigates your real GitHub repo, reads the actual source code at the broken commit, finds the exact bug line, and opens a GitHub issue with full root cause analysis — in under 2 minutes.

---

## The Problem It Solves

When something breaks at 2am, a junior engineer opens 4 tabs:

| Tab | What they're doing |
|---|---|
| Logs | Searching for the error pattern |
| GitHub | Finding what deployed recently |
| Source code | Reading the file that's crashing |
| Runbook | Hoping someone documented this before |

A senior engineer does the same thing — but in 15 minutes instead of 4 hours, because they know *where to look* and *how to connect signals*.

Postmortem encodes that senior engineer's intuition as a hypothesis-driven agent loop.

---

## What It Does

```
PHASE 1 — Understand the error
  parse_logs              → error type, class, method, count, classification hint
  get_error_spike_time    → exact minute errors crossed threshold (your correlation anchor)
  analyze_error_patterns  → primary vs cascading errors, affected endpoints

PHASE 2 — Find what changed
  get_recent_commits      → real commits from your GitHub repo (last 12h)
  correlate_deploy_with_spike → ranks commits by proximity to spike + file risk score
  get_pr_diff             → full unified diff of the culprit PR
  get_git_blame           → who last touched a specific file:line

PHASE 3 — Find the exact bug line  ← what makes this different
  search_repo_for_class   → find file path from class name in stack trace
  read_file_from_github   → source code AT the culprit commit SHA (not main)
  analyze_code_for_bug    → static analysis: exact line, why it fails, fix suggestion
  get_commit_file_diff    → confirm which lines were added in the bad commit

PHASE 4 — Document
  generate_runbook        → saves structured Markdown runbook to incidents/
  create_github_issue     → opens a real GitHub issue with full root cause report
```

The agent classifies the incident first (deploy regression, infrastructure, configuration, dependency failure, security, or unknown), then picks the appropriate investigation path. It states a hypothesis before every tool call and pivots when data contradicts it.

---

## Demo — What the Agent Finds

Given this incident report:

```
INCIDENT — P0
POST /api/refunds returning HTTP 500 — error rate 100%

2025-04-11 02:18:03 INFO  Deploy a3f9c12 completed — PaymentService v2.4.1
2025-04-11 02:31:04 ERROR NullPointerException: Cannot invoke String.toUpperCase() on null
  at PaymentService.processRefund(PaymentService.java:47)
2025-04-11 02:31:05 ERROR NullPointerException: Cannot invoke String.toUpperCase() on null
...
```

The agent runs 10–12 tool calls and produces:

```
Classification: DEPLOY_REGRESSION (92% confident)

[1]  parse_logs              → 8 errors | NullPointerException at PaymentService.java:47
[2]  get_error_spike_time    → spike at 2025-04-11 02:31
[3]  get_recent_commits      → 3 commits in last 12h
[4]  correlate_deploy_with_spike → PR #312 (score=185, 12.9min before spike) ← culprit
[5]  get_pr_diff(312)        → diff shows customer.getAddress().toUpperCase() added
[6]  search_repo_for_class   → PaymentService → src/services/PaymentService.java
[7]  read_file_from_github   → file at commit a3f9c12 (broken version)
[8]  analyze_code_for_bug    → line 47: address.toUpperCase() — address can be null
[9]  get_commit_file_diff    → ADDED: String address = customer.getAddress();
                               ADDED: logger.info(address.toUpperCase()); ← NPE
[10] generate_runbook        → saved → incidents/2025-04-11-payment-500.md
[11] create_github_issue     → Issue #8 created → github.com/you/repo/issues/8
```

---

## Architecture

```
postmortem/
├── main.py                    Entry point — interactive or demo mode
├── core/
│   └── graph.py               LangGraph StateGraph
│       ├── AgentState         TypedDict — messages, steps_taken, repo
│       ├── call_model         Gemini 2.0 Flash node
│       ├── call_tools         LangGraph ToolNode (auto-dispatch, no if/elif)
│       └── tools_condition    Built-in loop/exit edge
├── tools/
│   └── devops_tools.py        14 tools — all real API calls
│       ├── parse_logs
│       ├── get_error_spike_time
│       ├── analyze_error_patterns
│       ├── get_recent_commits
│       ├── correlate_deploy_with_spike
│       ├── get_pr_diff
│       ├── get_git_blame
│       ├── get_open_issues
│       ├── search_repo_for_class   ← NEW
│       ├── read_file_from_github   ← NEW
│       ├── get_commit_file_diff    ← NEW
│       ├── analyze_code_for_bug    ← NEW
│       ├── create_github_issue     ← NEW
│       └── generate_runbook
├── setup/
│   └── setup_auth_incident.py     Creates a JWT bug repo for testing
├── incidents/                     Runbooks saved here (git-ignored)
└── requirements.txt
```

### Why LangGraph instead of a raw while loop

| Raw loop | LangGraph |
|---|---|
| `while True:` with manual stop check | `tools_condition` built-in edge |
| `if/elif` tool routing | `ToolNode` handles all dispatch |
| No state typing | `AgentState` TypedDict — no silent bugs |
| Not resumable | Checkpointable at any node |
| Hard to extend | Add a node without touching other code |

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/postmortem
cd postmortem
pip install -r requirements.txt
```

### 2. Get API keys

| Key | Where | Notes |
|---|---|---|
| `GEMINI_API_KEY` | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) | Free tier available |
| `GITHUB_TOKEN` | [github.com/settings/tokens](https://github.com/settings/tokens) | Needs `repo` scope (full) |

```bash
export GEMINI_API_KEY=AIza...
export GITHUB_TOKEN=ghp_...
```

---

## Usage

### Interactive mode — pick from your real GitHub repos and issues

```bash
python main.py
```

1. Lists all your GitHub repos with open issues
2. You pick the repo
3. Lists all open issues — you pick the one to investigate
4. Agent auto-ranks merged PRs by relevance
5. You confirm the culprit PR (or let the agent decide)
6. Investigation runs — GitHub issue created at the end

### Demo mode — instant start with pre-built incidents

```bash
# NullPointerException in a payment service (DEPLOY_REGRESSION)
python main.py --type deploy

# JWT algorithm mismatch — all users locked out (SECURITY)
python main.py --type auth

# OutOfMemoryError — pod restarting every 15 min (INFRASTRUCTURE)
python main.py --type infra

# Missing environment variable — service won't start (CONFIGURATION)
python main.py --type config
```

### Target a specific repo directly

```bash
python main.py --repo sourabhm-25/payment-service
```

### Pass a custom incident description

```bash
python main.py "POST /api/orders returning 503 since 3am, connection refused to orders-db"
```

---

## Creating a Test Incident (JWT auth bug)

The `setup/setup_auth_incident.py` script creates a real GitHub repo with a planted bug:

**The bug:** A security hardening PR upgraded JWT signing from HS256 to RS256 in `JwtTokenProvider` but forgot to update `JwtTokenValidator` — which still tries to verify RS256-signed tokens with an HS256 secret. Every token validation throws `SignatureException`. All users are locked out.

```bash
export GITHUB_TOKEN=ghp_...
python setup/setup_auth_incident.py
```

This creates:
- Repo: `YOUR_USERNAME/auth-service`
- A merged PR: `security: harden JWT signing — upgrade HS256 to RS256`
- An open incident issue with logs

Then run the agent against it:

```bash
python main.py --type auth
# or interactively:
python main.py --repo YOUR_USERNAME/auth-service
```

The agent should find:
- `JwtTokenValidator.java` line 21: `.setSigningKey(SECRET)` using HS256 secret to verify RS256 token
- Fix: update the validator to use the RSA public key, or revert provider to HS256

---

## Incident Types Supported

| Type | Example | Agent path |
|---|---|---|
| `DEPLOY_REGRESSION` | NPE after a deploy, wrong logic | logs → spike → commits → correlate → diff → **read code** → issue |
| `INFRASTRUCTURE` | OOM, CPU saturation, disk full | logs → patterns → commits (infra files) → runbook → issue |
| `CONFIGURATION` | Missing env var, wrong feature flag | logs → commits (config files) → diff → runbook → issue |
| `DEPENDENCY_FAILURE` | DB down, upstream 503 | logs → patterns → open issues → runbook → issue |
| `SECURITY` | JWT mismatch, auth failures, 401 spike | logs → spike → commits → diff → **read code** → issue |
| `UNKNOWN` | Mixed signals | exhausts all paths, documents what was ruled out |

---

## The `correlate_deploy_with_spike` Scoring Algorithm

This is the core of what makes the agent work. It encodes a senior engineer's intuition as a scoring function:

```python
score = 100.0 (base)

# Time proximity to spike (sweet spot: 5–60 min before)
if 5 <= delta_min <= 60:    score += 60   # high confidence
elif delta_min < 5:         score += 20   # too close — may not have propagated
elif 60 < delta_min <= 120: score += 20   # possible
elif delta_min > 120:       score -= 20   # probably not this one

# High-risk files changed
if any(keyword in filename for keyword in [
    "service", "controller", "api", "auth", "payment",
    "jwt", "token", "handler", "middleware", "config"...
]):
    score += 25
```

A deploy regression typically scores 160–185. Infrastructure issues score low (no recent commits = no culprit found = agent pivots to infra path).

---

## The GitHub Issue It Creates

Every investigation ends with a real GitHub issue:

```markdown
## Root Cause
NullPointerException at PaymentService.java:47 —
address.toUpperCase() called on null return value of customer.getAddress()

## The Bug
```java
// line 47 — broken
String address = customer.getAddress();
logger.info("Refund for: " + address.toUpperCase()); // NPE when address is null
```

## The Fix
```java
String address = customer.getAddress();
String safeAddress = (address != null) ? address.toUpperCase() : "UNKNOWN";
logger.info("Refund for: " + safeAddress);
```

## What Changed (culprit commit)
Commit a3f9c12 by rahul@company.com — "fix: update refund calculation logic"
Merged via PR #312 at 02:18 AM (13 minutes before error spike)

## Affected Endpoint
POST /api/refunds

## Prevention
- Add unit test for processRefund() with null customer address
- Code review rule: always null-check before chaining method calls
- Consider using Objects.requireNonNullElse() as a pattern
```

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GEMINI_API_KEY` | Yes | Gemini API key from Google AI Studio |
| `GITHUB_TOKEN` | Yes | GitHub personal access token with `repo` scope |

---

## Inspired by

[AWS DevOps Agent](https://aws.amazon.com/devops-agent/) — built for enterprise at $50M scale.
Postmortem solves the same core pain for teams of 2–20 with zero cloud account setup required.

---

Built with [LangGraph](https://langchain-ai.github.io/langgraph/) + [Gemini 2.0 Flash](https://ai.google.dev/gemini-api/docs) + [GitHub REST API](https://docs.github.com/en/rest).