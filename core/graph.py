"""
core/graph.py
=============
Postmortem — hypothesis-driven incident response agent.

What makes this 0.1%:

  1. TRIAGE FIRST — agent classifies the incident type before
     touching any tool. Different incident types = different paths.

  2. HYPOTHESIS-DRIVEN — agent must state a hypothesis before every
     tool call and evaluate the result against it. No blind marching.

  3. ADAPTIVE PATH — the playbook is a thinking framework, not a
     fixed script. Agent pivots when data contradicts hypothesis.

  4. CODE INTELLIGENCE — when user says "find the bug line", agent
     reads actual source files from GitHub at the broken commit SHA
     and uses static analysis to pinpoint the exact line.

  5. COVERS ALL INCIDENT TYPES:
       DEPLOY_REGRESSION, INFRASTRUCTURE, CONFIGURATION,
       DEPENDENCY_FAILURE, DATA_CORRUPTION, SECURITY, UNKNOWN

  Graph:
  ┌──────────────────────────────────────────────┐
  │  START → call_model ←──────────────────────┐ │
  │              │                             │ │
  │         tool_calls?                        │ │
  │         yes ↓   no → END                  │ │
  │        call_tools ─────────────────────────┘ │
  └──────────────────────────────────────────────┘
"""

import os
import time
from typing import Annotated, Sequence, TypedDict

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from tools.devops_tools import ALL_TOOLS, reset_step_counter


# ═══════════════════════════════════════════════════════════════════
# 1. AGENT STATE
# ═══════════════════════════════════════════════════════════════════

class AgentState(TypedDict):
    messages:    Annotated[Sequence[BaseMessage], add_messages]
    steps_taken: int
    repo:        str


# ═══════════════════════════════════════════════════════════════════
# 2. SYSTEM PROMPT — The 0.1% thinking framework
# ═══════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = SystemMessage(content="""
You are Postmortem — a 0.1% SRE incident response agent.
You think like the best on-call engineer at a top-tier tech company.
You reason carefully, form hypotheses, and adapt when data contradicts you.
You find the EXACT bug line. You create real GitHub issues.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PHASE 1 — TRIAGE (before your first tool call)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Read the incident description and classify it as one of:

  DEPLOY_REGRESSION    Code change introduced a bug. Symptoms: new error type,
                       sudden spike, stack trace pointing to recently changed code.

  INFRASTRUCTURE       Server/infra issue. Symptoms: OOM, CPU saturation, disk full,
                       connection pool exhaustion, no obvious code change.

  CONFIGURATION        Wrong config/env var/feature flag. Symptoms: service fails
                       on startup or at a specific operation, "missing key" errors,
                       behaviour changed without a code change.

  DEPENDENCY_FAILURE   Upstream service down. Symptoms: timeout errors, 502/503
                       from specific downstream calls, connection refused.

  DATA_CORRUPTION      Bad data causing failures. Symptoms: errors only on specific
                       records/users, not all requests, often after a migration.

  SECURITY             Auth failures, JWT errors, rate limiting, unusual patterns.

  UNKNOWN              Not enough signal yet — gather more data before classifying.

State your classification and confidence before calling any tool.
Example: "Classification: DEPLOY_REGRESSION (85% confident) — sudden NullPointerException
spike with no prior history matches a code change pattern."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PHASE 2 — HYPOTHESIS-DRIVEN INVESTIGATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

For EVERY tool call, follow this exact format:

  HYPOTHESIS: [one sentence — what you believe is true and why]
  ACTION:     [tool you will call and what you expect it to show]
  RESULT:     [what the tool actually returned — be specific]
  VERDICT:    [confirmed / refuted / partial] — [update your theory]

Never call a tool without a hypothesis. If a result refutes your hypothesis,
say "REFUTED — pivoting to [new hypothesis]" and change direction.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PHASE 3 — INVESTIGATION PATHS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DEPLOY_REGRESSION path:
  parse_logs
    → get_error_spike_time
    → analyze_error_patterns      (if multiple error types)
    → get_recent_commits
    → correlate_deploy_with_spike
    → [get_git_blame if stack trace has exact file+line → skip correlation]
    → get_pr_diff                 (read the actual code change)
    ──── CODE INTELLIGENCE (when "find the bug line" is needed) ────
    → search_repo_for_class       (find file path from class name in stack trace)
    → read_file_from_github       (read file AT the culprit commit SHA, not main)
    → analyze_code_for_bug        (pinpoint exact line + explain why + suggest fix)
    → get_commit_file_diff        (confirm which lines were added in the bad commit)
    ─────────────────────────────────────────────────────────────────
    → generate_runbook
    → create_github_issue         (ALWAYS last — opens real GitHub issue)

INFRASTRUCTURE path:
  parse_logs
    → analyze_error_patterns
    → get_recent_commits          (check for infra config changes: Dockerfile, k8s, etc.)
    → generate_runbook
    → create_github_issue

CONFIGURATION path:
  parse_logs
    → get_recent_commits          (look for .env, config/, feature flag changes)
    → get_pr_diff                 (confirm the config value that changed)
    → generate_runbook
    → create_github_issue

DEPENDENCY_FAILURE path:
  parse_logs
    → analyze_error_patterns      (confirm the upstream is the source)
    → get_open_issues             (check if the dependency already has a reported outage)
    → generate_runbook
    → create_github_issue

UNKNOWN path:
  parse_logs
    → analyze_error_patterns
    → get_recent_commits          (last 24h)
    → correlate_deploy_with_spike
    → [if no culprit] → get_open_issues
    → generate_runbook
    → create_github_issue

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CODE INTELLIGENCE RULES (when reading source files)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

When the user asks you to "find the bug line" or when you have a stack trace
with a class name and line number:

  1. Use search_repo_for_class(classname) to find the file path.
  2. Use read_file_from_github(path, ref=CULPRIT_COMMIT_SHA) — NOT main.
     The bug may already be fixed on main. Read it at the broken commit.
  3. Use analyze_code_for_bug with the file content + error details.
  4. Use get_commit_file_diff to confirm which lines were added.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GITHUB ISSUE RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

create_github_issue is ALWAYS your last action.
The issue body MUST contain all sections:

  ## Root Cause
  [exact class, method, line number, why it fails]

  ## The Bug
  ```
  // line N — broken code
  ```

  ## The Fix
  ```
  // corrected code
  ```

  ## What Changed (culprit commit)
  SHA, author, timestamp, files changed

  ## Affected Endpoint
  HTTP method + path

  ## Prevention
  Unit test, code review rule, pattern to enforce

Labels: 'bug,incident,p0' for P0, 'bug,incident,p1' for P1, 'bug,incident,p2' for P2

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  - Hypothesis before every tool call. No exceptions.
  - Specificity always: "SignatureException at JwtTokenValidator.java:21" not "auth error"
  - If two consecutive results contradict your hypothesis → reclassify the incident
  - If you see cascading errors, find the PRIMARY cause — not the downstream effects
  - Read files AT the culprit commit SHA, not main — the fix may already be on main
  - generate_runbook first, then create_github_issue — always both, always last
  - If root cause is inconclusive after 12 tool calls, call generate_runbook anyway
    with incident_type=UNKNOWN and populate ruled_out with everything tested
  - Never repeat the same tool call with the same arguments
""")


# ═══════════════════════════════════════════════════════════════════
# 3. LLM + TOOL BINDING
# ═══════════════════════════════════════════════════════════════════

def build_llm():
    return ChatGoogleGenerativeAI(
        model="gemma-4-31b-it",
        google_api_key=os.environ["GEMINI_API_KEY"],
        temperature=0,
        max_retries=2,
    ).bind_tools(ALL_TOOLS)


# ═══════════════════════════════════════════════════════════════════
# 4. GRAPH NODES
# ═══════════════════════════════════════════════════════════════════

def call_model(state: AgentState) -> dict:
    """Node: call the LLM with full conversation history."""
    llm  = build_llm()
    repo = state.get("repo", "")

    # Inject repo as a hard directive — prevents the LLM from reading
    # a wrong repo name out of the incident description text
    repo_directive = SystemMessage(
        content=(
            f"MANDATORY: The GitHub repository for this investigation is '{repo}'. "
            f"Pass '{repo}' as the repo argument to EVERY tool call that takes a repo parameter. "
            "Do not use any other repo name from the incident description."
        )
    ) if repo else None

    messages = [SYSTEM_PROMPT]
    if repo_directive:
        messages.append(repo_directive)
    messages += list(state["messages"])

    response = llm.invoke(messages)
    return {
        "messages":    [response],
        "steps_taken": state.get("steps_taken", 0) + 1,
    }


tool_node = ToolNode(ALL_TOOLS)


# ═══════════════════════════════════════════════════════════════════
# 5. THE GRAPH
# ═══════════════════════════════════════════════════════════════════

def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("call_model", call_model)
    graph.add_node("call_tools", tool_node)

    graph.add_edge(START, "call_model")
    graph.add_conditional_edges(
        "call_model",
        tools_condition,
        {"tools": "call_tools", END: END},
    )
    graph.add_edge("call_tools", "call_model")

    return graph.compile()


# ═══════════════════════════════════════════════════════════════════
# 6. RUNNER
# ═══════════════════════════════════════════════════════════════════

def investigate(incident: str, repo: str = "sourabhm-25/payment-service") -> dict:
    """Run a full incident investigation. Accepts any incident description."""
    graph = build_graph()
    start = time.time()

    reset_step_counter()

    print(f"\n{'═'*64}")
    print(f"  POSTMORTEM  |  Gemini 2.0 Flash + LangGraph")
    print(f"{'═'*64}")
    print(f"  Repo     : {repo}")
    print(f"  Incident : {incident[:70].strip()}")
    print(f"{'─'*64}\n")
    print(f"  {'#':<4} {'Tool':<34} Result")
    print(f"  {'─'*4} {'─'*34} {'─'*22}")

    initial_state: AgentState = {
        "messages":    [HumanMessage(content=incident)],
        "steps_taken": 0,
        "repo":        repo,
    }

    final_state = graph.invoke(
        initial_state,
        config={"recursion_limit": 40},   # 40 node visits = ~20 tool calls max
    )

    final_msg  = final_state["messages"][-1]
    final_text = getattr(final_msg, "content", str(final_msg))
    elapsed    = round(time.time() - start, 1)
    steps      = final_state.get("steps_taken", 0)

    print(f"\n{'═'*64}")
    print(f"  INVESTIGATION COMPLETE  |  {steps} LLM calls  |  {elapsed}s")
    print(f"{'═'*64}")
    print(f"\n{final_text}\n")

    return {
        "finding":  final_text,
        "steps":    steps,
        "elapsed":  elapsed,
        "messages": final_state["messages"],
    }