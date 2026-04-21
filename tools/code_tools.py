"""
tools/code_tools.py
====================
Code intelligence tools — the thing that separates Postmortem
from every other DevOps agent.

Most agents stop at "PR #312 was deployed before the spike."
These tools go further:

  read_file_from_github  → fetch the actual source file from GitHub
  analyze_code_for_bug   → read the file + error class/method → find exact bug line
  search_repo_for_class  → find which file contains a class when you don't know the path
  get_commit_file_diff   → get the line-by-line diff for a specific file in a commit

Together they answer: "exactly which line, and exactly why."
"""

import os
import re
import json
import base64
import requests
from langchain_core.tools import tool
from pydantic import BaseModel, Field

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
_H = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json",
}


# ── Input schemas ─────────────────────────────────────────────────

class ReadFileInput(BaseModel):
    repo:   str = Field(description="GitHub repo, e.g. 'sourabhm-25/payment-service'")
    path:   str = Field(description="File path in repo, e.g. 'src/services/PaymentService.java'")
    ref:    str = Field(default="main", description="Branch or commit SHA. Default 'main'.")

class SearchClassInput(BaseModel):
    repo:      str = Field(description="GitHub repo to search in")
    classname: str = Field(description="Class or file name to find, e.g. 'PaymentService'")

class CommitFileInput(BaseModel):
    repo:     str = Field(description="GitHub repo")
    sha:      str = Field(description="Full commit SHA from get_recent_commits")
    filename: str = Field(description="Specific file path to get the diff for")

class BugAnalysisInput(BaseModel):
    error_class:  str = Field(description="Class from the stack trace, e.g. 'PaymentService'")
    error_method: str = Field(description="Method from the stack trace, e.g. 'processRefund'")
    error_line:   int = Field(default=0, description="Line number from stack trace if known, e.g. 47")
    error_message:str = Field(description="The full error message, e.g. 'NullPointerException: Cannot invoke String.toUpperCase() on null'")
    file_content: str = Field(description="Full source file content from read_file_from_github")


# ── Tools ─────────────────────────────────────────────────────────

@tool(args_schema=ReadFileInput)
def read_file_from_github(repo: str, path: str, ref: str = "main") -> str:
    """
    Read the actual source code of a file from GitHub.

    USE THIS when you know which class caused the error (from the stack trace)
    and want to read the code to find the exact bug.

    After calling search_repo_for_class to find the file path, call this
    to get the full source code. Then call analyze_code_for_bug.
    Returns the raw file content with line numbers prepended.
    """
    print(f"  [code] read_file_from_github — {repo}/{path} @ {ref}")
    try:
        resp = requests.get(
            f"https://api.github.com/repos/{repo}/contents/{path}",
            headers=_H,
            params={"ref": ref},
            timeout=15,
        )
        if resp.status_code == 404:
            return json.dumps({"error": f"File '{path}' not found in {repo}@{ref}. Try search_repo_for_class to find the correct path."})
        resp.raise_for_status()

        data    = resp.json()
        content = base64.b64decode(data["content"]).decode("utf-8", errors="replace")

        # Add line numbers — critical for bug identification
        lines = content.split("\n")
        numbered = "\n".join(f"{i+1:4d}  {line}" for i, line in enumerate(lines))

        return json.dumps({
            "path":        path,
            "ref":         ref,
            "total_lines": len(lines),
            "content":     numbered[:8000],   # trim if huge file
            "truncated":   len(numbered) > 8000,
        }, indent=2)

    except requests.RequestException as e:
        return json.dumps({"error": str(e)})


@tool(args_schema=SearchClassInput)
def search_repo_for_class(repo: str, classname: str) -> str:
    """
    Search the GitHub repo for a file matching a class name.

    USE THIS when you see 'PaymentService' in the stack trace but don't
    know the exact file path. Returns matching file paths so you can
    call read_file_from_github with the correct path.

    Example: stack trace says 'PaymentService.processRefund(PaymentService.java:47)'
    → search_repo_for_class(repo, 'PaymentService') → finds 'src/services/PaymentService.java'
    """
    print(f"  [code] search_repo_for_class — {classname} in {repo}")
    try:
        # GitHub code search API
        resp = requests.get(
            "https://api.github.com/search/code",
            headers={**_H, "Accept": "application/vnd.github.v3+json"},
            params={
                "q":        f"class {classname} repo:{repo}",
                "per_page": 10,
            },
            timeout=15,
        )

        if resp.status_code == 403:
            # Rate limited or no search access — fall back to tree search
            return _tree_search(repo, classname)

        if resp.status_code == 422:
            return _tree_search(repo, classname)

        resp.raise_for_status()
        items = resp.json().get("items", [])

        if not items:
            return _tree_search(repo, classname)

        results = [{"path": i["path"], "url": i["html_url"]} for i in items[:5]]
        return json.dumps({
            "classname": classname,
            "matches":   results,
            "message":   f"Found {len(results)} file(s) containing '{classname}'",
        }, indent=2)

    except requests.RequestException as e:
        return _tree_search(repo, classname)


@tool(args_schema=CommitFileInput)
def get_commit_file_diff(repo: str, sha: str, filename: str) -> str:
    """
    Get the exact line-by-line diff for ONE specific file in a commit.

    USE THIS after correlate_deploy_with_spike identifies the culprit commit.
    Pass the full commit SHA and the filename to see exactly what changed
    in that file — which lines were added (+) and removed (-).

    This is more precise than get_pr_diff when you already know which file
    the error is in.
    """
    print(f"  [code] get_commit_file_diff — {sha[:7]} / {filename}")
    try:
        resp = requests.get(
            f"https://api.github.com/repos/{repo}/commits/{sha}",
            headers=_H,
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()

        for f in data.get("files", []):
            if f["filename"] == filename or filename in f["filename"]:
                patch = f.get("patch", "No patch available (binary or too large)")
                # Annotate the patch with line context
                annotated = _annotate_patch(patch)
                return json.dumps({
                    "filename":  f["filename"],
                    "status":    f["status"],
                    "additions": f["additions"],
                    "deletions": f["deletions"],
                    "patch":     annotated,
                    "message":   f"Lines added (+) are new code. Lines removed (-) are what was there before.",
                }, indent=2)

        # File not in this commit
        all_files = [f["filename"] for f in data.get("files", [])]
        return json.dumps({
            "message":   f"'{filename}' not changed in commit {sha[:7]}",
            "all_files_in_commit": all_files,
        }, indent=2)

    except requests.RequestException as e:
        return json.dumps({"error": str(e)})


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

    USE THIS after read_file_from_github — pass the file content and the
    error details from the stack trace. This tool performs static analysis
    to identify the exact problematic line and explain WHY it fails.

    Returns: exact line number, the code on that line, why it fails,
    and the minimal fix needed.

    This is the tool that answers "which exact line caused the bug"
    without needing to run the code.
    """
    print(f"  [code] analyze_code_for_bug — {error_class}.{error_method}() line {error_line}")

    # Parse the file content (strip line numbers we added)
    lines = []
    raw_content = ""
    try:
        data = json.loads(file_content)
        raw = data.get("content", file_content)
    except Exception:
        raw = file_content

    for line in raw.split("\n"):
        # Strip the line number prefix we added (e.g. "  47  code here")
        m = re.match(r"^\s*(\d+)\s{2}(.*)", line)
        if m:
            lines.append((int(m.group(1)), m.group(2)))
            raw_content += m.group(2) + "\n"
        else:
            raw_content += line + "\n"

    # Find the method
    method_start = None
    method_lines = []
    in_method    = False
    brace_depth  = 0

    for lineno, code in lines:
        if error_method in code and ("void " in code or "public " in code
                                      or "private " in code or "def " in code
                                      or ")" in code):
            method_start = lineno
            in_method    = True
            brace_depth  = 0

        if in_method:
            method_lines.append((lineno, code))
            brace_depth += code.count("{") - code.count("}")
            if method_start and brace_depth <= 0 and lineno > method_start:
                break

    # If we have the exact line from stack trace, use it
    target_line = None
    target_code = ""
    if error_line > 0:
        for lineno, code in lines:
            if lineno == error_line:
                target_line = lineno
                target_code = code.strip()
                break

    # Static analysis: find null-risk patterns
    null_risks = []
    npe_patterns = [
        (r"\.(\w+)\(\)\.(\w+)\(\)",  "Chained call — first call may return null"),
        (r"(\w+)\.toUpperCase\(\)",   "toUpperCase() called — variable may be null"),
        (r"(\w+)\.toLowerCase\(\)",   "toLowerCase() called — variable may be null"),
        (r"(\w+)\.length\(\)",        ".length() called — variable may be null"),
        (r"(\w+)\.get\(",             ".get() result may be null"),
        (r"return .*\.(\w+)\(\);",    "Return value not null-checked"),
    ]

    check_lines = method_lines if method_lines else lines[:50]
    for lineno, code in check_lines:
        for pattern, reason in npe_patterns:
            if re.search(pattern, code):
                null_risks.append({
                    "line":   lineno,
                    "code":   code.strip(),
                    "reason": reason,
                    "is_error_line": lineno == error_line,
                })

    # Build the analysis
    analysis = {
        "error_summary": {
            "class":   error_class,
            "method":  error_method,
            "line":    error_line,
            "message": error_message,
        },
        "exact_bug_line": None,
        "method_found":   bool(method_lines),
        "method_code":    [(ln, c) for ln, c in method_lines[:30]],
        "null_risk_lines": null_risks,
        "fix_suggestion":  None,
    }

    if target_line:
        # Determine the fix based on the error
        fix = _suggest_fix(target_code, error_message)
        analysis["exact_bug_line"] = {
            "line_number": target_line,
            "code":        target_code,
            "explanation": _explain_bug(target_code, error_message),
            "fix":         fix,
        }
        analysis["fix_suggestion"] = fix

    elif null_risks:
        top = null_risks[0]
        analysis["exact_bug_line"] = {
            "line_number": top["line"],
            "code":        top["code"],
            "explanation": top["reason"],
            "fix":         _suggest_fix(top["code"], error_message),
        }

    return json.dumps(analysis, indent=2)


# ── Helpers ───────────────────────────────────────────────────────

def _tree_search(repo: str, classname: str) -> str:
    """Fallback: search the git tree for files matching classname."""
    try:
        resp = requests.get(
            f"https://api.github.com/repos/{repo}/git/trees/HEAD",
            headers=_H,
            params={"recursive": "1"},
            timeout=15,
        )
        if resp.status_code != 200:
            return json.dumps({"message": f"Could not search repo for '{classname}'"})

        tree  = resp.json().get("tree", [])
        lower = classname.lower()
        matches = [
            {"path": item["path"]}
            for item in tree
            if item["type"] == "blob"
            and lower in item["path"].lower()
            and not item["path"].startswith(".")
        ]

        if not matches:
            return json.dumps({
                "message": f"No files found matching '{classname}' in {repo}. "
                           f"The repo may be empty or the class name may differ."
            })

        return json.dumps({
            "classname": classname,
            "matches":   matches[:5],
            "message":   f"Found {len(matches)} file(s) matching '{classname}'",
        }, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)})


def _annotate_patch(patch: str) -> str:
    """Add clear labels to diff lines."""
    if not patch:
        return "No diff available"
    result = []
    for line in patch.split("\n"):
        if line.startswith("+") and not line.startswith("+++"):
            result.append(f"ADDED  : {line[1:]}")
        elif line.startswith("-") and not line.startswith("---"):
            result.append(f"REMOVED: {line[1:]}")
        else:
            result.append(f"CONTEXT: {line}")
    return "\n".join(result)


def _explain_bug(code: str, error_message: str) -> str:
    """Generate a plain-English explanation of why this line fails."""
    code = code.strip()

    if "NullPointerException" in error_message or "null" in error_message.lower():
        if ".toUpperCase()" in code:
            var = re.search(r"(\w+)\.toUpperCase\(\)", code)
            vname = var.group(1) if var else "variable"
            return (f"'{vname}' is null at runtime. Calling .toUpperCase() on a null "
                    f"String throws NullPointerException. The value comes from a method "
                    f"that can return null (e.g. getAddress(), findBy...) but no null "
                    f"check is performed before use.")
        if ".toLowerCase()" in code:
            return "Variable is null. .toLowerCase() cannot be called on null."
        if "." in code:
            return ("A method returns null but the result is used immediately "
                    "without a null check, causing NullPointerException.")

    if "ClassCastException" in error_message:
        return "Object is being cast to an incompatible type."

    if "ArrayIndexOutOfBoundsException" in error_message:
        return "Array index exceeds the array length."

    return f"This line causes: {error_message}"


def _suggest_fix(code: str, error_message: str) -> str:
    """Generate a concrete fix for the buggy line."""
    code = code.strip()

    if "NullPointerException" in error_message or "null" in error_message.lower():
        if ".toUpperCase()" in code:
            var = re.search(r"(\w+)\.toUpperCase\(\)", code)
            vname = var.group(1) if var else "value"
            return (f"Add null check before calling .toUpperCase():\n"
                    f'  if ({vname} != null) {{\n'
                    f'      // original code using {vname}.toUpperCase()\n'
                    f'  }}\n'
                    f"Or use Objects.requireNonNullElse({vname}, \"\").toUpperCase()\n"
                    f"Or use Optional: Optional.ofNullable({vname}).map(String::toUpperCase).orElse(\"\")")

        if "getAddress()" in code or "get" in code.lower():
            return ("Check for null before using the return value:\n"
                    "  String value = obj.getMethod();\n"
                    "  if (value == null) { handle it }\n"
                    "  // now safe to use value")

        return ("Add null check before using this value:\n"
                "  if (value != null) { ... }")

    return "Review the logic and add appropriate null/bounds checks."


ALL_CODE_TOOLS = [
    search_repo_for_class,
    read_file_from_github,
    get_commit_file_diff,
    analyze_code_for_bug,
]