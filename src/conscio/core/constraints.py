"""Data-driven constraint validation (v2).

Replaces the v1 one-word regex in ``ConflictMonitor`` with a structural
checker registry plus a flag-gated, batched LLM judge for semantic
constraints. Constraints come from two sources: persistent influence rows
(``goals.active_constraints()``) parsed via :meth:`ConstraintValidator.parse`,
and per-episode instructions extracted from the user input via
:meth:`ConstraintValidator.extract_episode_constraints`.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Literal


Checker = Callable[[str], tuple[bool, str]]


@dataclass
class ParsedConstraint:
    constraint_id: str  # influence id or "episode:<n>"
    text: str
    kind: Literal["structural", "semantic"]
    checker: Checker | None = None  # structural only


@dataclass
class ConstraintCheck:
    constraint_id: str
    text: str
    kind: str
    passed: bool | None  # None = not checkable (semantic, judge off / no verdict)
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "constraint_id": self.constraint_id,
            "text": self.text,
            "kind": self.kind,
            "passed": self.passed,
            "detail": self.detail,
        }


@dataclass
class ConstraintReport:
    checks: list[ConstraintCheck] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """True when every check with a verdict passes (None = recorded, not blocking)."""
        return all(check.passed for check in self.checks if check.passed is not None)

    @property
    def violations(self) -> list[ConstraintCheck]:
        return [check for check in self.checks if check.passed is False]

    def to_dicts(self) -> list[dict[str, Any]]:
        return [check.to_dict() for check in self.checks]


def _word_limit_checker(limit: int) -> Checker:
    def check(answer: str) -> tuple[bool, str]:
        words = len(answer.split())
        return words <= limit, f"{words} word(s), limit {limit}"

    return check


def _char_limit_checker(limit: int, *, strict: bool) -> Checker:
    def check(answer: str) -> tuple[bool, str]:
        length = len(answer)
        passed = length < limit if strict else length <= limit
        return passed, f"{length} chars, limit {'<' if strict else '<='}{limit}"

    return check


def _json_checker(answer: str) -> tuple[bool, str]:
    try:
        json.loads(answer.strip())
    except (json.JSONDecodeError, ValueError) as exc:
        return False, f"invalid JSON: {exc}"
    return True, "valid JSON"


def _bullet_checker(answer: str) -> tuple[bool, str]:
    lines = [line.strip() for line in answer.splitlines() if line.strip()]
    if not lines:
        return False, "empty answer"
    bad = [
        line
        for line in lines
        if not (line.startswith(("-", "*", "•")) or re.match(r"\d+[.)]\s", line))
    ]
    if bad:
        return False, f"{len(bad)} line(s) not bullet-prefixed: {bad[0][:60]!r}"
    return True, f"{len(lines)} bullet line(s)"


def _include_checker(needle: str, *, negated: bool) -> Checker:
    def check(answer: str) -> tuple[bool, str]:
        present = needle.lower() in answer.lower()
        if negated:
            return not present, f"{needle!r} {'present' if present else 'absent'} (must not include)"
        return present, f"{needle!r} {'present' if present else 'absent'} (must include)"

    return check


# Registry: (compiled regex over constraint text) -> checker builder.
# Order matters: first match wins in parse().
_ONE_WORD_RE = re.compile(r"\b(?:one word|single word|one-word)\b", re.I)
_AT_MOST_WORDS_RE = re.compile(r"\b(?:at most|no more than|under|fewer than)\s+(\d+)\s+words?\b", re.I)
_CHAR_LIMIT_RE = re.compile(
    r"\b(under|at most|no more than|fewer than|less than)\s+(\d+)\s+(?:characters|chars)\b", re.I
)
_JSON_RE = re.compile(r"\b(?:respond in json|valid json|json only|as json|in json)\b", re.I)
_BULLET_RE = re.compile(r"\b(?:bullet(?:ed)?(?:\s+(?:points?|list))?|list format)\b", re.I)
_INCLUDE_RE = re.compile(r"\bmust\s+(not\s+)?(?:include|mention)\s+[\"“']([^\"”']+)[\"”']", re.I)

_STRICT_CHAR_WORDS = {"under", "fewer than", "less than"}


def _build_one_word(match: re.Match[str]) -> Checker:
    return _word_limit_checker(1)


def _build_word_limit(match: re.Match[str]) -> Checker:
    return _word_limit_checker(int(match.group(1)))


def _build_char_limit(match: re.Match[str]) -> Checker:
    strict = match.group(1).lower() in _STRICT_CHAR_WORDS
    return _char_limit_checker(int(match.group(2)), strict=strict)


def _build_json(match: re.Match[str]) -> Checker:
    return _json_checker


def _build_bullets(match: re.Match[str]) -> Checker:
    return _bullet_checker


def _build_include(match: re.Match[str]) -> Checker:
    return _include_checker(match.group(2), negated=bool(match.group(1)))


STRUCTURAL_CHECKERS: list[tuple[re.Pattern[str], Callable[[re.Match[str]], Checker]]] = [
    (_ONE_WORD_RE, _build_one_word),
    (_AT_MOST_WORDS_RE, _build_word_limit),
    (_CHAR_LIMIT_RE, _build_char_limit),
    (_JSON_RE, _build_json),
    (_BULLET_RE, _build_bullets),
    (_INCLUDE_RE, _build_include),
]


def _sentence_around(text: str, position: int) -> str:
    """The sentence/clause of ``text`` containing ``position`` (for readable constraint text)."""
    start = max(
        (text.rfind(sep, 0, position) + 1 for sep in ".!?;\n"),
        default=0,
    )
    ends = [idx for sep in ".!?;\n" if (idx := text.find(sep, position)) != -1]
    end = min(ends) + 1 if ends else len(text)
    return text[start:end].strip()


_JUDGE_SYSTEM_PROMPT = (
    "You are a strict constraint judge. Given an answer and a list of constraints, "
    "decide for each constraint whether the answer satisfies it. Respond with ONLY a "
    'JSON array: [{"constraint_id": "...", "passed": true|false, "reason": "..."}].'
)


class ConstraintValidator:
    """Parses constraint rows / user input into checkable constraints and validates answers.

    Structural constraints (word counts, char limits, JSON, bullets, must-include)
    are checked deterministically. Anything unmatched is ``semantic``: checked only
    when ``judge_enabled`` and an ``llm`` is present, via one batched LLM call;
    otherwise recorded with ``passed=None`` (not blocking).
    """

    def __init__(self, *, llm: Any = None, judge_enabled: bool = False) -> None:
        self.llm = llm
        self.judge_enabled = judge_enabled

    def parse(self, rows: list[dict]) -> list[ParsedConstraint]:
        """Parse persistent constraint rows (from ``goals.active_constraints()``)."""
        constraints: list[ParsedConstraint] = []
        for index, row in enumerate(rows, start=1):
            text = str(row.get("content", "")).strip()
            if not text:
                continue
            constraint_id = str(row.get("id") or f"influence:{index}")
            checker = self._match_checker(text)
            if checker is not None:
                constraints.append(
                    ParsedConstraint(constraint_id, text, "structural", checker)
                )
            else:
                constraints.append(ParsedConstraint(constraint_id, text, "semantic", None))
        return constraints

    def extract_episode_constraints(self, user_input: str) -> list[ParsedConstraint]:
        """Extract per-episode structural constraints stated in the user input."""
        constraints: list[ParsedConstraint] = []
        seen: set[tuple[int, tuple[str | None, ...]]] = set()
        for pattern_index, (pattern, builder) in enumerate(STRUCTURAL_CHECKERS):
            for match in pattern.finditer(user_input):
                key = (pattern_index, match.groups())
                if key in seen:
                    continue
                seen.add(key)
                constraints.append(
                    ParsedConstraint(
                        constraint_id=f"episode:{len(constraints) + 1}",
                        text=_sentence_around(user_input, match.start()),
                        kind="structural",
                        checker=builder(match),
                    )
                )
        return constraints

    async def validate(
        self, answer: str, constraints: list[ParsedConstraint]
    ) -> ConstraintReport:
        checks: list[ConstraintCheck] = []
        semantic: list[ParsedConstraint] = []
        for constraint in constraints:
            if constraint.kind == "structural" and constraint.checker is not None:
                passed, detail = constraint.checker(answer)
                checks.append(
                    ConstraintCheck(constraint.constraint_id, constraint.text, "structural", passed, detail)
                )
            else:
                semantic.append(constraint)
        if semantic:
            checks.extend(await self._check_semantic(answer, semantic))
        return ConstraintReport(checks=checks)

    def _match_checker(self, text: str) -> Checker | None:
        for pattern, builder in STRUCTURAL_CHECKERS:
            match = pattern.search(text)
            if match:
                return builder(match)
        return None

    async def _check_semantic(
        self, answer: str, constraints: list[ParsedConstraint]
    ) -> list[ConstraintCheck]:
        if not (self.judge_enabled and self.llm is not None):
            return [
                ConstraintCheck(
                    c.constraint_id, c.text, "semantic", None, "semantic constraint; judge disabled"
                )
                for c in constraints
            ]
        verdicts = await self._judge(answer, constraints)
        checks: list[ConstraintCheck] = []
        for constraint in constraints:
            verdict = verdicts.get(constraint.constraint_id)
            if verdict is None:
                checks.append(
                    ConstraintCheck(
                        constraint.constraint_id,
                        constraint.text,
                        "semantic",
                        None,
                        "judge returned no verdict",
                    )
                )
            else:
                passed, reason = verdict
                checks.append(
                    ConstraintCheck(constraint.constraint_id, constraint.text, "semantic", passed, reason)
                )
        return checks

    async def _judge(
        self, answer: str, constraints: list[ParsedConstraint]
    ) -> dict[str, tuple[bool | None, str]]:
        """One batched LLM call judging all semantic constraints at temperature 0."""
        payload = {
            "answer": answer,
            "constraints": [
                {"constraint_id": c.constraint_id, "text": c.text} for c in constraints
            ],
        }
        messages = [
            {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        try:
            response = await self.llm.chat_async(messages, temperature=0.0, max_tokens=200)
        except Exception:
            return {}
        content = str(response.get("content", "") or "")
        items = self._parse_judge_json(content)
        verdicts: dict[str, tuple[bool | None, str]] = {}
        for item in items:
            if not isinstance(item, dict):
                continue
            constraint_id = str(item.get("constraint_id", ""))
            if not constraint_id:
                continue
            verdicts[constraint_id] = (
                self._coerce_bool(item.get("passed")),
                str(item.get("reason", "")),
            )
        return verdicts

    @staticmethod
    def _parse_judge_json(content: str) -> list[Any]:
        start = content.find("[")
        end = content.rfind("]")
        if start == -1 or end == -1 or end < start:
            return []
        try:
            parsed = json.loads(content[start : end + 1])
        except (json.JSONDecodeError, ValueError):
            return []
        return parsed if isinstance(parsed, list) else []

    @staticmethod
    def _coerce_bool(value: Any) -> bool | None:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in ("true", "yes", "pass", "passed"):
                return True
            if lowered in ("false", "no", "fail", "failed"):
                return False
        return None
