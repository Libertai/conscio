from __future__ import annotations

import unittest

from conscio.core.constraints import ConstraintValidator


class JudgeStubLLM:
    def __init__(self, content: str) -> None:
        self.content = content
        self.calls: list[list[dict]] = []
        self.kwargs: list[dict] = []

    async def chat_async(self, messages: list[dict], **kwargs) -> dict:
        self.calls.append(messages)
        self.kwargs.append(kwargs)
        return {"content": self.content}


class StructuredJudgeStubLLM(JudgeStubLLM):
    """Judge stub advertising a structured-output mode, like a RoleClient."""

    def __init__(self, content: str, mode: str = "json_object") -> None:
        super().__init__(content)
        self._mode = mode

    def response_format_support(self) -> str:
        return self._mode


class StructuralCheckerTests(unittest.IsolatedAsyncioTestCase):
    async def test_one_word_constraint_rejects_long_answer(self) -> None:
        validator = ConstraintValidator()
        constraints = validator.extract_episode_constraints(
            "Answer in one word: what is 2+2?"
        )

        self.assertEqual(len(constraints), 1)
        self.assertEqual(constraints[0].kind, "structural")
        report = await validator.validate("The answer is four.", constraints)

        self.assertFalse(report.passed)
        self.assertEqual(len(report.violations), 1)
        self.assertEqual(report.violations[0].constraint_id, "episode:1")

    async def test_one_word_constraint_accepts_single_word(self) -> None:
        validator = ConstraintValidator()
        constraints = validator.extract_episode_constraints("Answer in one word: 2+2?")

        report = await validator.validate("Four", constraints)

        self.assertTrue(report.passed)
        self.assertEqual(report.violations, [])

    async def test_at_most_n_words(self) -> None:
        validator = ConstraintValidator()
        constraints = validator.parse(
            [{"id": "inf-1", "content": "Reply in at most 3 words."}]
        )

        self.assertEqual(constraints[0].kind, "structural")
        ok = await validator.validate("Yes it is", constraints)
        bad = await validator.validate("No, that is not correct", constraints)

        self.assertTrue(ok.passed)
        self.assertFalse(bad.passed)

    async def test_char_limit(self) -> None:
        validator = ConstraintValidator()
        constraints = validator.parse(
            [{"id": "inf-2", "content": "Keep the summary under 20 characters."}]
        )

        ok = await validator.validate("Short answer.", constraints)
        bad = await validator.validate("x" * 21, constraints)

        self.assertTrue(ok.passed)
        self.assertFalse(bad.passed)
        self.assertIn("21 chars", bad.violations[0].detail)

    async def test_json_constraint(self) -> None:
        validator = ConstraintValidator()
        constraints = validator.parse(
            [{"id": "inf-3", "content": "Respond in valid JSON."}]
        )

        ok = await validator.validate('{"answer": 4}', constraints)
        bad = await validator.validate("the answer is 4", constraints)

        self.assertTrue(ok.passed)
        self.assertFalse(bad.passed)

    async def test_bullet_constraint(self) -> None:
        validator = ConstraintValidator()
        constraints = validator.parse(
            [{"id": "inf-4", "content": "Format the response as a bullet list."}]
        )

        ok = await validator.validate("- first\n- second\n* third", constraints)
        bad = await validator.validate("First, do this. Then do that.", constraints)

        self.assertTrue(ok.passed)
        self.assertFalse(bad.passed)

    async def test_must_include_and_must_not_include(self) -> None:
        validator = ConstraintValidator()
        constraints = validator.parse(
            [
                {"id": "inf-5", "content": 'The answer must include "Paris".'},
                {"id": "inf-6", "content": 'The answer must not mention "city".'},
            ]
        )

        ok = await validator.validate("Paris is the capital of France.", constraints)
        bad = await validator.validate("It is a large city in France.", constraints)

        self.assertTrue(ok.passed)
        self.assertFalse(bad.passed)
        self.assertEqual(
            {check.constraint_id for check in bad.violations}, {"inf-5", "inf-6"}
        )


class EpisodeExtractionTests(unittest.IsolatedAsyncioTestCase):
    async def test_extracts_multiple_constraints_with_episode_ids(self) -> None:
        validator = ConstraintValidator()
        constraints = validator.extract_episode_constraints(
            'Answer in one word and you must include "four".'
        )

        self.assertEqual([c.constraint_id for c in constraints], ["episode:1", "episode:2"])
        self.assertTrue(all(c.kind == "structural" for c in constraints))

        report = await validator.validate("Four", constraints)
        self.assertTrue(report.passed)

    def test_plain_input_extracts_nothing(self) -> None:
        validator = ConstraintValidator()
        self.assertEqual(validator.extract_episode_constraints("What is 2+2?"), [])


class SemanticConstraintTests(unittest.IsolatedAsyncioTestCase):
    async def test_semantic_constraint_returns_none_when_judge_off(self) -> None:
        validator = ConstraintValidator()
        constraints = validator.parse(
            [{"id": "inf-7", "content": "Be polite and encouraging."}]
        )

        self.assertEqual(constraints[0].kind, "semantic")
        report = await validator.validate("Whatever.", constraints)

        self.assertIsNone(report.checks[0].passed)
        self.assertTrue(report.passed)  # None checks are recorded, not blocking
        self.assertEqual(report.violations, [])

    async def test_batched_judge_resolves_semantic_constraints(self) -> None:
        llm = JudgeStubLLM(
            '[{"constraint_id": "inf-8", "passed": true, "reason": "tone ok"},'
            ' {"constraint_id": "inf-9", "passed": false, "reason": "mentions internals"}]'
        )
        validator = ConstraintValidator(llm=llm, judge_enabled=True)
        constraints = validator.parse(
            [
                {"id": "inf-8", "content": "Be polite and encouraging."},
                {"id": "inf-9", "content": "Never discuss internal implementation details."},
            ]
        )

        report = await validator.validate("Sure! The handler lives in runtime.py.", constraints)

        # one batched call for both constraints, temperature 0
        self.assertEqual(len(llm.calls), 1)
        self.assertEqual(llm.kwargs[0].get("temperature"), 0.0)
        self.assertIn("inf-8", llm.calls[0][-1]["content"])
        self.assertIn("inf-9", llm.calls[0][-1]["content"])
        by_id = {check.constraint_id: check for check in report.checks}
        self.assertTrue(by_id["inf-8"].passed)
        self.assertFalse(by_id["inf-9"].passed)
        self.assertIn("internals", by_id["inf-9"].detail)
        self.assertFalse(report.passed)

    async def test_judge_requests_structured_output_when_supported(self) -> None:
        llm = StructuredJudgeStubLLM(
            '[{"constraint_id": "inf-8", "passed": true, "reason": "ok"}]'
        )
        validator = ConstraintValidator(llm=llm, judge_enabled=True)
        constraints = validator.parse([{"id": "inf-8", "content": "Be polite."}])

        await validator.validate("Sure!", constraints)

        self.assertEqual(llm.kwargs[0].get("response_format"), {"type": "json_object"})

    async def test_judge_omits_response_format_when_unsupported(self) -> None:
        llm = StructuredJudgeStubLLM(
            '[{"constraint_id": "inf-8", "passed": true, "reason": "ok"}]', mode="none"
        )
        validator = ConstraintValidator(llm=llm, judge_enabled=True)
        constraints = validator.parse([{"id": "inf-8", "content": "Be polite."}])

        await validator.validate("Sure!", constraints)

        self.assertNotIn("response_format", llm.kwargs[0])

    async def test_judge_garbage_output_degrades_to_none(self) -> None:
        llm = JudgeStubLLM("I cannot judge this.")
        validator = ConstraintValidator(llm=llm, judge_enabled=True)
        constraints = validator.parse(
            [{"id": "inf-10", "content": "Stay on topic."}]
        )

        report = await validator.validate("Anything.", constraints)

        self.assertIsNone(report.checks[0].passed)
        self.assertTrue(report.passed)

    async def test_structural_and_semantic_mix(self) -> None:
        validator = ConstraintValidator()
        constraints = validator.parse(
            [
                {"id": "inf-11", "content": "Reply in at most 2 words."},
                {"id": "inf-12", "content": "Be kind."},
            ]
        )

        report = await validator.validate("Hello there friend", constraints)

        kinds = {check.constraint_id: check.kind for check in report.checks}
        self.assertEqual(kinds, {"inf-11": "structural", "inf-12": "semantic"})
        self.assertFalse(report.passed)
        self.assertEqual([v.constraint_id for v in report.violations], ["inf-11"])


if __name__ == "__main__":
    unittest.main()
