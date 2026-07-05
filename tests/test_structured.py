"""Tests for the structured-output helper (response_format decoration + hand-parse).

The design contract: ``response_format`` is sent only when the client advertises
support via a ``response_format_support()`` method; the reply content is ALWAYS
parsed with :func:`first_json_value`. Plain stubs have no such method and so must
never receive a ``response_format`` kwarg. Transport errors propagate; a parse
miss returns None.
"""
from __future__ import annotations

import unittest

from conscio.llm.structured import first_json_value, structured_json


class RecordingStubLLM:
    def __init__(self, content: str, support: str | None = None) -> None:
        self.content = content
        self.kwargs: list[dict] = []
        self._support = support

    async def chat_async(self, messages, **kwargs):
        self.kwargs.append(kwargs)
        return {"role": "assistant", "content": self.content}


class SupportingStubLLM(RecordingStubLLM):
    """A stub advertising a structured-output mode, like a RoleClient."""

    def response_format_support(self) -> str:
        return self._support or "none"


class RaisingStubLLM:
    async def chat_async(self, messages, **kwargs):
        raise RuntimeError("transport down")


class StructuredJsonTests(unittest.IsolatedAsyncioTestCase):
    async def test_plain_stub_gets_no_response_format(self) -> None:
        llm = RecordingStubLLM("Here is the array: [1] then more prose.")
        data = await structured_json(
            llm, [{"role": "user", "content": "give me a number"}], schema={"type": "object"}
        )

        self.assertEqual(data, [1])
        self.assertNotIn("response_format", llm.kwargs[0])

    async def test_json_schema_support_sends_schema(self) -> None:
        llm = SupportingStubLLM("[1]", support="json_schema")
        schema = {"type": "object", "properties": {"x": {"type": "integer"}}}
        data = await structured_json(
            llm, [{"role": "user", "content": "x"}], schema=schema, schema_name="widget"
        )

        self.assertEqual(data, [1])
        fmt = llm.kwargs[0].get("response_format")
        self.assertIsNotNone(fmt)
        self.assertEqual(fmt["type"], "json_schema")
        self.assertEqual(fmt["json_schema"]["name"], "widget")
        self.assertEqual(fmt["json_schema"]["schema"], schema)

    async def test_json_object_support_without_schema(self) -> None:
        # support "json_object" with a schema still downgrades to json_object
        # (no schema means json_object too).
        object_only = SupportingStubLLM("just [1]", support="json_object")
        await structured_json(
            object_only, [{"role": "user", "content": "x"}], schema={"type": "object"}
        )
        self.assertEqual(object_only.kwargs[0].get("response_format"), {"type": "json_object"})

        # support "json_schema" but schema=None → also json_object.
        schema_none = SupportingStubLLM("and [2]", support="json_schema")
        await structured_json(schema_none, [{"role": "user", "content": "x"}], schema=None)
        self.assertEqual(schema_none.kwargs[0].get("response_format"), {"type": "json_object"})

    async def test_support_none_sends_nothing(self) -> None:
        llm = SupportingStubLLM("[1]", support="none")
        await structured_json(llm, [{"role": "user", "content": "x"}], schema={"type": "object"})

        self.assertNotIn("response_format", llm.kwargs[0])

    async def test_parse_miss_returns_none(self) -> None:
        llm = RecordingStubLLM("I have no JSON for you.")
        data = await structured_json(llm, [{"role": "user", "content": "x"}])

        self.assertIsNone(data)

    async def test_transport_error_propagates(self) -> None:
        llm = RaisingStubLLM()
        with self.assertRaises(RuntimeError):
            await structured_json(llm, [{"role": "user", "content": "x"}])

    async def test_temperature_and_max_tokens_flow_through(self) -> None:
        llm = RecordingStubLLM("[1]")
        await structured_json(
            llm,
            [{"role": "user", "content": "x"}],
            temperature=0.3,
            max_tokens=99,
        )
        self.assertEqual(llm.kwargs[0]["temperature"], 0.3)
        self.assertEqual(llm.kwargs[0]["max_tokens"], 99)


class FirstJsonValueTests(unittest.TestCase):
    def test_fenced_json_block_wins(self) -> None:
        raw = 'Prose with [99] noise\n```json\n{"a": 1}\n```\nmore [42] noise'
        self.assertEqual(first_json_value(raw), {"a": 1})

    def test_fenced_block_without_lang_tag(self) -> None:
        raw = "intro\n```\n[1, 2, 3]\n```\noutro"
        self.assertEqual(first_json_value(raw), [1, 2, 3])

    def test_prose_embedded_object_parses(self) -> None:
        raw = 'The verdict is {"passed": true, "reason": "ok"} as I said.'
        self.assertEqual(first_json_value(raw), {"passed": True, "reason": "ok"})

    def test_balanced_brackets_survive_trailing_brackets_in_prose(self) -> None:
        # A greedy `\[.*\]` would wrongly swallow the trailing `]` in prose.
        raw = 'see [1, "x]"] then y'
        self.assertEqual(first_json_value(raw), [1, "x]"])

    def test_empty_returns_none(self) -> None:
        self.assertIsNone(first_json_value(""))
        self.assertIsNone(first_json_value("   "))

    def test_garbage_returns_none(self) -> None:
        self.assertIsNone(first_json_value("just words, no structure at all"))


if __name__ == "__main__":
    unittest.main()
