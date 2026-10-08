"""Tests for reasoning control and provider-specific request parameters."""

import pytest

from rag_facts_check import llm as llm_module
from rag_facts_check.llm import APILLM, AsyncAPILLM, reasoning_disable_params
from rag_facts_check.server import _build_extra_body


class TestReasoningDisableParams:
    """reasoning_disable_params() translates LLM_DISABLE_REASONING."""

    @pytest.mark.parametrize("value", [None, "", "0", "false", "no", "off", "  "])
    def test_disabled_values_send_nothing(self, value):
        assert reasoning_disable_params(value) == {}

    @pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on", "disable"])
    def test_truthy_values_use_default_strategy(self, value):
        assert reasoning_disable_params(value) == {
            "chat_template_kwargs": {"enable_thinking": False}
        }

    def test_chat_template_strategy(self):
        assert reasoning_disable_params("chat_template") == {
            "chat_template_kwargs": {"enable_thinking": False}
        }

    def test_reasoning_effort_strategy(self):
        assert reasoning_disable_params("reasoning_effort") == {"reasoning_effort": "none"}

    def test_both_strategy(self):
        assert reasoning_disable_params("both") == {
            "chat_template_kwargs": {"enable_thinking": False},
            "reasoning_effort": "none",
        }

    def test_unknown_value_leaves_reasoning_on(self):
        assert reasoning_disable_params("definitely-off") == {}


class TestBuildExtraBody:
    """_build_extra_body() merges the flag with LLM_EXTRA_BODY."""

    def test_no_params_by_default(self):
        assert _build_extra_body({}) == {}

    def test_flag_adds_params(self):
        assert _build_extra_body({"LLM_DISABLE_REASONING": "1"}) == {
            "chat_template_kwargs": {"enable_thinking": False}
        }

    def test_llm_extra_body_is_merged(self):
        body = _build_extra_body({"LLM_EXTRA_BODY": '{"stop": ["END"]}'})
        assert body == {"stop": ["END"]}

    def test_llm_extra_body_wins_on_conflicts(self):
        body = _build_extra_body(
            {
                "LLM_DISABLE_REASONING": "1",
                "LLM_EXTRA_BODY": '{"chat_template_kwargs": {"enable_thinking": true}}',
            }
        )
        assert body == {"chat_template_kwargs": {"enable_thinking": True}}

    def test_invalid_json_is_ignored(self):
        assert _build_extra_body({"LLM_EXTRA_BODY": "not json", "LLM_DISABLE_REASONING": "1"}) == {
            "chat_template_kwargs": {"enable_thinking": False}
        }

    def test_non_object_json_is_ignored(self):
        assert _build_extra_body({"LLM_EXTRA_BODY": "[1, 2]"}) == {}


class TestPayloadMerging:
    """extra_body reaches the request payload without overriding call params."""

    def test_apillm_sends_extra_body(self, monkeypatch):
        captured = {}

        class FakeResponse:
            def raise_for_status(self):
                pass

            def json(self):
                return {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}

        def fake_post(url, json=None, headers=None):
            captured["payload"] = json
            return FakeResponse()

        monkeypatch.setattr(llm_module.requests, "post", fake_post)

        llm = APILLM(
            "http://example/v1/chat/completions",
            model_name="m",
            chat_mode=True,
            extra_body=reasoning_disable_params("1"),
        )
        assert llm.generate("prompt", max_new_tokens=64, temperature=0.3) == "ok"

        payload = captured["payload"]
        assert payload["chat_template_kwargs"] == {"enable_thinking": False}
        assert payload["max_tokens"] == 64
        assert payload["temperature"] == 0.3

    async def test_async_apillm_sends_extra_body(self, monkeypatch):
        captured = {}

        async def fake_request_once(payload, headers):
            captured["payload"] = payload
            return "ok"

        llm = AsyncAPILLM(
            "http://example/v1/chat/completions",
            model_name="m",
            chat_mode=True,
            extra_body=reasoning_disable_params("both"),
        )
        monkeypatch.setattr(llm, "_request_once", fake_request_once)

        assert await llm.generate("prompt", max_new_tokens=64) == "ok"
        payload = captured["payload"]
        assert payload["chat_template_kwargs"] == {"enable_thinking": False}
        assert payload["reasoning_effort"] == "none"
        assert payload["max_tokens"] == 64
