"""Bounding-box judgments (changes/2026-10-15-bounding-box.md), offline: the
schema convention, the helper, Jev's Box on the typesafe wire, the cloud
wires, and the token-trie driver that answers a box by generation."""
from __future__ import annotations

import asyncio
import json
from dataclasses import replace

import pytest

from lm15 import Config, Message, Request, TypeSafeLM, bounding_box, judgments, yes_no
from lm15.errors import ProviderError, UnsupportedFeatureError
from lm15.judgments import (
    BOX_EDGES,
    anthropic_schema,
    gemini_schema,
    judgments_in_schema,
    measurable_judgments,
)
from lm15.providers import AnthropicLM, GeminiLM, OpenAIChatLM, OpenAILM
from lm15.providers.async_base import AsyncOpenAIChatLM
from lm15.testing import FakeResponse, FakeTransport
from lm15.types import image

PNG = "iVBORw0KGgo="
PHOTO = Message.user(image(data=PNG, media_type="image/png"))
FIND = judgments(mug=bounding_box("the coffee mug"), mug_visible=yes_no("Is there a coffee mug in the image?"))
BOX_ONLY = judgments(mug=bounding_box("the coffee mug"))


# ─── the convention (MAP-14 §1) ──────────────────────────────────────


def test_the_helper_emits_four_described_numbers_and_no_bounds() -> None:
    prop = bounding_box("the coffee mug")
    assert prop["type"] == "object" and prop["description"] == "the coffee mug"
    assert list(prop["properties"]) == list(BOX_EDGES) == ["top", "bottom", "left", "right"]
    assert prop["required"] == ["top", "bottom", "left", "right"] and prop["additionalProperties"] is False
    for edge, schema in prop["properties"].items():
        assert schema["type"] == "number" and "fraction" in schema["description"]
        # Anthropic answers 400 on minimum/maximum (receipted 2026-10-15).
        assert "minimum" not in schema and "maximum" not in schema
    assert "height" in prop["properties"]["top"]["description"] and "width" in prop["properties"]["left"]["description"]


def test_a_box_is_read_by_shape_and_declares_no_keys() -> None:
    found = judgments_in_schema(FIND["schema"])
    assert [(j.name, j.kind, j.keys, j.instruction) for j in found.values()] == [
        ("mug", "box", (), "the coffee mug"),
        ("mug_visible", "boolean", ("true", "false"), "Is there a coffee mug in the image?"),
    ]
    assert not found["mug"].measurable and list(measurable_judgments(found)) == ["mug_visible"]
    edge = {"type": "number"}
    shapes = {
        "hand_written": {"properties": {"left": edge, "top": edge, "right": edge, "bottom": edge}},  # any order, no type
        "bounded": {"type": "object", "properties": {k: {"type": "number", "minimum": 0, "maximum": 1} for k in BOX_EDGES}},
        "extra_edge": {"type": "object", "properties": {**{k: edge for k in BOX_EDGES}, "label": {"type": "string"}}},
        "integer_edges": {"type": "object", "properties": {k: {"type": "integer"} for k in BOX_EDGES}},
        "three_edges": {"type": "object", "properties": {k: edge for k in BOX_EDGES[:3]}},
        "array": {"type": "array", "properties": {k: edge for k in BOX_EDGES}},
        "pixels": {"type": "object", "properties": {"x": edge, "y": edge, "w": edge, "h": edge}},
    }
    read = judgments_in_schema({"type": "object", "properties": shapes})
    assert {name: j.kind for name, j in read.items()} == {"hand_written": "box", "bounded": "box"}


def test_the_rewriting_wires_leave_a_box_untouched() -> None:
    found = judgments_in_schema(FIND["schema"])
    assert anthropic_schema(FIND["schema"], found)["properties"]["mug"] == FIND["schema"]["properties"]["mug"]
    assert gemini_schema(FIND["schema"], found)["properties"]["mug"] == FIND["schema"]["properties"]["mug"]


# ─── typesafe: Jev's Box (MAP-14 §2) ─────────────────────────────────


def _reply(answers: dict) -> FakeResponse:
    return FakeResponse(status=200, headers=[("x-typesafe-request-id", "r-1")], body=json.dumps({
        "model": "jev-1.14.0", "answers": answers, "usage": {"input_tokens": 290, "output_tokens": 21}}).encode())


def test_typesafe_sends_a_box_question_with_the_opt_in_header() -> None:
    lm = TypeSafeLM(api_key="k")
    req = Request(model="jev-preview", messages=(PHOTO,), config=Config(response_format=FIND))
    built = lm.build_request(req, stream=False)
    wire = json.loads(built.body)
    assert wire["questions"] == {
        "mug": {"type": "box", "instructions": "the coffee mug"},  # no criteria: Box takes none
        "mug_visible": {"type": "noul", "instructions": "Is there a coffee mug in the image?"},
    }
    assert dict(built.headers)["TypeSafe-Danger-Zone-Images"] == "true"
    assert list(lm.plan(req)) == []
    # A Box question needs the header even with no image in the request
    # (live 400 without it, 2026-10-15); lm15 does not judge whether the
    # question is answerable.
    text_only = Request(model="jev-preview", messages=(Message.user("a kitchen"),), config=Config(response_format=BOX_ONLY))
    built = lm.build_request(text_only, stream=False)
    assert dict(built.headers)["TypeSafe-Danger-Zone-Images"] == "true" and "assets" not in json.loads(built.body)
    # A box without a description: the property name is the instruction, recorded.
    unnamed = {"type": "object", "properties": {k: {"type": "number"} for k in BOX_EDGES}}
    req = replace(text_only, config=Config(response_format=judgments(mug=unnamed)))
    assert json.loads(lm.build_request(req, stream=False).body)["questions"]["mug"] == {"type": "box", "instructions": "mug"}
    assert [(a.field, a.action) for a in lm.plan(req)] == [("config.response_format.schema.properties.mug.description", "defaulted")]


def test_typesafe_answers_the_box_verbatim_without_a_distribution() -> None:
    box = {"top": 0.25, "bottom": 0.82, "left": 0.44, "right": 1}
    answers = {"mug": {"type": "box", "box": box, "confidence": 0.92}, "mug_visible": {"type": "noul", "noul": 0.97}}
    lm = TypeSafeLM(api_key="k", transport=FakeTransport([_reply(answers)]))
    r = lm.complete(Request(model="jev-preview", messages=(PHOTO,), config=Config(response_format=FIND, probabilities="required")))
    assert r.data == {"mug": box, "mug_visible": True}
    assert isinstance(r.data["mug"]["right"], int)  # numbers as written
    assert set(r.probabilities) == {"mug_visible"} and r.method == "provider_classification"
    assert r.provider_data["typesafe"]["answers"]["mug"]["confidence"] == 0.92
    # Boxes only: no distribution exists, so none is reported (and `required` asked for nothing measurable).
    lm = TypeSafeLM(api_key="k", transport=FakeTransport([_reply({"mug": answers["mug"]})]))
    r = lm.complete(Request(model="jev-preview", messages=(PHOTO,), config=Config(response_format=BOX_ONLY, probabilities="required")))
    assert r.data == {"mug": box} and r.probabilities is None and r.method is None


@pytest.mark.parametrize("box,where", [
    ({"top": 0.1, "bottom": 0.2, "left": 0.3}, "answers.mug.box"),
    ({"top": 0.1, "bottom": 0.2, "left": 0.3, "right": 0.4, "x": 0}, "answers.mug.box"),
    ({"top": 0.1, "bottom": 1.2, "left": 0.3, "right": 0.4}, "answers.mug.box.bottom"),
    ({"top": "0.1", "bottom": 0.2, "left": 0.3, "right": 0.4}, "answers.mug.box.top"),
    ([0.1, 0.2, 0.3, 0.4], "answers.mug.box"),
])
def test_typesafe_refuses_a_malformed_box(box, where) -> None:
    lm = TypeSafeLM(api_key="k", transport=FakeTransport([_reply({"mug": {"type": "box", "box": box}})]))
    with pytest.raises(ProviderError) as exc:
        lm.complete(Request(model="jev-preview", messages=(PHOTO,), config=Config(response_format=BOX_ONLY)))
    assert where in str(exc.value)
    lm = TypeSafeLM(api_key="k", transport=FakeTransport([_reply({"mug": {"type": "noul", "noul": 0.5}})]))
    with pytest.raises(ProviderError, match="answers.mug.type"):
        lm.complete(Request(model="jev-preview", messages=(PHOTO,), config=Config(response_format=BOX_ONLY)))


# ─── cloud wires: structured output, verbatim ────────────────────────


def test_cloud_wires_send_the_box_verbatim_and_answer_a_data_part() -> None:
    req = Request(model="m", messages=(PHOTO,), config=Config(response_format=FIND, max_tokens=200))
    schema = FIND["schema"]
    o = json.loads(OpenAILM(api_key="k").build_request(req, stream=False).body)
    assert o["text"]["format"]["schema"] == schema
    a = json.loads(AnthropicLM(api_key="k").build_request(req, stream=False).body)
    assert a["output_config"]["format"]["schema"]["properties"]["mug"] == schema["properties"]["mug"]
    g = json.loads(GeminiLM(api_key="k").build_request(req, stream=False).body)
    assert g["generationConfig"]["responseJsonSchema"]["properties"]["mug"] == schema["properties"]["mug"]
    answer = '{"mug": {"top": 0.2, "bottom": 0.6, "left": 0.55, "right": 0.85}, "mug_visible": true}'
    chat = OpenAIChatLM(api_key="k", transport=FakeTransport([FakeResponse(status=200, body=json.dumps({
        "id": "c1", "model": "m", "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": answer}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1}}).encode())]))
    r = chat.complete(req)
    assert r.data == json.loads(answer) and r.probabilities is None


def test_probabilities_policy_concerns_only_keyed_judgments() -> None:
    lm = OpenAIChatLM(api_key="k")
    # Boxes only: nothing measurable was asked — no refusal, no record.
    for policy in ("if_available", "required"):
        req = Request(model="m", messages=(PHOTO,), config=Config(response_format=BOX_ONLY, probabilities=policy))
        assert list(lm.plan(req)) == []
    # A box beside a yes/no: the yes/no is what `required` cannot get here.
    with pytest.raises(UnsupportedFeatureError) as exc:
        lm.plan(Request(model="m", messages=(PHOTO,), config=Config(response_format=FIND, probabilities="required")))
    assert exc.value.feature == "config.probabilities"
    recorded = lm.plan(Request(model="m", messages=(PHOTO,), config=Config(response_format=FIND, probabilities="if_available")))
    assert [(a.field, a.action) for a in recorded] == [("config.probabilities", "dropped")]


# ─── the token trie: the box is generated, the yes/no measured ───────


class ScoringTransport(FakeTransport):
    generated = '{"mug": {"top": 0.2, "bottom": 0.6, "left": 0.55, "right": 0.85}, "mug_visible": false}'

    def reply(self, request):
        self.requests.append(request)
        payload = json.loads(request.body)
        if request.url.endswith("/tokenize"):
            answer = payload["messages"][-1]["content"]
            tokens = [1]
            if answer != "Answer:":
                tokens.append(10 if answer.endswith("true") else 11)
                if not payload["continue_final_message"]:
                    tokens.append(99)
            return json.dumps({"tokens": tokens}).encode()
        if request.url.endswith("/chat/completions"):
            return json.dumps({"id": "generated", "model": "m", "choices": [
                {"index": 0, "message": {"role": "assistant", "content": self.generated}, "finish_reason": "stop"}]}).encode()
        top = {f"token_id:{i}": (-0.1 if i == 10 else -3.0) for i in payload["logprob_token_ids"]}
        return json.dumps({"model": "m", "choices": [{"index": i, "logprobs": {"top_logprobs": [top]}}
                                                      for i in range(len(payload["prompt"]))]}).encode()

    def stream(self, request):
        return FakeResponse(status=200, body=self.reply(request))


class AsyncScoringTransport(ScoringTransport):
    def stream(self, request):
        from lm15.transports import AsyncTransportResponse

        body = self.reply(request)

        async def chunks():
            yield body

        async def release(body_consumed):
            pass

        return AsyncTransportResponse(status=200, reason="OK", headers=[], http_version="HTTP/1.1", chunks=chunks(), release=release)


@pytest.mark.parametrize("asynchronous", [False, True])
def test_trie_measures_the_yes_no_and_generation_answers_the_box(asynchronous) -> None:
    req = Request(model="m", messages=(Message.user("a kitchen"),), config=Config(response_format=FIND, probabilities="required"))
    transport = AsyncScoringTransport([]) if asynchronous else ScoringTransport([])
    cls = AsyncOpenAIChatLM if asynchronous else OpenAIChatLM
    lm = cls(api_key="k", compat="vllm", base_url="http://scoring/v1", transport=transport)
    assert [(a.field, a.action) for a in lm.plan(req)] == [("config.response_format", "client_side")]
    r = asyncio.run(lm.complete(req)) if asynchronous else lm.complete(req)
    assert r.data["mug"] == {"top": 0.2, "bottom": 0.6, "left": 0.55, "right": 0.85}
    assert r.data["mug_visible"] is True  # measured, overriding the generated pick
    assert set(r.probabilities) == {"mug_visible"} and r.method == "candidate_sequence_likelihood"
    urls = [t.url.rsplit("/", 1)[-1] for t in transport.requests]
    assert urls.count("completions") == 2 and urls[-1] == "completions"  # one scoring batch, one generation
    # A box missing from the generated answer is a malformed reply, never a guess.
    transport = AsyncScoringTransport([]) if asynchronous else ScoringTransport([])
    transport.generated = '{"mug_visible": false}'
    lm = cls(api_key="k", compat="vllm", base_url="http://scoring/v1", transport=transport)
    with pytest.raises(ProviderError, match="mug"):
        asyncio.run(lm.complete(req)) if asynchronous else lm.complete(req)


def test_boxes_only_on_a_scoring_server_take_the_ordinary_generated_path() -> None:
    transport = ScoringTransport([])
    transport.generated = '{"mug": {"top": 0.2, "bottom": 0.6, "left": 0.55, "right": 0.85}}'
    lm = OpenAIChatLM(api_key="k", compat="vllm", base_url="http://scoring/v1", transport=transport)
    req = Request(model="m", messages=(Message.user("a kitchen"),), config=Config(response_format=BOX_ONLY, probabilities="required"))
    r = lm.complete(req)
    assert r.data == json.loads(transport.generated) and r.probabilities is None
    assert [t.url.rsplit("/", 1)[-1] for t in transport.requests] == ["completions"]
    assert transport.requests[0].url.endswith("/chat/completions")
