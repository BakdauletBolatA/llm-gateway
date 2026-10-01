"""The router's rules are meant to be read: each decision comes with its reasons."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from llm_gateway.complexity import classify
from llm_gateway.schemas import ChatCompletionRequest
from llm_gateway.settings import ComplexityRouterConfig

CONFIG = ComplexityRouterConfig(enabled=True, small_route="small", large_route="large")


def ask(*contents: str) -> ChatCompletionRequest:
    roles = ["user", "assistant"]
    messages = [{"role": roles[i % 2], "content": c} for i, c in enumerate(contents)]
    return ChatCompletionRequest.model_validate({"messages": messages})


def test_a_short_factual_question_goes_to_the_small_model() -> None:
    decision = classify(ask("What is the capital of France?"), CONFIG)
    assert decision.tier == "small"
    assert decision.score == 0
    assert decision.reasons == []


def test_code_goes_to_the_large_model_and_says_why() -> None:
    decision = classify(ask("Write a Python function that merges two sorted lists."), CONFIG)
    assert decision.tier == "large"
    assert any(reason.startswith("code") for reason in decision.reasons)


def test_a_fenced_code_block_counts_as_code() -> None:
    decision = classify(ask("What does this do?\n```\nx = [i for i in y]\n```"), CONFIG)
    assert decision.tier == "large"


def test_a_request_to_reason_goes_to_the_large_model() -> None:
    decision = classify(ask("Explain step by step why the sky is blue."), CONFIG)
    assert decision.tier == "large"
    assert any(reason.startswith("reasoning") for reason in decision.reasons)


def test_arithmetic_counts_as_math() -> None:
    decision = classify(
        ask("A train travels 120 km in 1.5 hours. What is its average speed?"), CONFIG
    )
    assert decision.tier == "large"
    assert any(reason.startswith("math") for reason in decision.reasons)


def test_length_alone_is_not_enough_to_pay_for_the_large_model() -> None:
    long_but_plain = "Tell me about the history of the city of Paris. " * 20
    decision = classify(ask(long_but_plain), CONFIG)
    assert decision.score == 2
    assert decision.tier == "small"


def test_only_the_last_user_message_is_judged_for_content() -> None:
    decision = classify(ask("Write a Python function.", "Sure, here it is.", "Thanks!"), CONFIG)
    assert not any(reason.startswith("code") for reason in decision.reasons)


def test_a_long_conversation_adds_a_point() -> None:
    turns = ["hello", "hi", "how are you", "fine", "and you"]
    decision = classify(ask(*turns), CONFIG)
    assert any(reason.startswith("turns") for reason in decision.reasons)


def test_the_threshold_is_configurable() -> None:
    strict = CONFIG.model_copy(update={"large_threshold": 10})
    assert classify(ask("Write a Python function that merges two lists."), strict).tier == "small"


def test_a_router_without_two_distinct_routes_is_refused() -> None:
    with pytest.raises(ValidationError, match="different"):
        ComplexityRouterConfig(enabled=True, small_route="same", large_route="same")


@pytest.mark.parametrize(
    "prompt",
    [
        "What is 15% of 240?",
        "A tank is 40% full. How many litres are missing?",
        "A shop sells pens at 3 for $2. How much do 24 pens cost?",
        "Tom is twice as old as Anna and in 6 years they will be 42 in total. How old is Anna?",
    ],
)
def test_arithmetic_word_problems_are_not_mistaken_for_simple_questions(prompt: str) -> None:
    decision = classify(ask(prompt), CONFIG)
    assert decision.tier == "large", decision
    assert any(reason.startswith("math") for reason in decision.reasons)


def test_a_percent_sign_is_matched_even_though_it_is_not_a_word_character() -> None:
    assert classify(ask("What is 7% of 50?"), CONFIG).score >= 3


def test_plain_questions_that_merely_contain_a_number_stay_small() -> None:
    assert (
        classify(ask("What is the capital of the country whose phone prefix is 33?"), CONFIG).tier
        == "small"
    )


def test_a_quantity_question_without_a_number_is_not_arithmetic() -> None:
    assert classify(ask("How many days are in a leap year?"), CONFIG).tier == "small"


def test_the_reason_names_the_whole_number_not_its_last_digit() -> None:
    decision = classify(ask("What is 15% of 240?"), CONFIG)
    assert decision.reasons == ["math(15%):+3"]
