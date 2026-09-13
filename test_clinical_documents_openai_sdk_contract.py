from __future__ import annotations

import inspect

from openai import OpenAI


def test_openai_responses_sdk_supports_medical_report_contract_without_network_call():
    client = OpenAI(api_key="sk-synthetic-not-used")

    assert hasattr(client.responses, "parse")
    parse_parameters = inspect.signature(client.responses.parse).parameters
    for required in ("model", "input", "text_format", "store", "reasoning", "max_output_tokens"):
        assert required in parse_parameters

    create_parameters = inspect.signature(client.responses.create).parameters
    for required in ("model", "input", "tools", "store", "reasoning", "max_output_tokens"):
        assert required in create_parameters
