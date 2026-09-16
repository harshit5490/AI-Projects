from app.llm.mock import MockLLM


def test_mock_llm_returns_configured_response():
    expected_response = '{"name": "John Doe"}'

    llm = MockLLM(response=expected_response)

    response = llm.generate(
        system_prompt="Test system prompt",
        user_prompt="Test user prompt",
    )

    assert response == expected_response