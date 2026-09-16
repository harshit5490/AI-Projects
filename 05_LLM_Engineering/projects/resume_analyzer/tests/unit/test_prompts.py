from app.prompts.resume_prompts import (
    RESUME_ANALYSIS_SYSTEM_PROMPT,
    build_resume_analysis_prompt,
)


def test_system_prompt_contains_required_instructions():
    assert "valid JSON only" in RESUME_ANALYSIS_SYSTEM_PROMPT
    assert "Do not invent or assume candidate information." in RESUME_ANALYSIS_SYSTEM_PROMPT
    assert '"name": "string"' in RESUME_ANALYSIS_SYSTEM_PROMPT


def test_resume_text_is_inserted_into_prompt():
    resume_text = """
    John Doe
    Python Developer
    Skills: Python, FastAPI, Docker
    """

    prompt = build_resume_analysis_prompt(resume_text)

    assert resume_text in prompt
    assert "Analyze the following resume." in prompt


def test_resume_prompt_contains_output_instruction():
    prompt = build_resume_analysis_prompt("Sample resume text")

    assert "Return the analysis as valid JSON" in prompt