# LLM Engineering --- Complete Chapter Notes

> **Bootcamp section:** LLM Engineering\
> **Goal:** Build reliable applications around hosted LLMs.\
> **Status:** Theory complete; project phase starts with Resume
> Analyzer, followed by Email Generator.

------------------------------------------------------------------------

## 1. What is LLM Engineering?

LLM Engineering is the discipline of building reliable applications
around Large Language Models (LLMs).

Traditional ML usually focuses on: - Data collection and preprocessing -
Model training - Model evaluation - Model serving

LLM Engineering focuses heavily on: - Hosted LLM APIs - Prompt
engineering - Generation controls - Structured outputs - Tool/function
calling - API reliability - Evaluation - Cost and latency -
Application/API development - Deployment

### Typical architecture

``` text
User
  ↓
Application
  ↓
Prompt / Instructions
  ↓
LLM API
  ↓
LLM Response
  ↓
Parsing / Validation
  ↓
Application Logic
  ↓
Final Response
```

**Core principle:** The LLM is a component of the application, not the
entire application.

------------------------------------------------------------------------

# 2. Prompt Engineering Principles

A prompt is the instruction/context supplied to an LLM.

A reliable prompt normally separates:

1.  Role
2.  Task
3.  Context
4.  Constraints
5.  Output format

Example:

``` text
Role:
You are a professional resume analyzer.

Task:
Analyze the supplied resume.

Context:
The candidate is applying for an AI Engineer role.

Constraints:
- Do not invent information.
- Use only information present in the resume.
- Keep recommendations practical.

Output format:
Return valid JSON with name, skills, experience, and recommendations.
```

### Why structure matters

A vague prompt such as:

``` text
Analyze this resume.
```

leaves many decisions to the model.

A structured prompt improves: - Consistency - Reliability -
Testability - Maintainability

Prompt engineering is not about making prompts unnecessarily long. The
goal is to provide the right information and constraints clearly.

------------------------------------------------------------------------

# 3. System, User and Developer Instructions

Instructions can be separated by purpose.

## System instructions

Define high-level model behavior.

Example:

``` text
You are a professional resume analyzer.
Do not invent information.
Always return the requested structure.
```

## User instructions

Contain the user's request and data.

Example:

``` text
Analyze the following resume for an AI Engineer position.
Resume:
...
```

## Developer instructions

Define application-level rules such as: - Required output structure -
Business rules - Tool usage rules - Validation requirements

Conceptually:

``` text
System
   ↓
Developer
   ↓
User
   ↓
Data / Context
```

The exact instruction behavior depends on the API/model, but the
engineering principle is to keep application rules separate from
user-provided content.

------------------------------------------------------------------------

# 4. Few-Shot Prompting

Few-shot prompting means providing examples of desired behavior.

### Zero-shot

``` text
Classify this email as spam or not spam.
```

No example is provided.

### Few-shot

``` text
Example 1:
Email: "Win a free phone now!"
Classification: spam

Example 2:
Email: "Your interview is scheduled for Monday."
Classification: not_spam

Now classify:
Email: "Claim your reward immediately!"
```

### When useful

Few-shot prompting is useful when: - Output style is difficult to
describe - Classification rules are subtle - The model needs formatting
examples - More consistency is required

### Trade-off

Too many examples increase: - Token usage - Prompt size - Latency

Use examples that actually improve behavior.

------------------------------------------------------------------------

# 5. Structured / JSON Outputs

Applications often need predictable data rather than free-form text.

Example:

``` json
{
  "name": "Harshit",
  "role": "AI Engineer",
  "experience_years": 1,
  "skills": ["Python", "LLM", "FastAPI"]
}
```

## JSON parsing

``` python
import json

data = json.loads(response_text)
```

Successful JSON parsing does **not** mean the data is valid for the
application.

For example:

``` json
{
  "name": "Harshit",
  "experience_years": -5
}
```

This is valid JSON but may violate application rules.

## Pydantic schema validation

``` python
from pydantic import BaseModel, Field

class ResumeOutput(BaseModel):
    name: str
    role: str
    experience_years: float = Field(ge=0)
    skills: list[str]
```

Validation:

``` python
validated = ResumeOutput.model_validate(data)
```

### Validation pipeline

``` text
LLM response
     ↓
JSON parsing
     ↓
Schema validation
     ↓
Application logic
```

**Rule:** Never assume generated JSON is automatically correct.

------------------------------------------------------------------------

# 6. Prompt Templates

Prompt templates make prompts reusable.

Example:

``` python
template = """
Analyze the resume for the following role:

Role: {role}

Resume:
{resume}

Return the requested structured output.
"""
```

Usage:

``` python
prompt = template.format(
    role="AI Engineer",
    resume=resume_text
)
```

### Benefits

-   Reusability
-   Consistency
-   Easier testing
-   Easier maintenance
-   Separation of instructions and data

Keep instructions and dynamic data clearly separated.

------------------------------------------------------------------------

# 7. Temperature and Generation Controls

Temperature controls randomness/variability in generated output.

## Low temperature

Usually produces more predictable output.

Useful for: - Information extraction - Classification - Structured
JSON - Business workflows

## Higher temperature

Usually allows more variation.

Useful for: - Brainstorming - Creative writing - Alternative ideas -
Marketing copy

**Important:** Temperature does not make a model more intelligent. It
changes generation behavior.

Other generation controls can include output token limits and
provider-specific parameters.

Choose generation settings according to the application.

------------------------------------------------------------------------

# 8. OpenAI API

Hosted LLM providers expose APIs that applications can call.

Conceptually:

``` text
Application
   ↓
OpenAI SDK
   ↓
OpenAI API
   ↓
Model
   ↓
Response
```

Provider-specific code should ideally be isolated behind an LLM client
abstraction.

## API keys

Never hard-code secrets:

``` python
api_key = "my-secret-key"
```

Use environment variables.

`.env`:

``` text
OPENAI_API_KEY=your_key_here
```

Python:

``` python
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
```

### Important billing lesson

ChatGPT subscription billing and API usage billing are separate. Having
a ChatGPT plan does not automatically mean API credits are available.

------------------------------------------------------------------------

# 9. Gemini API

Gemini can also be accessed through an API.

Conceptually:

``` text
Application
   ↓
Google GenAI SDK
   ↓
Gemini API
   ↓
Gemini model
   ↓
Response
```

Example pattern:

``` python
from google import genai

client = genai.Client(api_key=api_key)

response = client.models.generate_content(
    model="MODEL_NAME",
    contents="Your prompt"
)
```

Use a model currently available to your account/provider.

------------------------------------------------------------------------

# 10. Provider Abstraction and Mock LLM

A strong application should avoid tightly coupling business logic to one
provider.

Conceptually:

``` text
                 ┌── OpenAI
Application ─── LLMClient
                 ├── Gemini
                 └── MockLLM
```

Example interface:

``` python
class LLMClient:
    def generate(self, prompt):
        raise NotImplementedError
```

### Why use a mock LLM?

A mock LLM allows development and testing without making real API calls.

Benefits: - No API cost - Faster tests - Deterministic behavior - Easier
failure testing - Provider independence

This architecture was especially useful in our bootcamp when API
credit/quota was unavailable.

------------------------------------------------------------------------

# 11. API Keys and Secret Management

Secrets include: - API keys - Passwords - Access tokens - Database
credentials

Never commit secrets to Git.

Recommended `.gitignore`:

``` text
.env
.venv/
__pycache__/
```

Development flow:

``` text
.env
  ↓
Environment variable
  ↓
Application
  ↓
Provider SDK
```

For production, use the secret-management facilities of the deployment
environment.

------------------------------------------------------------------------

# 12. Retries

External APIs can fail temporarily.

Examples: - Server errors - Temporary service unavailability - Provider
overload - Rate limiting

A retry attempts the operation again.

## Exponential backoff

Instead of retrying immediately:

``` text
Attempt 1 → wait 1 second
Attempt 2 → wait 2 seconds
Attempt 3 → wait 4 seconds
Attempt 4 → wait 8 seconds
```

Conceptually:

``` text
delay = base_delay × 2^attempt
```

## Jitter

Add a random component to prevent many clients from retrying at exactly
the same time.

``` text
delay = exponential_backoff + random_jitter
```

## Maximum retries

Never retry forever.

``` text
max_retries = 3
```

After the retry budget is exhausted, return a controlled error.

------------------------------------------------------------------------

# 13. Timeouts

An API request can take too long or hang.

A timeout prevents the application from waiting indefinitely.

``` text
Request
  ↓
Wait
  ↓
Timeout reached
  ↓
Stop request
  ↓
Handle failure
```

Timeouts should be configured for external network calls.

------------------------------------------------------------------------

# 14. Rate Limits

Providers may limit: - Requests per minute - Tokens per minute -
Concurrent requests - Overall usage

Rate-limit errors commonly appear as HTTP `429`.

A good response is: 1. Detect the rate limit. 2. Respect provider retry
information when available. 3. Wait using backoff. 4. Retry when
appropriate. 5. Fail gracefully if the limit persists.

### Do not retry every error

Usually retryable: - Temporary server failures - Service unavailable -
Rate limits

Usually not retryable: - Invalid API key - Invalid request - Invalid
model name - Permanent validation errors

**Error classification is part of reliability engineering.**

------------------------------------------------------------------------

# 15. SDK Retry Behavior

Some provider SDKs already perform automatic retries for selected
temporary failures.

Therefore, avoid accidental nested retry loops:

``` text
Your retry loop
    ↓
SDK retry loop
    ↓
API
```

This can create: - Too many requests - Unexpected latency - Longer
failure times

Before adding custom retries, understand the SDK's built-in behavior.

------------------------------------------------------------------------

# 16. Tool / Function Calling

An LLM can decide which predefined application function should be used.

Architecture:

``` text
User
 ↓
LLM
 ↓
Function Call
 ↓
Tool Router
 ↓
Python Function
 ↓
Tool Result
 ↓
LLM
 ↓
Final Answer
```

Example tools:

``` python
def add_numbers(a: int, b: int) -> int:
    return a + b

def multiply_numbers(a: int, b: int) -> int:
    return a * b
```

The LLM can request:

``` text
add_numbers
a = 10
b = 20
```

The application executes the allowed function and sends the result back
to the model.

------------------------------------------------------------------------

# 17. Manual Tool Calling

Tool calls can be handled explicitly.

Example registry:

``` python
tool_registry = {
    "add_numbers": add_numbers,
    "multiply_numbers": multiply_numbers,
}
```

Conceptually:

``` python
function_name = ...
arguments = ...

tool = tool_registry[function_name]
result = tool(**arguments)
```

### Security principle

Never execute arbitrary model-generated Python code.

Avoid:

``` python
eval(model_output)
```

Prefer:

``` text
LLM
 ↓
Allowed tool name
 ↓
Tool registry
 ↓
Known Python function
```

The registry acts as a control/security boundary.

------------------------------------------------------------------------

# 18. Complete Tool → Result → LLM Loop

A complete workflow:

``` text
1. User sends request
        ↓
2. LLM decides a tool is required
        ↓
3. LLM returns function name + arguments
        ↓
4. Application validates arguments
        ↓
5. Tool router selects allowed function
        ↓
6. Python function executes
        ↓
7. Tool result is produced
        ↓
8. Result is sent back to LLM
        ↓
9. LLM generates final response
```

This is a fundamental pattern for tool-using LLM applications.

------------------------------------------------------------------------

# 19. Tool Schemas

Tool schemas define valid arguments.

Example:

``` python
from pydantic import BaseModel, Field

class AddNumbersArgs(BaseModel):
    a: int
    b: int

class MultiplyNumbersArgs(BaseModel):
    a: int
    b: int

class GetUserInfoArgs(BaseModel):
    name: str = Field(min_length=1)
```

### Why schemas matter

LLM-generated arguments are untrusted input.

Validation protects against: - Missing arguments - Incorrect types -
Invalid values - Unexpected input

Pipeline:

``` text
LLM arguments
      ↓
Pydantic validation
      ↓
Validated arguments
      ↓
Tool execution
```

------------------------------------------------------------------------

# 20. Tool Registry and Tool Router

A scalable registry can contain function + schema information.

Conceptually:

``` python
tool_registry = {
    "add_numbers": {
        "function": add_numbers,
        "input_schema": AddNumbersArgs,
        "output_schema": AddNumbersOutput,
    }
}
```

A tool router can: 1. Find the requested tool. 2. Validate inputs. 3.
Execute the tool. 4. Validate outputs. 5. Return structured errors.

This centralizes tool control.

------------------------------------------------------------------------

# 21. Parsing and Validating Tool Outputs

Tool input validation is only half the problem.

Tool outputs should also be validated.

Example:

``` python
class AddNumbersOutput(BaseModel):
    result: int

class MultiplyNumbersOutput(BaseModel):
    result: int

class UserInfoOutput(BaseModel):
    name: str
    role: str
    experience: str
```

Validation:

``` python
validated_output = AddNumbersOutput.model_validate(result)
```

Complete boundary:

``` text
LLM
 ↓
Tool arguments
 ↓
Input validation
 ↓
Tool
 ↓
Output validation
 ↓
LLM
```

### Success and failure outputs

A successful lookup might return:

``` json
{
  "name": "Harshit",
  "role": "AI Engineer",
  "experience": "1 year"
}
```

A failure might return:

``` json
{
  "error": "User not found"
}
```

These structures are different. Production systems should explicitly
model success and failure instead of treating an error object as a
successful output.

------------------------------------------------------------------------

# 22. Basic LLM Evaluation

An LLM application needs repeatable evaluation.

A prompt working once does not mean the system is production-ready.

Evaluation asks:

``` text
Did the application produce the expected result?
```

## Simple keyword evaluation

Example:

``` text
Expected:
["Python", "FastAPI"]

Generated:
"Candidate has Python and FastAPI experience."

Matched:
["Python", "FastAPI"]

Missing:
[]
```

Pass rate:

``` text
pass rate = passed tests / total tests
```

## Test cases

Keep reusable test cases:

``` python
test_cases = [
    {
        "prompt": "...",
        "expected_keywords": [...]
    }
]
```

This allows repeated evaluation after prompt/code changes.

------------------------------------------------------------------------

# 23. Structured Evaluation and Failure Analysis

For structured applications:

``` text
LLM response
    ↓
JSON parsing
    ↓
Schema validation
    ↓
Application-specific checks
```

Failures should identify the stage.

Possible categories:

``` text
json_parsing
schema_validation
rule_validation
semantic_failure
```

This is much more useful than simply reporting `FAIL`.

------------------------------------------------------------------------

# 24. LLM-as-a-Judge

LLM-as-a-Judge uses an LLM to evaluate another LLM's response.

Example dimensions:

``` text
Correctness:           1–5
Relevance:             1–5
Completeness:          1–5
Instruction following: 1–5
Explanation:           text
```

Example schema:

``` python
class EvaluationResult(BaseModel):
    correctness: int
    relevance: int
    completeness: int
    instruction_following: int
    explanation: str
```

The judge receives: - Question - Expected answer - Generated answer

and returns a structured evaluation.

### Limitation

LLM-as-a-Judge is **not ground truth**.

The judge can also make mistakes or introduce bias.

Therefore combine:

``` text
Schema validation
+
Rule-based checks
+
Semantic evaluation
+
LLM-as-a-Judge
+
Human evaluation when necessary
```

------------------------------------------------------------------------

# 25. Cost Awareness

LLM APIs generally charge based on usage, commonly involving input and
output tokens.

Important cost drivers: - Input tokens - Output tokens - Number of
requests - Tool calls - Model selection - Prompt size

Conceptually:

``` text
Total Cost
≈
Input Usage × Input Rate
+
Output Usage × Output Rate
```

Actual provider pricing is provider/model-specific and changes over
time. Use current official pricing for production decisions.

### Cost optimization

Reduce unnecessary: - Prompt text - Repeated context - Output length -
Tool calls - LLM calls

Choose the least expensive model that meets the quality requirement.

------------------------------------------------------------------------

# 26. Latency Awareness

Latency is the time required to complete an operation.

For an LLM application:

``` text
Request
 ↓
Network latency
 ↓
LLM processing
 ↓
Token generation
 ↓
Tool calls
 ↓
Final response
```

Useful latency metrics: - P50 --- median latency - P95 --- 95th
percentile latency - P99 --- 99th percentile latency

P95 is useful for understanding slower user requests.

### Measuring latency

``` python
import time

start = time.perf_counter()

response = client.models.generate_content(...)

elapsed = time.perf_counter() - start

print(f"Latency: {elapsed:.2f}s")
```

### Useful monitoring fields

``` text
request_id
model
input_tokens
output_tokens
latency
status
error_type
retry_count
tool_calls
```

------------------------------------------------------------------------

# 27. Production Reliability Architecture

All the chapter concepts connect:

``` text
                    ┌──────────────────┐
                    │      User        │
                    └────────┬─────────┘
                             ↓
                    ┌──────────────────┐
                    │   Application    │
                    └────────┬─────────┘
                             ↓
                    ┌──────────────────┐
                    │ Prompt Template  │
                    └────────┬─────────┘
                             ↓
                    ┌──────────────────┐
                    │    LLM Client    │
                    └────────┬─────────┘
                             ↓
                    ┌──────────────────┐
                    │   Hosted LLM     │
                    └────────┬─────────┘
                             ↓
                 ┌───────────┴───────────┐
                 ↓                       ↓
          Structured Output         Tool Call
                 ↓                       ↓
          JSON Parsing              Validation
                 ↓                       ↓
          Schema Validation         Tool Router
                 ↓                       ↓
          Application Logic          Tool
                 │                       ↓
                 │                  Tool Output
                 │                       ↓
                 │                  Validation
                 └───────────┬───────────┘
                             ↓
                       Final Response
```

Across the system:

``` text
Secrets
Retries
Timeouts
Rate Limits
Evaluation
Cost Monitoring
Latency Monitoring
```

must also be handled.

------------------------------------------------------------------------

# 28. Failure Handling

Failures should be treated as expected engineering conditions.

Possible failure layers:

``` text
1. Configuration failure
2. Authentication failure
3. Network failure
4. Timeout
5. Rate limit
6. Provider/server failure
7. Invalid model/request
8. Invalid JSON
9. Schema validation failure
10. Tool input failure
11. Tool execution failure
12. Tool output validation failure
13. Evaluation failure
```

Good pattern:

``` text
External failure
      ↓
Detect
      ↓
Classify
      ↓
Retry if appropriate
      ↓
Validate
      ↓
Return controlled error
      ↓
Log / Monitor
```

------------------------------------------------------------------------

# 29. Engineering Principles Learned

### Principle 1 --- Treat LLM output as untrusted

Always validate generated data.

### Principle 2 --- Separate instructions from data

Use structured prompts and templates.

### Principle 3 --- Prefer structured outputs

JSON + schema validation is safer than arbitrary prose parsing.

### Principle 4 --- Tools must be controlled

Only predefined tools should be executable.

### Principle 5 --- Validate tool inputs and outputs

Both sides of a tool boundary matter.

### Principle 6 --- External APIs fail

Design for timeouts, retries, rate limits and server failures.

### Principle 7 --- Do not retry everything

Classify errors before retrying.

### Principle 8 --- Evaluate continuously

Prompt/code changes can change model behavior.

### Principle 9 --- LLM-as-a-Judge is not ground truth

Use multiple evaluation techniques.

### Principle 10 --- Monitor cost and latency

A system that works but is too expensive or slow is not
production-ready.

### Principle 11 --- Keep providers abstracted

A provider-independent interface makes testing and switching easier.

------------------------------------------------------------------------

# 30. Roadmap Alignment

The official roadmap lists **LLM Engineering --- 2 weeks**, with the
objective of building reliable applications around hosted LLMs.

Required topics:

-   [x] Prompt engineering principles
-   [x] System/user/developer instructions
-   [x] Few-shot prompting
-   [x] Structured/JSON outputs
-   [x] Prompt templates
-   [x] Temperature and generation controls
-   [x] OpenAI API
-   [x] Gemini API
-   [x] API keys and secret management
-   [x] Retries, timeouts and rate limits
-   [x] Tool/function calling
-   [x] Tool schemas
-   [x] Parsing and validating tool outputs
-   [x] Basic LLM evaluation
-   [x] Cost and latency awareness

The roadmap then specifies: - Resume Analyzer - Email Generator

Project deliverables: - Structured outputs - Prompt tests - Failure
handling - FastAPI integration where useful - Docker + deployment - Live
URL + GitHub README for each project

The roadmap completion checkpoint is: - Build reliable LLM API
integrations - Validate structured output - Implement tool calling -
Handle API failures - Deploy both projects

**Roadmap source:** `AI_GenAI_Engineer_Detailed_Roadmap_2026(3).docx`.

------------------------------------------------------------------------

# 31. Project Phase --- What We Do Next

The theory phase is complete.

## Project 1: Resume Analyzer

We will build it incrementally:

``` text
1. Requirements
        ↓
2. Folder structure
        ↓
3. Input / output contract
        ↓
4. Pydantic schemas
        ↓
5. Prompt design
        ↓
6. LLM client abstraction
        ↓
7. Resume analysis logic
        ↓
8. Prompt tests
        ↓
9. Failure handling
        ↓
10. FastAPI integration
        ↓
11. Docker
        ↓
12. Deployment
        ↓
13. Live URL
        ↓
14. GitHub README
```

Then we will build the **Email Generator** using the same engineering
principles.

------------------------------------------------------------------------

# 32. Quick Revision --- Mental Model

``` text
LLM ENGINEERING
│
├── Prompt
│   ├── Role
│   ├── Task
│   ├── Context
│   ├── Constraints
│   ├── Output format
│   └── Few-shot examples
│
├── Generation
│   ├── Temperature
│   └── Generation controls
│
├── Output
│   ├── JSON
│   ├── Parsing
│   └── Pydantic validation
│
├── Providers
│   ├── OpenAI
│   ├── Gemini
│   └── Provider abstraction
│
├── Reliability
│   ├── Retries
│   ├── Backoff
│   ├── Jitter
│   ├── Timeouts
│   └── Rate limits
│
├── Tools
│   ├── Function calling
│   ├── Tool registry
│   ├── Input schemas
│   ├── Input validation
│   ├── Execution
│   ├── Output schemas
│   └── Output validation
│
├── Evaluation
│   ├── Test cases
│   ├── Rule checks
│   ├── Structured validation
│   ├── LLM-as-a-Judge
│   └── Failure analysis
│
└── Production
    ├── Secrets
    ├── Cost
    ├── Latency
    ├── Monitoring
    ├── FastAPI
    ├── Docker
    └── Deployment
```

------------------------------------------------------------------------

# 33. Final Takeaway

The central idea of LLM Engineering is:

> **Do not simply call an LLM. Engineer a reliable system around it.**

A strong LLM application combines:

``` text
Clear Prompt
    +
Structured Output
    +
Validation
    +
Controlled Tools
    +
Failure Handling
    +
Evaluation
    +
Cost Awareness
    +
Latency Awareness
    =
Reliable LLM Application
```

This is the foundation for the Resume Analyzer and Email Generator
projects.
