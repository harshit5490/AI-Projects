import os
import json

from dotenv import load_dotenv
from google import genai
from google.genai import types

from tools import (
    add_numbers,
    multiply_numbers,
    get_user_info,
)


# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY is not set")


# ============================================================
# 2. CREATE GEMINI CLIENT
# ============================================================

client = genai.Client(
    api_key=api_key
)


# ============================================================
# 3. TOOL REGISTRY
# ============================================================

tool_registry = {
    "add_numbers": add_numbers,
    "multiply_numbers": multiply_numbers,
    "get_user_info": get_user_info,
}


# ============================================================
# 4. DEFINE AVAILABLE TOOLS
# ============================================================

tools = [
    add_numbers,
    multiply_numbers,
    get_user_info,
]


# ============================================================
# 5. GEMINI CONFIGURATION
# ============================================================

config = types.GenerateContentConfig(
    tools=tools
)


# ============================================================
# 6. USER REQUEST
# ============================================================

user_prompt = "What is 25 multiplied by 8?"


# ============================================================
# 7. FIRST REQUEST
# ============================================================

response = client.models.generate_content(
    model="gemini-3.5-flash",
    contents=user_prompt,
    config=config,
)


print("=" * 60)
print("USER REQUEST")
print("=" * 60)

print(user_prompt)


# ============================================================
# 8. CHECK WHETHER GEMINI REQUESTED A TOOL
# ============================================================

tool_parts = []

for candidate in response.candidates:

    if not candidate.content:
        continue

    for part in candidate.content.parts:

        if part.function_call:
            tool_parts.append(part)


# ============================================================
# 9. IF NO TOOL IS REQUIRED
# ============================================================

if not tool_parts:

    print("\n" + "=" * 60)
    print("FINAL ANSWER")
    print("=" * 60)

    print(response.text)


# ============================================================
# 10. EXECUTE REQUESTED TOOLS
# ============================================================

else:

    tool_responses = []

    for part in tool_parts:

        function_call = part.function_call

        function_name = function_call.name
        function_args = dict(function_call.args)

        print("\n" + "=" * 60)
        print("TOOL CALL")
        print("=" * 60)

        print("Function:", function_name)
        print("Arguments:", function_args)


        # ----------------------------------------------------
        # Find the function in the registry
        # ----------------------------------------------------

        function = tool_registry.get(function_name)

        if function is None:

            tool_result = {
                "error": f"Unknown tool: {function_name}"
            }

        else:

            try:

                # ------------------------------------------------
                # Execute Python function
                # ------------------------------------------------

                result = function(**function_args)

                tool_result = {
                    "result": result
                }

            except Exception as error:

                tool_result = {
                    "error": str(error)
                }


        # ----------------------------------------------------
        # Display tool result
        # ----------------------------------------------------

        print("\nTool Result:")

        print(
            json.dumps(
                tool_result,
                indent=2
            )
        )


        # ----------------------------------------------------
        # Prepare result for Gemini
        # ----------------------------------------------------

        tool_responses.append(
            types.Part.from_function_response(
                name=function_name,
                response=tool_result,
            )
        )


    # ========================================================
    # 11. SEND TOOL RESULT BACK TO GEMINI
    # ========================================================

    follow_up_response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=[
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(
                        text=user_prompt
                    )
                ],
            ),
            response.candidates[0].content,
            types.Content(
                role="tool",
                parts=tool_responses,
            ),
        ],
        config=config,
    )


    # ========================================================
    # 12. FINAL ANSWER
    # ========================================================

    print("\n" + "=" * 60)
    print("FINAL ANSWER")
    print("=" * 60)

    print(follow_up_response.text)