from pydantic import ValidationError

from tools import (
    add_numbers,
    multiply_numbers,
    get_user_info,
)

from tool_schemas import (
    AddNumbersArgs,
    MultiplyNumbersArgs,
    GetUserInfoArgs,
)

from tool_output_schemas import (
    AddNumbersOutput,
    MultiplyNumbersOutput,
    UserInfoOutput,
)


# ============================================================
# TOOL REGISTRY
# ============================================================

TOOL_REGISTRY = {

    "add_numbers": {
        "function": add_numbers,
        "input_schema": AddNumbersArgs,
        "output_schema": AddNumbersOutput,
    },

    "multiply_numbers": {
        "function": multiply_numbers,
        "input_schema": MultiplyNumbersArgs,
        "output_schema": MultiplyNumbersOutput,
    },

    "get_user_info": {
        "function": get_user_info,
        "input_schema": GetUserInfoArgs,
        "output_schema": UserInfoOutput,
    },
}


# ============================================================
# EXECUTE TOOL
# ============================================================

def execute_tool(
    tool_name: str,
    arguments: dict
):

    # --------------------------------------------------------
    # 1. Check tool
    # --------------------------------------------------------

    tool = TOOL_REGISTRY.get(tool_name)

    if tool is None:

        return {
            "success": False,
            "error": f"Unknown tool: {tool_name}",
        }


    function = tool["function"]
    input_schema = tool["input_schema"]
    output_schema = tool["output_schema"]


    # --------------------------------------------------------
    # 2. Validate INPUT
    # --------------------------------------------------------

    try:

        validated_args = input_schema.model_validate(
            arguments
        )

    except ValidationError as error:

        return {
            "success": False,
            "error": "Invalid tool arguments",
            "details": error.errors(),
        }


    # --------------------------------------------------------
    # 3. Execute TOOL
    # --------------------------------------------------------

    try:

        raw_result = function(
            **validated_args.model_dump()
        )

    except Exception as error:

        return {
            "success": False,
            "error": "Tool execution failed",
            "details": str(error),
        }


    # --------------------------------------------------------
    # 4. Validate OUTPUT
    # --------------------------------------------------------

    try:

        validated_output = output_schema.model_validate(
            raw_result
            if isinstance(raw_result, dict)
            else {"result": raw_result}
        )

    except ValidationError as error:

        return {
            "success": False,
            "error": "Invalid tool output",
            "details": error.errors(),
        }


    # --------------------------------------------------------
    # 5. Return validated result
    # --------------------------------------------------------

    return {
        "success": True,
        "result": validated_output.model_dump(),
    }