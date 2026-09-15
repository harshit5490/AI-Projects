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


# ============================================================
# TOOL REGISTRY
# ============================================================

TOOL_REGISTRY = {
    "add_numbers": {
        "function": add_numbers,
        "schema": AddNumbersArgs,
    },

    "multiply_numbers": {
        "function": multiply_numbers,
        "schema": MultiplyNumbersArgs,
    },

    "get_user_info": {
        "function": get_user_info,
        "schema": GetUserInfoArgs,
    },
}


# ============================================================
# TOOL EXECUTION
# ============================================================

def execute_tool(
    tool_name: str,
    arguments: dict
):

    # --------------------------------------------------------
    # Check whether tool exists
    # --------------------------------------------------------

    tool = TOOL_REGISTRY.get(tool_name)

    if tool is None:

        return {
            "success": False,
            "error": f"Unknown tool: {tool_name}"
        }


    function = tool["function"]
    schema = tool["schema"]


    # --------------------------------------------------------
    # Validate arguments
    # --------------------------------------------------------

    try:

        validated_args = schema.model_validate(
            arguments
        )

    except ValidationError as error:

        return {
            "success": False,
            "error": "Invalid tool arguments",
            "details": error.errors(),
        }


    # --------------------------------------------------------
    # Execute tool
    # --------------------------------------------------------

    try:

        result = function(
            **validated_args.model_dump()
        )

        return {
            "success": True,
            "result": result,
        }

    except Exception as error:

        return {
            "success": False,
            "error": "Tool execution failed",
            "details": str(error),
        }