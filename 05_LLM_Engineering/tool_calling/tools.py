def add_numbers(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


def multiply_numbers(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


def get_user_info(name: str) -> dict:
    """Return basic information about a user."""

    users = {
        "harshit": {
            "name": "Harshit",
            "role": "AI Engineer",
            "experience": "1 year"
        },
        "rahul": {
            "name": "Rahul",
            "role": "Software Engineer",
            "experience": "2 years"
        }
    }

    return users.get(
        name.lower(),
        {
            "error": f"User '{name}' not found."
        }
    )