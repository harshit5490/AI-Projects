import re
from bs4 import BeautifulSoup


def clean_text(text: str) -> str:
    """
    Clean a review while preserving sentiment-relevant information.
    """

    # Remove HTML
    text = BeautifulSoup(text, "html.parser").get_text(" ")

    # Remove URLs
    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text
    )

    # Normalize whitespace
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text