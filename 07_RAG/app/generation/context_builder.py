
from app.schemas.retrieval import RetrievedChunk


class ContextBuilder:
    """Builds bounded context from retrieved document chunks."""

    def build(
        self,
        chunks: list[RetrievedChunk | dict],
        max_context_chars: int = 12_000,
    ) -> str:
        if max_context_chars <= 0:
            raise ValueError(
                "max_context_chars must be greater than 0."
            )

        sections: list[str] = []
        used_chars = 0

        for item in chunks:
            chunk = (
                item
                if isinstance(item, RetrievedChunk)
                else RetrievedChunk.model_validate(item)
            )

            page_label = (
                str(chunk.page)
                if chunk.page is not None
                else "N/A"
            )

            section = (
                f"[Source: {chunk.filename} | "
                f"Page: {page_label} | "
                f"Chunk: {chunk.chunk_index}]\n"
                f"{chunk.text.strip()}"
            )

            separator = "\n\n"
            extra_chars = len(section) + (
                len(separator) if sections else 0
            )
            remaining = max_context_chars - used_chars

            if extra_chars <= remaining:
                sections.append(section)
                used_chars += extra_chars
                continue

            if remaining > 0:
                prefix = separator if sections else ""
                available = (
                    max_context_chars
                    - used_chars
                    - len(prefix)
                )

                if available > 0:
                    partial_section = section[:available]

                    if partial_section.strip():
                        sections.append(partial_section)
            break

        return "\n\n".join(sections)
