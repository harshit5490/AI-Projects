class RecursiveCharacterChunker:

    def __init__(
        self,
        chunk_size: int,
        chunk_overlap: int = 0,
        separators: list[str] | None = None,
    ):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")

        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")

        if chunk_overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap must be smaller than chunk_size"
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.separators = separators or [
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ]

    def split(self, text: str) -> list[str]:
        if not text.strip():
            return []

        chunks = self._split_recursive(
            text=text,
            separators=self.separators,
        )

        return self._apply_overlap(chunks)

    def _split_recursive(
        self,
        text: str,
        separators: list[str],
    ) -> list[str]:

        text = text.strip()

        if len(text) <= self.chunk_size:
            return [text]

        if not separators:
            return self._split_by_size(text)

        separator = separators[0]

        if separator:
            pieces = text.split(separator)
        else:
            pieces = list(text)

        chunks = []
        current_parts = []
        current_length = 0

        for piece in pieces:

            piece = piece.strip()

            if not piece:
                continue

            separator_length = (
                len(separator)
                if current_parts
                else 0
            )

            candidate_length = (
                current_length
                + separator_length
                + len(piece)
            )

            if candidate_length <= self.chunk_size:

                current_parts.append(piece)

                current_length = candidate_length

                continue

            if current_parts:

                chunks.append(
                    separator.join(current_parts).strip()
                )

                current_parts = []
                current_length = 0

            if len(piece) > self.chunk_size:

                sub_chunks = self._split_recursive(
                    text=piece,
                    separators=separators[1:],
                )

                chunks.extend(sub_chunks)

            else:

                current_parts = [piece]
                current_length = len(piece)

        if current_parts:

            chunks.append(
                separator.join(current_parts).strip()
            )

        return chunks

    def _split_by_size(self, text: str) -> list[str]:

        return [
            text[start:start + self.chunk_size].strip()
            for start in range(
                0,
                len(text),
                self.chunk_size,
            )
            if text[start:start + self.chunk_size].strip()
        ]

    def _apply_overlap(
        self,
        chunks: list[str],
    ) -> list[str]:

        if self.chunk_overlap == 0:
            return chunks

        overlapped_chunks = [chunks[0]]

        for index in range(1, len(chunks)):

            previous_chunk = chunks[index - 1]
            current_chunk = chunks[index]

            available_space = (
                self.chunk_size
                - len(current_chunk)
                - 1
            )

            if available_space <= 0:
                overlapped_chunks.append(current_chunk)
                continue

            overlap_size = min(
                self.chunk_overlap,
                available_space,
            )

            overlap = previous_chunk[-overlap_size:]

            # Move the overlap to a word boundary.
            if " " in overlap:
                overlap = overlap.split(" ", 1)[1]

            combined = (
                overlap
                + " "
                + current_chunk
            ).strip()

            overlapped_chunks.append(combined)

        return overlapped_chunks