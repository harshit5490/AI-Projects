from abc import ABC, abstractmethod


class BaseDocumentParser(ABC):

    @abstractmethod
    def parse(self, file_path: str) -> str:
        """Parse a document and return its text."""
        pass