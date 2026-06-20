from typing import Any


class OutputParser:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, generated_text: str, *args, **kwargs) -> Any | None:
        """
        Parses the generated_text of an LLM to extract some meaningful result.
        Returns None on parsing failure.
        """
        raise NotImplementedError


class NoOutputParser(OutputParser):
    def __init__(self, check_string_type: bool = True):
        """
        Returns 'generated_text' directly without performing any parsing.

        :param check_string_type: If True, checks that LLM's generated text is of
            type 'str'. This can be useful for catching errors that cause LLM to
            instead return None, for example.
        """
        super().__init__()
        self.check = check_string_type

    def __call__(self, generated_text: str, *args, **kwargs) -> str:
        if self.check and not isinstance(generated_text, str):
            raise ValueError(
                f"'{type(generated_text)}' instead of type 'str': {generated_text}"
            )
        return generated_text
