from dataclasses import dataclass, field
from typing import Any
from enum import Enum


class MessageType(Enum):
    SYSTEM = "SYSTEM"
    USER = "USER"
    ASSISTANT = "ASSISTANT"


@dataclass
class Message:
    message_type: MessageType
    text: str


@dataclass
class PromptData:
    # Chat-like list of messages making up the current prompt (if any).
    messages: list[Message] | None

    # The unique identifier for this prompt within a prompt group.
    prompt_id: int

    # The unique ID of the prompt group that the current prompt belongs to.
    group_id: int

    # Any additional relevant data.
    additional_data: dict[str, Any] = field(default_factory=dict)


class IDGenerator:
    def __init__(self):
        self.prompt_id = self.group_id = 0

    def next_prompt_id(self, no_increment: bool = False) -> int:
        value = self.prompt_id
        if not no_increment:
            self.prompt_id += 1
        return value

    def next_group_id(self, no_increment: bool = False) -> int:
        value = self.group_id
        if not no_increment:
            self.group_id += 1
            self.prompt_id = 0
        return value
