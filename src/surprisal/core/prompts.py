from dataclasses import dataclass

SystemPromptID = str
UserTemplateID = str
SystemPrompt = str


@dataclass
class SystemPromptsConfig:
    prompts: dict[SystemPromptID, SystemPrompt]


@dataclass
class UserTemplate:
    template: str
    indicator: str


@dataclass
class UserTemplatesConfig:
    templates: dict[UserTemplateID, UserTemplate]
