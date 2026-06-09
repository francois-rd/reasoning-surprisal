from dataclasses import dataclass, field
from random import Random
import re

from ..core import Logprob, Logprobs, RankedLogprob

from .base import LLM, LLMOutput, Nickname
from .prompting import PromptData


@dataclass
class OutputOption:
    text: str  # The text to output.
    weight: int  # The weighting of this text relative to others.


@dataclass
class DummyConfig:
    # Distribution over output options.
    output_distribution: list[OutputOption] = field(default_factory=list)
    seed: int = 42  # Random seed for distribution sampling.


class DummyLLM(LLM):
    pattern = re.compile(r"(\w+\s*)|(\W+)")

    def __init__(
        self,
        nickname: Nickname,
        llm_cfg: DummyConfig,
        *args,
        trim_indicator: str | None = None,
        **kwargs,
    ):
        super().__init__(nickname, *args, **kwargs)
        self.cfg = llm_cfg
        self.indicator = trim_indicator
        self.rng = Random(self.cfg.seed)
        self.pop = [o.text for o in self.cfg.output_distribution]
        self.weights = [o.weight for o in self.cfg.output_distribution]

    def invoke(
        self,
        prompt_data: PromptData,
        *args,
        add_logprobs: bool = False,
        add_prompt_logprobs: bool = False,
        **kwargs,
    ) -> LLMOutput:
        text = self.rng.choices(self.pop, self.weights)[0]
        prompt_data.additional_data.get("generated_text", text)
        message = prompt_data.additional_data.get("error_message", None)
        data = prompt_data.additional_data.get("derived_data", {})
        if add_logprobs and add_prompt_logprobs:
            data["logprobs"] = self._fake_logprobs(text)
            data["prompt_logprobs"] = self._fake_logprobs(prompt_data.messages[1].text)
        elif add_logprobs and not add_prompt_logprobs:
            data["logprobs"] = self._fake_logprobs(text)
        elif not add_logprobs and add_prompt_logprobs:
            full_text = prompt_data.messages[1].text + "\n" + text
            data["prompt_logprobs"] = self._fake_logprobs(full_text)
            quit()
        else:
            pass  # Add nothing.
        return LLMOutput(generated_text=text, error_message=message, derived_data=data)

    def _fake_logprobs(self, text: str) -> Logprobs:
        logprobs = []
        for first, second in self.pattern.findall(text):
            if len(first) == len(second) == 0:
                raise ValueError(f"Bad regex: Empty split into tokens: {text}")
            elif len(first) != 0 and len(second) != 0:
                raise ValueError(f"Bad regex: Multi split into tokens: {text}")
            elif len(first) == 0:
                token = second
            else:
                token = first
            logprob = RankedLogprob(
                chosen=Logprob(token=token, rank=2, logprob=-1),
                others={},
                ranking="absolute",
            )
            logprobs.append(logprob)
        logprobs = Logprobs(sequence=logprobs)
        logprobs.maybe_trim(self.indicator, raise_if_not_unique=True)
        return logprobs
