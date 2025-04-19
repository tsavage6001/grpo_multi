from abc import ABC, abstractmethod
from typing import Any, Dict, List, Sequence, Callable
import logging

from datasets import Dataset

from verifiers import RewardFunc
from ..imports import LLM, SamplingParams  # type: ignore

class Environment(ABC):

    def __init__(self, **kwargs: Any):
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.logger = logging.getLogger(f"verifiers.envs.{self.__class__.__name__}")
        self.tokenizer = None
        self.dataset = None
        self.eval_dataset = None
        self.eot_id = 151643
        self.message_end_id = 151645
        self.reward_funcs = []
        self.reward_weights = []

    @abstractmethod
    def get_dataset(self, **kwargs: Any) -> Dataset | None:
        pass

    @abstractmethod
    def get_eval_dataset(self, **kwargs: Any) -> Dataset | None:
        pass

    @abstractmethod
    def get_reward_funcs(self, **kwargs: Any) -> List[RewardFunc]:
        pass

    @abstractmethod
    def get_reward_weights(self, **kwargs: Any) -> List[float]:
        pass

    def generate_conversations(
        self,
        prompts: List[List[Dict[str, Any]]],
        llm: LLM,
        sampling_params: SamplingParams,
        frozen_user: LLM,
        frozen_user_sampling_params: SamplingParams,
        num_generations: int = 1,
        num_turns: int = 20,
    ) -> List[List[Dict[str, str]]]:
        """
        Simulates multi-turn conversations between the assistant (llm) and a frozen user model.
        """
        from copy import deepcopy

        all_conversations = []
        for init_prompt in prompts:
            for _ in range(num_generations):
                conversation = deepcopy(init_prompt)
                for t in range(num_turns):
                    # Assistant (trainable model)
                    assistant_prompt = self.tokenizer.apply_chat_template(conversation, tokenize=False)
                    assistant_response = llm.generate_one(assistant_prompt, sampling_params)
                    conversation.append({"role": "assistant", "content": assistant_response})

                    # Frozen user (patient)
                    user_prompt = self.tokenizer.apply_chat_template(conversation, tokenize=False)
                    user_response = frozen_user.generate_one(user_prompt, frozen_user_sampling_params)
                    conversation.append({"role": "user", "content": user_response})

                all_conversations.append(conversation)

        return all_conversations
