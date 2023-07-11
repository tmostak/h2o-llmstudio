import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from llm_studio.src.datasets.text_causal_language_modeling_ds import (
    CustomDataset as LLMCustomDataset,
)

logger = logging.getLogger(__name__)


class CustomDataset(LLMCustomDataset):
    """
    Dataset for DPO optimization.
    The data is assumed to be in hierarchical form of the following format:

    Beginning of a chat-answer interaction (parent_id is not set):
        instruction                    What kind of noises did dinosaurs make?
        output               Humans and dinosaurs didn’t live at the same t...
        id                                610e4ad5-09c4-4055-9ff4-948fe6b4f832
        parent_id                                                         None
        chosen_response                                                   None
        rejected_response                                                 None

    Within a chat-answer interaction (parent_id points for the previous prompt-answer sample):
        instruction                                               yes they did
        output               to guess, and that would probably require lots...
        id                                573e8d77-550a-4889-8ff4-1e8d8944897c
        parent_id                         610e4ad5-09c4-4055-9ff4-948fe6b4f832
        chosen_response                                                   None
        rejected_response                                                 None


    Last question. Output should be empty, chosen and rejected responses should be given:
        instruction          Do have a phone number or email address for hi...
        output
        id                                e0edeaf1-166d-4683-8609-dcba6fafc520
        parent_id                         e7e96d54-006d-4b34-a9ed-479c3ec3068c
        chosen_response       He doesn’t have a publicly available phone nu...
        rejected_response     If you want to contact Ryan Reynolds by phone...
    """

    def __init__(self, df: pd.DataFrame, cfg: Any, mode: str = "train"):
        """
        Args:
            df: input DataFrame
            cfg: config with all the hyperparameters
            mode: dataset mode. One of {"train", "validation"}
        """
        assert (
            cfg.dataset.limit_chained_samples
        ), "Need to enable limit_chained_samples for dpo training"

        super().__init__(df=df, cfg=cfg, mode=mode)
        self.chosen_answers = (
            self.df[self.cfg.dataset.chosen_response_column].astype(str).values.tolist()
        )
        self.rejected_answer = (
            self.df[self.cfg.dataset.rejected_response_column]
            .astype(str)
            .values.tolist()
        )

    def __getitem__(self, idx: int) -> Dict:
        """Reads a single text observation."""
        sample = super().__getitem__(idx)
        sample.pop("reward_model_prompt_text", None)
        sample.pop("reward_model_prompt_text", None)

        idx = self.indices[idx]
        sample["chosen_answer_ids"] = self.encode(
            self.tokenizer,
            text=self.chosen_answers[idx],
            max_length=self.cfg.tokenizer.max_length_answer,
            truncation_side="right",
        )["input_ids"]
        sample["rejected_answer_ids"] = self.encode(
            self.tokenizer,
            text=self.rejected_answer[idx],
            max_length=self.cfg.tokenizer.max_length_answer,
            truncation_side="right",
        )["input_ids"]
        return sample
