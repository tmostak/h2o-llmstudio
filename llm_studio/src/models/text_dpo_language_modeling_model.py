import logging
from typing import Any, Dict

import torch
from torch import nn
from transformers import AutoModelForCausalLM

from llm_studio.src.metrics.text_causal_language_modeling_metrics import Perplexity
from llm_studio.src.utils.data_utils import batch_padding
from llm_studio.src.utils.modeling_utils import (
    create_nlp_backbone,
    generate_text,
    prepare_lora,
)

logger = logging.getLogger(__name__)


class Model(nn.Module):
    """
    Model for DPO language modeling problem type.
    """

    def __init__(self, cfg: Any):
        """
        Args:
            cfg: config with all the hyperparameters
        """

        super(Model, self).__init__()

        self.cfg = cfg
        self.backbone, self.backbone_config = create_nlp_backbone(
            cfg, model_class=AutoModelForCausalLM
        )

        if cfg.training.lora:
            self.backbone = prepare_lora(cfg=cfg, backbone=self.backbone)

        self.loss_fn = self.cfg.training.loss_class.get(
            self.cfg.training.loss_function
        )(self.cfg)
        if self.cfg.prediction.metric == "Perplexity":
            self.perplexity = Perplexity(self.cfg, reduce=False)

    def generate(self, batch: Dict, cfg: Any, streamer=None):
        return generate_text(self.backbone, batch, cfg, streamer, self.training)

    def forward(
        self,
        batch: Dict,
        padding: bool = True,
    ) -> Dict:
        # disable cache if gradient checkpointing is enabled
        if self.cfg.architecture.gradient_checkpointing:
            self.backbone.config.use_cache = False

        outputs: Dict = {}

        for answer in ["rejected", ""]

        if padding:
            batch = batch_padding(
                self.cfg,
                batch,
                self.training,
                mask_key="attention_mask",
                pad_keys=[
                    "input_ids",
                    "attention_mask",
                    "special_tokens_mask",
                    "labels",
                ],
            )

        output = self.backbone(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )

        if "labels" in batch:
            loss = self.loss_fn(output.logits, batch["labels"])
            outputs["loss"] = loss

        if self.cfg.prediction.metric == "Perplexity":
            outputs["perplexity"] = self.perplexity(output.logits, batch["labels"])

        # enable cache again if gradient checkpointing is enabled
        if self.cfg.architecture.gradient_checkpointing:
            self.backbone.config.use_cache = True

        return outputs
