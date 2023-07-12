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
    unwrap_model,
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

        for answer in ["rejected", "chosen"]:
            if padding:
                batch = batch_padding(
                    self.cfg,
                    batch,
                    self.training,
                    mask_key=f"{answer}_attention_mask",
                    pad_keys=[
                        f"{answer}_input_ids",
                        f"{answer}_attention_mask",
                        f"{answer}_labels",
                    ],
                )
            logits = self.backbone(
                input_ids=batch[f"{answer}_input_ids"],
                attention_mask=batch[f"{answer}_attention_mask"],
            ).logits
            outputs[f"{answer}_logps"] = self.get_batch_logps(logits, batch[f"{answer}_labels"])

            with self.backbone.disable_adapter():
                reference_logits = self.backbone(
                    input_ids=batch[f"{answer}_input_ids"],
                    attention_mask=batch[f"{answer}_attention_mask"],
                ).logits
                outputs[f"{answer}_reference_logps"] = self.get_batch_logps(reference_logits,
                                                                            batch[f"{answer}_labels"])

        loss = self.loss_fn(policy_chosen_logps=outputs[f"chosen_logps"],
                            policy_rejected_logps=outputs[f"rejected_logps"],
                            reference_chosen_logps=outputs[f"chosen_reference_logps"],
                            reference_rejected_logps=outputs[f"rejected_reference_logps"],
                            beta=0.1)
        outputs["loss"] = loss

        # enable cache again if gradient checkpointing is enabled
        if self.cfg.architecture.gradient_checkpointing:
            self.backbone.config.use_cache = True

        return outputs

    def get_batch_logps(self,
                        logits: torch.FloatTensor, labels: torch.LongTensor, average_log_prob: bool = False
                        ) -> torch.FloatTensor:
        # Implementation from https://github.com/eric-mitchell/direct-preference-optimization
        """Compute the log probabilities of the given labels under the given logits.

        Args:
            logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
            labels: Labels for which to compute the log probabilities. Label tokens with a value of -100 are ignored. Shape: (batch_size, sequence_length)
            average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.

        Returns:
            A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
        """
        assert logits.shape[:-1] == labels.shape

        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]
        loss_mask = labels != -100

        # dummy token; we'll ignore the losses on these tokens later
        labels[labels == -100] = 0

        per_token_logps = torch.gather(
            logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)
        ).squeeze(2)

        if average_log_prob:
            return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
        else:
            return (per_token_logps * loss_mask).sum(-1)
