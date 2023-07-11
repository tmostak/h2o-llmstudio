import gc
import logging
import os
import re
import shutil
from collections import OrderedDict
from typing import Any, Dict

import torch
from peft import LoraConfig, get_peft_model
from peft.utils import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullyShardedDataParallel,
    MixedPrecision,
)
from torch.nn.parallel import DistributedDataParallel
from transformers import (
    AutoConfig,
    AutoModel,
    BitsAndBytesConfig,
    GenerationMixin,
    StoppingCriteria,
    StoppingCriteriaList,
)
from transformers.utils import logging as transformers_logging

from llm_studio.src.datasets.text_utils import get_tokenizer
from llm_studio.src.utils.data_utils import batch_padding

logger = logging.getLogger(__name__)


def unwrap_model(model: torch.nn.Module):
    options = (torch.nn.parallel.DistributedDataParallel, torch.nn.DataParallel)

    while isinstance(model, options):
        model = model.module

    return model


def check_disk_space_for_model_saving(model: torch.nn.Module, path: str):
    total, used, free = shutil.disk_usage(path)

    model_size_in_bytes = 0
    for param in model.parameters():
        if param.data.dtype in [torch.int8, torch.uint8]:
            model_size_in_bytes += param.numel() * 1
        elif param.data.dtype in [torch.float16, torch.bfloat16]:
            model_size_in_bytes += param.numel() * 2
        elif param.data.dtype == torch.float32:
            model_size_in_bytes += param.numel() * 4
        else:
            # If the data type is not supported, calculate it as float32.
            model_size_in_bytes += param.numel() * 4
            logger.warning(f"Unsupported data type: {param.data.dtype}")

    if model_size_in_bytes * 1.03 < free:  # leave a 3% margin here.
        logger.info("Enough space available for saving model weights.")
    else:
        raise ValueError(
            f"Not enough space available for saving model weights. "
            f"Required space: {model_size_in_bytes * 1.03 / (1024 * 1024):.2f}MB, "
            f"Available space: {free / (1024 * 1024):.2f}MB."
        )


def load_model_weights(
    model: torch.nn.Module, model_weights: Dict, strict: bool, cfg: Any
):
    orig_num_items = len(model_weights)
    model_state_dict = model.state_dict()

    # needed to load models trained in int4/int8 with other dtypes
    model_weights = {
        k: v
        if not (
            cfg.architecture.backbone_dtype not in ("int4", "int8")
            and (v.dtype is torch.int8 or v.dtype is torch.uint8)
        )
        else model_state_dict[k]
        for k, v in model_weights.items()
        if not (
            ("SCB" in k or "weight_format" in k)
            and cfg.architecture.backbone_dtype not in ("int4", "int8")
        )
    }

    # Need to ignore int4/int8 weights so undo strict loading requirement
    if len(model_weights) != orig_num_items:
        strict = False

    model_weights = {re.sub(r"^module\.", "", k): v for k, v in model_weights.items()}
    model_weights = {k.replace("_orig_mod.", ""): v for k, v in model_weights.items()}

    # manual fix for int8 weights
    if cfg.architecture.backbone_dtype == "int8":
        model_weights = {
            k: v.to(cfg.environment._device) if "weight_format" not in k else v
            for k, v in model_weights.items()
        }

    try:
        model.load_state_dict(OrderedDict(model_weights), strict=True)
    except Exception as e:
        if strict:
            raise e
        else:
            if cfg.environment._local_rank == 0:
                logger.warning(
                    "Only a part of the pretrained weights was loaded. "
                    "Some layers can't be initialized with pretrained "
                    f"weights: {e}"
                )

            for layer_name in re.findall("size mismatch for (.*?):", str(e)):
                model_weights.pop(layer_name, None)
            model.load_state_dict(OrderedDict(model_weights), strict=False)
    return model


def load_checkpoint(
    cfg: Any, model: torch.nn.Module, strict: bool = True, weights_path: str = None
):
    """Load checkpoint

    Args:
        cfg: config file
        model: model to load weights to
        strict: whether to apply strict matching for weights
        weights_path: custom path to the weights.
            If None, cfg.architecture.pretrained_weights is used
    Returns:
        epoch: current epoch
    """

    if weights_path is None:
        weights_path = cfg.architecture.pretrained_weights

    model_weights = torch.load(weights_path, map_location="cpu")["model"]

    model = load_model_weights(model, model_weights, strict, cfg)

    del model_weights
    gc.collect()

    if cfg.environment._local_rank == 0:
        logger.info(f"Weights loaded from: {weights_path}")


def wrap_model_distributed(model: torch.nn.Module, cfg: Any, fsdp: bool):
    if fsdp:
        auto_wrap_policy = None

        mixed_precision_policy = None
        dtype = None
        if cfg.environment.mixed_precision:
            dtype = torch.float16
        if dtype is not None:
            mixed_precision_policy = MixedPrecision(
                param_dtype=dtype, reduce_dtype=dtype, buffer_dtype=dtype
            )
        model = FullyShardedDataParallel(
            model,
            # sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
            # cpu_offload=CPUOffload(offload_params=True),
            auto_wrap_policy=auto_wrap_policy,
            mixed_precision=mixed_precision_policy,
            device_id=cfg.environment._local_rank,
            # use_orig_params=False
            limit_all_gathers=True,
        )
    else:
        find_unused_parameters = cfg.environment.find_unused_parameters
        if getattr(cfg.architecture, "gradient_checkpointing", None):
            find_unused_parameters = False
        model = DistributedDataParallel(
            model,
            device_ids=[cfg.environment._local_rank],
            find_unused_parameters=find_unused_parameters,
        )

    return model


def create_nlp_backbone(cfg, model_class=AutoModel) -> Any:
    """
    Creates a backbone model for NLP tasks.
    This is needed for Gradient Checkpointing in DDP mode.
    """
    kwargs = dict()
    try:
        config = AutoConfig.from_pretrained(
            cfg.llm_backbone,
            trust_remote_code=cfg.environment.trust_remote_code,
            use_auth_token=os.getenv("HUGGINGFACE_TOKEN"),
            revision=cfg.environment.huggingface_branch,
        )
        kwargs["use_auth_token"] = os.getenv("HUGGINGFACE_TOKEN")
    except TypeError:
        # TypeError: RWForCausalLM.__init__() got
        # an unexpected keyword argument 'use_auth_token'
        config = AutoConfig.from_pretrained(
            cfg.llm_backbone,
            trust_remote_code=cfg.environment.trust_remote_code,
            revision=cfg.environment.huggingface_branch,
        )
    config.hidden_dropout_prob = cfg.architecture.intermediate_dropout
    config.attention_probs_dropout_prob = cfg.architecture.intermediate_dropout

    tokenizer = get_tokenizer(cfg)

    if config.eos_token_id != tokenizer.eos_token_id:
        logger.warning(
            "EOS token id not matching between config and tokenizer. "
            "Overwriting with tokenizer id."
        )
        config.eos_token_id = tokenizer.eos_token_id
    if config.pad_token_id != tokenizer.pad_token_id:
        logger.warning(
            "PAD token id not matching between config and tokenizer. "
            "Overwriting with tokenizer id."
        )
        config.pad_token_id = tokenizer.pad_token_id
    # no warning needed as not used
    if config.bos_token_id != tokenizer.bos_token_id:
        config.bos_token_id = tokenizer.bos_token_id

    quantization_config = None
    if cfg.architecture.backbone_dtype == "int8":
        kwargs["device_map"] = {"": cfg.environment._device}  # type: ignore
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=0.0,
        )
        # need to force pretrained
        cfg.architecture.pretrained = True
        kwargs["torch_dtype"] = torch.float16  # type: ignore
    elif cfg.architecture.backbone_dtype == "int4":
        kwargs["device_map"] = {"": cfg.environment._device}  # type: ignore
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
        # need to force pretrained
        cfg.architecture.pretrained = True
        kwargs["torch_dtype"] = torch.float16  # type: ignore
    else:
        kwargs["torch_dtype"] = getattr(torch, cfg.architecture.backbone_dtype)

    logger.info(f"Using {cfg.architecture.backbone_dtype} for backbone")

    kwargs["trust_remote_code"] = cfg.environment.trust_remote_code

    if cfg.architecture.pretrained:
        backbone = model_class.from_pretrained(
            cfg.llm_backbone,
            revision=cfg.environment.huggingface_branch,
            config=config,
            quantization_config=quantization_config,
            **kwargs,
        )
    else:
        kwargs.pop("use_auth_token", None)
        backbone = model_class.from_config(config, **kwargs)

    if cfg.tokenizer._vocab_length > config.vocab_size:
        logger.info(f"Resizing token embeddings to {cfg.tokenizer._vocab_length}")
        backbone.resize_token_embeddings(cfg.tokenizer._vocab_length)

    if cfg.training.lora:
        # if used, gradient checkpointing will be enabled below
        loaded_in_kbit = getattr(backbone, "is_loaded_in_8bit", False) or getattr(
            backbone, "is_loaded_in_4bit", False
        )

        for name, param in backbone.named_parameters():
            # freeze base model's layers
            param.requires_grad = False

        # cast all non INT8 parameters to fp32
        if loaded_in_kbit:
            for param in backbone.parameters():
                if (param.dtype == torch.float16) or (param.dtype == torch.bfloat16):
                    param.data = param.data.to(torch.float32)
    else:
        if cfg.architecture.backbone_dtype != "float32":
            if cfg.environment.mixed_precision:
                logger.info("Disabling mixed precision as dtype not set to float32.")
                cfg.environment.mixed_precision = False
            if cfg.architecture.backbone_dtype != "bfloat16":
                logger.warning(
                    "Pure float16 or int8 training will "
                    "likely lead to unstable training without adapters."
                )

    if cfg.architecture.gradient_checkpointing:
        backbone.gradient_checkpointing_enable()

    if backbone.generation_config.eos_token_id != tokenizer.eos_token_id:
        logger.warning(
            "EOS token id not matching between generation config and tokenizer. "
            "Overwriting with tokenizer id."
        )
        backbone.generation_config.eos_token_id = tokenizer.eos_token_id
    if backbone.generation_config.pad_token_id != tokenizer.pad_token_id:
        logger.warning(
            "PAD token id not matching between generation config and tokenizer. "
            "Overwriting with tokenizer id."
        )
        backbone.generation_config.pad_token_id = tokenizer.pad_token_id
    # no warning needed as not used
    if backbone.generation_config.bos_token_id != tokenizer.bos_token_id:
        backbone.generation_config.bos_token_id = tokenizer.bos_token_id

    return backbone, config


def prepare_lora(cfg, backbone):
    target_modules = (
        [
            lora_target_module.strip()
            for lora_target_module in cfg.training.lora_target_modules.strip().split(  # noqa: E501
                ","
            )
        ]
        if cfg.training.lora_target_modules
        else None
    )
    if (
        not target_modules
        and backbone.config.model_type
        not in TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING
    ):
        # extend LORA automatic target module mapping.
        target_modules = {
            "RefinedWebModel": [
                "query_key_value",
                "dense_h_to_4h",
                "dense_4h_to_h",
                "dense",
            ],
        }.get(backbone.config.model_type)
    lora_config = LoraConfig(
        r=cfg.training.lora_r,
        lora_alpha=cfg.training.lora_alpha,
        target_modules=target_modules,
        lora_dropout=cfg.training.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    if cfg.architecture.gradient_checkpointing:
        backbone.enable_input_require_grads()
    backbone = get_peft_model(backbone, lora_config)
    backbone.print_trainable_parameters()
    return backbone


def generate_text(backbone, batch, cfg, streamer, training):
    if training and getattr(cfg.training, "use_rlhf", False):
        # The KL-div estimation assumes sampling and specific settings
        do_sample = True
        temperature = cfg.training.ppo_generate_temperature
        top_k = 0.0
        top_p = 1.0
        repetition_penalty = 1.0
    else:
        do_sample = cfg.prediction.do_sample
        temperature = float(cfg.prediction.temperature)
        top_k = cfg.prediction.top_k
        top_p = float(cfg.prediction.top_p)
        repetition_penalty = float(cfg.prediction.repetition_penalty)
    mask_key = "prompt_attention_mask"
    pad_keys = [
        "prompt_input_ids",
        "prompt_attention_mask",
    ]
    batch = batch_padding(
        cfg,
        batch,
        training,
        mask_key=mask_key,
        pad_keys=pad_keys,
    )
    input_ids = batch["prompt_input_ids"]
    attention_mask = batch["prompt_attention_mask"]
    # Adding GenerationMixin type annotation for faster lookup
    generation_function: GenerationMixin.generate = backbone.generate
    verbosity = transformers_logging.get_verbosity()
    stopping_criteria = StoppingCriteriaList(
        [
            TokenStoppingCriteria(
                stop_word_ids=cfg.tokenizer._stop_words_ids,
                prompt_input_ids_len=input_ids.shape[1],
            )
        ]
    )
    # force to use cache and disable gradient checkpointing if enabled
    backbone.config.use_cache = True
    if cfg.architecture.gradient_checkpointing:
        backbone.gradient_checkpointing_disable()
    transformers_logging.set_verbosity_error()
    output = generation_function(
        inputs=input_ids,
        attention_mask=attention_mask,
        generation_config=backbone.generation_config,
        min_new_tokens=cfg.prediction.min_length_inference,
        max_new_tokens=cfg.prediction.max_length_inference,
        do_sample=do_sample,
        num_beams=cfg.prediction.num_beams,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        top_k=top_k,
        top_p=top_p,
        stopping_criteria=stopping_criteria,
        renormalize_logits=True,
        return_dict_in_generate=False,
        use_cache=True,
        streamer=streamer,
    )
    transformers_logging.set_verbosity(verbosity)
    # enable checkpointing again
    if cfg.architecture.gradient_checkpointing:
        backbone.gradient_checkpointing_enable()
    # remove the prompt tokens
    output = output[:, input_ids.shape[1] :]
    return output


class TokenStoppingCriteria(StoppingCriteria):
    """
    Stopping criteria based on tokens.
    Will stop generation when each generated sample contains at least one of the
    stop_word_ids.
    """

    def __init__(self, stop_word_ids, prompt_input_ids_len):
        super().__init__()
        self.prompt_input_ids_len = prompt_input_ids_len
        if stop_word_ids is None:
            stop_word_ids = []
        self.stop_word_ids = stop_word_ids

    def should_stop(
        self,
        generated_ids: torch.Tensor,
        stop_word_id: torch.Tensor,
    ):
        if len(stop_word_id.shape) == 0:
            return (
                torch.mean(((generated_ids == stop_word_id).sum(1) > 0).float()) == 1
            ).item()
        else:
            return (
                self.get_num_vector_found_in_matrix_rows(stop_word_id, generated_ids)
                == generated_ids.shape[0]
            )

    @staticmethod
    def get_num_vector_found_in_matrix_rows(vector, matrix):
        """
        Count the number of times a vector is found in a matrix row.
        If the vector is found in a row, the search stops and the next row is searched.
        """
        assert len(vector.shape) == 1
        assert len(matrix.shape) == 2

        found = 0
        for row in matrix:
            # stride through the vector
            for i in range(len(row) - len(vector) + 1):
                # check if the vector contains the tensor
                if torch.all(row[i : i + len(vector)] == vector):
                    found += 1
                    break

        return found

    def __call__(self, input_ids: torch.Tensor, scores: torch.FloatTensor, **kwargs):
        generated_ids: torch.Tensor = input_ids[:, self.prompt_input_ids_len :]
        for stop_word_id in self.stop_word_ids:
            if self.should_stop(generated_ids, stop_word_id.to(generated_ids.device)):
                if generated_ids.shape[1] == 1:
                    logger.warning(
                        f"Stopping criteria triggered for {stop_word_id} at first "
                        "generated token."
                    )
                return True
        return False
