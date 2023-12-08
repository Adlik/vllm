"""Utilities for selecting and loading models."""
import contextlib
from typing import Type
import gc

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from vllm.config import ModelConfig
from vllm.model_executor.models import *  # pylint: disable=wildcard-import
from vllm.model_executor.weight_utils import (get_quant_config,
                                              initialize_dummy_weights)
from vllm.model_executor.parallel_utils.layers import (ColumnParallelLinear,
                                                       RowParallelLinear)

# TODO(woosuk): Lazy-load the model classes.
_MODEL_REGISTRY = {
    "AquilaModel": AquilaForCausalLM,
    "AquilaForCausalLM": AquilaForCausalLM,  # AquilaChat2
    "BaiChuanForCausalLM": BaiChuanForCausalLM,  # baichuan-7b
    "BaichuanForCausalLM": BaichuanForCausalLM,  # baichuan-13b
    "BloomForCausalLM": BloomForCausalLM,
    "ChatGLMModel": ChatGLMForCausalLM,
    "FalconForCausalLM": FalconForCausalLM,
    "GPT2LMHeadModel": GPT2LMHeadModel,
    "GPTBigCodeForCausalLM": GPTBigCodeForCausalLM,
    "GPTJForCausalLM": GPTJForCausalLM,
    "GPTNeoXForCausalLM": GPTNeoXForCausalLM,
    "InternLMForCausalLM": InternLMForCausalLM,
    "LlamaForCausalLM": LlamaForCausalLM,
    "LLaMAForCausalLM": LlamaForCausalLM,  # For decapoda-research/llama-*
    "MistralForCausalLM": MistralForCausalLM,
    # transformers's mpt class has lower case
    "MptForCausalLM": MptForCausalLM,
    "MPTForCausalLM": MptForCausalLM,
    "OPTForCausalLM": OPTForCausalLM,
    "QWenLMHeadModel": QWenLMHeadModel,
    "RWForCausalLM": FalconForCausalLM,
    "YiForCausalLM": YiForCausalLM,
}

# FIXME(woosuk): Remove this once all models support quantization.
_MODEL_CLASSES_SUPPORT_QUANTIZATION = [
    LlamaForCausalLM,
    MistralForCausalLM,
]


@contextlib.contextmanager
def _set_default_torch_dtype(dtype: torch.dtype):
    """Sets the default torch dtype to the given dtype."""
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(old_dtype)


def _get_model_architecture(config: PretrainedConfig) -> Type[nn.Module]:
    architectures = getattr(config, "architectures", [])
    for arch in architectures:
        if arch in _MODEL_REGISTRY:
            return _MODEL_REGISTRY[arch]
    raise ValueError(
        f"Model architectures {architectures} are not supported for now. "
        f"Supported architectures: {list(_MODEL_REGISTRY.keys())}")


def _replace_quant_params(model,
                          modules_to_not_convert="lm_head",
                          auto_quant_mode="llm_int8"):
    """
    modules_to_not_convert (`str`, *optional*, defaults to `lm_head`):
            Name of the module to not convert in `Linear8bitLt`.
            In practice we keep the `lm_head` in full precision
            for numerical stability reasons.
    """
    if not isinstance(modules_to_not_convert, list):
        modules_to_not_convert = [modules_to_not_convert]
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            _replace_quant_params(module, modules_to_not_convert, auto_quant_mode)
        if isinstance(module,(ColumnParallelLinear, RowParallelLinear)) and \
            name not in modules_to_not_convert:
            param = module._parameters["weight"]
            if auto_quant_mode =="llm_int8":
                import bitsandbytes as bnb
                module.state = bnb.MatmulLtState()
                # Necessary for stacked layers
                module.state.threshold = 6.0
                module.state.has_fp16_weights = False
                module.state.memory_efficient_backward = False
                module.state.use_pool = True

                new_value = bnb.nn.Int8Params(param.data,
                                              requires_grad=False,
                                              has_fp16_weights=False)
            elif auto_quant_mode == "weight_int4":
                from vllm.model_executor.quantization_utils.auto_quant import \
                    Int4Params
                new_value = Int4Params(param.data)
            module._parameters["weight"] = new_value
            del param
            torch.cuda.empty_cache()


def get_model(model_config: ModelConfig) -> nn.Module:
    model_class = _get_model_architecture(model_config.hf_config)

    # Get the quantization config.
    quant_config = None
    if model_config.quantization is not None:
        if model_class not in _MODEL_CLASSES_SUPPORT_QUANTIZATION:
            raise ValueError(
                f"Quantization is not supported for {model_class}.")
        quant_config = get_quant_config(model_config.quantization,
                                        model_config.model,
                                        model_config.download_dir)
        capability = torch.cuda.get_device_capability()
        capability = capability[0] * 10 + capability[1]
        if capability < quant_config.get_min_capability():
            raise ValueError(
                f"The quantization method {model_config.quantization} is not "
                "supported for the current GPU. "
                f"Minimum capability: {quant_config.get_min_capability()}. "
                f"Current capability: {capability}.")
        supported_dtypes = quant_config.get_supported_act_dtypes()
        if model_config.dtype not in supported_dtypes:
            raise ValueError(
                f"{model_config.dtype} is not supported for quantization "
                f"method {model_config.quantization}. Supported dtypes: "
                f"{supported_dtypes}")
    

    with _set_default_torch_dtype(model_config.dtype):
        # Create a model instance.
        # The weights will be initialized as empty tensors.
        if model_class in _MODEL_CLASSES_SUPPORT_QUANTIZATION:
            model = model_class(model_config.hf_config, quant_config,
                                auto_quant_mode=model_config.auto_quant_mode)
        else:
            model = model_class(model_config.hf_config,
                                auto_quant_mode=model_config.auto_quant_mode)
        if model_config.load_format == "dummy":
            model = model.cuda()
            # NOTE(woosuk): For accurate performance evaluation, we assign
            # random values to the weights.
            initialize_dummy_weights(model)
        else:
            # Load the weights from the cached or downloaded files.
            modules_to_not_convert = ["lm_head"]
            model.load_weights(model_config.model, model_config.download_dir,
                               model_config.load_format, model_config.revision)
            if model_config.auto_quant_mode in [
                    "llm_int8", "weight_int4"
            ]:
                _replace_quant_params(
                    model,
                    modules_to_not_convert=modules_to_not_convert,
                    auto_quant_mode=model_config.auto_quant_mode)
            model = model.cuda()
            gc.collect()
            torch.cuda.empty_cache()
            print("Memory allocated:",
                  torch.cuda.memory_allocated(torch.cuda.current_device()))
            print("Memory reserved:",
                  torch.cuda.memory_reserved(torch.cuda.current_device()))
    return model.eval()
