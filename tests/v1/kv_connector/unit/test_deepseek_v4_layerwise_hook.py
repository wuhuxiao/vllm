# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import torch

import vllm.model_executor.layers.attention.kv_transfer_utils as kv_transfer_utils
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1
from vllm.model_executor.layers.deepseek_v4_attention import deepseek_v4_attention


def test_deepseek_v4_attention_triggers_layerwise_kv_hooks(monkeypatch):
    calls = []
    connector = Mock(spec=KVConnectorBase_V1)
    connector.has_connector_metadata.return_value = True
    connector.wait_for_layer_load.side_effect = lambda layer_name: calls.append(
        f"wait_for_layer_load:{layer_name}"
    )
    connector.save_kv_layer.side_effect = (
        lambda layer_name, _kv_cache, _attn_metadata: calls.append(
            f"save_kv_layer:{layer_name}"
        )
    )
    monkeypatch.setattr(kv_transfer_utils, "has_kv_transfer_group", lambda: True)
    monkeypatch.setattr(kv_transfer_utils, "is_v1_kv_transfer_group", lambda: True)
    monkeypatch.setattr(kv_transfer_utils, "get_kv_transfer_group", lambda: connector)

    attention_impl = Mock(side_effect=lambda *_args: calls.append("attention_impl"))
    swa_kv_cache = torch.empty(1)
    swa_attn_metadata = object()
    layer = SimpleNamespace(
        compress_ratio=4,
        swa_cache_layer=SimpleNamespace(
            prefix="model.layers.0.attn.swa_cache",
            kv_cache=swa_kv_cache,
        ),
        mla_attn=SimpleNamespace(
            prefix="model.layers.0.attn",
        ),
        attention_impl=attention_impl,
    )
    forward_context = SimpleNamespace(
        attn_metadata={
            "model.layers.0.attn.swa_cache": swa_attn_metadata,
        },
        no_compile_layers={
            "model.layers.0.attn.deepseek_v4_multi_head_latent_attention": layer
        }
    )
    monkeypatch.setattr(
        "vllm.model_executor.layers.deepseek_v4_attention.get_forward_context",
        lambda: forward_context,
    )

    hidden_states = torch.empty(1, 1)
    positions = torch.empty(1)
    out = torch.empty(1, 1)

    deepseek_v4_attention(
        hidden_states,
        positions,
        out,
        "model.layers.0.attn.deepseek_v4_multi_head_latent_attention",
    )

    connector.wait_for_layer_load.assert_any_call("model.layers.0.attn.swa_cache")
    connector.wait_for_layer_load.assert_called_once()
    attention_impl.assert_called_once_with(hidden_states, positions, out)
    connector.save_kv_layer.assert_any_call(
        "model.layers.0.attn.swa_cache", swa_kv_cache, swa_attn_metadata
    )
    connector.save_kv_layer.assert_called_once()
    assert calls == [
        "wait_for_layer_load:model.layers.0.attn.swa_cache",
        "attention_impl",
        "save_kv_layer:model.layers.0.attn.swa_cache",
    ]
