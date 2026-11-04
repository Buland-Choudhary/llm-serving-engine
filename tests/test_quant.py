from __future__ import annotations

import pytest
import torch
from torch import nn

from llmserve.models.loader import load_model
from llmserve.quant import Int4Linear, Int8Linear, _int4_quantize, quantize_model, weight_bytes
from llmserve.sampling import SamplingParams
from tests.conftest import run_all
from tests.test_model_parity import _prefill_logits


def test_int8_linear_matches_dequantized_reference():
    torch.manual_seed(0)
    lin = nn.Linear(128, 96, bias=True)
    q = Int8Linear(lin)
    x = torch.randn(5, 128)
    ref = x @ q.dequantize().t() + lin.bias
    torch.testing.assert_close(q(x), ref, atol=1e-4, rtol=1e-4)
    assert (q.dequantize() - lin.weight).abs().max() <= q.scale.max() / 2 + 1e-6
    # large inputs take the dequantize + GEMM path: same result
    xl = torch.randn(40, 128)
    torch.testing.assert_close(q(xl), xl @ q.dequantize().t() + lin.bias, atol=1e-4, rtol=1e-4)


def test_int4_linear_matches_dequantized_reference():
    torch.manual_seed(0)
    lin = nn.Linear(128, 64, bias=False)
    m = Int4Linear(lin, group_size=32)
    qi, scale, zero = _int4_quantize(lin.weight.detach(), 32)
    deq = ((qi.view(64, -1, 32).float() - 8) * scale[..., None] + zero[..., None]).view(64, 128)
    x = torch.randn(3, 128)
    torch.testing.assert_close(m(x), x @ deq.t(), atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(m.dequantize(), deq, atol=1e-5, rtol=1e-5)
    xl = torch.randn(40, 128)  # dequantize + GEMM path
    torch.testing.assert_close(m(xl), xl @ deq.t(), atol=1e-4, rtol=1e-4)
    # 3-D input (batch, seq, features) keeps its leading shape
    assert m(torch.randn(2, 3, 128)).shape == (2, 3, 64)


@pytest.mark.parametrize("method,tol", [("int8", 0.05), ("int4", 0.35)])
def test_quantized_model_close_to_fp32(tiny_llama_dir, method, tol):
    torch.manual_seed(1)
    ids = torch.randint(2, 256, (20,))
    ref = _prefill_logits(load_model(tiny_llama_dir, linear_backend="torch"), ids)
    qm = load_model(tiny_llama_dir, quantization=method, quantize_lm_head=True, group_size=32)
    got = _prefill_logits(qm, ids)
    rel = (got - ref).norm() / ref.norm()
    assert rel < tol, rel
    assert weight_bytes(qm) < weight_bytes(load_model(tiny_llama_dir, linear_backend="torch"))


def test_awq_runs_and_is_no_worse_on_calibration(tiny_llama_dir):
    torch.manual_seed(2)
    calib = [torch.randint(2, 256, (64,)).tolist() for _ in range(4)]
    ids = torch.tensor(calib[0])
    ref = _prefill_logits(load_model(tiny_llama_dir, linear_backend="torch"), ids)
    plain = _prefill_logits(load_model(tiny_llama_dir, quantization="int4", group_size=32), ids)
    awq = _prefill_logits(
        load_model(tiny_llama_dir, quantization="int4-awq", group_size=32, calib_token_ids=calib),
        ids,
    )
    assert (awq - ref).norm() <= (plain - ref).norm() * 1.05


def test_engine_with_int8(tiny_llama_dir):
    from llmserve.engine import EngineConfig, LLMEngine

    m = load_model(tiny_llama_dir, quantization="int8", max_positions=256)
    eng = LLMEngine(
        EngineConfig(model=str(tiny_llama_dir), num_kv_blocks=64, block_size=4, max_model_len=256),
        model=m,
    )
    rid = eng.add_request(
        [5, 6, 7, 8], SamplingParams(temperature=0, max_tokens=10, ignore_eos=True)
    )
    assert len(run_all(eng)[rid].outputs[0].token_ids) == 10


def test_unknown_method():
    with pytest.raises(ValueError):
        quantize_model(nn.Sequential(nn.Linear(4, 4)), "int3")


@pytest.mark.parametrize("rows", [1, 3, 16])
def test_own_int8_gemv_matches_reference(rows):
    from llmserve.kernels import load_int8_gemv

    if load_int8_gemv() is None:
        pytest.skip("no C++ toolchain")
    torch.manual_seed(rows)
    lin = nn.Linear(200, 77, bias=False)  # odd sizes exercise the loop tails
    q = Int8Linear(lin)
    x = torch.randn(rows, 200)
    ref = x @ q.dequantize().t()
    torch.testing.assert_close(q(x), ref, atol=1e-4, rtol=1e-4)
    q.kernel = "aten"
    torch.testing.assert_close(q(x), ref, atol=1e-4, rtol=1e-4)
