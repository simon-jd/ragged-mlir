"""Real CPU NJT max semantics, using the shared independent float64 oracle."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from baselines.torch_reference import njt_max
from oracle import assert_matches, quantize_bf16


def max_case(case, d):
    offsets = np.array([0, 0, 1, 4, 6, 6], dtype=np.int64)
    values = -np.arange(1, 6*d + 1, dtype=np.float32).reshape(6, d) / 8
    if case == "all_empty":
        values, offsets = np.empty((0, d), np.float32), np.array([0, 0, 0, 0], np.int64)
    elif case == "zero_segments":
        values, offsets = np.empty((0, d), np.float32), np.array([0], np.int64)
    elif case == "negative_infinity":
        values[:] = -np.inf
    elif case == "positive_infinity":
        values[1] = np.inf
    elif case == "nan":
        values[2] = np.nan
    elif case == "finite_minimum":
        values[:] = -np.finfo(np.float32).max
    elif case != "negative_finite":
        raise ValueError(case)
    return values, offsets


@pytest.mark.parametrize("case", ["negative_finite", "negative_infinity", "positive_infinity", "nan",
                                  "finite_minimum", "all_empty", "zero_segments"])
@pytest.mark.parametrize("dtype", ["f32", "bf16"])
@pytest.mark.parametrize("d", [1, 7, 16, 1024])
@pytest.mark.parametrize("with_hints", [False, True])
def test_njt_max_cpu_oracle(case, dtype, d, with_hints):
    values, offsets = max_case(case, d)
    if dtype == "bf16":
        values = quantize_bf16(values)
    tensor = torch.from_numpy(values).to(torch.float32 if dtype == "f32" else torch.bfloat16)
    lengths = np.diff(offsets)
    hints = {"min_len": int(lengths.min()), "max_len": int(lengths.max())} if with_hints and len(lengths) else {}
    actual = njt_max(tensor, torch.from_numpy(offsets), **hints)
    assert actual.dtype == tensor.dtype and actual.device == tensor.device
    assert actual.shape == (len(offsets) - 1, d)
    assert_matches(actual.float().numpy(), values, offsets, operation="max", input_dtype=dtype)


def test_padding_identity_repairs_empty_and_short_nonempty_negative_infinity_segments():
    values = torch.tensor([[-float("inf"), -float("inf")], [-3, float("nan")], [float("inf"), -2]])
    offsets = torch.tensor([0, 0, 1, 3, 3], dtype=torch.int64)
    nt = torch.nested.nested_tensor_from_jagged(values, offsets=offsets, min_seqlen=0, max_seqlen=2)
    raw = torch.amax(nt, dim=1)
    repaired = njt_max(values, offsets, min_len=0, max_len=2)
    assert torch.all(raw[[0, 1, 3]] == torch.finfo(torch.float32).min)
    assert torch.all(torch.isneginf(repaired[[0, 1, 3]]))
    torch.testing.assert_close(repaired[2], raw[2], rtol=0, atol=0, equal_nan=True)


def test_fullgraph_capture_keeps_njt_construction_padding_and_reduction():
    graphs = []
    def backend(graph, _):
        graphs.append(graph)
        return graph.forward
    def function(values, offsets):
        return njt_max(values, offsets, min_len=0, max_len=3)
    values, offsets = max_case("negative_finite", 7)
    output = torch.compile(function, backend=backend, fullgraph=True)(torch.from_numpy(values), torch.from_numpy(offsets))
    assert_matches(output.numpy(), values, offsets, operation="max", input_dtype="f32")
    assert len(graphs) == 1
    targets = [str(node.target) for node in graphs[0].graph.nodes if node.op == "call_function"]
    assert any("amax" in target for target in targets)
    assert any("to_padded" in target for target in targets)
    assert any("nested" in target for target in targets)
