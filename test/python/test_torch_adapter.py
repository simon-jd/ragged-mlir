"""Real PyTorch registration/shape/NJT tests; these do not execute CUDA kernels."""
import pytest

torch = pytest.importorskip("torch")

from ragged.errors import ValidationError
from ragged.torch_adapter import jagged_components, register_torch_ops


@pytest.fixture(scope="module")
def ops():
    return register_torch_ops()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("offset_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("kind", ["sum", "max"])
def test_real_meta_reduce_shape(ops, dtype, offset_dtype, kind):
    values = torch.empty((17, 13), dtype=dtype, device="meta")
    offsets = torch.empty((5,), dtype=offset_dtype, device="meta")
    output = ops["segment_reduce"](values, offsets, kind, "shape-only-no-program")
    assert output.shape == (4, 13)
    assert output.dtype == dtype
    assert output.device.type == "meta"


def test_real_meta_softmax_shape_and_registration_idempotence(ops):
    values = torch.empty((0, 1024), dtype=torch.bfloat16, device="meta")
    offsets = torch.empty((1,), dtype=torch.int64, device="meta")
    output = ops["segment_softmax"](values, offsets, "shape-only-no-program")
    assert output.shape == (0, 1024)
    assert register_torch_ops() == ops


def test_fullgraph_capture_of_opaque_custom_op_using_meta_tensors(ops):
    graphs = []
    def backend(graph, example_inputs):
        graphs.append(graph)
        return graph.forward
    def function(values, offsets):
        return torch.ops.ragged_mlir.segment_reduce(values, offsets, "sum", "shape-only-no-program")
    compiled = torch.compile(function, backend=backend, fullgraph=True)
    values = torch.empty((17, 13), device="meta")
    offsets = torch.empty((5,), dtype=torch.int64, device="meta")
    assert compiled(values, offsets).shape == (4, 13)
    assert len(graphs) == 1
    assert any(node.op == "call_function" and "ragged_mlir.segment_reduce" in str(node.target)
               for node in graphs[0].graph.nodes)


@pytest.mark.parametrize("kind", ["mean", "min"])
def test_fake_rejects_unsupported_reduction(ops, kind):
    with pytest.raises(ValidationError, match="kind"):
        ops["segment_reduce"](torch.empty((3, 4), device="meta"),
                              torch.empty((3,), dtype=torch.int64, device="meta"), kind, "token")


def test_njt_contiguous_adapter_is_zero_copy():
    values = torch.randn(6, 4)
    offsets = torch.tensor([0, 2, 2, 6])
    nt = torch.nested.nested_tensor_from_jagged(values, offsets)
    extracted_values, extracted_offsets = jagged_components(nt)
    assert extracted_values.data_ptr() == values.data_ptr()
    assert extracted_offsets.data_ptr() == offsets.data_ptr()


def test_njt_holes_are_explicitly_rejected():
    values = torch.randn(6, 4)
    offsets = torch.tensor([0, 3, 6])
    nt = torch.nested.nested_tensor_from_jagged(values, offsets, lengths=torch.tensor([2, 1]))
    with pytest.raises(ValidationError, match="holes"):
        jagged_components(nt)


def test_njt_nonleading_ragged_dimension_rejected():
    values = torch.randn(4, 6)
    offsets = torch.tensor([0, 2, 6])
    nt = torch.nested.nested_tensor_from_jagged(values, offsets, jagged_dim=2)
    with pytest.raises(ValidationError, match="first non-batch"):
        jagged_components(nt)


def test_dense_tensor_is_not_njt():
    with pytest.raises(ValidationError, match="torch.jagged"):
        jagged_components(torch.randn(2, 3, 4))


@pytest.mark.parametrize("kind", ["sum", "max", "softmax"])
def test_checked_mode_reaches_registered_program_without_changing_output_shape(monkeypatch, kind):
    """A host call stub records mode routing; it performs no GPU calculation."""
    from types import SimpleNamespace
    from ragged import ProgramSpec
    from ragged import torch_adapter
    calls = []
    program = SimpleNamespace(spec=ProgramSpec(kind,feature_dim=4),
                              artifact=SimpleNamespace(key=SimpleNamespace(digest="mode-routing-"+kind)),
                              run=lambda values,offsets,out,**kwargs: calls.append(kwargs))
    monkeypatch.setattr(torch_adapter, "_programs", {})
    values,offsets = torch.empty((6,4)),torch.tensor([0,2,6])
    if kind == "softmax":
        output = torch_adapter.segment_softmax(values,offsets,program=program,mode="checked")
    else:
        output = torch_adapter.segment_reduce(values,offsets,kind,program=program,mode="checked")
    assert calls == [{"mode":"checked"}]
    assert output.shape == (6 if kind == "softmax" else 2,4)


def test_checked_mode_is_a_constant_in_fullgraph_custom_op(ops):
    graphs = []
    def backend(graph, _):
        graphs.append(graph)
        return graph.forward
    def operation(values,offsets):
        return ops["segment_softmax"](values,offsets,"shape-only-no-program","checked")
    compiled = torch.compile(operation,backend=backend,fullgraph=True)
    assert compiled(torch.empty((6,4),device="meta"),torch.empty((3,),dtype=torch.int64,device="meta")).shape == (6,4)
    nodes = [node for node in graphs[0].graph.nodes if "ragged_mlir.segment_softmax" in str(node.target)]
    assert len(nodes) == 1 and nodes[0].args[-1] == "checked"


def test_fake_rejects_unknown_execution_mode(ops):
    with pytest.raises(ValidationError,match="mode"):
        ops["segment_softmax"](torch.empty((6,4),device="meta"),
                                torch.empty((3,),dtype=torch.int64,device="meta"),"token","unsafe")


def test_masked_softmax_fullgraph_and_export_use_optional_tensor_operand(ops):
    class Operation(torch.nn.Module):
        def forward(self,values,offsets,mask):
            return ops["segment_softmax"](values,offsets,"shape-only-no-program","checked",mask)
    args = (torch.empty((6,4),dtype=torch.bfloat16,device="meta"),
            torch.empty((3,),dtype=torch.int64,device="meta"),torch.empty((6,4),device="meta"))
    graphs = []
    def backend(graph,_):
        graphs.append(graph)
        return graph.forward
    output = torch.compile(Operation(),backend=backend,fullgraph=True)(*args)
    assert output.shape == (6,4) and output.dtype == torch.bfloat16
    nodes = [node for node in graphs[0].graph.nodes if "ragged_mlir.segment_softmax" in str(node.target)]
    assert len(nodes) == 1 and isinstance(nodes[0].args[-1],torch.fx.Node)
    exported = torch.export.export(Operation(),args)
    assert any("ragged_mlir.segment_softmax" in str(node.target) for node in exported.graph.nodes)


@pytest.mark.parametrize("problem", ["bf16","rank","broadcast","noncontiguous","device","grad"])
def test_mask_fake_rejects_unsupported_tensor_contract(ops,problem):
    values = torch.empty((6,4),dtype=torch.bfloat16,device="meta")
    offsets = torch.empty((3,),dtype=torch.int64,device="meta")
    mask = torch.empty((6,4),device="meta")
    if problem == "bf16": mask = mask.to(torch.bfloat16)
    elif problem == "rank": mask = torch.empty((24,),device="meta")
    elif problem == "broadcast": mask = torch.empty((1,4),device="meta")
    elif problem == "noncontiguous": mask = torch.empty((4,6),device="meta").t()
    elif problem == "device": mask = torch.empty((6,4))
    elif problem == "grad": mask.requires_grad_()
    with pytest.raises((ValidationError,RuntimeError)):
        ops["segment_softmax"](values,offsets,"shape-only-no-program","validated",mask)


def test_njt_mask_routes_dense_values_mask_to_program(monkeypatch):
    from types import SimpleNamespace
    from ragged import ProgramSpec
    from ragged import torch_adapter
    calls = []
    program = SimpleNamespace(spec=ProgramSpec("softmax",feature_dim=4,mask=True),
                              artifact=SimpleNamespace(key=SimpleNamespace(digest="njt-mask-routing")),
                              run=lambda values,offsets,out,**kwargs: calls.append(kwargs))
    monkeypatch.setattr(torch_adapter,"_programs",{})
    values,offsets,mask = torch.empty((6,4)),torch.tensor([0,2,6]),torch.empty((6,4))
    njt = torch.nested.nested_tensor_from_jagged(values,offsets)
    out = torch_adapter.njt_softmax(njt,program=program,mode="checked",mask=mask)
    assert out.values().shape == values.shape and out.offsets().data_ptr() == offsets.data_ptr()
    assert len(calls) == 1 and calls[0]["mask"] is mask and calls[0]["mode"] == "checked"
    with pytest.raises(ValidationError,match="mask presence"):
        torch_adapter.bind(program)(values,offsets)
