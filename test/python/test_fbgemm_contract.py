"""Host admission/routing tests, never a simulation of foreign GPU numerics."""
from types import SimpleNamespace
import hashlib

import pytest

from baselines import fbgemm_reference as adapter


@pytest.fixture(autouse=True)
def clear_package_cache():
    adapter._load.cache_clear()
    yield
    adapter._load.cache_clear()


@pytest.mark.parametrize("version", ["1.8.0", "1.8.0+cu130", "1.8.0+git.abc"])
def test_pinned_release_and_local_build_identity(version):
    assert adapter._release(version) == version


@pytest.mark.parametrize("version", [None, 180, "1.7.0", "1.8.1", "1.8.0rc1", "1.8.0.dev0", "1.8.0\n", "1.8.0+"])
def test_other_versions_are_failures(version):
    with pytest.raises(RuntimeError, match="requires release 1.8.0"):
        adapter._release(version)


@pytest.mark.parametrize("n,d,b,bound", [(0, 1, 0, None), (0, 7, 4, 0), (17, 1024, 5, None),
                                       (17, 7, 3, 10), (17, 7, 3, 99)])
def test_shape_only_bound_selection(n, d, b, bound):
    info = adapter.configuration(n, d, b, max_len=bound)
    assert info["max_L"] == (n if bound is None else bound)
    assert info["max_L_source"] == ("shape_N_upper_bound" if bound is None else "caller_verified_length_bound")
    assert info["device_semantics"] == "unqualified"
    assert info["accumulation_dtype"] == "f32"
    if bound is not None:
        assert "caller must have verified" in info["no_truncation"]


@pytest.mark.parametrize("n,d,b,bound", [
    (True, 1, 1, None), (1, True, 1, None), (1, 1, False, None), (-1, 1, 1, None),
    (1, 0, 1, None), (1, 1025, 1, None), (1, 1, 0, None),
    (adapter.MAX_LENGTH + 1, 1, 1, None), (adapter.INT32_MAX // 1024 + 1, 1024, 1, None),
    (0, 1, adapter.INT32_MAX, None), (1, 1, 1, True), (1, 1, 1, -1),
    (1, 1, 1, adapter.MAX_LENGTH + 1), (1, 1, 1, 0), (1, 1, 1, 1.0),
])
def test_shape_and_safe_integer_range_rejections(n, d, b, bound):
    with pytest.raises(ValueError):
        adapter.configuration(n, d, b, max_len=bound)


def test_safe_integer_boundaries_without_allocating_tensors():
    assert adapter.configuration(adapter.MAX_LENGTH, 1, 1)["max_L"] == adapter.MAX_LENGTH
    assert adapter.configuration(0, 1, adapter.INT32_MAX - 1)["max_L"] == 0
    assert adapter.configuration(adapter.INT32_MAX // 1024, 1024, 1)["max_L"] > 0


def test_only_missing_distribution_is_optional_unavailability(monkeypatch):
    def missing(name):
        assert name == "fbgemm-gpu"
        raise adapter.metadata.PackageNotFoundError(name)
    monkeypatch.setattr(adapter.metadata, "version", missing)
    with pytest.raises(adapter.OptionalDependencyUnavailable, match="not installed"):
        adapter.toolchain()


def test_wrong_installed_version_rejected_before_import(monkeypatch):
    monkeypatch.setattr(adapter.metadata, "version", lambda _: "1.7.0")
    monkeypatch.setattr(adapter.importlib, "import_module", lambda _: pytest.fail("wrong package was imported"))
    with pytest.raises(RuntimeError, match="requires release") as error:
        adapter.toolchain()
    assert not isinstance(error.value, adapter.OptionalDependencyUnavailable)


@pytest.mark.parametrize("error", [ImportError("undefined symbol: fake_torch_abi"),
                                  ModuleNotFoundError("installed package has a missing dependency"),
                                  OSError("cannot open native shared object")])
def test_installed_import_errors_remain_failures(monkeypatch, error):
    monkeypatch.setattr(adapter.metadata, "version", lambda _: "1.8.0")
    def broken_import(_):
        raise error
    monkeypatch.setattr(adapter.importlib, "import_module", broken_import)
    with pytest.raises(type(error), match=str(error)) as actual:
        adapter.toolchain()
    assert actual.value is error


def fake_install(monkeypatch, tmp_path, *, module_version="1.8.0+test", operator=True, native=True):
    init = tmp_path / "__init__.py"
    init.write_text("# synthetic package identity fixture, not executable FBGEMM\n")
    library = tmp_path / "libfixture.so"
    if native:
        library.write_bytes(b"synthetic provenance fixture, not a shared library")
    package = SimpleNamespace(__version__=module_version, __file__=str(init))
    operation = object()
    torch = SimpleNamespace(__version__="fixture", version=SimpleNamespace(cuda=None),
                            ops=SimpleNamespace(fbgemm=SimpleNamespace(**({"jagged_softmax": operation} if operator else {}))))
    monkeypatch.setattr(adapter.metadata, "version", lambda _: "1.8.0+test")
    def fake_import(name):
        assert name in ("fbgemm_gpu", "torch")
        return {"fbgemm_gpu": package, "torch": torch}[name]
    monkeypatch.setattr(adapter.importlib, "import_module", fake_import)
    return init, library, operation


@pytest.mark.parametrize("kwargs,match", [({"module_version": "1.7.0"}, "requires release"),
                                       ({"operator": False}, "lacks public"),
                                       ({"native": False}, "no auditable native")])
def test_installed_package_contract_failures(monkeypatch, tmp_path, kwargs, match):
    fake_install(monkeypatch, tmp_path, **kwargs)
    with pytest.raises(RuntimeError, match=match) as error:
        adapter.toolchain()
    assert not isinstance(error.value, adapter.OptionalDependencyUnavailable)


def test_provenance_hashes_inventory_and_defensive_copy(monkeypatch, tmp_path):
    init, library, operation = fake_install(monkeypatch, tmp_path)
    record = adapter.toolchain()
    assert adapter._load()[0] is operation
    assert record["fbgemm_gpu"] == "1.8.0+test"
    assert record["package_init_sha256"] == hashlib.sha256(init.read_bytes()).hexdigest()
    assert record["native_libraries"] == [{"path": str(library), "sha256": hashlib.sha256(library.read_bytes()).hexdigest()}]
    assert record["compatibility"] == "unverified" and record["device_semantics"] == "unqualified"
    assert "inventory" in record["native_libraries_scope"] and "each benchmark case" in record["device_semantics_scope"]
    record["known_risks"].clear()
    record["native_libraries"][0]["sha256"] = "modified"
    assert adapter.toolchain()["known_risks"]
    assert adapter.toolchain()["native_libraries"][0]["sha256"] != "modified"


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("offset_dtype", ["int32", "int64"])
def test_invocation_routes_f32_and_original_offsets_then_casts(dtype, offset_dtype):
    torch = pytest.importorskip("torch")
    values = torch.tensor([[0.25, -2.5], [1.5, 3.5]], dtype=getattr(torch, dtype))
    offsets = torch.tensor([0, 0, 2], dtype=getattr(torch, offset_dtype))
    calls = []
    def inspect_only(work, actual_offsets, maximum):
        # Identity return is intentional: this tests routing, not softmax.
        assert work.dtype == torch.float32 and actual_offsets is offsets and maximum == 2
        torch.testing.assert_close(work, values.float(), rtol=0, atol=0)
        calls.append(work)
        return work, actual_offsets
    output = adapter._invoke(values, offsets, 2, inspect_only)
    assert len(calls) == 1 and output.dtype == values.dtype
    torch.testing.assert_close(output, values, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("batch", [0, 4])
def test_empty_packed_shape_returns_without_foreign_kernel(dtype, batch):
    torch = pytest.importorskip("torch")
    values = torch.empty((0, 7), dtype=getattr(torch, dtype))
    result = adapter._invoke(values, torch.zeros(batch + 1, dtype=torch.int64), 0,
                             lambda *_: pytest.fail("empty input called a foreign kernel"))
    assert result.shape == values.shape and result.dtype == values.dtype


@pytest.mark.parametrize("bad", ["bare_tensor", "tuple_length", "shape", "dtype", "offset_copy"])
def test_output_structure_failures_are_not_numerical_fallbacks(bad):
    torch = pytest.importorskip("torch")
    values, offsets = torch.ones((2, 7)), torch.tensor([0, 2])
    result = {"bare_tensor": values, "tuple_length": (values,), "shape": (values[:1], offsets),
              "dtype": (values.double(), offsets), "offset_copy": (values, offsets.clone())}[bad]
    with pytest.raises(RuntimeError, match="contract|must return"):
        adapter._invoke(values, offsets, 2, lambda *_: result)


@pytest.mark.parametrize("bad", ["cpu", "rank", "offset_rank", "empty_offsets", "strided_values",
                                "strided_offsets", "float64", "offset_float", "grad", "operation"])
def test_real_cpu_input_admission_rejects_without_optional_package(monkeypatch, bad):
    torch = pytest.importorskip("torch")
    values, offsets, op = torch.ones((2, 7)), torch.tensor([0, 2]), "softmax"
    if bad == "rank": values = values.flatten()
    if bad == "offset_rank": offsets = offsets[:, None]
    if bad == "empty_offsets": offsets = offsets[:0]
    if bad == "strided_values": values = torch.ones((2, 14))[:, ::2]
    if bad == "strided_offsets": offsets = torch.tensor([0, 99, 2, 99])[::2]
    if bad == "float64": values = values.double()
    if bad == "offset_float": offsets = offsets.float()
    if bad == "grad": values.requires_grad_(True)
    if bad == "operation": op = "sum"
    monkeypatch.setattr(adapter, "_load", lambda: pytest.fail("invalid inputs reached optional load"))
    with pytest.raises(ValueError):
        adapter.call(values, offsets, op)
