"""Public API regression tests."""

from __future__ import annotations


def test_runner_exports_are_available():
    import hicosmo

    assert callable(hicosmo.run_from_config)
    assert hasattr(hicosmo, "runner")
    assert hasattr(hicosmo, "InferenceRunner")


def test_list_cosmologies_returns_string_names():
    from hicosmo import list_cosmologies

    names = list_cosmologies()
    assert isinstance(names, list)
    assert names
    assert all(isinstance(name, str) for name in names)
    assert "LCDM" in names


def test_cli_prepare_namespace_includes_core_symbols():
    import hicosmo.cli as cli

    namespace = cli.prepare_namespace()
    for key in ["hicosmo", "InferenceRunner", "list_likelihoods", "list_cosmologies"]:
        assert key in namespace
