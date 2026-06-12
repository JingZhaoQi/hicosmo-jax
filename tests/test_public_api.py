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


def test_high_level_api_import_order_remains_callable():
    import importlib

    importlib.import_module("hicosmo.hicosmo")
    from hicosmo import hicosmo

    assert callable(hicosmo)
    assert hasattr(hicosmo, "InferenceRunner")


def test_high_level_submodule_import_remains_module_after_root_import():
    import subprocess
    import sys

    code = """
from hicosmo import hicosmo as api
import hicosmo.hicosmo as hicosmo_module
assert callable(api)
assert hasattr(hicosmo_module, "InferenceRunner")
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_high_level_api_samples_only_free_and_nuisance_parameters():
    from hicosmo import hicosmo

    mcmc = hicosmo(
        "LCDM",
        "bao",
        free_params=["H0", "Omega_m"],
        num_samples=4,
        num_chains=1,
        chain_name=None,
        verbose=False,
        enable_checkpoints=False,
        chain_method="sequential",
    )

    assert set(mcmc.backend.parameters) == {"H0", "Omega_m", "H0_rd"}
    assert mcmc.param_config.parameters["Omega_b"].free is False
    assert mcmc.param_config.parameters["H0_rd"].free is True
    assert mcmc.enable_checkpoints is False
    assert mcmc.chain_method == "sequential"


def test_high_level_api_accepts_w0wacdm_alias_for_cpl_model():
    from hicosmo import hicosmo

    mcmc = hicosmo(
        "W0WACDM",
        "bao",
        free_params=["H0", "Omega_m", "w0", "wa"],
        num_samples=4,
        num_chains=1,
        chain_name=None,
        verbose=False,
        enable_checkpoints=False,
        chain_method="sequential",
    )

    assert set(mcmc.backend.parameters) == {"H0", "Omega_m", "w0", "wa", "H0_rd"}
    assert mcmc.param_config.parameters["Omega_b"].free is False


def test_public_api_uses_shared_component_registries_for_extensions():
    from hicosmo.api_registry import resolve_cosmology_class, resolve_likelihoods
    from hicosmo.models import LCDM
    from hicosmo.runner.components import LIKELIHOOD_REGISTRY, THEORY_REGISTRY

    class ToyLCDM(LCDM):
        pass

    class ToyLikelihood:
        def __init__(self, cosmology_class=None):
            self.cosmology_class = cosmology_class

        def __call__(self, **params):
            return 0.0

    def toy_likelihood_factory(cosmology_class=None):
        return ToyLikelihood(cosmology_class=cosmology_class)

    THEORY_REGISTRY.register("__test_toy_model__", lambda **kwargs: ToyLCDM)
    LIKELIHOOD_REGISTRY.register("__test_toy_like__", toy_likelihood_factory)

    assert resolve_cosmology_class("__test_toy_model__") is ToyLCDM
    resolved = resolve_likelihoods(ToyLCDM, "__test_toy_like__")
    assert len(resolved) == 1
    assert isinstance(resolved[0], ToyLikelihood)
    assert resolved[0].cosmology_class is ToyLCDM


def test_resolve_likelihoods_preserves_mixed_input_order():
    from hicosmo.api_registry import resolve_likelihoods
    from hicosmo.models import LCDM
    from hicosmo.runner.components import LIKELIHOOD_REGISTRY

    class ToyLikelihood:
        def __init__(self, label, cosmology_class=None):
            self.label = label
            self.cosmology_class = cosmology_class

        def __call__(self, **params):
            return 0.0

    def make_factory(label):
        def factory(cosmology_class=None):
            return ToyLikelihood(label, cosmology_class=cosmology_class)

        return factory

    explicit = ToyLikelihood("explicit")
    LIKELIHOOD_REGISTRY.register("__test_order_like_a__", make_factory("a"))
    LIKELIHOOD_REGISTRY.register("__test_order_like_b__", make_factory("b"))

    resolved = resolve_likelihoods(
        LCDM,
        ["__test_order_like_a__", explicit, "__test_order_like_b__"],
    )

    assert [like.label for like in resolved] == ["a", "explicit", "b"]
    assert resolved[0].cosmology_class is LCDM
    assert resolved[1] is explicit
    assert resolved[1].cosmology_class is LCDM
    assert resolved[2].cosmology_class is LCDM


def test_resolve_likelihoods_accepts_tuple_specs():
    from hicosmo.api_registry import resolve_likelihoods
    from hicosmo.models import LCDM
    from hicosmo.runner.components import LIKELIHOOD_REGISTRY

    class ToyLikelihood:
        def __init__(self, label, cosmology_class=None):
            self.label = label
            self.cosmology_class = cosmology_class

        def __call__(self, **params):
            return 0.0

    def make_factory(label):
        def factory(cosmology_class=None):
            return ToyLikelihood(label, cosmology_class=cosmology_class)

        return factory

    LIKELIHOOD_REGISTRY.register("__test_tuple_like_a__", make_factory("a"))
    LIKELIHOOD_REGISTRY.register("__test_tuple_like_b__", make_factory("b"))

    resolved = resolve_likelihoods(
        LCDM,
        ("__test_tuple_like_a__", "__test_tuple_like_b__"),
    )

    assert [like.label for like in resolved] == ["a", "b"]
    assert all(like.cosmology_class is LCDM for like in resolved)
