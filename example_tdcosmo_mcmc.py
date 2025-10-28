#!/usr/bin/env python3
"""Run hierarchical TDCOSMO inference with NumPyro and compare to reference chains."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist

from hicosmo.samplers import init_hicosmo, MCMCSampler
from hicosmo.models.lcdm import LCDM
from hicosmo.likelihoods import TDCOSMOLikelihood
from hicosmo.visualization import HIcosmoViz
from numpyro.infer import SA
from numpyro.infer.util import init_to_value

REFERENCE_CHAIN = Path("hierarchy_analysis_2020_public/TDCOSMO_sample/tdcosmo_chain_alpha_free.h5")


def load_reference_statistics(path: Path) -> Dict[str, Dict[str, float]]:
    """Load published TDCOSMO constraints for comparison."""
    import h5py

    if not path.exists():
        raise FileNotFoundError(path)

    with h5py.File(path, "r") as f:
        chain = f["mcmc/chain"][:]
    flat = chain.reshape(-1, chain.shape[-1])
    stats = {}
    names = ["H0", "Omega_m"]
    for idx, name in enumerate(names):
        samples = flat[:, idx]
        stats[name] = {
            "median": float(np.median(samples)),
            "p16": float(np.percentile(samples, 16)),
            "p84": float(np.percentile(samples, 84)),
        }
    return stats


def summarize_samples(samples: Dict[str, np.ndarray], params: Tuple[str, ...]) -> Dict[str, Dict[str, float]]:
    """Compute median and 16/84 percentiles for selected parameters."""
    summary: Dict[str, Dict[str, float]] = {}
    for name in params:
        vals = np.asarray(samples[name])
        summary[name] = {
            "median": float(np.median(vals)),
            "p16": float(np.percentile(vals, 16)),
            "p84": float(np.percentile(vals, 84)),
        }
    return summary


def print_summary(summary: Dict[str, Dict[str, float]]) -> None:
    header = f"{'Parameter':<12} | {'Median':>8} | {'-σ':>6} | {'+σ':>6}"
    print("\n" + header)
    print("-" * len(header))
    for name, stats in summary.items():
        median = stats["median"]
        err_lo = median - stats["p16"]
        err_hi = stats["p84"] - median
        print(f"{name:<12} | {median:8.2f} | {err_lo:6.2f} | {err_hi:6.2f}")
    print()


def pretty_print_comparison(new: Dict[str, Dict[str, float]], ref: Dict[str, Dict[str, float]]) -> None:
    """Print side-by-side comparison between HIcosmo and reference results."""
    header = (
        f"{'Parameter':<12} | {'HIcosmo median ±σ':<28} | "
        f"{'TDCOSMO median ±σ':<28} | Δ-median"
    )
    print("\n" + header)
    print("-" * len(header))
    for name in new:
        new_med = new[name]["median"]
        new_err_lo = new_med - new[name]["p16"]
        new_err_hi = new[name]["p84"] - new_med
        ref_med = ref[name]["median"]
        ref_err_lo = ref_med - ref[name]["p16"]
        ref_err_hi = ref[name]["p84"] - ref_med
        delta = new_med - ref_med
        print(
            f"{name:<12} | "
            f"{new_med:6.2f} -{new_err_lo:5.2f} +{new_err_hi:5.2f} | "
            f"{ref_med:6.2f} -{ref_err_lo:5.2f} +{ref_err_hi:5.2f} | "
            f"{delta:+6.2f}"
        )
    print()


def make_tdcosmo_model(like: TDCOSMOLikelihood, use_omega_m_prior: bool = False):
    """Construct NumPyro model mirroring the TDCOSMO hierarchical analysis."""

    lambda_lower, lambda_upper = [jnp.array(v, dtype=jnp.float32) for v in like.lambda_bounds]
    a_lower, a_upper = [jnp.array(v, dtype=jnp.float32) for v in like.anisotropy_bounds]
    kappa_bounds = {
        name: (jnp.array(lens.kappa_min, dtype=jnp.float32), jnp.array(lens.kappa_max, dtype=jnp.float32))
        for name, lens in like.lens_data.items()
    }

    def model() -> None:
        # Cosmological parameters
        H0 = numpyro.sample("H0", dist.Uniform(50.0, 110.0))
        Omega_m = numpyro.sample("Omega_m", dist.Uniform(0.05, 0.5))
        if use_omega_m_prior and like.omega_m_prior is not None:
            mu, sigma = like.omega_m_prior
            numpyro.factor("Omega_m_prior", dist.Normal(mu, sigma).log_prob(Omega_m))

        # Hyper-parameters for internal MST and anisotropy
        lambda_mean = numpyro.sample("lambda_int_mean", dist.Uniform(lambda_lower, lambda_upper))
        log_lambda_sigma = numpyro.sample(
            "log_lambda_sigma", dist.Uniform(jnp.log(0.001), jnp.log(0.5))
        )
        lambda_sigma = jnp.exp(log_lambda_sigma)
        alpha_lambda = numpyro.sample("alpha_lambda", dist.Uniform(-1.0, 1.0))

        a_mean = numpyro.sample("a_ani_mean", dist.Uniform(a_lower, a_upper))
        log_a_sigma = numpyro.sample(
            "log_a_ani_sigma", dist.Uniform(jnp.log(0.01), jnp.log(1.0))
        )
        a_sigma = jnp.exp(log_a_sigma)

        if like.log_scatter_prior:
            numpyro.factor("lambda_sigma_prior", -log_lambda_sigma)
            numpyro.factor("a_sigma_prior", -log_a_sigma)
        numpyro.factor("a_mean_prior", -jnp.log(jnp.maximum(a_mean, 1e-6)))

        # Collect parameters for likelihood evaluation
        params: Dict[str, jnp.ndarray] = {
            "lambda_int_mean": lambda_mean,
            "lambda_int_sigma": lambda_sigma,
            "alpha_lambda": alpha_lambda,
            "a_ani_mean": a_mean,
            "a_ani_sigma": a_sigma,
        }

        for name, lens in like.lens_data.items():
            low, high = kappa_bounds[name]
            base = dist.Uniform(low, high)
            kappa = numpyro.sample(f"kappa_ext_{name}", base)
            numpyro.factor(f"kappa_pdf_{name}", lens.kappa_logpdf(kappa) - base.log_prob(kappa))
            numpyro.factor(f"kappa_gauss_{name}", like.lens_priors[name].logpdf(kappa))

            lambda_loc = lambda_mean + alpha_lambda * jnp.asarray(lens.lambda_scaling, dtype=jnp.float32)
            lambda_param = numpyro.sample(
                f"lambda_int_{name}",
                dist.Normal(lambda_loc, lambda_sigma),
            )
            a_param = numpyro.sample(
                f"a_ani_{name}",
                dist.Normal(a_mean, a_sigma),
            )

            params[f"kappa_ext_{name}"] = kappa
            params[f"lambda_int_{name}"] = lambda_param
            params[f"a_ani_{name}"] = a_param

        numpyro.deterministic("lambda_int_sigma", lambda_sigma)
        numpyro.deterministic("a_ani_sigma", a_sigma)

        cosmo = LCDM(H0=H0, Omega_m=Omega_m, Omega_b=like.default_omega_b)
        log_like = like.log_likelihood(cosmo, include_priors=False, **params)
        numpyro.factor("tdcosmo_likelihood", log_like)

    return model


def build_initial_params(like: TDCOSMOLikelihood) -> Dict[str, jnp.ndarray]:
    defaults = like.default_parameters()
    init: Dict[str, np.ndarray] = {
        "H0": np.array(73.0, dtype=np.float32),
        "Omega_m": np.array(0.3, dtype=np.float32),
        "lambda_int_mean": np.array(1.0, dtype=np.float32),
        "log_lambda_sigma": np.log(np.array(0.05, dtype=np.float32)),
        "alpha_lambda": np.array(0.0, dtype=np.float32),
        "a_ani_mean": np.array(1.0, dtype=np.float32),
        "log_a_ani_sigma": np.log(np.array(0.1, dtype=np.float32)),
    }
    for name in like.lens_names:
        lens = like.lens_data[name]
        low = lens.kappa_min
        high = lens.kappa_max
        base = float(defaults[f"kappa_ext_{name}"])
        safe = float(np.clip(base, low + 1e-4, high - 1e-4))
        init[f"kappa_ext_{name}"] = np.array(safe, dtype=np.float32)
        lam_base = float(defaults[f"lambda_int_{name}"])
        lam_safe = float(np.clip(lam_base, like.lambda_bounds[0] + 1e-3, like.lambda_bounds[1] - 1e-3))
        init[f"lambda_int_{name}"] = np.array(lam_safe, dtype=np.float32)
        ani_base = float(defaults[f"a_ani_{name}"])
        ani_safe = float(np.clip(ani_base, like.anisotropy_bounds[0] + 1e-3, like.anisotropy_bounds[1] - 1e-3))
        init[f"a_ani_{name}"] = np.array(ani_safe, dtype=np.float32)
    return init


def run_inference(like: TDCOSMOLikelihood, use_omega_m_prior: bool, num_samples: int, num_warmup: int,
                  num_chains: int, seed: int) -> Dict[str, np.ndarray]:
    model = make_tdcosmo_model(like, use_omega_m_prior=use_omega_m_prior)
    init_vals = build_initial_params(like)
    kernel = SA(
        model,
        init_strategy=init_to_value(values=init_vals),
    )
    sampler = MCMCSampler(
        model,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        chain_method="sequential",
        kernel=kernel,
    )
    rng_key = jax.random.PRNGKey(seed)
    sampler.run(rng_key)
    raw_samples = sampler.get_samples()
    return {key: np.asarray(jax.device_get(val)) for key, val in raw_samples.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run HIcosmo TDCOSMO hierarchical inference.")
    parser.add_argument("--omega-prior", action="store_true", help="Apply Gaussian prior on Omega_m")
    parser.add_argument("--samples", type=int, default=4000, help="Total samples per chain")
    parser.add_argument("--warmup", type=int, default=4000, help="Warm-up steps per chain")
    parser.add_argument("--chains", type=int, default=2, help="Number of MCMC chains")
    parser.add_argument("--seed", type=int, default=17, help="Random seed for NumPyro")
    args = parser.parse_args()

    init_hicosmo()
    like = TDCOSMOLikelihood(omega_m_prior=(0.298, 0.022))

    print("Running NumPyro inference ...")
    samples = run_inference(
        like,
        use_omega_m_prior=args.omega_prior,
        num_samples=args.samples,
        num_warmup=args.warmup,
        num_chains=args.chains,
        seed=args.seed,
    )

    target_params = ("H0", "Omega_m")
    summary = summarize_samples(samples, target_params)
    print("HIcosmo posterior summary:")
    print_summary(summary)

    if REFERENCE_CHAIN.exists() and not args.omega_prior:
        ref_stats = load_reference_statistics(REFERENCE_CHAIN)
        print("Comparison with published TDCOSMO chain (alpha free, uniform Ω_m):")
        pretty_print_comparison(summary, ref_stats)
    else:
        print("Reference comparison skipped (chain missing or Omega_m prior enabled).")

    # Corner plot for key parameters
    viz = HIcosmoViz()
    corner_params = list(target_params)
    viz.corner(samples, params=corner_params, filename="tdcosmo_corner.pdf")

    # Persist samples for downstream analysis
    output_path = Path("results/tdcosmo_numpyro_samples.npz")
    np.savez(output_path, **{k: samples[k] for k in samples})
    print(f"Samples saved to {output_path}")


if __name__ == "__main__":
    main()
