#!/usr/bin/env python3
"""
HIcosmo Interactive Command-Line Interface
==========================================

Provides an interactive REPL environment for cosmological parameter estimation.

Usage:
    $ hicosmo

This launches an interactive Python session with HIcosmo pre-loaded.
"""

import sys
import os
from pathlib import Path
import code
import readline
import rlcompleter

try:
    from rich.console import Console
    from rich.text import Text

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


def get_default_output_paths():
    """Get default output directory paths (without creating them)."""
    cwd = Path.cwd()

    return {
        "results": cwd / "hicosmo_results",
        "chains": cwd / "mcmc_chains",
        "plots": cwd / "plots",
    }


def create_gradient_logo():
    """Create ASCII art logo with blue gradient."""
    if not RICH_AVAILABLE:
        return "\n  HIcosmo\n"

    # Hand-crafted minimalist ASCII logo with blue gradient
    # Clean and simple design
    LOGO_LINES = [
        "\033[38;2;78;168;255m██╗  ██╗ ██╗ ██████╗  ██████╗ ███████╗ ███╗   ███╗  ██████╗ \033[0m",
        "\033[38;2;99;154;255m██║  ██║ ██║██╔════╝ ██╔═══██╗██╔════╝ ████╗ ████║ ██╔═══██╗\033[0m",
        "\033[38;2;113;145;255m███████║ ██║██║      ██║   ██║███████╗ ██╔████╔██║ ██║   ██║\033[0m",
        "\033[38;2;120;141;255m██╔══██║ ██║██║      ██║   ██║╚════██║ ██║╚██╔╝██║ ██║   ██║\033[0m",
        "\033[38;2;127;136;255m██║  ██║ ██║╚██████╗ ╚██████╔╝███████║ ██║ ╚═╝ ██║ ╚██████╔╝\033[0m",
        "\033[38;2;127;136;255m╚═╝  ╚═╝ ╚═╝ ╚═════╝  ╚═════╝ ╚══════╝ ╚═╝     ╚═╝  ╚═════╝ \033[0m",
    ]

    # Print using standard print to preserve ANSI codes
    print()  # Top padding
    for line in LOGO_LINES:
        print(f"  {line}")

    return ""


def create_banner():
    """Create welcome banner with minimalist gradient logo."""

    # Print the gradient logo if rich is available
    if RICH_AVAILABLE:
        create_gradient_logo()
        logo_text = ""  # Logo already printed
    else:
        logo_text = "\n  HIcosmo\n"

    banner = logo_text + """
  Neutral Hydrogen (HI) Cosmology Parameter Constraints & Forecast
  Powered by JAX + NumPyro
  ───────────────────────────────────────────────────────────────

Working directory: {cwd}

Quick start:
  >>> inference = hicosmo(
  ...     cosmology='LCDM',
  ...     likelihood='sn',
  ...     free_params=['H0', 'Omega_m']
  ... )
  >>> samples = inference.run()
  >>> inference.summary()
  >>> inference.corner_plot('plots/corner.pdf')

Commands:
  list_cosmologies()   - Show all supported cosmology models & parameters
  list_likelihoods()   - Show all supported likelihood strings

Default output directories (created when needed):
  - Results: {results}
  - Chains:  {chains}
  - Plots:   {plots}

Type 'help(hicosmo)' for documentation, 'exit()' to quit.
"""

    # Get default output paths (without creating directories)
    workspace = get_default_output_paths()

    return banner.format(
        cwd=Path.cwd(),
        results=workspace["results"],
        chains=workspace["chains"],
        plots=workspace["plots"],
    )


def prepare_namespace():
    """Prepare interactive namespace with common imports."""
    namespace = {}

    # Import HIcosmo API
    try:
        from hicosmo import hicosmo, InferenceRunner, list_likelihoods, list_cosmologies
        from hicosmo.models import LCDM
        from hicosmo.likelihoods import SN_likelihood

        namespace.update(
            {
                "hicosmo": hicosmo,
                "InferenceRunner": InferenceRunner,
                "list_likelihoods": list_likelihoods,
                "list_cosmologies": list_cosmologies,
                "LCDM": LCDM,
                "SN_likelihood": SN_likelihood,
            }
        )

        # Try to import additional models
        try:
            from hicosmo.models import wCDM, CPL

            namespace["wCDM"] = wCDM
            namespace["CPL"] = CPL
        except ImportError:
            pass

        # Try to import additional likelihoods
        try:
            from hicosmo.likelihoods import (
                BAO_likelihood,
                Planck2018DistancePriorsLikelihood,
            )

            namespace["BAO_likelihood"] = BAO_likelihood
            namespace["Planck2018DistancePriorsLikelihood"] = (
                Planck2018DistancePriorsLikelihood
            )
        except ImportError:
            pass

    except ImportError as e:
        print(f"Warning: Could not import HIcosmo: {e}")
        print("Please ensure HIcosmo is installed: pip install -e .")
        sys.exit(1)

    # Add common scientific libraries
    try:
        import numpy as np
        import matplotlib.pyplot as plt

        namespace["np"] = np
        namespace["plt"] = plt
    except ImportError:
        pass

    # Add Path for convenience
    namespace["Path"] = Path

    return namespace


def main():
    """Main entry point for HIcosmo interactive shell."""
    # Enable tab completion
    readline.set_completer(rlcompleter.Completer().complete)
    readline.parse_and_bind("tab: complete")

    # Prepare namespace
    namespace = prepare_namespace()

    # Create banner
    banner = create_banner()

    # Start interactive console
    console = code.InteractiveConsole(locals=namespace)

    try:
        console.interact(banner=banner, exitmsg="\n👋 Thanks for using HIcosmo!\n")
    except SystemExit:
        print("\n👋 Thanks for using HIcosmo!\n")
    except KeyboardInterrupt:
        print("\n\n👋 Thanks for using HIcosmo!\n")


if __name__ == "__main__":
    main()
