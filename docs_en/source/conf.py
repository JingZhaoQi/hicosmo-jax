import os
import sys

# Project root directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

project = "HIcosmo"
author = "HIcosmo Team"
release = "0.1.0"

language = "en"
master_doc = "index"
source_suffix = ".rst"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.intersphinx",
]

autosummary_generate = True
autodoc_typehints = "description"
autosectionlabel_prefix_document = True

autodoc_mock_imports = []

# CI / Read the Docs environment does not provide scientific deps
if os.environ.get("READTHEDOCS", "").lower() == "true":
    autodoc_mock_imports = [
        "numpy",
        "jax",
        "jaxlib",
        "numpyro",
        "getdist",
        "matplotlib",
        "scipy",
        "astropy",
        "h5py",
        "torch",
        "camb",
        "classy",
        "dynesty",
        "emcee",
        "corner",
        "pandas",
        "tqdm",
        "rich",
        "yaml",
        "hicosmo",
    ]
    autosummary_generate = False

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

templates_path = ["_templates"]
html_static_path = ["_static"]

html_theme = "sphinx_rtd_theme"
html_title = "HIcosmo Documentation"

html_css_files = ["lang_toggle.css"]
html_js_files = ["lang_toggle.js"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}
