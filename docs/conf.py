"""
Configuration file for the Sphinx documentation builder.
"""

#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

from sphinx_astropy.conf.v2 import *

project = "wfss-tools"
copyright = "2025, Peter J. Watson"
author = "Peter J. Watson"

import sys
from importlib.metadata import version as get_version

release: str = get_version("wfss-tools")
# for example take major/minor
version: str = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# extensions = [
#     "sphinx.ext.autosummary",
#     "sphinx.ext.autodoc",
#     "sphinx.ext.autosectionlabel",
#     "sphinx.ext.intersphinx",
#     "numpydoc",
#     "sphinx_copybutton",
#     "sphinx_automodapi.automodapi",
# ]

automodapi_inheritance_diagram = False

# autoapi_dirs = ["../src"]

# Avoid ambiguous section headings, prefix with document name
autosectionlabel_prefix_document = True

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'furo'
# html_theme = 'sphinx_rtd'
html_theme = "pydata_sphinx_theme"
html_static_path = []

# Don't show typehints in description or signature
autodoc_typehints = "none"
autodoc_member_order = "bysource"

# Create xrefs for parameter types in docstrings
numpydoc_xref_param_type = True
# Don't make a table of contents for all class methods and attributes
numpydoc_class_members_toctree = False
# Don't show all class members in the methods and attributes list
numpydoc_show_class_members = False
# Don't show inherited class members either
numpydoc_show_inherited_class_members = False

version_link = f"{sys.version_info.major}.{sys.version_info.minor}"
# intersphinx_mapping += {
#     "python": (
#         f"https://docs.python.org/{version_link}",
#         None,
#     ),  # link to used Python version
#     "numpy": ("https://numpy.org/doc/stable", None),
#     "scipy": ("https://docs.scipy.org/doc/scipy", None),
#     "matplotlib": ("https://matplotlib.org/stable", None),
#     "astropy": ("https://docs.astropy.org/en/stable", None),
intersphinx_mapping |= {
    "grizli": ("https://grizli.readthedocs.io/en/latest", None),
    "bagpipes": ("https://bagpipes.readthedocs.io/en/latest", None),
    "pypher": ("https://pypher.readthedocs.io/en/latest/", None),
    "stwcs": ("https://stwcs.readthedocs.io/en/latest/", None),
}

# Any `...` defaults to a link
default_role = "autolink"
