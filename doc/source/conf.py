# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Private Evolution"
copyright = "2024, Zinan Lin"
author = "Zinan Lin"
release = "0.0.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

templates_path = ["_templates"]
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"  # 'alabaster'
html_static_path = ["_static"]
html_favicon = "icon/favicon.ico"
html_logo = "icon/icon.png"

html_theme_options = {"navigation_depth": -1, "collapse_navigation": False}

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.ifconfig",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "sphinx.ext.napoleon",
    "sphinx_toolbox.more_autodoc.autonamedtuple",
]

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = True
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

numfig = True
nitpicky = True
nitpick_ignore = [
    ("py:class", "optional"),
    ("py:class", "abc.ABC"),
    ("py:class", "np.ndarray"),
    ("py:class", "fastchat.conversation.Conversation"),
    ("py:class", "fastchat.model.model_adapter.BaseModelAdapter"),
    ("py:class", "torch.utils.data.dataset.Dataset"),
    ("py:class", "torch.nn.modules.module.Module"),
    ("py:class", "Module"),
    ("py:class", "pandas.DataFrame"),
    ("py:class", "improved_diffusion.respace.SpacedDiffusion"),
    ("py:class", "improved_diffusion.unet.UNetModel"),
    ("py:class", "omegaconf.dictconfig.DictConfig"),
    ("py:class", "python_avatars.Avatar"),
    ("py:class", "torch.utils.data.DataLoader"),
    ("py:class", "torch.nn.Module"),
]
