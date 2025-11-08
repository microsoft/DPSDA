APIs
====

API reference: :doc:`/api/pe.api`

:py:class:`pe.api.API` is responsible for implementing the foundation model APIs. It has the following key methods:

* :py:meth:`pe.api.API.random_api`: Randomly generates the synthetic samples for the initial samples of the **Private Evolution** algorithm.
* :py:meth:`pe.api.API.variation_api`: Generates the variations of the given synthetic samples for the initial or the next **Private Evolution** iteration.

Available APIs
--------------

Currently, the following APIs are implemented:

* Images

    * :py:class:`pe.api.StableDiffusion`: The APIs of `Stable Diffusion`_ (introduced in [#pe1]_).
    * :py:class:`pe.api.ImprovedDiffusion`: The APIs of the `improved diffusion model`_ (introduced in [#pe1]_).
    * :py:class:`pe.api.DrawText`: The APIs that render text on images (introduced in [#pe3]_).
    * :py:class:`pe.api.Avatar`: The APIs that generate avatars using `the Python Avatars library <python_avatars_>`__ (introduced in [#pe3]_).
    * :py:class:`pe.api.NearestImage`: The APIs that utilize a static image dataset (introduced in [#pe3]_).

* Text
    
    * :py:class:`pe.api.LLMAugPE`: The APIs for text generation using LLMs (introduced in [#pe2]_). When constructing the instance of this API, an LLM instance is required. The LLM instances follow the interface of :py:class:`pe.llm.LLM`. Currently, the following LLMs are implemented:

        * :py:class:`pe.llm.OpenAILLM`: The LLMs from OpenAI APIs.
        * :py:class:`pe.llm.AzureOpenAILLM`: The LLMs from Azure OpenAI APIs.
        * :py:class:`pe.llm.HuggingfaceLLM`: The open-source LLMs from Huggingface.

Adding Your Own APIs
--------------------

To add your own APIs, you need to create a class that inherits from :py:class:`pe.api.API` and implements the :py:meth:`pe.api.API.random_api` and :py:meth:`pe.api.API.variation_api` methods.


.. rubric:: Citations

.. [#pe1] `Differentially Private Synthetic Data via Foundation Model APIs 1: Images (ICLR 2024) <pe1_paper_>`__.
.. [#pe2] `Differentially Private Synthetic Data via Foundation Model APIs 2: Text (ICML 2024 Spotlight) <pe2_paper_>`__.
.. [#pe3] `Differentially Private Synthetic Data via APIs 3: Using Simulators Instead of Foundation Models <pe3_paper_>`__.


.. _improved diffusion model: https://github.com/openai/improved-diffusion
.. _Stable Diffusion: https://huggingface.co/CompVis/stable-diffusion-v1-4

.. _python_avatars: https://github.com/ibonn/python_avatars

.. _pe1_paper: https://arxiv.org/abs/2305.15560
.. _pe2_paper: https://arxiv.org/abs/2403.01749
.. _pe3_paper: https://arxiv.org/abs/2502.05505
