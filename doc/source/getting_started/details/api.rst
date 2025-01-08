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

    * :py:class:`pe.api.StableDiffusion`: The APIs of `Stable Diffusion`_.
    * :py:class:`pe.api.ImprovedDiffusion`: The APIs of the `improved diffusion model`_.

* Text
    
    * :py:class:`pe.api.LLMAugPE`: The APIs for text generation using LLMs. When constructing the instance of this API, an LLM instance is required. The LLM instances follow the interface of :py:class:`pe.llm.LLM`. Currently, the following LLMs are implemented:

        * :py:class:`pe.llm.OpenAILLM`: The LLMs from OpenAI APIs.
        * :py:class:`pe.llm.AzureOpenAILLM`: The LLMs from Azure OpenAI APIs.
        * :py:class:`pe.llm.HuggingfaceLLM`: The open-source LLMs from Huggingface.

Adding Your Own APIs
--------------------

To add your own APIs, you need to create a class that inherits from :py:class:`pe.api.API` and implements the :py:meth:`pe.api.API.random_api` and :py:meth:`pe.api.API.variation_api` methods.

    
.. _improved diffusion model: https://github.com/openai/improved-diffusion
.. _Stable Diffusion: https://huggingface.co/CompVis/stable-diffusion-v1-4
