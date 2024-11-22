APIs
====

API reference: :doc:`/api/pe.api`

:py:class:`pe.api.api.API` is responsible for implementing the foundation model APIs. It has the following key methods:

* :py:meth:`pe.api.api.API.random_api`: Randomly generates the synthetic samples for the initial samples of the **Private Evolution** algorithm.
* :py:meth:`pe.api.api.API.variation_api`: Generates the variations of the given synthetic samples for the initial or the next **Private Evolution** iteration.

Available APIs
--------------

Currently, the following APIs are implemented:

* Images

    * :py:class:`pe.api.image.stable_diffusion_api.StableDiffusion`: The APIs of `Stable Diffusion`_.
    * :py:class:`pe.api.image.improved_diffusion_api.ImprovedDiffusion`: The APIs of the `improved diffusion model`_.

* Text
    
    * Coming soon!

Adding Your Own APIs
--------------------

To add your own APIs, you need to create a class that inherits from :py:class:`pe.api.api.API` and implements the :py:meth:`pe.api.api.API.random_api` and :py:meth:`pe.api.api.API.variation_api` methods.

    
.. _improved diffusion model: https://github.com/openai/improved-diffusion
.. _Stable Diffusion: https://huggingface.co/CompVis/stable-diffusion-v1-4
