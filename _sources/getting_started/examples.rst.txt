Examples
========

Here are some examples of how to use the **Private Evolution** library.

Images
------

These examples follow the experimental settings in the paper `Differentially Private Synthetic Data via Foundation Model APIs 1: Images (ICLR 2024) <paper_>`__.

* **CIFAR10**: `This example <CIFAR10 example_>`__ shows how to generate differentially private synthetic images for the `CIFAR10 dataset`_ using the APIs from a pre-trained `ImageNet diffusion model`_.

* **Camelyon17**: `This example <Camelyon17 example_>`__ shows how to generate differentially private synthetic images for the `Camelyon17 dataset`_ using the APIs from a pre-trained `ImageNet diffusion model`_.

* **Cat**: `This example <Cat example_>`__ shows how to generate differentially private synthetic images of the `Cat dataset`_ using the APIs from `Stable Diffusion`_.

Text
----

Coming soon!

.. _ImageNet diffusion model: https://github.com/openai/improved-diffusion
.. _Stable Diffusion: https://huggingface.co/CompVis/stable-diffusion-v1-4
.. _Cat dataset: https://www.kaggle.com/datasets/fjxmlzn/cat-cookie-doudou
.. _CIFAR10 dataset: https://www.cs.toronto.edu/~kriz/cifar.html
.. _Camelyon17 dataset: https://camelyon17.grand-challenge.org/
.. _CIFAR10 example: https://github.com/microsoft/DPSDA/blob/main/example/image/cifar10.py
.. _Camelyon17 example: https://github.com/microsoft/DPSDA/blob/main/example/image/camelyon17.py
.. _Cat example: https://github.com/microsoft/DPSDA/blob/main/example/image/cat.py
.. _paper: https://arxiv.org/abs/2305.15560