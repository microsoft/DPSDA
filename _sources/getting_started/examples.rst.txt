Examples
========

Here are some examples of how to use the **Private Evolution** library.

Images
------

These examples follow the experimental settings in the paper `Differentially Private Synthetic Data via Foundation Model APIs 1: Images (ICLR 2024) <pe1_paper_>`__.

* **CIFAR10 dataset**: `This example <CIFAR10 example_>`__ shows how to generate differentially private synthetic images for the `CIFAR10 dataset`_ using the APIs from a pre-trained `ImageNet diffusion model`_.

* **Camelyon17 dadtaset**: `This example <Camelyon17 example_>`__ shows how to generate differentially private synthetic images for the `Camelyon17 dataset`_ using the APIs from a pre-trained `ImageNet diffusion model`_.

* **Cat dataset**: `This example <Cat example_>`__ shows how to generate differentially private synthetic images of the `Cat dataset`_ using the APIs from `Stable Diffusion`_.

Text
----

These examples follow the experimental settings in the paper `Differentially Private Synthetic Data via Foundation Model APIs 2: Text (ICML 2024 Spotlight) <pe2_paper_>`__.

* **Yelp dataset**: These examples show how to generate differentially private synthetic text for the `Yelp dataset`_ using LLM APIs from:

    * **OpenAI APIs**: `See example <Yelp OpenAI example_>`__
    * **Huggingface models**: `See example <Yelp Huggingface example_>`__

* **OpenReview dataset**: These examples show how to generate differentially private synthetic text for the `OpenReview dataset`_ using LLM APIs from:

    * **OpenAI APIs**: `See example <Openreview OpenAI example_>`__
    * **Huggingface models**: `See example <Openreview Huggingface example_>`__

* **PubMed dataset**: These examples show how to generate differentially private synthetic text for the `PubMed dataset`_ using LLM APIs from:

    * **OpenAI APIs**: `See example <PubMed OpenAI example_>`__
    * **Huggingface models**: `See example <PubMed Huggingface example_>`__


.. _ImageNet diffusion model: https://github.com/openai/improved-diffusion
.. _Stable Diffusion: https://huggingface.co/CompVis/stable-diffusion-v1-4

.. _Cat dataset: https://www.kaggle.com/datasets/fjxmlzn/cat-cookie-doudou
.. _CIFAR10 dataset: https://www.cs.toronto.edu/~kriz/cifar.html
.. _Camelyon17 dataset: https://camelyon17.grand-challenge.org/
.. _Yelp dataset: https://github.com/AI-secure/aug-pe/tree/main/data
.. _OpenReview dataset: https://github.com/AI-secure/aug-pe/tree/main/data
.. _PubMed dataset: https://github.com/AI-secure/aug-pe/tree/main/data

.. _CIFAR10 example: https://github.com/microsoft/DPSDA/blob/main/example/image/cifar10_diffusion.py
.. _Camelyon17 example: https://github.com/microsoft/DPSDA/blob/main/example/image/camelyon17_diffusion.py
.. _Cat example: https://github.com/microsoft/DPSDA/blob/main/example/image/cat_diffusion.py
.. _Yelp OpenAI example: https://github.com/microsoft/DPSDA/blob/main/example/text/yelp_openai/main.py
.. _Yelp Huggingface example: https://github.com/microsoft/DPSDA/blob/main/example/text/yelp_huggingface/main.py
.. _Openreview OpenAI example: https://github.com/microsoft/DPSDA/blob/main/example/text/openreview_openai/main.py
.. _Openreview Huggingface example: https://github.com/microsoft/DPSDA/blob/main/example/text/openreview_huggingface/main.py
.. _PubMed OpenAI example: https://github.com/microsoft/DPSDA/blob/main/example/text/pubmed_openai/main.py
.. _PubMed Huggingface example: https://github.com/microsoft/DPSDA/blob/main/example/text/pubmed_huggingface/main.py

.. _pe1_paper: https://arxiv.org/abs/2305.15560
.. _pe2_paper: https://arxiv.org/abs/2403.01749