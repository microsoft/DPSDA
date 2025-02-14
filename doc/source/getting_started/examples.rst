Examples
========

Here are some examples of how to use the **Private Evolution** library.

Images
------

* Using **foundation models (diffusion models)** as the APIs. These examples follow the experimental settings in the paper `Differentially Private Synthetic Data via Foundation Model APIs 1: Images (ICLR 2024) <pe1_paper_>`__.

    * **CIFAR10 dataset**: `This example <CIFAR10 example_>`__ shows how to generate differentially private synthetic images for the `CIFAR10 dataset`_ using the APIs from a pre-trained `ImageNet diffusion model`_.
    * **Camelyon17 dataset**: `This example <Camelyon17 example_>`__ shows how to generate differentially private synthetic images for the `Camelyon17 dataset`_ using the APIs from a pre-trained `ImageNet diffusion model`_.
    * **Cat dataset**: `This example <Cat example_>`__ shows how to generate differentially private synthetic images for the `Cat dataset`_ using the APIs from `Stable Diffusion`_.

* Using **simulators** as the APIs. These examples follow the experimental settings in the paper `Differentially Private Synthetic Data via APIs 3: Using Simulators Instead of Foundation Models <pe3_paper_>`__.

    * **MNIST dataset**: `This example <MNIST example_>`__ shows how to generate differentially private synthetic images for the `MNIST dataset`_ using a text render.
    * **CelebA dataset (simulator-generated data)**: `This example <CelebA DigiFace1M example_>`__ shows how to generate differentially private synthetic images for the `CelebA dataset`_ using `the generated data from a computer graphics-based renderer for face images <DigiFace1M_>`__.
    * **CelebA dataset (weak simulator)**: `This example <CelebA avatar example_>`__ shows how to generate differentially private synthetic images for the `CelebA dataset`_ using `a rule-based avatar generator <python_avatars_>`__.

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

.. _DigiFace1M: https://github.com/microsoft/DigiFace1M
.. _python_avatars: https://github.com/ibonn/python_avatars

.. _Cat dataset: https://www.kaggle.com/datasets/fjxmlzn/cat-cookie-doudou
.. _CIFAR10 dataset: https://www.cs.toronto.edu/~kriz/cifar.html
.. _Camelyon17 dataset: https://camelyon17.grand-challenge.org/

.. _MNIST dataset: https://pytorch.org/vision/main/generated/torchvision.datasets.MNIST.html
.. _CelebA dataset: https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

.. _Yelp dataset: https://github.com/AI-secure/aug-pe/tree/main/data
.. _OpenReview dataset: https://github.com/AI-secure/aug-pe/tree/main/data
.. _PubMed dataset: https://github.com/AI-secure/aug-pe/tree/main/data

.. _CIFAR10 example: https://github.com/microsoft/DPSDA/blob/main/example/image/diffusion_model/cifar10_improved_diffusion.py
.. _Camelyon17 example: https://github.com/microsoft/DPSDA/blob/main/example/image/diffusion_model/camelyon17_improved_diffusion.py
.. _Cat example: https://github.com/microsoft/DPSDA/blob/main/example/image/diffusion_model/cat_stable_diffusion.py

.. _MNIST example: https://github.com/microsoft/DPSDA/blob/main/example/image/simulator/mnist_text_render.py
.. _CelebA DigiFace1M example: https://github.com/microsoft/DPSDA/blob/main/example/image/simulator/celeba_digiface1m.py
.. _CelebA avatar example: https://github.com/microsoft/DPSDA/blob/main/example/image/simulator/celeba_avatar.py

.. _Yelp OpenAI example: https://github.com/microsoft/DPSDA/blob/main/example/text/yelp_openai/main.py
.. _Yelp Huggingface example: https://github.com/microsoft/DPSDA/blob/main/example/text/yelp_huggingface/main.py
.. _Openreview OpenAI example: https://github.com/microsoft/DPSDA/blob/main/example/text/openreview_openai/main.py
.. _Openreview Huggingface example: https://github.com/microsoft/DPSDA/blob/main/example/text/openreview_huggingface/main.py
.. _PubMed OpenAI example: https://github.com/microsoft/DPSDA/blob/main/example/text/pubmed_openai/main.py
.. _PubMed Huggingface example: https://github.com/microsoft/DPSDA/blob/main/example/text/pubmed_huggingface/main.py


.. _pe1_paper: https://arxiv.org/abs/2305.15560
.. _pe2_paper: https://arxiv.org/abs/2403.01749
.. _pe3_paper: https://arxiv.org/abs/2502.05505