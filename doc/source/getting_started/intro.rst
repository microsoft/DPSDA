What is Private Evolution?
===============================

**Private Evolution** (PE in short) is an algorithm for **generating differentially private synthetic data without the need of any ML model training**. 

Given a dataset, **Private Evolution** can generate a new synthetic dataset that is statistically similar to the original dataset, while ensuring a rigorous privacy guarantee called `differential privacy (DP) <https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf>`_, which implies that the privacy of individuals in the original dataset is protected. It is particularly useful in situations where the original data is sensitive or confidential, such as medical records, financial data, or personal information. The DP synthetic dataset can replace the original data in various use cases where privacy is a concern, for example:

* Sharing them with other parties for collaboration and research.
* Using them in downstream algorithms (e.g., training ML models) in the normal non-private pipeline.
* Inspecting the data directly for easier product debugging and development.

Key Features
------------

Compared to other DP synthetic data alternatives, **Private Evolution** has the following key features:

* ✅ **No training needed!** **Private Evolution** only requires the inference APIs of foundation models or non-neural-network data synthesis tools. Therefore, it can leverage any state-of-the-art black-box models (e.g., GPT-4), open-source models (e.g., Stable Diffusion, Llama), or tools (e.g., computer graphics-based image synthesis tools).
* ✅ **Protects privacy even from the API provider.** Even when using APIs from a third-party provider, you can rest assured that the information of individuals in the original dataset is still protected, as all API queries made from **Private Evolution** are also differentially private.
* ✅ **Works across images, text, etc.** **Private Evolution** can generate synthetic data for various data types, including images and text. More data modalities are coming soon!
* ✅ **Could even match/beat SoTA training-based methods in data quality.** **Private Evolution** can generate synthetic data that is statistically similar to the original data, and in some cases, it can even match or beat the state-of-the-art training-based methods in data quality even though it does not require any training.

What This Library Provides
--------------------------

**This library is the official Python package of Private Evolution**. It allows you to generate differentially private synthetic data (e.g., images, text) using the **Private Evolution** algorithm. This library is designed to be easy to use, flexible, modular, and extensible. It provides several popular foundation models and data synthesis tools, and you can easily extend it to work with your own foundation models (and/or APIs), data synthesis tools, data types, or new **Private Evolution** algorithms if needed.

The source code of this **Private Evolution** library is available at https://github.com/microsoft/DPSDA.

Citations
---------

If you use **Private Evolution** in your research or work, please cite the following papers:

.. literalinclude:: pe1.bib
    :language: bibtex

.. literalinclude:: pe2.bib
    :language: bibtex

.. literalinclude:: pe3.bib
    :language: bibtex

Please see https://github.com/fjxmlzn/private-evolution-papers for the full list of **Private Evolution** papers and code repositories done by the community.
