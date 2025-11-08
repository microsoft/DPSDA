Installation
============

PIP
---

The Main Package
^^^^^^^^^^^^^^^^

To install the core package of **Private Evolution**, please use the following command:

.. code-block:: bash

    pip install "private-evolution @ git+https://github.com/microsoft/DPSDA.git"

Image Generation
^^^^^^^^^^^^^^^^

If you are using **Private Evolution** to generate **images**, use the following command instead to install the package with the necessary dependencies:

.. code-block:: bash

    pip install "private-evolution[image] @ git+https://github.com/microsoft/DPSDA.git"

Text Generation
^^^^^^^^^^^^^^^

If you are using **Private Evolution** to generate **text**, use the following command instead to install the package with the necessary dependencies:

.. code-block:: bash

    pip install "private-evolution[text] @ git+https://github.com/microsoft/DPSDA.git"

Multiple dependencies can also be combined, such as:

.. code-block:: bash

    pip install "private-evolution[image,text] @ git+https://github.com/microsoft/DPSDA.git"

Editable Mode
^^^^^^^^^^^^^

To install **Private Evolution** in editable mode, please use the following command:

.. code-block:: bash

    git clone https://github.com/microsoft/DPSDA.git
    cd DPSDA
    pip install -e .[option]

where `option` can be `image`, `text`, or `image,text`.


Faiss
-----

**Private Evolution** requires a nearest neighbor search process. By default, it uses the sklearn_ package for this purpose. However, for faster computation, we recommend using the faiss_ package.
To install `faiss 1.8.0`, please use the following command:

.. code-block:: bash

    conda install -y -c pytorch -c nvidia faiss-gpu=1.8.0

Please check out the faiss_ website for the latest information on how to install the package.

..
    Docker
    ------

    We provide Docker images for **Private Evolution** with all dependencies (including faiss_) pre-installed. To pull the Docker image, please use the following command:

    TODO

.. _faiss: https://faiss.ai/
.. _sklearn: https://scikit-learn.org/dev/modules/generated/sklearn.neighbors.NearestNeighbors.html