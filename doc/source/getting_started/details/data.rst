Data
====

API reference: :doc:`/api/pe.data`.

:py:class:`pe.data.data.Data` is the base class for holding the synthetic samples or the private samples, along with their metadata. Different components are mostly communicated through objects of this class.
:py:class:`pe.data.data.Data` has two key attributes:

* ``data_frame``: A pandas_ DataFrame that holds the samples. Each row in the DataFrame is a sample, and each column is part of the sample (e.g., the image, the text, the label) and other information of the sample (e.g., its embedding produced by :doc:`embedding`).
* ``metadata``: A OmegaConf_ that holds the metadata of the samples, such as the **Private Evolution** iteration number when the samples are generated, and the label names of the classes.

Available Datasets
------------------

For convenience, some well-known datasets are already packaged as `pe.data.data.Data` classes:

* Image datasets

    * :py:class:`pe.data.image.cifar10.Cifar10`: The `CIFAR10 dataset`_.
    * :py:class:`pe.data.image.camelyon17.Camelyon17`: The `Camelyon17 dataset`_.
    * :py:class:`pe.data.image.cat.Cat`: The `Cat dataset`_.
    * In addition, you can easily load a custom image dataset from a (nested) directory with the image files using :py:meth:`pe.data.image.image.load_image_folder`.

* Text datasets
    
        * Coming soon!

Using Your Own Datasets
-----------------------
To apply **Private Evolution** to your own private dataset, you need to create a :py:class:`pe.data.data.Data` object that holds your dataset, with two parameters, ``data_frame`` and ``metadata``, passed to the constructor:

* ``data_frame``: A pandas_ DataFrame that holds the samples. Each row in the DataFrame is a sample. The following columns must be included:

    * :py:attr:`pe.constant.data.LABEL_ID_COLUMN_NAME`: The label (class) ID of the sample. The label IDs must be in {0, 1, ..., K-1} if there are K classes. If you are targeting unconditional generation, the values of this column can just be zeros.

    The ``data_frame`` can have any numbers of additional columns that hold the data of the samples, as long as the modules you are using (e.g., :doc:`api`, :doc:`Callbacks <callback_and_logger>`) can recognize them.

* ``metadata``: A dictionary that holds the metadata of the samples. The following keys must be included:

    * ``label_names``: A list of strings that holds the names of the classes. The length of the list must be equal to K.

    In addition, you can include any other keys that hold the metadata of the samples if needed.


.. _OmegaConf: https://omegaconf.readthedocs.io/en/latest/
.. _pandas: https://pandas.pydata.org/
.. _Cat dataset: https://www.kaggle.com/datasets/fjxmlzn/cat-cookie-doudou
.. _CIFAR10 dataset: https://www.cs.toronto.edu/~kriz/cifar.html
.. _Camelyon17 dataset: https://camelyon17.grand-challenge.org/
