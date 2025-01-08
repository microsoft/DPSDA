Using Your Own Data/APIs
========================


To apply **Private Evolution** in your own data/domain/applications, most likely you only need to provide your own data (an object of :py:class:`pe.data.Data`) and APIs (an object of :py:class:`pe.api.API`).
The **Private Evolution** library is preloaded with popular data and APIs. You can also easily bring your own data and APIs. Here is how you can do it.

Data
----

* **Preloaded datasets**: Some well-known datasets are already packaged as :py:class:`pe.data.Data` classes. Please refer to :doc:`this document <details/data>` for more details.
* **New image datasets**: You can easily load a custom image dataset from a (nested) directory with the image files using :py:meth:`pe.data.load_image_folder`.
* **New text datasets**: You can easily load a custom text dataset from a CSV file using :py:class:`pe.data.TextCSV`.
* **Beyond the above**: You can create a :py:class:`pe.data.Data` object that holds your dataset, with two parameters, ``data_frame`` and ``metadata``, passed to the constructor. The ``data_frame`` is a pandas DataFrame that holds the samples, and the ``metadata`` is a dictionary that holds the metadata of the samples. Please refer to :doc:`this document <details/data>` for more details.

APIs
----

* **Preloaded APIs**: Some well-known APIs used in prior **Private Evolution** papers are already packaged as :py:class:`pe.api.API` classes. Please refer to :doc:`this document <details/api>` for more details.
* **Beyond the above**: You can create a class that inherits from :py:class:`pe.api.API` and implements the :py:meth:`pe.api.API.random_api` and :py:meth:`pe.api.API.variation_api` methods. Please refer to :doc:`this document <details/api>` for more details.
 