Embeddings
==========

API reference: :doc:`/api/pe.embedding`.

:py:class:`pe.embedding.embedding.Embedding` is responsible for computing the embeddings of the (synthetic or private) samples. It has the following key methods/attributes:

* :py:meth:`pe.embedding.embedding.Embedding.compute_embedding`: Computes the embeddings of the (synthetic or private) samples.
* :py:attr:`pe.embedding.embedding.Embedding.column_name`: The column name to be used when saving the embeddings in the ``data_frame`` of `pe.data.data.Data`.

Available Embeddings
--------------------

Currently, the following embeddings are implemented:

* Images

    * :py:class:`pe.embedding.image.inception.Inception`: The embeddings computed using the Inception model.

* Text
    
    * Coming soon!
