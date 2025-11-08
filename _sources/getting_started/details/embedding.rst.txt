Embeddings
==========

API reference: :doc:`/api/pe.embedding`.

:py:class:`pe.embedding.embedding.Embedding` is responsible for computing the embeddings of the (synthetic or private) samples. It has the following key methods/attributes:

* :py:meth:`pe.embedding.Embedding.compute_embedding`: Computes the embeddings of the (synthetic or private) samples.
* :py:attr:`pe.embedding.Embedding.column_name`: The column name to be used when saving the embeddings in the ``data_frame`` of `pe.data.Data`.

Available Embeddings
--------------------

Currently, the following embeddings are implemented:

* Images

    * :py:class:`pe.embedding.Inception`: The embeddings computed using the Inception model.
    * :py:class:`pe.embedding.FLDInception`: The embeddings computed using the Inception model following the procedure in the `FLD`_ library.

* Text
    
    * :py:class:`pe.embedding.SentenceTransformer`: The embeddings computed using the `Sentence Transformers`_ library.


.. _Sentence Transformers: https://sbert.net/
.. _FLD: https://github.com/marcojira/fld