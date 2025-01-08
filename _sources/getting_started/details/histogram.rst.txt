Histograms
==========

API reference: :doc:`/api/pe.histogram`.

:py:class:`pe.histogram.Histogram` is responsible for generating the histograms over the synthetic samples. It has the following key methods:

* :py:meth:`pe.histogram.Histogram.compute_histogram`: Generates the histograms over the synthetic samples using private samples.

Available Histograms
--------------------

Currently, the following histograms are implemented:

* :py:class:`pe.histogram.NearestNeighbors`: This histogram algorithm projects the synthetic samples and the private samples into an embedding space and computes the nearest neighbor(s) of each private sample in the synthetic samples. The histogram value for each synthetic sample is the number of times it is the nearest neighbor(s) of a private sample.