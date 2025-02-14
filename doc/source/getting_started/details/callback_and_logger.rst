Callbacks and Loggers
======================

API reference: :doc:`/api/pe.callback` and :doc:`/api/pe.logger`.

:py:class:`pe.callback.Callback` can be configured to be called after each **Private Evolution** iteration with the synthetic data as the input. It is useful for computing metrics, saving the synthetic samples, monitoring the progress, etc. Each :py:class:`pe.callback.Callback` can return a list of results (float numbers, images, matplotlib plots, etc.) in the form of :py:class:`pe.metric_item.MetricItem` (see :py:mod:`pe.metric_item`). All :py:class:`pe.metric_item.MetricItem` from all :py:class:`pe.callback.Callback` will be passed through each of the :py:class:`pe.logger.Logger` modules, which will then log the results in the desired way.

Available Callbacks
-------------------

Currently, the following callbacks are implemented:

* For any data modality

    * :py:class:`pe.callback.ComputeFID`: Computes the FID between the synthetic samples and the private samples.
    * :py:class:`pe.callback.SaveCheckpoints`: Saves the checkpoint of current synthetic samples to files.
    * :py:class:`pe.callback.ComputePrecisionRecall`: Computes the `precision and recall metrics`_ between the synthetic samples and the private samples.

* Images

    * :py:class:`pe.callback.SampleImages`: Samples some images from each class.
    * :py:class:`pe.callback.SaveAllImages`: Saves all synthetic images to files.
    * :py:class:`pe.callback.DPImageBenchClassifyImages`: Trains classifiers on the synthetic images and evaluates them on the private images, following the settings in `DPImageBench`_.

* Text

    * :py:class:`pe.callback.SaveTextToCSV`: Save all text samples to a CSV file.


Available Loggers
-----------------

Currently, the following loggers are implemented:

* :py:class:`pe.logger.CSVPrint`: Saves the float results to a CSV file.
* :py:class:`pe.logger.LogPrint`: Prints the float results to the console and/or files using the logging module.
* :py:class:`pe.logger.ImageFile`: Saves the images to files.
* :py:class:`pe.logger.MatplotlibPDF`: Saves the matplotlib plots to PDF files.

.. _precision and recall metrics: https://arxiv.org/abs/1904.06991
.. _DPImageBench: https://github.com/2019ChenGong/DPImageBench