Callbacks and Loggers
======================

API reference: :doc:`/api/pe.callback` and :doc:`/api/pe.logger`.

:py:class:`pe.callback.callback.Callback` can be configured to be called after each **Private Evolution** iteration with the synthetic data as the input. It is useful for computing metrics, saving the synthetic samples, monitoring the progress, etc. Each :py:class:`pe.callback.callback.Callback` can return a list of results (float numbers, images, matplotlib plots, etc.) in the form of :py:class:`pe.metric_item.MetricItem` (see :py:mod:`pe.metric_item`). All :py:class:`pe.metric_item.MetricItem` from all  :py:class:`pe.callback.callback.Callback` will be passed through each of the :py:class:`pe.logger.logger.Logger` modules, which will then log the results in the desired way.

Available Callbacks
-------------------

Currently, the following callbacks are implemented:

* For any data modality

    * :py:class:`pe.callback.common.compute_fid.ComputeFID`: Computes the FID between the synthetic samples and the private samples.
    * :py:class:`pe.callback.common.save_checkpoints.SaveCheckpoints`: Saves the checkpoint of current synthetic samples to files.

* Images

    * :py:class:`pe.callback.image.sample_images.SampleImages`: Samples some images from each class.
    * :py:class:`pe.callback.image.save_all_images.SaveAllImages`: Saves all synthetic images to files.

* Text

    * Coming soon!


Available Loggers
-----------------

Currently, the following loggers are implemented:

* :py:class:`pe.logger.csv_print.CSVPrint`: Saves the float results to a CSV file.
* :py:class:`pe.logger.log_print.LogPrint`: Prints the float results to the console and/or files using the logging module.
* :py:class:`pe.logger.image_file.ImageFile`: Saves the images to files.
* :py:class:`pe.logger.matplotlib_pdf.MatplotlibPDF`: Saves the matplotlib plots to PDF files.
