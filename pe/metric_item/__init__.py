import matplotlib.pyplot as plt

scopes = []


class metric_scope(object):
    """The context manager to manage the metric scope."""

    def __init__(self, name):
        self._name = name

    def __enter__(self):
        scopes.append(self._name)

    def __exit__(self, type, value, traceback):
        scopes.pop()


class MetricItem(object):
    """The base class for the metric item."""

    def __init__(self, name, value):
        """Constructor.

        :param name: The name of the metric item
        :type name: str
        :param value: The value of the metric item
        :type value: object
        """
        self._name = "/".join(scopes + [name])
        self._value = value

    @property
    def name(self):
        """Get the name of the metric item.

        :return: The name of the metric item
        :rtype: str
        """
        return self._name

    @property
    def value(self):
        """Get the value of the metric item.

        :return: The value of the metric item
        :rtype: object
        """
        return self._value

    def clean_up(self):
        """Clean up the metric item."""
        ...


class MatplotlibMetricItem(MetricItem):
    """The metric item for Matplotlib figures."""

    def clean_up(self):
        plt.close(self._value)


class FloatMetricItem(MetricItem):
    """The metric item for a single float value."""

    ...


class FloatListMetricItem(MetricItem):
    """The metric item for a list of float values."""

    ...


class ImageMetricItem(MetricItem):
    """The metric item for an image."""

    ...


class ImageListMetricItem(MetricItem):
    """The metric item for a list of images."""

    def __init__(self, num_images_per_row=None, *args, **kwargs):
        """Constructor.

        :param num_images_per_row: The number of images per row when saving to the file, defaults to None
        :type num_images_per_row: int, optional
        """
        super().__init__(*args, **kwargs)
        self._num_images_per_row = num_images_per_row

    @property
    def num_images_per_row(self):
        """Get the number of images per row when saving to the file.

        :return: The number of images per row when saving to the file
        :rtype: int or None
        """
        return self._num_images_per_row
