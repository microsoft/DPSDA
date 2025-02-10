import os

from pe.callback.callback import Callback


class SaveCheckpoints(Callback):
    """The callback that saves checkpoints of the synthetic data."""

    def __init__(
        self,
        output_folder,
        iteration_format="09d",
    ):
        """Constructor.

        :param output_folder: The output folder that will be used to save the checkpoints
        :type output_folder: str
        :param iteration_format: The format of the iteration number, defaults to "09d"
        :type iteration_format: str, optional
        """
        self._output_folder = output_folder
        self._iteration_format = iteration_format

    def __call__(self, syn_data):
        """This function is called after each PE iteration that saves checkpoints of the synthetic data.

        :param syn_data: The synthetic data
        :type syn_data: :py:class:`pe.data.Data`
        """
        syn_data.save_checkpoint(self._get_checkpoint_path(syn_data.metadata.iteration))

    def _get_checkpoint_path(self, iteration):
        """Get the checkpoint path.

        :param iteration: The PE iteration number
        :type iteration: int
        :return: The checkpoint path
        :rtype: str
        """
        os.makedirs(self._output_folder, exist_ok=True)
        iteration_string = format(iteration, self._iteration_format)
        checkpoint_path = os.path.join(
            self._output_folder,
            iteration_string,
        )
        return checkpoint_path
