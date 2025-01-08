from abc import ABC, abstractmethod
from functools import reduce


class LLM(ABC):
    """The abstract class for large language models (LLMs)."""

    @abstractmethod
    def get_responses(self, requests, **generation_args):
        """Get the responses from the LLM.

        :param requests: The requests
        :type requests: list[:py:class:`pe.llm.request.Request`]
        :param \\*\\*generation_args: The generation arguments
        :type \\*\\*generation_args: str
        :return: The responses
        :rtype: list[str]
        """
        ...

    @property
    def generation_arg_map(self):
        """Get the mapping from the generation arguments to arguments for this specific LLM.

        :return: The mapping from the generation arguments to the large language model arguments
        :rtype: dict
        """
        return {}

    def get_generation_args(self, *args):
        """Get the generation arguments from a list of dictionaries.

        :param \\*args: A list of generation arguments. The later ones will overwrite the earlier ones.
        :type \\*args: dict
        :return: The generation arguments
        :rtype: dict
        """
        generation_args = reduce(lambda x, y: {**x, **y}, args)
        generation_args = {
            k if k not in self.generation_arg_map else self.generation_arg_map[k]: v
            for k, v in generation_args.items()
        }
        return generation_args
