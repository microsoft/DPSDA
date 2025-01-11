from openai import AzureOpenAI
from openai import BadRequestError
from openai import AuthenticationError
from openai import NotFoundError
from openai import PermissionDeniedError
from azure.identity import AzureCliCredential, get_bearer_token_provider
import os
from tenacity import retry
from tenacity import retry_if_not_exception_type
from tenacity import stop_after_attempt
from tenacity import wait_random_exponential
from tenacity import before_sleep_log
import json
import logging
from concurrent.futures import ThreadPoolExecutor
import random

from pe.logging import execution_logger
from .llm import LLM


class AzureOpenAILLM(LLM):
    """A wrapper for Azure OpenAI LLM APIs. The following environment variables are required:

    * ``AZURE_OPENAI_API_KEY``: Azure OpenAI API key. You can get it from https://portal.azure.com/. Multiple keys can
      be separated by commas, and a key will be selected randomly for each request. The key can also be "AZ_CLI", in
      which case the Azure CLI will be used to authenticate the requests, and the environment variable
      ``AZURE_OPENAI_API_SCOPE`` needs to be set. See Azure OpenAI authentication documentation for more information:
      https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/switching-endpoints#microsoft-entra-id-authentication
    * ``AZURE_OPENAI_API_ENDPOINT``: Azure OpenAI endpoint. You can get it from https://portal.azure.com/.
    * ``AZURE_OPENAI_API_VERSION``: Azure OpenAI API version. You can get it from https://portal.azure.com/."""

    def __init__(self, dry_run=False, num_threads=1, **generation_args):
        """Constructor.

        :param dry_run: Whether to enable dry run. When dry run is enabled, the responses are fake and the APIs are
            not called. Defaults to False
        :type dry_run: bool, optional
        :param num_threads: The number of threads to use for making concurrent API calls, defaults to 1
        :type num_threads: int, optional
        :param \\*\\*generation_args: The generation arguments that will be passed to the OpenAI API
        :type \\*\\*generation_args: str
        """
        self._dry_run = dry_run
        self._num_threads = num_threads
        self._generation_args = generation_args

        self._api_keys = self._get_environment_variable("AZURE_OPENAI_API_KEY").split(",")
        self._clients = []
        for api_key in self._api_keys:
            if api_key == "AZ_CLI":
                credential = get_bearer_token_provider(
                    AzureCliCredential(), self._get_environment_variable("AZURE_OPENAI_API_SCOPE")
                )
                client = AzureOpenAI(
                    azure_ad_token_provider=credential,
                    api_version=self._get_environment_variable("AZURE_OPENAI_API_VERSION"),
                    azure_endpoint=self._get_environment_variable("AZURE_OPENAI_API_ENDPOINT"),
                )
            else:

                client = AzureOpenAI(
                    api_key=self._get_environment_variable("AZURE_OPENAI_API_KEY"),
                    api_version=self._get_environment_variable("AZURE_OPENAI_API_VERSION"),
                    azure_endpoint=self._get_environment_variable("AZURE_OPENAI_API_ENDPOINT"),
                )
            self._clients.append(client)
        execution_logger.info(f"Using {len(self._api_keys)} AzureOpenAI API keys")

    @property
    def generation_arg_map(self):
        """Get the mapping from the generation arguments to arguments for this specific LLM.

        :return: The mapping that maps ``max_completion_tokens`` to ``max_tokens``
        :rtype: dict
        """
        return {"max_completion_tokens": "max_tokens"}

    def _get_environment_variable(self, name):
        """Get the environment variable.

        :param name: The name of the environment variable
        :type name: str
        :raises ValueError: If the environment variable is not set
        :return: The value of the environment variable
        :rtype: str
        """
        if name not in os.environ or os.environ[name] == "":
            raise ValueError(f"{name} environment variable is not set.")
        return os.environ[name]

    def get_responses(self, requests, **generation_args):
        """Get the responses from the LLM.

        :param requests: The requests
        :type requests: list[:py:class:`pe.llm.request.Request`]
        :param \\*\\*generation_args: The generation arguments. The priority of the generation arguments from the
            highest to the lowerest is in the order of: the arguments set in the requests > the arguments passed to
            this function > and the arguments passed to the constructor
        :type \\*\\*generation_args: str
        :return: The responses
        :rtype: list[str]
        """
        messages_list = [request.messages for request in requests]
        generation_args_list = [
            self.get_generation_args(self._generation_args, generation_args, request.generation_args)
            for request in requests
        ]
        with ThreadPoolExecutor(max_workers=self._num_threads) as executor:
            responses = list(executor.map(self._get_response_for_one_request, messages_list, generation_args_list))
        return responses

    @retry(
        retry=retry_if_not_exception_type(
            (
                BadRequestError,
                AuthenticationError,
                NotFoundError,
                PermissionDeniedError,
            )
        ),
        wait=wait_random_exponential(min=8, max=500),
        stop=stop_after_attempt(30),
        before_sleep=before_sleep_log(execution_logger, logging.DEBUG),
    )
    def _get_response_for_one_request(self, messages, generation_args):
        """Get the response for one request.

        :param messages: The messages
        :type messages: list[str]
        :param generation_args: The generation arguments
        :type generation_args: dict
        :return: The response
        :rtype: str
        """
        if self._dry_run:
            response = f"Dry run enabled. The request is {json.dumps(messages)}"
        else:
            client = random.choice(self._clients)
            full_response = client.chat.completions.create(
                messages=messages,
                **generation_args,
            )
            response = full_response.choices[0].message.content
        return response
