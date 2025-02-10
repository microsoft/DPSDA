import json
import random
import copy
import pandas as pd
import tiktoken
import numpy as np

from pe.api import API
from pe.api.util import ConstantList
from pe.logging import execution_logger
from pe.data import Data
from pe.llm import Request
from pe.constant.data import TEXT_DATA_COLUMN_NAME
from pe.constant.data import LLM_REQUEST_MESSAGES_COLUMN_NAME
from pe.constant.data import LLM_PARAMETERS_COLUMN_NAME
from pe.constant.data import LABEL_ID_COLUMN_NAME


class LLMAugPE(API):
    """The text API that uses open-source or API-based LLMs. This algorithm is initially proposed in the ICML 2024
    Spotlight paper, "Differentially Private Synthetic Data via Foundation Model APIs 2: Text"
    (https://arxiv.org/abs/2403.01749)"""

    def __init__(
        self,
        llm,
        random_api_prompt_file,
        variation_api_prompt_file,
        min_word_count=0,
        word_count_std=None,
        token_to_word_ratio=None,
        max_completion_tokens_limit=None,
        blank_probabilities=None,
        tokenizer_model="gpt-3.5-turbo",
    ):
        """Constructor.

        :param llm: The LLM utilized for the random and variation generation
        :type llm: :py:class:`pe.llm.LLM`
        :param random_api_prompt_file: The prompt file for the random API. See the explanations to
            ``variation_api_prompt_file`` for the format of the prompt file
        :type random_api_prompt_file: str
        :param variation_api_prompt_file: The prompt file for the variation API. The file is in JSON format and
            contains the following fields:

            * ``message_template``: A list of messages that will be sent to the LLM. Each message contains the
              following fields:

              * ``content``: The content of the message. The content can contain variable placeholders (e.g.,
                {variable_name}). The variable_name can be label name in the original data that will be replaced by
                the actual label value; or "sample" that will be replaced by the input text to the variation API;
                or "masked_sample" that will be replaced by the masked/blanked input text to the variation API
                when the blanking feature is enabled; or "word_count" that will be replaced by the target word
                count of the text when the word count variation feature is enabled; or other variables
                specified in the replacement rules (see below).
              * ``role``: The role of the message. The role can be "system",  "user", or "assistant".
            * ``replacement_rules``: A list of replacement rules that will be applied one by one to update the variable
              list. Each replacement rule contains the following fields:

              * ``constraints``: A dictionary of constraints that must be satisfied for the replacement rule to be
                applied. The key is the variable name and the value is the variable value.
              * ``replacements``: A dictionary of replacements that will be used to update the variable list if the
                constraints are satisfied. The key is the variable name and the value is the variable value or a
                list of variable values to choose from in a uniform random manner.
        :type variation_api_prompt_file: str
        :param min_word_count: The minimum word count for the variation API, defaults to 0
        :type min_word_count: int, optional
        :param word_count_std: The standard deviation for the word count for the variation API. If None, the word count
            variation feature is disabled and "{word_count}" variable will not be provided to the prompt. Defaults to
            None
        :type word_count_std: float, optional
        :param token_to_word_ratio: The token to word ratio for the variation API. If not None, the maximum completion
            tokens will be set to ``token_to_word_ratio`` times the target word count when the word count variation
            feature is enabled. Defaults to None
        :type token_to_word_ratio: float, optional
        :param max_completion_tokens_limit: The maximum completion tokens limit for the variation API, defaults to None
        :type max_completion_tokens_limit: int, optional
        :param blank_probabilities: The token blank probabilities for the variation API utilized at each PE iteration.
            If a single float is provided, the same blank probability will be used for all iterations. If None, the
            blanking feature is disabled and "{masked_sample}" variable will not be provided to the prompt. Defaults
            to None
        :type blank_probabilities: float or list[float], optional
        :param tokenizer_model: The tokenizer model used for blanking, defaults to "gpt-3.5-turbo"
        :type tokenizer_model: str, optional
        """
        super().__init__()
        self._llm = llm

        self._random_api_prompt_file = random_api_prompt_file
        with open(random_api_prompt_file, "r") as f:
            self._random_api_prompt_config = json.load(f)

        self._variation_api_prompt_file = variation_api_prompt_file
        with open(variation_api_prompt_file, "r") as f:
            self._variation_api_prompt_config = json.load(f)

        self._min_word_count = min_word_count
        self._word_count_std = word_count_std
        self._token_to_word_ratio = token_to_word_ratio
        self._max_completion_tokens_limit = max_completion_tokens_limit
        if isinstance(blank_probabilities, list):
            self._blank_probabilities = blank_probabilities
        else:
            self._blank_probabilities = ConstantList(blank_probabilities)

        self._encoding = tiktoken.encoding_for_model(tokenizer_model)
        self._mask_token = self._encoding.encode("_")[0]

    def _construct_prompt(self, prompt_config, variables):
        """Applying the replacement rules to construct the final prompt messages.

        :param prompt_config: The prompt configuration
        :type prompt_config: dict
        :param variables: The inital variables to be used in the prompt messages
        :type variables: dict
        :return: The constructed prompt messages
        :rtype: list[dict]
        """
        if "replacement_rules" in prompt_config:
            for replacement_rule in prompt_config["replacement_rules"]:
                constraints = replacement_rule["constraints"]
                replacements = replacement_rule["replacements"]
                satisfied = True
                for key, value in constraints.items():
                    if key not in variables or variables[key] != value:
                        satisfied = False
                        break
                if satisfied:
                    for key, value in replacements.items():
                        if isinstance(value, list):
                            value = random.choice(value)
                        variables[key] = value
        messages = copy.deepcopy(prompt_config["message_template"])
        for message in messages:
            message["content"] = message["content"].format(**variables)
        return messages

    def random_api(self, label_info, num_samples):
        """Generating random synthetic data.

        :param label_info: The info of the label
        :type label_info: omegaconf.dictconfig.DictConfig
        :param num_samples: The number of random samples to generate
        :type num_samples: int
        :return: The data object of the generated synthetic data
        :rtype: :py:class:`pe.data.Data`
        """
        label_name = label_info.name
        execution_logger.info(f"RANDOM API: creating {num_samples} samples for label {label_name}")

        variables = label_info.column_values
        execution_logger.info("RANDOM API: producing LLM requests")
        messages_list = [
            self._construct_prompt(prompt_config=self._random_api_prompt_config, variables=copy.deepcopy(variables))
            for _ in range(num_samples)
        ]
        requests = [Request(messages=messages) for messages in messages_list]
        execution_logger.info("RANDOM API: getting LLM responses")
        responses = self._llm.get_responses(requests)
        execution_logger.info("RANDOM API: constructing data")
        data_frame = pd.DataFrame(
            {
                TEXT_DATA_COLUMN_NAME: responses,
                LLM_REQUEST_MESSAGES_COLUMN_NAME: [json.dumps(messages) for messages in messages_list],
                LABEL_ID_COLUMN_NAME: 0,
            }
        )
        metadata = {"label_info": [label_info]}
        execution_logger.info(f"RANDOM API: finished creating {num_samples} samples for label {label_name}")
        return Data(data_frame=data_frame, metadata=metadata)

    def _blank_sample(self, sample, blank_probability):
        """Blanking the input text.

        :param sample: The input text
        :type sample: str
        :param blank_probability: The token blank probability
        :type blank_probability: float
        :return: The blanked input text
        :rtype: str
        """
        input_ids = np.asarray(self._encoding.encode(sample))
        masked_indices = np.random.uniform(size=len(input_ids)) < blank_probability
        input_ids[masked_indices] = self._mask_token
        return self._encoding.decode(input_ids)

    def variation_api(self, syn_data):
        """Generating variations of the synthetic data.

        :param syn_data: The data object of the synthetic data
        :type syn_data: :py:class:`pe.data.Data`
        :return: The data object of the variation of the input synthetic data
        :rtype: :py:class:`pe.data.Data`
        """
        execution_logger.info(f"VARIATION API: creating variations for {len(syn_data.data_frame)} samples")

        samples = syn_data.data_frame[TEXT_DATA_COLUMN_NAME].tolist()
        label_ids = syn_data.data_frame[LABEL_ID_COLUMN_NAME].tolist()

        iteration = getattr(syn_data.metadata, "iteration", -1)
        blank_probability = self._blank_probabilities[iteration + 1]

        execution_logger.info("VARIATION API: producing LLM requests")
        messages_list = []
        requests = []
        generation_args_list = []
        for sample, label_id in zip(samples, label_ids):
            variables = {"sample": sample}
            variables.update(syn_data.metadata.label_info[label_id].column_values)
            generation_args = {}

            if blank_probability is not None:
                variables["masked_sample"] = self._blank_sample(sample=sample, blank_probability=blank_probability)

            if self._word_count_std is not None:
                word_count = len(sample.split())
                new_word_count = word_count + int(np.random.normal(loc=0, scale=self._word_count_std))
                new_word_count = max(self._min_word_count, new_word_count)
                variables["word_count"] = new_word_count

                if self._token_to_word_ratio is not None:
                    max_completion_tokens = int(new_word_count * self._token_to_word_ratio)
                    if self._max_completion_tokens_limit is not None:
                        max_completion_tokens = min(max_completion_tokens, self._max_completion_tokens_limit)
                    generation_args["max_completion_tokens"] = max_completion_tokens

            messages = self._construct_prompt(prompt_config=self._variation_api_prompt_config, variables=variables)
            messages_list.append(messages)
            generation_args_list.append(generation_args)
            requests.append(Request(messages=messages, generation_args=generation_args))
        execution_logger.info("VARIATION API: getting LLM responses")
        responses = self._llm.get_responses(requests)
        execution_logger.info("VARIATION API: constructing data")
        data_frame = pd.DataFrame(
            {
                TEXT_DATA_COLUMN_NAME: responses,
                LLM_REQUEST_MESSAGES_COLUMN_NAME: [json.dumps(messages) for messages in messages_list],
                LLM_PARAMETERS_COLUMN_NAME: [json.dumps(generation_args) for generation_args in generation_args_list],
                LABEL_ID_COLUMN_NAME: label_ids,
            }
        )
        execution_logger.info(f"VARIATION API: finished creating variations for {len(syn_data.data_frame)} samples")
        return Data(data_frame=data_frame, metadata=syn_data.metadata)
