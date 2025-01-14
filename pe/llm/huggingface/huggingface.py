import torch
import transformers
from pe.logging import execution_logger
from fastchat.model.model_adapter import get_conversation_template

from ..llm import LLM


class HuggingfaceLLM(LLM):
    """A wrapper for Huggingface LLMs."""

    def __init__(self, model_name_or_path, batch_size=128, dry_run=False, **generation_args):
        """Constructor.

        :param model_name_or_path: The model name or path of the Huggingface model. Note that we use the FastChat
            library (https://github.com/lm-sys/FastChat) to manage the conversation template. If the conversation
            template of your desired model is not available in FastChat, please register the conversation template in
            the FastChat library. See the following link for an example:
            https://github.com/microsoft/DPSDA/blob/main/pe/llm/huggingface/register_fastchat/gpt2.py
        :type model_name_or_path: str
        :param batch_size: The batch size to use for generating the responses, defaults to 128
        :type batch_size: int, optional
        :param dry_run: Whether to enable dry run. When dry run is enabled, the responses are fake and the LLMs are
            not called. Defaults to False
        :type dry_run: bool, optional
        :param \\*\\*generation_args: The generation arguments that will be passed to the OpenAI API
        :type \\*\\*generation_args: str
        """
        self._dry_run = dry_run
        self._generation_args = generation_args

        self._model_name_or_path = model_name_or_path
        self._batch_size = batch_size

        self._tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path, device_map="auto")
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._tokenizer.padding_side = "left"

        self._model = transformers.AutoModelForCausalLM.from_pretrained(
            model_name_or_path, device_map="auto", torch_dtype=torch.float16
        )
        if self._model.config.pad_token_id is None:
            self._model.config.pad_token_id = self._model.config.eos_token_id
        self._model.eval()

        self._conv_template = self._get_conv_template()
        self._stop_str = self._conv_template.stop_str
        self._stop_token_ids = self._conv_template.stop_token_ids or []
        self._stop_token_ids.append(self._tokenizer.eos_token_id)

    @property
    def generation_arg_map(self):
        """Get the mapping from the generation arguments to arguments for this specific LLM.

        :return: The mapping that maps ``max_completion_tokens`` to ``max_new_tokens``
        :rtype: dict
        """
        return {"max_completion_tokens": "max_new_tokens"}

    def _get_conv_template(self):
        """Get the conversation template.

        :return: The empty conversation template for this model from FastChat
        :rtype: :py:class:`fastchat.conversation.Conversation`
        """
        template = get_conversation_template(self._model_name_or_path)
        template.messages = []
        template.system_message = ""
        return template

    def _get_prompt(self, messages):
        """Get the prompt from the messages.

        :param messages: The messages
        :type messages: list[dict]
        :raises ValueError: If the role is invalid
        :return: The prompt
        :rtype: str
        """
        template = self._conv_template.copy()
        for message in messages:
            if message["role"] == "system":
                template.set_system_message(message["content"])
            elif message["role"] == "user":
                template.append_message(role=template.roles[0], message=message["content"])
            elif message["role"] == "assistant":
                template.append_message(role=template.roles[1], message=message["content"])
            else:
                raise ValueError(f"Invalid role: {message['role']}")
        template.append_message(role=template.roles[1], message=None)
        return template.get_prompt()

    def get_responses(self, requests, **generation_args):
        """Get the responses from the LLM.

        :param requests: The requests
        :type requests: list[:py:class:`pe.llm.Request`]
        :param \\*\\*generation_args: The generation arguments. The priority of the generation arguments from the
            highest to the lowerest is in the order of: the arguments set in the requests > the arguments passed to
            this function > and the arguments passed to the constructor
        :type \\*\\*generation_args: str
        :return: The responses
        :rtype: list[str]
        """
        execution_logger.info("HuggingfaceLLM: producing prompts")
        prompt_list = []
        generation_args_list = []
        for request in requests:
            prompt_list.append(self._get_prompt(request.messages))
            generation_args_list.append(
                self.get_generation_args(self._generation_args, generation_args, request.generation_args)
            )
        execution_logger.info("HuggingfaceLLM: getting responses")
        responses = [None] * len(requests)
        # Group requests according to generation_args
        generation_args_fronzen_set_list = [
            frozenset(generation_args.items()) for generation_args in generation_args_list
        ]
        generation_args_set = list(set(generation_args_fronzen_set_list))
        generation_args_to_set_index = {g: i for i, g in enumerate(generation_args_set)}
        grouped_request_indices = [[] for i in range(len(generation_args_set))]
        for i, generation_args in enumerate(generation_args_fronzen_set_list):
            grouped_request_indices[generation_args_to_set_index[generation_args]].append(i)
        for group in grouped_request_indices:
            sub_prompt_list = [prompt_list[j] for j in group]
            sub_response_list = self._get_responses(sub_prompt_list, generation_args_list[group[0]])
            for i, j in enumerate(group):
                responses[j] = sub_response_list[i]
        assert None not in responses
        return responses

    @torch.no_grad
    def _get_responses(self, prompt_list, generation_args):
        """Get the responses from the LLM.

        :param prompt_list: The prompts
        :type prompt_list: list[str]
        :param generation_args: The generation arguments
        :type generation_args: dict
        :return: The responses
        :rtype: list[str]
        """
        if self._dry_run:
            responses = [f"Dry run enabled. The request is {prompt}" for prompt in prompt_list]
        else:
            input_ids = self._tokenizer(
                prompt_list, return_tensors="pt", padding=True, padding_side="left"
            ).input_ids.to(self._model.device)
            responses = []
            for i in range(0, len(input_ids), self._batch_size):
                batch_input_ids = input_ids[i : i + self._batch_size]
                batch_responses = self._model.generate(
                    batch_input_ids,
                    stop_strings=self._stop_str,
                    eos_token_id=self._stop_token_ids,
                    do_sample=True,
                    **generation_args,
                )
                batch_responses = self._tokenizer.batch_decode(
                    batch_responses[:, input_ids.shape[1] :],
                    clean_up_tokenization_spaces=True,
                    skip_special_tokens=True,
                )
                responses.extend(batch_responses)
        return responses
