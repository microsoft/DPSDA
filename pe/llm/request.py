from collections import namedtuple


Request = namedtuple("Request", ["messages", "generation_args"], defaults=[[], {}])
""" The request to the LLM.

:param messages: The messages to the LLM
:type messages: list[dict]
:param generation_args: The generation arguments to the LLM
:type generation_args: dict
"""
