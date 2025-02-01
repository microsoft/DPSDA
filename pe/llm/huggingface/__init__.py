from generalimport import is_imported
from .register_fastchat.gpt2 import register as register_gpt2


if is_imported("fastchat"):
    register_gpt2()
