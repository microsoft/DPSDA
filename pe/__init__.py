import generalimport
from generalimport import generalimport as optionalimport

# This is a temporary hack to support libraries such as azure.identity
generalimport.top._assert_no_dots = lambda names: ...

optionalimport(
    "blobfile",
    "torch",
    "imageio",
    "clip",
    "diffusers",
    "wilds",
    "improved_diffusion",
    message=(
        "Please install private_evolution with [image] dependencies. "
        "See https://microsoft.github.io/DPSDA/getting_started/installation.html for more details."
    ),
)
optionalimport(
    "gdown",
    "openai",
    "tenacity",
    "azure.identity",
    "tiktoken",
    "dotenv",
    "sentence_transformers",
    "protobuf",
    "sentencepiece",
    "fastchat",
    "transformers",
    "accelerate",
    message=(
        "Please install private_evolution with [text] dependencies. "
        "See https://microsoft.github.io/DPSDA/getting_started/installation.html for more details."
    ),
)
