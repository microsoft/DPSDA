from generalimport import generalimport

generalimport(
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
generalimport(
    "gdown",
    "openai",
    "tenacity",
    "azure-identity",
    "tiktoken",
    "python-dotenv",
    "sentence-transformers",
    "protobuf",
    "sentencepiece",
    "fschat",
    "transformers",
    "accelerate",
    message=(
        "Please install private_evolution with [text] dependencies. "
        "See https://microsoft.github.io/DPSDA/getting_started/installation.html for more details."
    ),
)
