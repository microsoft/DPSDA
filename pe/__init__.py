from generalimport import generalimport as optionalimport


optionalimport(
    "blobfile",
    "torch",
    "imageio",
    "clip",
    "diffusers",
    "wilds",
    "improved_diffusion",
    "python_avatars",
    "cairosvg",
    message=(
        "Please install private_evolution with [image] dependencies. "
        "See https://microsoft.github.io/DPSDA/getting_started/installation.html for more details."
    ),
)
optionalimport(
    "gdown",
    "openai",
    "tenacity",
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
