from fastchat.model.model_adapter import BaseModelAdapter
from fastchat.model.model_adapter import register_model_adapter
from fastchat.conversation import get_conv_template
from fastchat.conversation import register_conv_template
from fastchat.conversation import Conversation
from fastchat.conversation import SeparatorStyle


def register():
    """Register the GPT-2 model adapter for fastchat."""

    class GPT2Adapter(BaseModelAdapter):
        """The GPT-2 model adapter for fastchat."""

        def match(self, model_path):
            return "gpt2" in model_path.lower()

        def load_model(self, model_path, from_pretrained_kwargs):
            raise NotImplementedError

        def get_default_conv_template(self, model_path):
            return get_conv_template("gpt2")

    register_conv_template(
        Conversation(
            name="gpt2",
            system_message="",
            roles=("", ""),
            sep_style=SeparatorStyle.NO_COLON_SINGLE,
            sep="",
        )
    )
    register_model_adapter(GPT2Adapter)
