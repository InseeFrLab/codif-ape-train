import logging

import nltk

from ..base import PreTokenizer

nltk.data.path.append("nltk_data/")

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler()],
)


class BertPreTokenizer(PreTokenizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        logger.info(
            "Using BertPreTokenizer. This is a dummy pre-tokenizer that outputs the raw text as is."
        )

    def clean_text_feature(
        self,
        text: list[str],
    ) -> list[str]:
        """
        Dummy: BertPreTokenizer does not modify the text.
        The tokenizer from HuggingFace directly handles raw text.
        """

        return text
