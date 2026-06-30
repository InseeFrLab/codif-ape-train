from typing import List

from tokenizers import Tokenizer
from torchTextClassifiers.tokenizers import NGramTokenizer, WordPieceTokenizer
from transformers import PreTrainedTokenizerFast

from src.utils.data import constants
from src.utils.io import get_file_system
from src.utils.logger import get_logger

logger = get_logger(name=__name__)

TOKENIZER_CLASSES = {
    "WordPiece": WordPieceTokenizer,
    "HuggingFace": NGramTokenizer,
}


def load_tokenizer(revision, tokenizer_type, vocab_size, training_text: List[str] = None, **kwargs):
    data_path = constants[revision][-1]
    tokenizer_bucket = "tokenizers/"
    tokenizer_path = data_path + tokenizer_bucket + f"{tokenizer_type}_{vocab_size}.json"

    fs = get_file_system()
    if fs.exists(tokenizer_path) is False:
        logger.info(
            f"Tokenizer {tokenizer_type} with {vocab_size} tokens not found at {tokenizer_path}."
        )
        tokenizer_class = TOKENIZER_CLASSES.get(tokenizer_type)
        if tokenizer_class is None:
            raise ValueError(f"Unsupported tokenizer type: {tokenizer_type}")
        tokenizer_instance = tokenizer_class(vocab_size=vocab_size, **kwargs)
        if hasattr(tokenizer_instance, "train"):
            logger.info(f"Training {tokenizer_type} tokenizer with {vocab_size} tokens.")
            if training_text is None:
                raise ValueError("training_text must be provided to train the tokenizer.")

            tokenizer_instance.train(
                training_text, save_path="my_tokenizer", s3_save_path=tokenizer_path, filesystem=fs
            )
        return tokenizer_instance
    else:
        with fs.open(tokenizer_path, "rb") as f:
            json_str = f.read().decode("utf-8")

        tokenizer_obj = Tokenizer.from_str(json_str)

        tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer_obj)

    tokenizer.vocab_size = len(tokenizer)

    logger.info("Loaded tokenizer from %s", tokenizer_path)

    return tokenizer
