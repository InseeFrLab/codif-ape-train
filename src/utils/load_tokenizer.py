import os
from typing import List

from torchTextClassifiers.tokenizers import NGramTokenizer, WordPieceTokenizer

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
    tokenizer_path = data_path + "tokenizers/" + f"{tokenizer_type}_{vocab_size}.json"

    tokenizer_class = TOKENIZER_CLASSES.get(tokenizer_type)
    if tokenizer_class is None:
        raise ValueError(f"Unsupported tokenizer type: {tokenizer_type}")

    fs = get_file_system()
    if fs.exists(tokenizer_path):
        logger.info("Loading tokenizer from %s", tokenizer_path)
        return tokenizer_class.load_from_s3(tokenizer_path, fs)

    logger.info(
        "Tokenizer %s with %s tokens not found at %s.", tokenizer_type, vocab_size, tokenizer_path
    )
    if training_text is None:
        raise ValueError("training_text must be provided to train the tokenizer.")

    logger.info("Training %s tokenizer with %s tokens.", tokenizer_type, vocab_size)
    tokenizer_instance = tokenizer_class(vocab_size=vocab_size, **kwargs)
    if hasattr(tokenizer_instance, "train"):
        tokenizer_instance.train(training_text)

    local_dir = "my_tokenizer"
    if hasattr(tokenizer_instance, "save_pretrained"):
        tokenizer_instance.save_pretrained(local_dir)
    else:
        tokenizer_instance.tokenizer.save_pretrained(local_dir)

    parent_dir = os.path.dirname(tokenizer_path)
    if not fs.exists(parent_dir):
        fs.mkdirs(parent_dir)
    fs.put(f"{local_dir}/tokenizer.json", tokenizer_path)
    logger.info("Tokenizer uploaded to S3 at %s", tokenizer_path)

    return tokenizer_instance
