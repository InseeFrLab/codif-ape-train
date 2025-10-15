from transformers import PreTrainedTokenizerFast

from src.utils.data import constants
from src.utils.io import get_file_system
from tokenizers import Tokenizer

from .logger import get_logger

logger = get_logger(name=__name__)


def load_tokenizer(revision, tokenizer_type, num_tokens, **kwargs):
    data_path = constants[revision][-1]
    tokenizer_bucket = "tokenizers/"
    tokenizer_path = data_path + tokenizer_bucket + f"{tokenizer_type}_{num_tokens}.json"

    fs = get_file_system()
    if fs.exists(tokenizer_path) is False:
        raise FileNotFoundError(
            f"Tokenizer not found at {tokenizer_path}. Please train it first (see src/train_tokenizers)."
        )

    with fs.open(tokenizer_path, "rb") as f:
        json_str = f.read().decode("utf-8")

    tokenizer_obj = Tokenizer.from_str(json_str)

    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer_obj)

    tokenizer.num_tokens = len(tokenizer)

    logger.info(f"Loaded {tokenizer_type} tokenizer with {num_tokens} tokens from {tokenizer_path}")

    return tokenizer
