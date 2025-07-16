import os

from src.utils.data import get_file_system, get_train_raw_data
from tokenizers import (
    Tokenizer,
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
)


def train_WordPiece(n_vocab=10000, revision="NAF2025"):
    tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))

    tokenizer.normalizer = normalizers.BertNormalizer(lowercase=True)

    tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()

    special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]"]

    trainer = trainers.WordPieceTrainer(vocab_size=n_vocab, special_tokens=special_tokens)

    training_text = get_train_raw_data(revision=revision)["libelle"].values.tolist()
    tokenizer.train_from_iterator(training_text, trainer=trainer)
    tokenizer.post_processor = processors.BertProcessing(
        ("[CLS]", tokenizer.token_to_id("[CLS]")),
        ("[SEP]", tokenizer.token_to_id("[SEP]")),
    )
    tokenizer.decoder = decoders.WordPiece(prefix="##")

    tokenizer.enable_padding(pad_id=tokenizer.token_to_id("[PAD]"), pad_token="[PAD]")

    # Save
    os.makedirs("tokenizers", exist_ok=True)
    local_path = f"tokenizers/WordPiece_{n_vocab}.json"
    tokenizer.save(local_path)

    # Dump on S3
    fs = get_file_system()
    s3_path = f"projet-ape/tokenizers/{revision}/WordPiece_{n_vocab}.json"
    parent_dir = os.path.dirname(s3_path)
    if not fs.exists(parent_dir):
        fs.makedirs(parent_dir)
    fs.put(local_path, s3_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train a WordPiece tokenizer.")
    parser.add_argument("--n_vocab", type=int, default=10000, help="Vocabulary size")
    parser.add_argument("--revision", type=str, default="NAF2025", help="Revision of the NAF data")
    args = parser.parse_args()

    train_WordPiece(n_vocab=args.n_vocab, revision=args.revision)
