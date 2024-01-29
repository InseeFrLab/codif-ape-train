"""
Utilitary functions for the Pytorch classifier.
"""
import ctypes
import os

import fasttext
import numpy as np

from utils.data import get_file_system, get_root_path


def get_hash(subword):
    h = ctypes.c_uint32(2166136261).value
    for c in subword:
        c = ctypes.c_int8(ord(c)).value
        h = ctypes.c_uint32(h ^ c).value
        h = ctypes.c_uint32(h * 16777619).value
    return h


def get_word_ngram_id(hashes, bucket, nwords):
    hashes = [ctypes.c_int32(hash_value).value for hash_value in hashes]
    h = ctypes.c_uint64(hashes[0]).value
    for j in range(1, len(hashes)):
        h = ctypes.c_uint64((h * 116049371)).value
        h = ctypes.c_uint64(h + hashes[j]).value
    return h % bucket + nwords


def indices_matrix(sentence: str, model):
    """
    Returns ?

    Args:
        sentence (str): _description_
        model (_type_): _description_
    """
    nwords = len(model.get_words())
    buckets = model.f.getArgs().bucket
    word_ngrams = model.f.getArgs().wordNgrams

    indices = []
    words = []
    word_ngram_ids = []

    for word in model.get_line(sentence)[0]:
        indices = indices + model.get_subwords(word)[1].tolist()
        words.append(word)

    for word_ngram_len in range(2, word_ngrams + 1):
        for i in range(len(words) - word_ngram_len + 1):
            hashes = tuple(get_hash(word) for word in words[i : i + word_ngram_len])
            word_ngram_id = int(get_word_ngram_id(hashes, buckets, nwords))
            word_ngram_ids.append(word_ngram_id)

    all_indices = indices + word_ngram_ids
    return np.asarray(all_indices)


def load_ft_from_minio(path: str):
    """
    Returns ?

    Args:
        path (str): _description_
    """

    root_path = get_root_path()
    if not os.path.exists(root_path / "model/"):
        os.mkdir(root_path / "model/")

    local_path = str(root_path / "model/pretrained_model.bin")
    fs = get_file_system()
    fs.get(path, local_path)
    model = fasttext.load_model(local_path)
    os.remove(local_path)
    return model
