import logging
import re
import string

import nltk
import unidecode
from joblib import Parallel, delayed
from nltk.corpus import stopwords as ntlk_stopwords
from nltk.stem.snowball import SnowballStemmer
from tqdm import tqdm

from ..base import PreTokenizer

nltk.data.path.append("nltk_data/")

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler()],
)

FRENCH_STOPWORDS = set(ntlk_stopwords.words("french")) | set(string.ascii_lowercase)


class NgramPreTokenizer(PreTokenizer):
    def __init__(self, remove_stop_words=True, stem=True, **kwargs):
        super().__init__(**kwargs)

        self.remove_stop_words = remove_stop_words
        self.stem = stem

    @staticmethod
    def _build_text_preprocessor(text, remove_stop_words=False, stem=False, n_jobs=None):
        stemmer = SnowballStemmer("french") if stem else None
        if remove_stop_words:
            stopwords = FRENCH_STOPWORDS
        else:
            stopwords = set()

        def preprocess(doc: str) -> str:
            # 1. Remove accents (é -> e, ç -> c, etc.)
            doc = unidecode.unidecode(doc)
            # 2. Lowercase
            doc = doc.lower()
            # 3. Remove punctuation, keep letters (a-z), numbers (0-9), and French chars (àâçéèêëîïôûùüÿñæœ), plus spaces
            doc = re.sub(r"[^a-z0-9àâçéèêëîïôûùüÿñæœ\s]", " ", doc)
            # 4. Collapse multiple spaces into one and strip leading/trailing spaces
            doc = re.sub(r"\s+", " ", doc).strip()
            # 5. Tokenize
            words = doc.split()
            # 6. Remove one-letter tokens (e.g., stray letters)
            words = [w for w in words if len(w) > 1]
            # 7. Stopword removal
            if remove_stop_words:
                words = [w for w in words if w not in stopwords]
            # 8. Stemming
            if stem:
                words = [stemmer.stem(w) for w in words]
            return " ".join(words)

        return preprocess

    def clean_text_feature(
        self,
        text: list[str],
        n_jobs: int = -1,
        threshold: int = 50_000,
    ) -> list[str]:
        """
        Hybrid text cleaning for FastText.
        Uses list comprehension for small corpora,
        joblib.Parallel for large corpora.

        Args:
            text (list[str]): List of documents.
            remove_stop_words (bool): If True, remove stopwords.
            stem (bool): If True, apply stemming.
            n_jobs (int): Number of CPU cores (-1 = all).
            threshold (int): Switch to parallel if len(text) >= threshold.

        Returns:
            list[str]: Cleaned text.
        """

        # workaround using a closure function for better parallelization
        preprocess = self._build_text_preprocessor(self.remove_stop_words, self.stem)

        if len(text) < threshold:
            # Small corpus → fastest with list comprehension
            return [preprocess(doc) for doc in text]
        else:
            # Large corpus → parallelize across CPUs
            return Parallel(n_jobs=n_jobs)(
                delayed(preprocess)(doc) for doc in tqdm(text, desc="Preprocessing")
            )
