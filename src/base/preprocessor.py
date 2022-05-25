"""
"""
from abc import ABC, abstractmethod


class Preprocessor(ABC):
    """ """

    def __init__(self):
        """ """

    def preprocess(self, data):
        """ """
        # General preprocessing to implement
        data = data

        # Specific preprocessing for model
        data = self.preprocess_for_model(data)

    @abstractmethod
    def preprocess_for_model(self, data):
        """ """
        raise NotImplementedError()
