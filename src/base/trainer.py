"""
"""
from abc import ABC, abstractmethod


class Trainer(ABC):
    """ """

    def __init__(self):
        """ """

    @abstractmethod
    def train(self):
        """ """
        raise NotImplementedError()
