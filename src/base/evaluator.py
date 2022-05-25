"""
"""
from abc import ABC, abstractmethod


class Evaluator(ABC):
    """ """

    def __init__(self):
        """ """

    @abstractmethod
    def evaluate(self):
        """ """
        raise NotImplementedError()
