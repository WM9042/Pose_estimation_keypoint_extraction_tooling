from abc import ABC, abstractmethod

class DataSplitManager(ABC):
    @abstractmethod
    def parseSplitInfo():
        pass