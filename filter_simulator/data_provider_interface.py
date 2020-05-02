from abc import ABC, abstractmethod

from .common import FrameList


class IDataProvider(ABC):
    @property
    @abstractmethod
    def frame_list(self) -> FrameList:
        pass
    # end def
# end class
