from abc import ABC, abstractmethod

from simulation_data.sim_data import SimulationData


class IDataProvider(ABC):
    @property
    @abstractmethod
    def sim_data(self) -> SimulationData:
        pass
    # end def
# end class
