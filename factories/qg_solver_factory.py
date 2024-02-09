from abc import ABC, abstractmethod
from solvers.interfaces.qg_solver import *

class QGSolverFactory(ABC):
    @abstractmethod
    def create_solver(self, **kwargs):
        pass
