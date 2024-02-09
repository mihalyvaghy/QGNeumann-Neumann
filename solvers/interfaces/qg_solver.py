from abc import ABC, abstractmethod

class QGSolver(ABC):
    @abstractmethod
    def solve(self):
        pass

    @abstractmethod
    def check_continuity(self):
        pass

    @abstractmethod
    def check_Neumann_Kirchhoff(self):
        pass

    @abstractmethod
    def get_solution(self):
        pass

    @abstractmethod
    def get_vertex_values(self):
        pass
