import numpy as np
from scipy import integrate
from solvers.interfaces.qg_solver import *

class QGEdge(ABC):
    @abstractmethod
    def solve(self):
        pass

    @abstractmethod
    def update_bc_values(self, lbc_val, rbc_val):
        pass

    @abstractmethod
    def approx_Neumann(self, endpoint):
        pass

    def endpoints(self):
        return [self.out_, self.in_]

    def Dirichlet(self, endpoint):
        return self.u[0] if endpoint == self.out_ else self.u[-1]

    def Neumann(self, endpoint):
        if endpoint == self.out_:
            return self.c_value[0]*(self.u[1]-self.u[0]) / self.h
        else:
            return self.c_value[-1]*(self.u[-2]-self.u[-1]) / self.h

    def midpoint_value(self):
        return self.u[len(self.u)//2]

class QGIterativeSolver(QGSolver):
    def solve(self):
        iter = 0
        change = self.change_threshold + 1
        prev_U = self._get_U()

        if self.adaptive:
            while iter < self.maxiter and change > self.change_threshold:
                iter += 1
                self._iterate()
                U = self._get_U()
                change = self._compute_change(prev_U, U)
                prev_U = U
        else:
            while iter < self.maxiter:
                iter += 1
                self._iterate()
                U = self._get_U()

        return iter

    @abstractmethod
    def check_continuity(self):
        pass

    @abstractmethod
    def check_Neumann_Kirchhoff(self):
        pass

    @abstractmethod
    def _iterate(self):
        pass

    @abstractmethod
    def _get_U(self):
        pass

    def _compute_change(self, prev_U, U):
        return np.sum([integrate.simpson(np.square(prev_u-u), self.x)+
                       integrate.simpson(np.square(np.diff(prev_u-u)/self.h), self.x[:-1]) for prev_u, u in zip(prev_U, U)])

    def _compute_dirichlet(self, node_values):
        node_values = np.array([node_values])
        return np.mean(np.square(np.tril(node_values-node_values.T)))
