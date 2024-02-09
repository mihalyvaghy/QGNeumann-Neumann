from factories.qg_solver_factory import *
from solvers.fem.qg_fem import *

class QGFEMFactory(QGSolverFactory):
    def __init__(self):
        pass

    def create_solver(self, **kwargs):
        return QGFEM(kwargs['adjacency'], kwargs['N'], kwargs['change_threshold'], kwargs['iterative'])
