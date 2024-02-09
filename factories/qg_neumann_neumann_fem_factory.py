from factories.qg_solver_factory import *
from solvers.fem.qg_neumann_neumann_fem import *

class QGNeumannNeumannFEMFactory(QGSolverFactory):
    def __init__(self):
        pass

    def create_solver(self, **kwargs):
        return QGNeumannNeumannFEM(kwargs['adjacency'],
                                   kwargs['bcs'],
                                   kwargs['N'],
                                   kwargs['change_threshold'],
                                   kwargs['maxiter'],
                                   kwargs['theta'],
                                   kwargs['weights'],
                                   kwargs['adaptive'])
