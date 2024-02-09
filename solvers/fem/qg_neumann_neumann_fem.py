import scipy as sp
from solvers.fem.fem import *
from solvers.interfaces.qg_neumann_neumann import *

class QGNNFEMEdge(QGEdge):
    def __init__(self, out_, in_, x, h, c_value, f_value, A, lbc, rbc):
        self.out_ = out_
        self.in_ = in_
        self.x = x
        self.h = h
        self.lbc = lbc
        self.rbc = rbc
        self.lweight = 1
        self.rweight = 1
        self.c_value = c_value
        self.f_value = f_value
        self.A = A
        self.Al = [self.A[1][0], self.A[0][1]]
        self.Ar = [self.A[2][-2], self.A[1][-1]]
        self.A = apply_bcs_to_stiffness_matrix(self.A, self.lbc, self.rbc)
        self.b = load_vector_value(self.f_value, self.h, self.lbc, self.rbc)
        self.u = np.zeros(self.x.shape)

    def solve(self):
        self.u = sp.linalg.solve_banded((1,1), self.A, self.b)

    def update_bc_values(self, lbc_value, rbc_value):
        self.lbc.value = lbc_value/self.lweight
        self.rbc.value = rbc_value/self.rweight
        apply_bcs_to_load_vector(self.b, self.f_value, self.h, self.lbc, self.rbc)

    def set_weights(self, lweight, rweight):
        self.lweight = lweight
        self.rweight = rweight

    def approx_Neumann(self, endpoint):
        if endpoint == self.out_:
            return -(np.dot(self.Al, self.u[:2])-self.h*self.f_value[0]/2)
        else:
            return -(np.dot(self.Ar, self.u[-2:])-self.h*self.f_value[-1]/2)

class QGNeumannNeumannFEM(QGNeumannNeumann):
    def __init__(self, adj, bcs, N, change_threshold, maxiter, theta, use_weights, adaptive):
        self.N = N
        self.x0 = 0
        self.xend = 1
        self.x = np.linspace(self.x0, self.xend, self.N+1)
        self.h = (self.xend-self.x0)/self.N
        self.adj = np.array(adj)
        self.bcs = bcs
        self.change_threshold = change_threshold/N
        self.maxiter = maxiter
        self.theta = theta
        self.adaptive = adaptive
        self.vertex_values = np.zeros(len(self.adj))
        self.dirichlet_edges, self.neumann_edges, self.vertex2edge = self._generate_edges()
        if use_weights:
            self.vertex_weights = self._degrees()
        else:
            self.vertex_weights = np.ones(len(self.adj))
        self._set_bc_weights()

    def _new_edges(self, out_, in_, lbc, rbc):
        params = self.adj[out_][in_]
        c_value = np.vectorize(params['c'])(self.x)
        v_value = np.vectorize(params['v'])(self.x)
        f_value = np.vectorize(params['f'])(self.x)
        A = stiffness_matrix_hamiltonian_value(c_value, v_value, self.h, lbc, rbc)
        dirichlet_edge = QGNNFEMEdge(out_, in_, self.x, self.h, c_value, f_value, A, lbc, rbc)
        neumann_edge = QGNNFEMEdge(out_, in_, self.x, self.h, c_value, np.zeros_like(self.x), A, BC(BCType.NEUMANN), BC(BCType.NEUMANN))
        return dirichlet_edge, neumann_edge
