import numpy as np
import scipy.sparse as sp
from solvers.interfaces.qg_solver import *

class QGFEM(QGSolver):
    def __init__(self, adj, N, change_threshold, iterative):
        self.N =N
        self.x = np.linspace(0, 1, N+1)
        self.h = 1/N
        self.adj = np.array(adj)
        self.change_threshold = change_threshold/N
        self.iterative = iterative
        self.edges = [[i,j] for i in range(len(adj)) for j in range(len(adj)) if adj[i][j]]
        self.H = self._compute_H()
        self.F = self._compute_F()
        self.u = []

    def solve(self):
        N = self.N
        if self.iterative:
            u, _ = sp.linalg.bicgstab(self.H, self.F, tol=self.change_threshold, atol=self.change_threshold, M=sp.linalg.LinearOperator(self.H.shape, sp.linalg.spilu(self.H).solve))
        else:
            u = sp.linalg.spsolve(self.H, self.F)

        u_vertex = u[-len(self.adj):]
        self.vertex_values = u_vertex
        for i, edge in enumerate(self.edges):
            out_, in_ = edge
            self.u.append(np.hstack([u_vertex[out_], u[i*(N-1):(i+1)*(N-1)], u_vertex[in_]]))

    def check_continuity(self):
        nodes_values = [[] for _ in range(len(self.adj))]
        for i, edge in enumerate(self.edges):
            out_, in_ = edge
            nodes_values[out_].append(self.u[i][0])
            nodes_values[in_].append(self.u[i][-1])
        return [self._compute_dirichlet(node_values) for node_values in nodes_values]

    def check_Neumann_Kirchhoff(self):
        x, h = self.x, self.h
        node_kn = [0 for _ in range(len(self.adj))]
        for i, edge in enumerate(self.edges):
            out_, in_ = edge
            c = np.vectorize(self.adj[out_, in_]['c'])(x)
            u = self.u[i]
            node_kn[out_] += c[0]*(u[1]-u[0])/h
            node_kn[in_] += c[-1]*(u[-2]-u[-1])/h
        return node_kn

    def get_solution(self):
        return [self.x]*len(self.edges), self.u

    def get_vertex_values(self):
        return self.vertex_values

    def _compute_H(self):
        A = self._compute_A()
        B = self._compute_B()
        G = self._compute_G(B)
        return sp.bmat([[A, B],
                        [B.T, G]], format='csc')

    def _compute_F(self):
        f_edge = np.vstack([self._edge_load_vector(self.adj[*edge]) for edge in self.edges])
        f_vertex = self._vertex_load_vector()
        return np.vstack([f_edge, f_vertex])
    
    def _compute_A(self):
        return sp.block_diag([self._stiffness_matrix(self.adj[*edge]) for edge in self.edges], format='csc')

    def _stiffness_matrix(self, edge):
        N, x, h = self.N, self.x, self.h
        c, v = np.vectorize(edge['c'])(x), np.vectorize(edge['v'])(x)
        diag = np.convolve(c, [1, 2, 1], 'valid')/(2*h)+h*v[1:-1]
        diagp1 = -np.roll(np.convolve(c[1:], [1, 1], 'valid')/(2*h), 1)
        diagm1 = -np.roll(np.convolve(c[:-1], [1, 1], 'valid')/(2*h), -1)
        return sp.spdiags([diag, diagp1, diagm1], [0, 1, -1], format='csc')

    def _compute_B(self):
        N, x, h = self.N, self.x, self.h
        B = sp.lil_matrix((len(self.edges)*(self.N-1), len(self.adj)))
        for i, edge in enumerate(self.edges): 
            out_, in_ = edge
            c = np.vectorize(self.adj[out_, in_]['c'])(x)
            B[i*(N-1), out_] = -(c[0]+c[1])#start
            B[(i+1)*(N-1)-1, in_] = -(c[-2]+c[-1])#end
        return B.tocsc()/(2*h)

    def _compute_G(self, B):
        x, h = self.x, self.h
        G = -np.sum(B, axis=0)
        for out_, in_ in self.edges:
            v = np.vectorize(self.adj[out_, in_]['v'])(x)
            G[0,out_] += v[0]*h/2
            G[0,in_] += v[-1]*h/2
        return sp.spdiags(G, 0, len(self.adj), len(self.adj))

    def _edge_load_vector(self, edge):
        f = edge['f']
        x, h = self.x, self.h
        return h*np.array([[f(xi) for xi in x[1:-1]]]).T

    def _vertex_load_vector(self):
        # sparse matrix
        x, h = self.x, self.h
        f_vertex = np.zeros((len(self.adj), 1))
        for out_, in_ in self.edges:
            f = np.vectorize(self.adj[out_, in_]['f'])(x)
            f_vertex[out_, 0] += f[0]*h/2
            f_vertex[in_, 0] += f[-1]*h/2
        return f_vertex

    def _compute_dirichlet(self, node_values):
        node_values = np.array([node_values])
        return np.mean(np.square(np.tril(node_values-node_values.T)))
