from solvers.bcs import *
from solvers.interfaces.qg_iterative_solver import *

class QGNeumannNeumann(QGIterativeSolver):
    def check_continuity(self):
        nodes_values = [[] for _ in range(len(self.adj))]
        for edge in self.dirichlet_edges:
            out_, in_ = edge.endpoints()
            nodes_values[out_].append(edge.Dirichlet(out_))
            nodes_values[in_].append(edge.Dirichlet(in_))
        return [self._compute_dirichlet(node_values) for node_values in nodes_values]

    def check_Neumann_Kirchhoff(self):
        nkd = self.check_Neumann_Kirchhoff_vector(self.dirichlet_edges)
        nkn = self.check_Neumann_Kirchhoff_vector(self.neumann_edges)
        return [nkde+nkne for nkde, nkne in zip(nkd, nkn)]

    def check_Neumann_Kirchhoff_vector(self, edges):
        node_kn = [0 for _ in range(len(self.adj))]
        for edge in edges:
            out_, in_ = edge.endpoints()
            node_kn[out_] += edge.Neumann(out_)
            node_kn[in_] += edge.Neumann(in_)
        return node_kn

    def get_solution(self):
        return [edge.x for edge in self.dirichlet_edges], [edge.u for edge in self.dirichlet_edges]

    def get_vertex_values(self):
        return self.vertex_values

    @abstractmethod
    def _new_edges(self, out_, in_, lbc, rbc):
        pass

    def _get_U(self):
        return np.stack([edge.u for edge in self.dirichlet_edges])

    def _iterate(self):
        self._update_vertex_values()

        for edge in self.dirichlet_edges:
            self._update_dirichlet_edge_bc_values(edge)
            edge.solve()

        for edge in self.neumann_edges:
            self._update_neumann_edge_bc_values(edge)
            edge.solve()

    def _update_vertex_values(self):
        node_values = np.zeros(len(self.adj))
        for edge in self.neumann_edges:
            out_, in_ = edge.endpoints()
            node_values[out_] += edge.Dirichlet(out_)
            node_values[in_] += edge.Dirichlet(in_)
        self.vertex_values -= self.theta*np.divide(node_values, self.vertex_weights)

    def _update_dirichlet_edge_bc_values(self, edge): 
        out_, in_ = edge.endpoints()
        lbc_value = self.bcs[out_].value if out_ in self.bcs else self.vertex_values[out_]
        rbc_value = self.bcs[in_].value if in_ in self.bcs else self.vertex_values[in_]
        edge.update_bc_values(lbc_value, rbc_value)

    def _update_neumann_edge_bc_values(self, edge): 
        out_, in_ = edge.endpoints()
        lbc_value = self.bcs[out_].value if out_ in self.bcs else -self._compute_new_neumann_bc(out_)
        rbc_value = self.bcs[in_].value if in_ in self.bcs else self._compute_new_neumann_bc(in_)
        edge.update_bc_values(lbc_value, rbc_value)

    def _compute_new_neumann_bc(self, v):
        return -np.sum([self.dirichlet_edges[edge].approx_Neumann(v) for edge in self.vertex2edge[v]])

    def _generate_edges(self):
        dirichlet_edges = []
        neumann_edges = []
        vertex2edge = {}

        for i in range(len(self.adj)):
            for j in range(len(self.adj)):
                if self.adj[i][j]:
                    lbc, rbc = self._initialize_dirichlet_edge_bcs(i, j)
                    dirichlet_edge, neumann_edge = self._new_edges(i, j, lbc, rbc)
                    dirichlet_edges.append(dirichlet_edge)
                    neumann_edges.append(neumann_edge)

                    new_edge_index = len(dirichlet_edges)-1
                    if i in vertex2edge:
                        vertex2edge[i].append(new_edge_index)
                    else:
                        vertex2edge[i] = [new_edge_index]
                    if j in vertex2edge:
                        vertex2edge[j].append(new_edge_index)
                    else:
                        vertex2edge[j] = [new_edge_index]

        return dirichlet_edges, neumann_edges, vertex2edge

    def _initialize_dirichlet_edge_bcs(self, out_, in_):
        lbc = self.bcs[out_] if out_ in self.bcs else BC(BCType.DIRICHLET)
        rbc = self.bcs[in_] if in_ in self.bcs else BC(BCType.DIRICHLET)
        return lbc, rbc

    def _degrees(self):
        return [len(edges) for edges in self.vertex2edge.values()]

    def _set_bc_weights(self):
        for edge in self.neumann_edges:
            out_, in_ = edge.endpoints()
            edge.set_weights(self.vertex_weights[out_], self.vertex_weights[in_])
