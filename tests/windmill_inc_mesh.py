import argparse
import networkx as nx
from utils.tester import *

parser = argparse.ArgumentParser(
        prog='QG solver for stars',
        description='Run Neumann-Neumann algorithm and QG FEM on windmill graphs with various meshes',
        epilog='Bottom text.')

parser.add_argument('--adaptive', action='store_true', default=False,
                    help='use adaptive iteration')
parser.add_argument('--change', action='store', default=0,
                    help='power of ten decreased tolerance', type=int)
parser.add_argument('--maxiter', action='store', default=100,
                    help='maximum number of iterations', type=int)
parser.add_argument('--K', action='store', default=14,
                    help='clique size')
parser.add_argument('--M', action='store', default=14,
                    help='number of cliques', type=int)
parser.add_argument('--N_max', action='store', default=13,
                    help='power of two of maximum number of elements', type=int)
parser.add_argument('--N_min', action='store', default=8,
                    help='poewr of two of minimum number of elements', type=int)
parser.add_argument('--N_step', action='store', default=1,
                    help='element number step', type=int)
parser.add_argument('--plots', action='store', default=[],
                    help='choose plots', nargs='*',
                    choices=['cont', 'h1', 'iter', 'l2', 'nk', 'time', 'qg'])
parser.add_argument('--ref_iter', action='store_true', default=False,
                    help='use iterative reference solver')
parser.add_argument('--reps', action='store', default=1,
                    help='number of repetitions', type=int)

args = parser.parse_args()

Ns = np.arange(args.N_min, args.N_max+1, args.N_step)
theta = 0.95
filename = 'windmill_inc_mesh'

G = nx.windmill_graph(args.M, args.K)
adj = nx.to_numpy_array(G).astype(object)
adj -= np.tril(adj)
for i in range(len(adj)):
    for j in range(len(adj)):
        if adj[i,j]:
            adj[i,j] = {'c': lambda x: 1/(2-x**2), 'v': lambda x: 2-x, 'f': lambda x: 2+np.cos(5*np.pi*x)}
coords = list(nx.spring_layout(G).values())

filename = f'{filename}_{args.M}_{args.K}'
#run_mesh_experiment(args, Ns, adj, coords, theta, filename)
plot_mesh_experiment(filename, args.reps, True, False)
