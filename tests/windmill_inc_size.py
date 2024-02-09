import argparse
import networkx as nx
from utils.tester import *

parser = argparse.ArgumentParser(
        prog='QG solver for stars',
        description='Run Neumann-Neumann algorithm and QG FEM on windmill graphs of various clique sizes',
        epilog='Bottom text.')

parser.add_argument('--adaptive', action='store_true', default=False,
                    help='use adaptive iteration')
parser.add_argument('--change', action='store', default=0,
                    help='power of ten decreased tolerance', type=int)
parser.add_argument('--K_max', action='store', default=15,
                    help='maximum clique size', type=int)
parser.add_argument('--K_min', action='store', default=5,
                    help='minimum clique size', type=int)
parser.add_argument('--K_step', action='store', default=2,
                    help='clique size step', type=int)
parser.add_argument('--maxiter', action='store', default=100,
                    help='maximum number of iterations', type=int)
parser.add_argument('--mesh', action='store', default=10,
                    help='power of two of test mesh size', type=int)
parser.add_argument('--N', action='store', default=11,
                    help='number of cliques')
parser.add_argument('--plots', action='store', default=[],
                    help='choose plots', nargs='*',
                    choices=['cont', 'h1', 'iter', 'l2', 'nk', 'time', 'qg'])
parser.add_argument('--ref_iter', action='store_true', default=False,
                    help='use iterative reference solver')
parser.add_argument('--reps', action='store', default=1,
                    help='number of repetitions', type=int)

args = parser.parse_args()

sizes = np.arange(args.K_min, args.K_max+1, args.K_step)
theta = 0.95
filename = 'windmill_inc_size'

def create_graph(size):
    G = nx.windmill_graph(args.N, size)
    adj = nx.to_numpy_array(G).astype(object)
    adj -= np.tril(adj)
    for i in range(len(adj)):
        for j in range(len(adj)):
            if adj[i,j]:
                adj[i,j] = {'c': lambda x: 1/(2-x**2), 'v': lambda x: 2-x, 'f': lambda x: 2+np.cos(5*np.pi*x)}
    coords = list(nx.spring_layout(G).values())
    return adj, coords

filename = f'{filename}_{args.mesh}'
run_size_experiment(args, sizes, create_graph, theta, filename)
plot_size_experiment(filename, args.reps, True, False)
