import matplotlib.pyplot as plt
import numpy as np
import pickle

plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'axes.titlesize': 16,
    'axes.labelsize': 14
    })

class PlotterQG:
    def __init__(self, vertices, edges, xs, us, labels, title='', xlabel='', ylabel='', zlabel = ''):
        self.edges = edges
        self.vertices = np.array(vertices)
        self.xs = xs
        self.us = us
        self.labels = labels
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.zlabel = zlabel
        self.colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k']
        self.plot()

    def plot(self):
        plt.figure()
        ax = plt.axes(projection='3d')
        self._plotQG(ax)

        for x, u, color, label in zip(self.xs, self.us, self.colors, self.labels):
            self._plotU(ax, x, u, color, label)

        ax.set_zlim(*self._zlim())
        ax.view_init(azim=-130, elev=25)
        self._no_duplication_legend()
        plt.title(self.title)
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        ax.set_zlabel(self.zlabel)

    def show(self):
        plt.show()

    def _zlim(self):
        zlim = [self.us[0][0][0], self.us[0][0][0]]
        for u in self.us:
            for ui in u:
                zlim[0] = min(zlim[0], np.min(ui))
                zlim[1] = max(zlim[1], np.max(ui))
        if zlim[0] > 0:
            zlim[0] = 0
        elif zlim[1] < 0:
            zlim[1] = 0
        return zlim

    def _plotQG(self, ax):
        for edge in self.edges:
            in_, out_ = edge
            x = [self.vertices[out_][0], self.vertices[in_][0]]
            y = [self.vertices[out_][1], self.vertices[in_][1]]
            ax.plot(x, y, [0, 0], 'k')
        ax.scatter3D(self.vertices[:,0], self.vertices[:,1], 0, color='k')

    def _calc_line(self, x1, x2, x):
        return x1+(x2-x1)*x/x[-1]

    def _draw_edge(self, edge, x):
        out_, in_ = edge
        X = self._calc_line(self.vertices[out_][0], self.vertices[in_][0], x)
        Y = self._calc_line(self.vertices[out_][1], self.vertices[in_][1], x)
        return X, Y

    def _plotU(self, ax, x, u, color, label):
        for edge, xi, ui in zip(self.edges, x, u):
            X, Y = self._draw_edge(edge, xi)
            ax.plot(X, Y, ui, color=color, label=label)

    def _no_duplication_legend(self):
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())

def plot_size_experiment(filename, reps, save, show):
    with open(f'tests/data/{filename}_{reps}_times.pkl', 'rb') as f:
        times = pickle.load(f)
    with open(f'tests/data/{filename}.pkl', 'rb') as f:
        results = pickle.load(f)
    with open(f'tests/data/{filename}_ref.pkl', 'rb') as f:
        ref_results = pickle.load(f)

    sizes = list(results.keys())
    solvers = list(results[sizes[0]].keys())

    plot_size_cont(ref_results, results, sizes, solvers, save, filename)
    plot_size_h1(results, sizes, solvers, save, filename)
    plot_size_iter(results, sizes, solvers, save, filename)
    plot_size_l2(results, sizes, solvers, save, filename)
    plot_size_nk(ref_results, results, sizes, solvers, save, filename)
    plot_size_time(ref_results, results, sizes, solvers, save, f'{filename}_{reps}')

    if show:
        plt.show()

def plot_size_cont(ref_results, results, sizes, solvers, save, filename):
    continuities = {solver_type: [np.sum(results[size][solver_type]['continuity_error']) for size in sizes] for solver_type in solvers}
    continuities['FEM'] = [np.sum(ref_results[size]['continuity_error']) for size in sizes]
    plt.figure()
    for solver_type in continuities:
        plt.plot(sizes, continuities[solver_type], label=solver_type, marker='.')
    plt.grid(visible=True)
    plt.title('Cumulative continuity error')
    plt.legend()
    if save:
        plt.savefig(f'tests/figures/{filename}_size_cont.png', dpi=600)
        plt.savefig(f'tests/figures/{filename}_size_cont.eps')

def plot_size_h1(results, sizes, solvers, save, filename):
    h1errors = {solver_type: [results[size][solver_type]['h1error'] for size in sizes] for solver_type in solvers}
    plt.figure()
    for solver_type in h1errors:
        plt.plot(sizes, h1errors[solver_type], label=solver_type, marker='.')
    plt.grid(visible=True)
    plt.title('$H^1(\mathsf{G})$ error')
    plt.legend()
    if save:
        plt.savefig(f'tests/figures/{filename}_size_H1.png', dpi=600)
        plt.savefig(f'tests/figures/{filename}_size_H1.eps')

def plot_size_iter(results, sizes, solvers, save, filename):
    iterations = {solver_type: [results[size][solver_type]['iteration'] for size in sizes] for solver_type in solvers}
    plt.figure()
    for solver_type in iterations:
        plt.plot(sizes, iterations['NNF'], label=solver_type, marker='.')
    plt.grid(visible=True)
    plt.title('Number of iterations')
    if save:
        plt.savefig(f'tests/figures/{filename}_size_iterations.png', dpi=600)
        plt.savefig(f'tests/figures/{filename}_size_iterations.eps')

def plot_size_l2(results, sizes, solvers, save, filename):
    l2errors = {solver_type: [results[size][solver_type]['l2error'] for size in sizes] for solver_type in solvers}
    plt.figure()
    for solver_type in l2errors:
        plt.plot(sizes, l2errors[solver_type], label=solver_type, marker='.')
    plt.grid(visible=True)
    plt.title('$L^2(\mathsf{G})$ error')
    plt.legend()
    if save:
        plt.savefig(f'tests/figures/{filename}_size_L2.png', dpi=600)
        plt.savefig(f'tests/figures/{filename}_size_L2.eps')

def plot_size_nk(ref_results, results, sizes, solvers, save, filename):
    neumann_kirchhoffs = {solver_type: [np.sum(np.absolute(results[size][solver_type]['neumann_kirchhoff_error'])) for size in sizes] for solver_type in solvers}
    neumann_kirchhoffs['FEM'] = [np.sum(np.absolute(ref_results[size]['neumann_kirchhoff_error'])) for size in sizes] 
    plt.figure()
    for solver_type in neumann_kirchhoffs:
        plt.plot(sizes, neumann_kirchhoffs[solver_type], label=solver_type, marker='.')
    plt.grid(visible=True)
    plt.title('Cumulative Neumann-Kirchhoff error')
    plt.legend()
    if save:
        plt.savefig(f'tests/figures/{filename}_size_NK.png', dpi=600)
        plt.savefig(f'tests/figures/{filename}_size_NK.eps')

def plot_size_time(ref_results, results, sizes, solvers, save, filename):
    times = {solver_type: [results[size][solver_type]['time'] for size in sizes] for solver_type in solvers}
    times['FEM'] = [ref_results[size]['time'] for size in sizes]
    plt.figure()
    for solver_type in times:
        plt.plot(sizes, times[solver_type], label=solver_type, marker='.')
    plt.grid(visible=True)
    plt.title('Computation time')
    plt.legend()
    if save:
        plt.savefig(f'tests/figures/{filename}_size_time.png', dpi=600)
        plt.savefig(f'tests/figures/{filename}_size_time.eps')

def plot_mesh_experiment(filename, reps, save, show):
    with open(f'tests/data/{filename}_{reps}_times.pkl', 'rb') as f:
        times = pickle.load(f)
    with open(f'tests/data/{filename}.pkl', 'rb') as f:
        results = pickle.load(f)
    with open(f'tests/data/{filename}_ref.pkl', 'rb') as f:
        ref_results = pickle.load(f)

    Ns = list(results.keys())
    solvers = list(results[Ns[0]].keys())

    plot_mesh_cont(ref_results, results, Ns, solvers, save, filename)
    plot_mesh_h1(results, Ns, solvers, save, filename)
    plot_mesh_iter(results, Ns, solvers, save, filename)
    plot_mesh_l2(results, Ns, solvers, save, filename)
    plot_mesh_nk(ref_results, results, Ns, solvers, save, filename)
    plot_mesh_time(ref_results, results, Ns, solvers, save, f'{filename}_{reps}')

    if show:
        plt.show()

def plot_mesh_cont(ref_results, results, Ns, solvers, save, filename):
    continuities = {solver_type: [np.sum(results[size][solver_type]['continuity_error']) for size in Ns] for solver_type in solvers}
    continuities['FEM'] = [np.sum(ref_results[size]['continuity_error']) for size in Ns]
    plt.figure()
    for solver_type in continuities:
        plt.semilogx(Ns, continuities[solver_type], base=2, label=solver_type, marker='.')
    plt.grid(visible=True)
    plt.title('Cumulative continuity error')
    plt.legend()
    if save:
        plt.savefig(f'tests/figures/{filename}_mesh_cont.png', dpi=600)
        plt.savefig(f'tests/figures/{filename}_mesh_cont.eps')

def plot_mesh_h1(results, Ns, solvers, save, filename):
    h1errors = {solver_type: [results[size][solver_type]['h1error'] for size in Ns] for solver_type in solvers}
    plt.figure()
    for solver_type in h1errors:
        plt.semilogx(Ns, h1errors[solver_type], base=2, label=solver_type, marker='.')
    plt.grid(visible=True)
    plt.title('$H^1(\mathsf{G})$ error')
    plt.legend()
    if save:
        plt.savefig(f'tests/figures/{filename}_mesh_H1.png', dpi=600)
        plt.savefig(f'tests/figures/{filename}_mesh_H1.eps')

def plot_mesh_iter(results, Ns, solvers, save, filename):
    iterations = {solver_type: [results[size][solver_type]['iteration'] for size in Ns] for solver_type in solvers}
    plt.figure()
    for solver_type in iterations:
        plt.semilogx(Ns, iterations['NNF'], base=2, label=solver_type, marker='.')
    plt.grid(visible=True)
    plt.title('Number of iterations')
    if save:
        plt.savefig(f'tests/figures/{filename}_mesh_iterations.png', dpi=600)
        plt.savefig(f'tests/figures/{filename}_mesh_iterations.eps')

def plot_mesh_l2(results, Ns, solvers, save, filename):
    l2errors = {solver_type: [results[size][solver_type]['l2error'] for size in Ns] for solver_type in solvers}
    plt.figure()
    for solver_type in l2errors:
        plt.semilogx(Ns, l2errors[solver_type], base=2, label=solver_type, marker='.')
    plt.grid(visible=True)
    plt.title('$L^2(\mathsf{G})$ error')
    plt.legend()
    if save:
        plt.savefig(f'tests/figures/{filename}_mesh_L2.png', dpi=600)
        plt.savefig(f'tests/figures/{filename}_mesh_L2.eps')

def plot_mesh_nk(ref_results, results, Ns, solvers, save, filename):
    neumann_kirchhoffs = {solver_type: [np.sum(np.absolute(results[size][solver_type]['neumann_kirchhoff_error'])) for size in Ns] for solver_type in solvers}
    neumann_kirchhoffs['FEM'] = [np.sum(np.absolute(ref_results[size]['neumann_kirchhoff_error'])) for size in Ns] 
    plt.figure()
    for solver_type in neumann_kirchhoffs:
        plt.semilogx(Ns, neumann_kirchhoffs[solver_type], base=2, label=solver_type, marker='.')
    plt.grid(visible=True)
    plt.title('Cumulative Neumann-Kirchhoff error')
    plt.legend()
    if save:
        plt.savefig(f'tests/figures/{filename}_mesh_NK.png', dpi=600)
        plt.savefig(f'tests/figures/{filename}_mesh_NK.eps')

def plot_mesh_time(ref_results, results, Ns, solvers, save, filename):
    times = {solver_type: [results[size][solver_type]['time'] for size in Ns] for solver_type in solvers}
    times['FEM'] = [ref_results[size]['time'] for size in Ns]
    plt.figure()
    for solver_type in times:
        plt.semilogx(Ns, times[solver_type], base=2, label=solver_type, marker='.')
    plt.grid(visible=True)
    plt.title('Computation time')
    plt.legend()
    if save:
        plt.savefig(f'tests/figures/{filename}_mesh_time.png', dpi=600)
        plt.savefig(f'tests/figures/{filename}_mesh_time.eps')
