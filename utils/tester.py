from factories.qg_fem_factory import *
from factories.qg_neumann_neumann_fem_factory import *
import pickle
from scipy import integrate
import time
from utils.plotter import *

def run_size_experiment(args, sizes, create_graph, thetas, filename):
    if isinstance(thetas, float):
        thetas = [thetas for _ in sizes]

    saved_ref_results = {}
    saved_results = {}
    times = {'FEM': [], 'NNF': []}
    factories = {'NNF': QGNeumannNeumannFEMFactory()}
    params = {'NNF': {'adaptive': args.adaptive,
                      'bcs': {},
                      'change_threshold': 10**-args.change,
                      'maxiter': args.maxiter,
                      'N': 2**args.mesh,
                      'weights': True}}

    for ix, size in enumerate(sizes):
        saved_ref_results[size] = {}
        saved_results[size] = {}
        adj, coords = create_graph(size)

        for solver_type in params:
            params[solver_type]['adjacency'] = adj
            params[solver_type]['theta'] = thetas[ix]

        for _ in range(args.reps-1):
            ref_solver, ref_results = run_reference({'adjacency': adj, 'N': 2**args.mesh,
                                                     'change_threshold': 10**-args.change,
                                                     'iterative': args.ref_iter},
                                                    False, False)
            times['FEM'].append(ref_results['time'])

            for solver_type in params:
                results = run_example(factories[solver_type], params[solver_type], False, False)
                times[solver_type].append(results['time'])

        ref_solver, saved_ref_results[size] = run_reference({'adjacency': adj, 'N': 2**args.mesh,
                                                             'change_threshold': 10**-args.change,
                                                             'iterative': args.ref_iter},
                                                            True, True)
        times['FEM'].append(saved_ref_results[size]['time'])
        saved_ref_results[size]['coords'] = coords
        saved_ref_results[size]['edges'] = ref_solver.edges
        X_fem, U_fem = ref_solver.get_solution()

        for solver_type in params:
            saved_results[size][solver_type] = run_example(factories[solver_type], params[solver_type], True, True, U_fem)
            times[solver_type].append(saved_results[size][solver_type]['time'])

    with open(f'tests/data/{filename}_{args.reps}_times.pkl', 'wb') as f:
        pickle.dump(times, f)
    with open(f'tests/data/{filename}.pkl', 'wb') as f:
        pickle.dump(saved_results, f)
    with open(f'tests/data/{filename}_ref.pkl', 'wb') as f:
        pickle.dump(saved_ref_results, f)

def run_mesh_experiment(args, Ns, adj, coords, thetas, filename):
    if isinstance(thetas, float):
        thetas = [thetas for _ in Ns]

    saved_ref_results = {}
    saved_results = {}
    times = {'FEM': [], 'NNF': []}
    factories = {'NNF': QGNeumannNeumannFEMFactory()}
    params = {'NNF': {'adaptive': args.adaptive,
                      'bcs': {},
                      'change_threshold': 10**-args.change,
                      'maxiter': args.maxiter,
                      'adjacency': adj,
                      'weights': True}}

    for ix, N in enumerate(Ns):
        saved_ref_results[N] = {}
        saved_results[N] = {}

        for solver_type in params:
            params[solver_type]['N'] = 2**N
            params[solver_type]['theta'] = thetas[ix]

        for _ in range(args.reps-1):
            ref_solver, ref_results = run_reference({'adjacency': adj, 'N': 2**N,
                                                     'change_threshold': 10**-args.change,
                                                     'iterative': args.ref_iter},
                                                    False, False)
            times['FEM'].append(ref_results['time'])

            for solver_type in params:
                results = run_example(factories[solver_type], params[solver_type], False, False)
                times[solver_type].append(results['time'])

        ref_solver, saved_ref_results[N] = run_reference({'adjacency': adj, 'N': 2**N,
                                                             'change_threshold': 10**-args.change,
                                                             'iterative': args.ref_iter},
                                                            True, True)
        times['FEM'].append(saved_ref_results[N]['time'])
        saved_ref_results[N]['coords'] = coords
        saved_ref_results[N]['edges'] = ref_solver.edges
        X_fem, U_fem = ref_solver.get_solution()

        for solver_type in params:
            saved_results[N][solver_type] = run_example(factories[solver_type], params[solver_type], True, True, U_fem)
            times[solver_type].append(saved_results[N][solver_type]['time'])

    with open(f'tests/data/{filename}_{args.reps}_times.pkl', 'wb') as f:
        pickle.dump(times, f)
    with open(f'tests/data/{filename}.pkl', 'wb') as f:
        pickle.dump(saved_results, f)
    with open(f'tests/data/{filename}_ref.pkl', 'wb') as f:
        pickle.dump(saved_ref_results, f)

def run_reference(params, store, evaluate):
    qg_fem_factory = QGFEMFactory()
    tic = time.perf_counter()
    qg_fem = qg_fem_factory.create_solver(**params)
    qg_fem.solve()
    toc = time.perf_counter()
    if evaluate:
        results = eval_reference(qg_fem, store)
    else:
        results = {}
    results['time'] = toc-tic
    print(f'Reference computed in {toc-tic:0.4f} seconds')
    return qg_fem, results

def eval_reference(solver, store):
    results = {'continuity_error': solver.check_continuity(),
               'neumann_kirchhoff_error': solver.check_Neumann_Kirchhoff()}
    if store:
        results['x'], results['u'] = solver.get_solution()
    return results

def run_example(factory, params, store, evaluate, U_ref=None):
    tic = time.perf_counter()
    qg_solver = factory.create_solver(**params)
    iteration = qg_solver.solve()
    toc = time.perf_counter()
    if evaluate:
        results = eval_example(qg_solver, store, U_ref)
    else:
        results = {}
    results['iteration'] = iteration
    results['time'] = toc-tic
    print(f'Solution computed in {toc-tic:0.4f} seconds')
    return results

def eval_example(solver, store, U_ref):
    X, U = solver.get_solution()
    x = X[0]
    l2error = 0
    h1error = 0
    for u, u_ref in zip(U, U_ref):
        u_N = len(u)-1
        l2tmp = integrate.simpson(np.square(u-u_ref), x)
        h1tmp = integrate.simpson(np.square(np.diff(u-u_ref)/np.diff(x)[0]), x[:-1])
        l2error += l2tmp
        h1error += l2tmp+h1tmp
    results = {'l2error': np.sqrt(l2error),
               'h1error': np.sqrt(h1error),
               'continuity_error': solver.check_continuity(),
               'neumann_kirchhoff_error': solver.check_Neumann_Kirchhoff()}
    if store:
        results['x'], results['u'] = X, U
    return results
