import numpy as np
import scipy as sp
from solvers.bcs import *

def load_vector_fun(f, x, lbc, rbc):
    h = x[1]-x[0]
    f = np.vectorize(f)(x)
    return load_vector_value(f, h, lbc, rbc)

def load_vector_value(f, h, lbc, rbc):
    N = len(f)
    b = np.zeros(N)
    b[1:N-1] = h*f[1:N-1]
    return apply_bcs_to_load_vector(b, f, h, lbc, rbc)

def apply_bcs_to_stiffness_matrix(A, lbc, rbc):
    if lbc.type is BCType.DIRICHLET:
        A[1][0] = 1
        A[0][1] = 0
    if rbc.type is BCType.DIRICHLET:
        A[1][-1] = 1
        A[2][-2] = 0
    return A

def apply_bcs_to_load_vector(b, f, h, lbc, rbc):
    if lbc.type is BCType.DIRICHLET:
        b[0] = lbc.value 
    elif lbc.type is BCType.NEUMANN:
        b[0] = h*f[0]/2 - lbc.value 
    if rbc.type is BCType.DIRICHLET:
        b[-1] = rbc.value 
    elif rbc.type is BCType.NEUMANN:
        b[-1] = h*f[-1]/2 + rbc.value 
    return b

############################ hamiltonian ############################
def stiffness_matrix_hamiltonian_fun(c, v, x, lbc, rbc):
    h = x[1]-x[0]
    c = np.vectorize(c)(x)
    v = np.vectorize(v)(x)
    return stiffness_matrix_hamiltonian_value(c, v, h, lbc, rbc)

def stiffness_matrix_hamiltonian_value(c, v, h, lbc, rbc):
    N = len(c)
    d = np.convolve(c, [1, 2, 1], 'same')/(2*h)+h*v
    du = -np.convolve(c, [1, 1], 'same')/(2*h)
    dl = np.copy(du)
    d[0] = (c[0]+c[1]) / (2*h) + h/2*v[0]
    d[-1] = (c[-2]+c[-1]) / (2*h) + h/2*v[-1]
    dl = np.roll(dl, -1)
    return [du, d, dl]

def FEM_hamiltonian(c, v, f, lbc, rbc, a, b, N):
    x = np.linspace(a, b, N+1)
    A = stiffness_matrix_hamiltonian_fun(c, v, x, lbc, rbc)
    b = load_vector_fun(f, x, lbc, rbc)
    return sp.linalg.solve_banded((1,1), A, b)

############################### heat ###############################
def stiffness_matrix_heat_fun(c, x, lbc, rbc):
    N = len(x)
    h = x[1]-x[0]
    A = sp.lil_matrix((N, N))
    for i in range(1,N-1):
        A[i, i] = (c(x[i-1])+2*c(x[i])+c(x[i+1])) / (2*h)
        A[i, i-1] = -(c(x[i-1])+c(x[i])) / (2*h)
        A[i, i+1] = -(c(x[i])+c(x[i+1])) / (2*h)
    if lbc.type is BCType.DIRICHLET:
        A[0, 0] = 1
    elif lbc.type is BCType.NEUMANN:
        A[0, 0] = (c(x[0])+c(x[1])) / (2*h)
        A[0, 1] = -A[0, 0]
    if rbc.type is BCType.DIRICHLET:
        A[-1, -1] = 1
    elif rbc.type is BCType.NEUMANN:
        A[-1, -1] = (c(x[-2])+c(x[-1])) / (2*h)
        A[-1, -2] = -A[-1, -1]
    return A.tocsc()

def FEM_heat(c, f, lbc, rbc, a, b, N):
    x = np.linspace(a, b, N+1)
    A = stiffness_matrix_heat_fun(c, x, lbc, rbc)
    b = load_vector_fun(f, x, lbc, rbc)
    return sp.linalg.spsolve(A, b)
