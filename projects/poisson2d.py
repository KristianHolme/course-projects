import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import scipy.sparse as sparse
from sympy.utilities.lambdify import implemented_function
from poisson import Poisson

x, y = sp.symbols('x,y')

class Poisson2D:
    r"""Solve Poisson's equation in 2D::

        \nabla^2 u(x, y) = f(x, y), x, y in [0, Lx] x [0, Ly]

    with homogeneous Dirichlet boundary conditions.
    """

    def __init__(self, Lx, Ly, Nx, Ny):
        self.px = Poisson(Lx, Nx) # we can reuse some of the code from the 1D case
        self.py = Poisson(Ly, Ny)

    def create_mesh(self):
        """Return a 2D Cartesian mesh
        """
        smesh = np.meshgrid(self.px.x, self.py.x, sparse=True, indexing='ij')
        self.smesh = smesh
        return smesh

    def laplace(self):
        """Return a vectorized Laplace operator"""
        D2x = self.px.D2()
        D2y = self.py.D2()
        laplace = (sparse.kron(D2x, sparse.eye(self.py.N+1)) +
            sparse.kron(sparse.eye(self.px.N+1), D2y))
        return laplace

    def assemble(self, f=None, bc=None):
        """Return assemble coefficient matrix A and right hand side vector b"""
        smesh = self.create_mesh()
        xij, yij = smesh
        A = self.laplace()
        b = sp.lambdify((x, y), f)(xij, yij).ravel()
        
        
        
        B = np.ones((self.px.N+1, self.py.N+1), dtype=bool)
        B[1:-1, 1:-1] = 0
        bnds = np.where(B.ravel() == 1)[0]
        A = A.tolil()
        for i in bnds:
            A[i] = 0
            A[i, i] = 1
        A = A.tocsr()
        
        #Set bdry condition
        B_sparse = sparse.csr_matrix(B)
        boundary_indices = B_sparse.nonzero()
        bdry_x_inds, bdry_y_inds = boundary_indices
        bdry_x = xij[bdry_x_inds,0]
        bdry_y = yij[0,bdry_y_inds]
        
        if bc == None:
            b[bnds] = 0
        else:
            # Evaluate boundary condition
            bc_func = sp.lambdify((x, y), bc)
            b[bnds] = bc_func(bdry_x, bdry_y)
        
        return A, b

    def l2_error(self, u, ue):
        """Return l2-error

        Parameters
        ----------
        u : array
            The numerical solution (mesh function)
        ue : Sympy function
            The analytical solution
        """
        xij, yij = self.smesh
        uj = sp.lambdify((x, y), ue)(xij, yij)
        dx = self.px.dx
        dy = self.py.dx
        np.sqrt(dx*dy*np.sum((u - uj)**2))
        return np.sqrt(dx*dy*np.sum((uj-u)**2))

    def __call__(self, f=implemented_function('f', lambda x, y: 2)(x, y), bc = None):
        """Solve Poisson's equation with a given righ hand side function

        Parameters
        ----------
        f : Sympy function
            The right hand side function f(x, y)

        Returns
        -------
        The solution as a Numpy array

        """
        A, b = self.assemble(f=f, bc = bc)
        return sparse.linalg.spsolve(A, b.ravel()).reshape((self.px.N+1, self.py.N+1))
    
    def plot(self, u, ue):
        xij, yij = self.smesh
        uj = sp.lambdify((x, y), ue)(xij, yij)
        fig = plt.figure()
        ax = fig.add_subplot(121, projection='3d')
        ax2 = fig.add_subplot(122, projection='3d')
        # Plotting the basic 3D surface
        ax.plot_surface(xij, yij, u, cmap='viridis', label='Numerical')
        ax.set_title('Numerical')
        ax2.plot_surface(xij, yij, uj, cmap='viridis', label='Exact')
        ax2.set_title('Exact')
        
        plt.show()


def test_homogenous():
    tol = 1e-4
    Lx = 1
    Ly = 1
    Nx = 100
    Ny = 200
    sol = Poisson2D(Lx=Lx, Ly = Ly, Nx = Nx, Ny=Ny)
    ue = x*(1-x)*y*(1-y)*sp.exp(sp.cos(4*sp.pi*x)*sp.sin(2*sp.pi*y))
    f = ue.diff(x, 2) + ue.diff(y, 2)
    #u(x, y) = x(1-x)y(1-y)\exp(\cos(4 \pi x)\sin(2 \pi y))
    u = sol(f=f)
    error = sol.l2_error(u, ue)
    # sol.plot(u, ue)
    assert error < tol
    
def test_heterogenous():
    tol = 1e-4
    Lx = 0.6
    Ly = 1.3
    Nx = 100
    Ny = 200
    sol = Poisson2D(Lx=Lx, Ly = Ly, Nx = Nx, Ny=Ny)
    ue = x*(1-x)*y*(1-y)*sp.exp(sp.cos(4*sp.pi*x)*sp.sin(2*sp.pi*y))
    f = ue.diff(x, 2) + ue.diff(y, 2)
    #u(x, y) = x(1-x)y(1-y)\exp(\cos(4 \pi x)\sin(2 \pi y))
    u = sol(f=f, bc = ue)
    error = sol.l2_error(u, ue)
    sol.plot(u, ue)
    assert error < tol
    

if __name__ == '__main__':
    test_homogenous()
    test_heterogenous()