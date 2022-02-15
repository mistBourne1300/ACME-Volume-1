# newtons_method.py
"""Volume 1: Newton's Method.
<Name>
<Class>
<Date>
"""

from hashlib import new
from matplotlib.cbook import maxdict
import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg as la
from scipy.optimize import newton
from autograd import grad, jacobian
from autograd import numpy as anp


# Problems 1, 3, and 5
def newton(f, x0, Df, tol=1e-5, maxiter=15, alpha=1.):
    """Use Newton's method to approximate a zero of the function f.

    Parameters:
        f (function): a function from R^n to R^n (assume n=1 until Problem 5).
        x0 (float or ndarray): The initial guess for the zero of f.
        Df (function): The derivative of f, a function from R^n to R^(nxn).
        tol (float): Convergence tolerance. The function should returns when
            the difference between successive approximations is less than tol.
        maxiter (int): The maximum number of iterations to compute.
        alpha (float): Backtracking scalar (Problem 3).

    Returns:
        (float or ndarray): The approximation for a zero of f.
        (bool): Whether or not Newton's method converged.
        (int): The number of iterations computed.
    """
    if np.isscalar(x0):
        x1 = x0 - alpha*f(x0)/Df(x0)
        counter = 0
        while np.abs(x1-x0) > tol:
            x0 = x1
            x1 = x0 - alpha*f(x0)/Df(x0)
            counter += 1
            if counter > maxiter:
                return x1, False, counter
        return x1, True, counter
    else:
        x1 = x0 - alpha*la.solve(Df(x0), f(x0))
        counter = 0
        while counter < maxiter:
            x0 = x1
            x1 = x0 - alpha*la.solve(Df(x0), f(x0))
            counter += 1
            if la.norm(x0-x1) < tol:
                return x1, True, counter
        
        return x1, False, counter



# Problem 2
def prob2(N1, N2, P1, P2):
    """Use Newton's method to solve for the constant r that satisfies

                P1[(1+r)**N1 - 1] = P2[1 - (1+r)**(-N2)].

    Use r_0 = 0.1 for the initial guess.

    Parameters:
        P1 (float): Amount of money deposited into account at the beginning of
            years 1, 2, ..., N1.
        P2 (float): Amount of money withdrawn at the beginning of years N1+1,
            N1+2, ..., N1+N2.
        N1 (int): Number of years money is deposited.
        N2 (int): Number of years money is withdrawn.

    Returns:
        (float): the value of r that satisfies the equation.
    """
    f = lambda r: P1*((1 + r)**N1 - 1) - P2*(1-(1+r)**(-N2))
    df = lambda r: P1*(N1*(1+r)**(N1-1)) - P2*(N2*(1+r)**(-N2-1))
    return newton(f, 0.1, df)[0]



# Problem 4
def optimal_alpha(f, x0, Df, tol=1e-5, maxiter=15):
    """Run Newton's method for various values of alpha in (0,1].
    Plot the alpha value against the number of iterations until convergence.

    Parameters:
        f (function): a function from R^n to R^n (assume n=1 until Problem 5).
        x0 (float or ndarray): The initial guess for the zero of f.
        Df (function): The derivative of f, a function from R^n to R^(nxn).
        tol (float): Convergence tolerance. The function should returns when
            the difference between successive approximations is less than tol.
        maxiter (int): The maximum number of iterations to compute.

    Returns:
        (float): a value for alpha that results in the lowest number of
            iterations.
    """
    alphas = np.linspace(0.001,1, num = 100, endpoint=False)
    newtons_cradle = np.vectorize(newton, excluded = ['f', 'x0', 'Df', 'tol', 'maxiter'])
    plt.plot(alphas, newtons_cradle(f, x0, Df, tol, maxiter, alphas)[2])
    plt.show()
    return alphas[np.argmin(newtons_cradle(f,x0,Df, tol, maxiter, alphas)[2])]



# Problem 6
def prob6():
    """Consider the following Bioremediation system.

                              5xy − x(1 + y) = 0
                        −xy + (1 − y)(1 + y) = 0

    Find an initial point such that Newton’s method converges to either
    (0,1) or (0,−1) with alpha = 1, and to (3.75, .25) with alpha = 0.55.
    Return the intial point as a 1-D NumPy array with 2 entries.
    """
    f = lambda x: anp.array([5*x[1]*x[0] - x[0]*(1+x[1]), -x[0]*x[1] + (1-x[1])*(1 + x[1])])
    df = jacobian(f)
    exxes = anp.linspace(-1/4.,0)
    whys = anp.linspace(0,1/4.)
    zero55 = anp.array([3.75, .25])
    zero11 = anp.array([0,1])
    zero12 = anp.array([0,-1])
    for x in exxes:
        for y in whys:
            x0 = anp.array([x,y])
            # print(f'testing {x0}')
            try:
                try55 = newton(f, x0, df, alpha = .55)
                try1 = newton(f, x0, df, alpha = 1)
                # print(try55)
                if not try55[1] or not try1[1]:
                    # print("\tdid not converge, continuing")
                    # print(try1)
                    # print(try1)
                    continue
                if (anp.allclose(try1[0], zero11) or anp.allclose(try1[0], zero12)) and anp.allclose(try55[0], zero55):
                    return x0
            except:
                print("\tmatrix was singular")
                continue



# Problem 7
def plot_basins(f, Df, zeros, domain, res=1000, iters=15):
    """Plot the basins of attraction of f on the complex plane.

    Parameters:
        f (function): A function from C to C.
        Df (function): The derivative of f, a function from C to C.
        zeros (ndarray): A 1-D array of the zeros of f.
        domain ([r_min, r_max, i_min, i_max]): A list of scalars that define
            the window limits and grid domain for the plot.
        res (int): A scalar that determines the resolution of the plot.
            The visualized grid has shape (res, res).
        iters (int): The exact number of times to iterate Newton's method.
    """
    x_real = np.linspace(domain[0], domain[1], res)
    x_imag = np.linspace(domain[2], domain[3], res)
    XREAL, XIMAG = np.meshgrid(x_real, x_imag)
    X0 = XREAL + 1j*XIMAG
    for i in range(iters):
        X0 = X0 - f(X0)/Df(X0)
    
    y = np.zeros((res,res))

    for i in range(res):
        for j in range(res):
            vals = np.abs(zeros - X0[i,j])
            y[i,j] = np.argmin(vals)
    
    plt.pcolormesh(y, cmap = 'Dark2')
    plt.show()
            



def test5():
    f = lambda x: anp.array((x[0]*x[1],x[0]-x[1]))
    Df = jacobian(f)
    x0 = np.array((0.1,0.1))
    print(newton(f,x0,Df,maxiter=30))


if __name__ == "__main__":
    # f = lambda x: x**5 - 3
    # df = lambda x: 5*x**4

    # soln = newton(f, 1, df, maxiter=100, tol = 1e-10)
    # print(soln)
    # print(f'f(soln): {f(soln[0])}')
    # # domain = np.linspace(-1,2)
    # # plt.plot(domain, f(domain), 'k')
    # # plt.show()


    # print(prob2(30, 20, 2000, 8000))



    f = lambda x: np.sign(x) * np.power(np.abs(x), 1./3)
    # df = grad(f)
    # print(f'a = 1: {newton(f, 0.1, df)}')
    # print(f'a = .4: {newton(f, 0.1, df, alpha = .4)}')

    # print(optimal_alpha(f, .1, df))

    # f = lambda x: np.array([x[0]**2 + x[1]**2, x[0]+x[1]])

    # test5()

    # x0 = prob6()
    # print(f'x0: {x0}')
    # f = lambda x: anp.array([5*x[1]*x[0] - x[0]*(1+x[1]), -x[0]*x[1] + (1-x[1])*(1 + x[1])])
    # df = jacobian(f)
    # print(f'.55: {newton(f,x0,df, alpha = .55)}\n1:{newton(f,x0,df)}')

    # This is how I called problem 7 to test.
    f = lambda x: x**3 -1
    Df = lambda x: 3*x**2

    plot_basins(f,Df,np.array((1,
            -.5+.866025j,-.5-.866025j
    )),[-1,1,-1,1])
    # This will give you the zeros and the derivative without Autograd

    pass