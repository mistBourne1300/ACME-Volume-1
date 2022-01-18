# sympy_intro.py
"""Python Essentials: Introduction to SymPy.
<Name>
<Class>
<Date>
"""
from mpmath.libmp.libmpf import to_pickable
from mpmath.matrices.eigen import eig
from numpy.linalg.linalg import _eigvalsh_dispatcher
import sympy as sy
import numpy as np
import matplotlib.pyplot as plt

# Problem 1
def prob1():
    """Return an expression for

        (2/5)e^(x^2 - y)cosh(x+y) + (3/7)log(xy + 1).

    Make sure that the fractions remain symbolic.
    """
    # create symbols
    x,y = sy.symbols('x, y')
    # return expression
    return sy.Rational(2,5) * sy.exp(x**2 - y) * sy.cosh(x+y) + sy.Rational(3,7) * sy.log(x*y + 1)



# Problem 2
def prob2():
    """Compute and simplify the following expression.

        product_(i=1 to 5)[ sum_(j=i to 5)[j(sin(x) + cos(x))] ]
    """
    # create symbols
    x,i,j = sy.symbols('x,i,j')

    # create, simplify and return expression
    expr = sy.product(sy.summation(j*(sy.sin(x) + sy.cos(x)), (j,i,5)), (i,1,5))
    return sy.simplify(expr)


# Problem 3
def prob3(N):
    """Define an expression for the Maclaurin series of e^x up to order N.
    Substitute in -y^2 for x to get a truncated Maclaurin series of e^(-y^2).
    Lambdify the resulting expression and plot the series on the domain
    y in [-3,3]. Plot e^(-y^2) over the same domain for comparison.
    """
    # lambda function plotting actual function
    g = lambda x: np.exp(-domain**2)

    # create symbols
    x,y,n = sy.symbols('x y n')

    # create summation expression and substitute int -y^2 for x
    expr = sy.summation((x**n)/sy.factorial(n), (n,0,N))
    truncated = expr.subs(x,-y**2)
    # create callable function
    f = sy.lambdify(y, truncated, 'numpy')

    # plot and show
    domain = np.linspace(-2,2)
    plt.plot(domain, f(domain))
    plt.plot(domain, g(domain))
    plt.title(f"Taylor series approx. for e^(-y^2) (n={N})")
    plt.show()



# Problem 4
def prob4():
    """The following equation represents a rose curve in cartesian coordinates.

    0 = 1 - [(x^2 + y^2)^(7/2) + 18x^5 y - 60x^3 y^3 + 18x y^5] / (x^2 + y^2)^3

    Construct an expression for the nonzero side of the equation and convert
    it to polar coordinates. Simplify the result, then solve it for r.
    Lambdify a solution and use it to plot x against y for theta in [0, 2pi].
    """
    # create symbols and expression
    x,y,r,t = sy.symbols('x y r theta')
    expr = 1 - ((x**2 + y**2)**sy.Rational(7,2) + 18*(x**5)*y - 60*(x**3)*(y**3) + 18*x*(y**5))/((x**2 + y**2)**3)

    # substitute in polar coordinates
    polar = expr.subs({x:(r*sy.cos(t)), y:r*sy.sin(t)})
    polar_simp = polar.trigsimp()

    # solve for r and create a callable function 
    r = sy.lambdify(t, sy.solve(polar_simp, r)[0], 'numpy')
    theta = np.linspace(0,2*np.pi, 500)

    # plot
    plt.plot(r(theta) * np.cos(theta), r(theta)*np.sin(theta))
    plt.show()


# Problem 5
def prob5():
    """Calculate the eigenvalues and eigenvectors of the following matrix.

            [x-y,   x,   0]
        A = [  x, x-y,   x]
            [  0,   x, x-y]

    Returns:
        (dict): a dictionary mapping eigenvalues (as expressions) to the
            corresponding eigenvectors (as SymPy matrices).
    """
    # create symbols and matrix
    x,y,l = sy.symbols('x y lambda')
    M = sy.Matrix([ [x-y,   x,   0],
                    [  x, x-y,   x],
                    [  0,   x, x-y]])
    
    # lambda*I matrix
    lamI = sy.Matrix([  [l,0,0],
                        [0,l,0],
                        [0,0,l]])
    
    # have sympy solve for the matrix
    eigvals = sy.solve(sy.det(M-lamI), l)

    eigvects = []
    for e in eigvals:
        lame = lamI.subs(l,e)
        eigvects.append((M-lame).nullspace())

    return {eigvals[n]:eigvects[n] for n in range(len(eigvals))}


# Problem 6
def prob6():
    """Consider the following polynomial.

        p(x) = 2*x^6 - 51*x^4 + 48*x^3 + 312*x^2 - 576*x - 100

    Plot the polynomial and its critical points. Determine which points are
    maxima and which are minima.

    Returns:
        (set): the local minima.
        (set): the local maxima.
    """
    # create symbols
    x = sy.symbols('x')
    domain = np.linspace(-5,5)
    expr = 2*x**6 - 51*x**4 + 48*x**3 + 312*x**2 - 576*x - 100
    # calculate derivatives
    deriv1 = sy.diff(expr,x)
    deriv2 = sy.diff(deriv1,x)

    # get the callable functions for the original polynomial and the double derivative
    double_prime = sy.lambdify(x,deriv2, 'numpy')
    p = sy.lambdify(x, expr, 'numpy')

    # calculate critical points
    crits = sy.solve(deriv1,x)

    # create sets and add the minima and maxima
    top_secret = set()
    bottom_secret = set()
    for c in crits:
        if double_prime(c) > 0:
            bottom_secret.add(c)
        elif double_prime(c) < 0:
            top_secret.add(c)
        else:
            top_secret.add(c)

    # plot
    plt.plot(domain, p(domain), 'b')    

    for classified in top_secret:
        plt.scatter(classified, p(classified), c='r', s=10)

    for classified in bottom_secret:
        plt.scatter(classified, p(classified), c='k', s=10)
        
    
    plt.show()
    return top_secret, bottom_secret



# Problem 7
def prob7():
    """Calculate the integral of f(x,y,z) = (x^2 + y^2 + z^2)^2 over the
    sphere of radius r. Lambdify the resulting expression and plot the integral
    value for r in [0,3]. Return the value of the integral when r = 2.

    Returns:
        (float): the integral of f over the sphere of radius 2.
    """
    # create symbols and expressions with substitutions
    x,y,z,r,t,p,R = sy.symbols('x,y,z,r,t,p,R')
    expr = (x**2 + y**2 + z**2)**2
    circ_expr = expr.subs({x:r*sy.sin(p)*sy.cos(t), y:r*sy.sin(p)*sy.sin(t), z:r*sy.cos(p)})

    # calculate jacobian of h function
    h = sy.Matrix([r*sy.sin(p)*sy.cos(t), r*sy.sin(p)*sy.sin(t), r*sy.cos(p)])
    jacob = sy.det(h.jacobian([r,p,t]))
    
    # create callable function and plot
    int_R = sy.lambdify(R, sy.integrate(circ_expr*jacob, (r,0,R), (t,0,2*sy.pi), (p,0,sy.pi)), 'numpy')
    domain = np.linspace(0,3)
    plt.plot(domain, int_R(domain))
    plt.show()
    return int_R(2)



if __name__ == "__main__":
    # sy.init_printing()
    # print(prob1())
    # print(prob2())
    # domain = np.linspace(-2,2)
    # plt.plot(domain, np.exp(-domain**2), 'k')
    # prob3(10)
    # prob4()
    # print(prob5())
    # print(prob6())
    # print(prob7())
    pass