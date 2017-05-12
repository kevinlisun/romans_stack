#!/usr/bin/env python

#configure plotting
#%matplotlib inline
#%config InlineBackend.figure_format = 'svg'

import matplotlib;matplotlib.rcParams['figure.figsize'] = (8,5)
import matplotlib;matplotlib.rcParams['text.usetex'] = True
import matplotlib;matplotlib.rcParams['font.size'] = 16
import matplotlib;matplotlib.rcParams['font.family'] = 'serif'
import GPy
import numpy as np
from matplotlib import pyplot as plt
np.random.seed(1)


k = GPy.kern.RBF(1, variance=7., lengthscale=0.2)
X = np.random.rand(200,1)

#draw the latent function value
f = np.random.multivariate_normal(np.zeros(200), k.K(X))

plt.plot(X, f, 'bo')
plt.title('latent function values');plt.xlabel('$x$');plt.ylabel('$f(x)$')

lik = GPy.likelihoods.Bernoulli()
p = lik.gp_link.transf(f) # squash the latent function
plt.plot(X, p, 'ro')
plt.title('latent probabilities');plt.xlabel('$x$');plt.ylabel('$\sigma(f(x))$')


Y = lik.samples(f).reshape(-1,1)
print Y.shape

plt.plot(X, Y, 'kx', mew=2);plt.ylim(-0.1, 1.1)
plt.title('Bernoulli draws');plt.xlabel('$x$');plt.ylabel('$y \in \{0,1\}$')


m = GPy.core.GP(X=X,
                Y=Y, 
                kernel=k, 
                inference_method=GPy.inference.latent_function_inference.expectation_propagation.EP(),
                likelihood=lik)
print m


m.plot()
plt.plot(X, p, 'ro')
plt.ylabel('$y, p(y=1)$');plt.xlabel('$x$')

m.plot_f()
plt.plot(X, f, 'bo')
plt.ylabel('$f(x)$');plt.xlabel('$x$')

print m, '\n'
for i in range(5):
    m.optimize('bfgs', max_iters=100) #first runs EP and then optimizes the kernel parameters
    print 'iteration:', i,
    print m
    print ""


m.plot()
plt.plot(X, p, 'ro', label='Truth')
plt.ylabel('$y, p(y=1)$');plt.xlabel('$x$')
plt.legend()
m.plot_f()
plt.plot(X, f, 'bo', label='Truth')
plt.ylabel('$f(x)$');plt.xlabel('$x$')
plt.legend()

print m

probs = m.predict(X)[0]

print probs
