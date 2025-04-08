import numpy as np
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.GraphUtils import GraphUtils

rng = np.random.default_rng()
nobs = 30000
eps = rng.normal(size=(nobs, 3))
B = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]])
I = np.eye(3)
Z = (np.linalg.inv(I - B.T) @ eps.T).T
np.linalg.inv(np.cov(Z.T))
Z_hat = np.zeros_like(Z)
Z_hat[:, 0] = Z[:, 0]
Z_hat[:, 1] = Z[:, 1] + Z[:, 2]
Z_hat[:, 2] = Z[:, 1] - Z[:, 2]

cg = pc(Z, verbose=True)
cg.G.graph
pyd = GraphUtils.to_pydot(cg.G)
pyd.write_png("simple_test_latent.png")

cg = pc(Z_hat)
pyd = GraphUtils.to_pydot(cg.G)
pyd.write_png("simple_test_rep.png")
