import numpy as np

def fit_line(points):                                                           
    xy = points[:,0:2]

    m = np.mean(xy, axis=0)
    cov = np.cov((xy - m).T).T

    evals, evecs = np.linalg.eig(cov)

    n = evecs[:, 1]
    if evals[0] < evals[1]:
        n = evecs[:,0] 

    ratio = np.max(evals) / (np.min(evals) + 1e-8)

    n = n / np.linalg.norm(n)

    d = n.dot(m)
    a, b = n.flatten()

    return a, b, d, ratio

