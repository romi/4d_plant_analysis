import copy
from sklearn.neighbors import KDTree

from utils import pairwise_distance
import numpy as np
import scipy.sparse as sps


def fast_sinkhorn_filter(KTar, KSrc, options={}):
    if 'p' not in options:
        options['p'] = 1

    if 'knn' not in options:
        options['knn'] = 20

    if 'distmax' not in options:
        options['distmax'] = 5

    if 'maxiter' not in options:
        options['maxiter'] = 500

    if 'kernel_type' not in options:
        options['kernel_type'] = 'sparse'

    if options['kernel_type'] == 'full':
        pair_distance = pairwise_distance(KSrc, KTar) ** options['p']
        lam = 200 / np.max(pair_distance)
        K = np.exp( -lam * pair_distance)
    else:
        knn = min(options['knn'], KTar.shape[0], KSrc.shape[0])
        distmax = options['distmax']

        tree = KDTree(KSrc)
        distance, J = tree.query(KTar, k=knn, return_distance=True)
        lam = distmax / np.max(distance)
        Kdist = np.exp(-lam * distance ** options['p'])
        _, I = np.meshgrid(np.arange(knn), np.arange(KTar.shape[0]))
        K1 = sps.csr_matrix((Kdist.flatten(), (I.flatten(), J.flatten())), shape=(KTar.shape[0], KSrc.shape[0]))

        tree = KDTree(KTar)
        distance, J = tree.query(KSrc, k=knn, return_distance=True)
        lam = distmax / np.max(distance)
        Kdist = np.exp(-lam * distance ** options['p'])
        _, I = np.meshgrid(np.arange(knn), np.arange(KSrc.shape[0]))
        K2 = sps.csr_matrix((Kdist.flatten(), (I.flatten(), J.flatten())), shape=(KSrc.shape[0], KTar.shape[0]))

        K = K1.transpose() + K2

    a = np.ones((KSrc.shape[0], 1))
    a = a/np.sum(a)

    b = np.ones((KTar.shape[0], 1))
    b = b/np.sum(b)

    if options['maxiter'] > 0:
        L, u, v = sinkhornTransport(a, b, K, lam, maxiter=options['maxiter'], verbose=True)
        S = sps.diags(u.flatten()) * K * sps.diags(v.flatten())
    else:
        S = K

    T12 = np.array(np.argmax(S, axis=1)).reshape(-1)
    T21 = np.array(np.argmax(S, axis=0)).reshape(-1)
    return T12, T21


def sinkhornTransport(a, b, K, lam, maxiter, tolerance=0.05, verbose=True):
    stoppingCriterion = 'marginalDifference'
    p_norm = np.infty

    if a.shape[1] == 1:
        one_vs_N = True
    elif a.shape[1] == b.shape[1]:
        one_vs_N = False
    else:
        return

    I = a > 0
    someZeroValues = False
    if not (I.all()): # need to update some vectors and matrices if a does not have full support
        someZeroValues = True
        K = K[I,:]
        # U = U[I,:]
        a = a[I]
    ainvK = sps.csr_matrix(K/a)

    iter = 0
    u = np.ones((a.shape[0], b.shape[1])) / (a.shape[0])
    if stoppingCriterion == 'distanceRelativeDecrease':
        Dold = np.ones((1, b.shape[1]))

    while iter < maxiter:
        if one_vs_N:
            u = 1 / ( ainvK.dot(b / K.transpose().dot(u) ))
            # u = 1. / (ainvK * (b. / (u'*K)'));
        iter += 1
        if iter % 20 == 0 or iter == maxiter:
            v = b / K.transpose().dot(u)
            u = 1 / (ainvK.dot(v))
            criterion = np.linalg.norm(np.sum(np.abs(v* K.transpose().dot(u) - b)))
            if criterion < tolerance or np.isnan(criterion):
                break;
            iter += 1
            if verbose:
                print(f"Iteration: {iter}, criterion: {criterion}, " )
    D = 1
    alpha = np.log(u)
    beta = np.log(v)
    beta[beta < 10**-5] = 0
    L = 1/lam * (a.transpose().dot(alpha) + np.sum(b*beta))

    uu = copy.deepcopy(u)
    u = np.zeros((I.shape[0], b.shape[1]))
    u = uu[I, None]
    return L, u, v



