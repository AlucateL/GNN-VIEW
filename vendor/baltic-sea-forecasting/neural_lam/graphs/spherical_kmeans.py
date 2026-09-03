"""
Modified from https://github.com/jasonlaska/spherecluster.
"""

# Standard library
import warnings

# Third-party
import numpy as np
import scipy.sparse as sp
from joblib import Parallel, delayed
from sklearn.cluster import KMeans
from sklearn.cluster._kmeans import (
    _check_sample_weight,
    _kmeans_plusplus,
    _tolerance,
)
from sklearn.preprocessing import normalize
from sklearn.utils import check_array, check_random_state
from sklearn.utils.extmath import row_norms, squared_norm
from sklearn.utils.validation import _num_samples


def _spherical_labels_inertia(X, sample_weight, centers):
    """Assignment and inertia using dot products (unit-norm X and centers).
    Squared chord dist = 2(1 - cos_sim), so argmin distance = argmax cos_sim.
    """
    sim = np.asarray(X @ centers.T)  # (n, k); works for sparse X
    labels = np.argmax(sim, axis=1)
    sim_best = sim[np.arange(X.shape[0]), labels]
    inertia = float(np.sum(sample_weight * 2.0 * (1.0 - sim_best)))
    return labels, inertia


def _spherical_kmeans_single_lloyd(
    X,
    n_clusters,
    sample_weight=None,
    max_iter=300,
    init="k-means++",
    verbose=False,
    x_squared_norms=None,
    random_state=None,
    tol=1e-4,
):
    """
    Modified from sklearn.cluster.k_means_.k_means_single_lloyd.
    """
    random_state = check_random_state(random_state)

    sample_weight = _check_sample_weight(sample_weight, X, dtype=np.float32)

    best_labels, best_inertia, best_centers = None, None, None

    # init
    if isinstance(init, str) and init == "k-means++":
        centers, _ = _kmeans_plusplus(
            X, n_clusters, x_squared_norms, sample_weight, random_state
        )
    elif isinstance(init, str) and init == "random":
        seeds = random_state.choice(X.shape[0], n_clusters, replace=False)
        centers = X[seeds]
    else:
        centers = init.copy()
    centers = centers.astype(np.float32)
    if verbose:
        print("Initialization complete")

    n_samples = X.shape[0]

    # iterations
    for i in range(max_iter):
        centers_old = centers.copy()

        # argmin chord² = argmax cos_sim (dot product for unit vectors)
        labels, inertia = _spherical_labels_inertia(X, sample_weight, centers)

        # center update: weighted sum per cluster via sparse (n, k) matrix
        W = sp.csr_matrix(
            (sample_weight, (np.arange(n_samples), labels)),
            shape=(n_samples, n_clusters),
        )
        cluster_weights = np.asarray(W.sum(axis=0)).ravel()
        result = X.T @ W
        weighted_centers = (
            result.A if sp.issparse(result) else np.asarray(result)
        )
        # (d, k) -> (k, d); avoid div by zero for empty clusters
        denom = np.where(cluster_weights > 0, cluster_weights, 1.0)
        centers = (weighted_centers / denom).T
        empty = cluster_weights == 0
        centers[empty] = centers_old[empty]

        # l2-normalize centers (this is the main contribution here)
        centers = normalize(centers)

        if verbose:
            print("Iteration %2d, inertia %.3f" % (i, inertia))

        if best_inertia is None or inertia < best_inertia:
            best_labels = labels.copy()
            best_centers = centers.copy()
            best_inertia = inertia

        center_shift_total = squared_norm(centers_old - centers)
        if center_shift_total <= tol:
            if verbose:
                print(
                    "Converged at iteration %d: "
                    "center shift %e within tolerance %e"
                    % (i, center_shift_total, tol)
                )
            break

    if center_shift_total > 0:
        # rerun E-step in case of non-convergence so that predicted labels
        # match cluster centers
        best_labels, best_inertia = _spherical_labels_inertia(
            X, sample_weight, best_centers
        )

    return best_labels, best_inertia, best_centers, i + 1


def spherical_k_means(
    X,
    n_clusters,
    sample_weight=None,
    init="k-means++",
    n_init=10,
    max_iter=300,
    verbose=False,
    tol=1e-4,
    random_state=None,
    copy_x=True,
    n_jobs=1,
    algorithm="auto",
    return_n_iter=False,
):
    """Modified from sklearn.cluster.k_means_.k_means."""
    if n_init <= 0:
        raise ValueError(
            "Invalid number of initializations."
            " n_init=%d must be bigger than zero." % n_init
        )
    random_state = check_random_state(random_state)

    if max_iter <= 0:
        raise ValueError(
            "Number of iterations should be a positive number,"
            " got %d instead" % max_iter
        )

    best_inertia = np.inf
    # avoid forcing order when copy_x=False
    order = "C" if copy_x else None
    X = check_array(
        X, accept_sparse="csr", dtype=np.float32, order=order, copy=copy_x
    )
    # verify that the number of samples given is larger than k
    if _num_samples(X) < n_clusters:
        raise ValueError(
            "n_samples=%d should be >= n_clusters=%d"
            % (_num_samples(X), n_clusters)
        )
    tol = _tolerance(X, tol)

    if hasattr(init, "__array__"):
        init = check_array(init, dtype=X.dtype.type, order="C", copy=True)
        if init.shape != (n_clusters, X.shape[1]):
            raise ValueError(
                "The shape of the initial centers (%s) does not match the "
                "number of clusters %i and number of features %i."
                % (str(init.shape), n_clusters, X.shape[1])
            )

        if n_init != 1:
            warnings.warn(
                "Explicit initial center position passed: "
                "performing only one init in k-means instead of n_init=%d"
                % n_init,
                RuntimeWarning,
                stacklevel=2,
            )
            n_init = 1

    # precompute squared norms of data points
    x_squared_norms = row_norms(X, squared=True)

    if n_jobs == 1:
        # For a single thread, less memory is needed if we just store one set
        # of the best results (as opposed to one set per run per thread).
        for it in range(n_init):
            # run a k-means once
            labels, inertia, centers, n_iter_ = _spherical_kmeans_single_lloyd(
                X,
                n_clusters,
                sample_weight,
                max_iter=max_iter,
                init=init,
                verbose=verbose,
                tol=tol,
                x_squared_norms=x_squared_norms,
                random_state=random_state,
            )

            # determine if these results are the best so far
            if best_inertia is None or inertia < best_inertia:
                best_labels = labels.copy()
                best_centers = centers.copy()
                best_inertia = inertia
                best_n_iter = n_iter_
    else:
        # parallelisation of k-means runs
        seeds = random_state.randint(np.iinfo(np.int32).max, size=n_init)
        results = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(_spherical_kmeans_single_lloyd)(
                X,
                n_clusters,
                sample_weight,
                max_iter=max_iter,
                init=init,
                verbose=verbose,
                tol=tol,
                x_squared_norms=x_squared_norms,
                # Change seed to ensure variety
                random_state=seed,
            )
            for seed in seeds
        )

        # Get results with the lowest inertia
        labels, inertia, centers, n_iters = zip(*results)
        best = np.argmin(inertia)
        best_labels = labels[best]
        best_inertia = inertia[best]
        best_centers = centers[best]
        best_n_iter = n_iters[best]

    if return_n_iter:
        return best_centers, best_labels, best_inertia, best_n_iter
    else:
        return best_centers, best_labels, best_inertia


class SphericalKMeans(KMeans):
    """Spherical K-Means clustering

    Modification of sklearn.cluster.KMeans where cluster centers are normalized
    (projected onto the sphere) in each iteration.

    Parameters
    ----------

    n_clusters : int, optional, default: 8
        The number of clusters to form as well as the number of
        centroids to generate.

    max_iter : int, default: 300
        Maximum number of iterations of the k-means algorithm for a
        single run.

    n_init : int, default: 10
        Number of time the k-means algorithm will be run with different
        centroid seeds. The final results will be the best output of
        n_init consecutive runs in terms of inertia.

    init : {'k-means++', 'random' or an ndarray}
        Method for initialization, defaults to 'k-means++':
        'k-means++' : selects initial cluster centers for k-mean
        clustering in a smart way to speed up convergence. See section
        Notes in k_init for more details.
        'random': choose k observations (rows) at random from data for
        the initial centroids.
        If an ndarray is passed, it should be of shape (n_clusters, n_features)
        and gives the initial centers.

    tol : float, default: 1e-4
        Relative tolerance with regards to inertia to declare convergence

    n_jobs : int
        The number of jobs to use for the computation. This works by computing
        each of the n_init runs in parallel.
        If -1 all CPUs are used. If 1 is given, no parallel computing code is
        used at all, which is useful for debugging. For n_jobs below -1,
        (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but one
        are used.

    random_state : integer or numpy.RandomState, optional
        The generator used to initialize the centers. If an integer is
        given, it fixes the seed. Defaults to the global numpy random
        number generator.

    verbose : int, default 0
        Verbosity mode.

    copy_x : boolean, default True
        When pre-computing distances it is more numerically accurate to center
        the data first.  If copy_x is True, then the original data is not
        modified.  If False, the original data is modified, and put back before
        the function returns, but small numerical differences may be introduced
        by subtracting and then adding the data mean.

    normalize : boolean, default True
        Normalize the input to have unnit norm.

    Attributes
    ----------

    cluster_centers_ : array, [n_clusters, n_features]
        Coordinates of cluster centers

    labels_ :
        Labels of each point

    inertia_ : float
        Sum of distances of samples to their closest cluster center.
    """

    def __init__(
        self,
        n_clusters=8,
        init="k-means++",
        n_init=10,
        max_iter=300,
        tol=1e-4,
        n_jobs=1,
        verbose=0,
        random_state=None,
        copy_x=True,
        normalize=True,
    ):
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.verbose = verbose
        self.random_state = random_state
        self.copy_x = copy_x
        self.n_jobs = n_jobs
        self.normalize = normalize

    def fit(self, X, y=None, sample_weight=None):
        """Compute k-means clustering.

        Parameters
        ----------

        X : array-like or sparse matrix, shape=(n_samples, n_features)

        y : Ignored
            not used, present here for API consistency by convention.

        sample_weight : array-like, shape (n_samples,), optional
            The weights for each observation in X. If None, all observations
            are assigned equal weight (default: None)
        """
        if self.normalize:
            X = normalize(X)

        random_state = check_random_state(self.random_state)

        # TODO: add check that all data is unit-normalized

        self.cluster_centers_, self.labels_, self.inertia_, self.n_iter_ = (
            spherical_k_means(
                X,
                n_clusters=self.n_clusters,
                sample_weight=sample_weight,
                init=self.init,
                n_init=self.n_init,
                max_iter=self.max_iter,
                verbose=self.verbose,
                tol=self.tol,
                random_state=random_state,
                copy_x=self.copy_x,
                n_jobs=self.n_jobs,
                return_n_iter=True,
            )
        )

        return self
