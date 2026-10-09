"""
Tools for performing NNLS optimisation.
"""

import numba
import numpy as np
from numba import njit
from numpy.typing import ArrayLike
from scipy import sparse
from wfss_tools.grism import float_dtype

__all__ = ["CDNNLS", "fnnls", "fennls"]

try:
    import fnnlsEigen as fe

    HAS_EIGEN = True
except:
    HAS_EIGEN = False


def fennls(
    A: ArrayLike, x: ArrayLike, max_iterations: int = 1000, tolerance: float = 1e-6
) -> ArrayLike:
    """
    A wrapper around `fnnlsEigen.fnnls`.

    Parameters
    ----------
    A : ArrayLike
        The coefficient array.
    x : ArrayLike
        The RHS vector.
    max_iterations : int | None, optional
        How many iterations to run, by default 1000.
    tolerance : float, optional
        The tolerance to use for the stopping condition, by default
        ``1e-6``.

    Returns
    -------
    ArrayLike
        The solution vector.
    """
    if not HAS_EIGEN:
        raise ImportError("FNNLS Eigen is not installed.")
    if float_dtype == np.float32:
        fnnls_fn = fe.fnnlsf
    else:
        fnnls_fn = fe.fnnls

    try:
        return fnnls_fn(
            np.ascontiguousarray(A, dtype=float_dtype),
            x.astype(float_dtype),
            max_iterations=max_iterations,
            tolerance=tolerance,
        )
    except Exception as e:
        print(e)
        return np.zeros(A.shape[1])


def fnnls(
    A: ArrayLike, x: ArrayLike, max_iterations: int = 1000, tolerance: float = 1e-6
) -> ArrayLike:
    """
    A wrapper around a numpy implementation of the FNNLS algorithm.

    Parameters
    ----------
    A : ArrayLike
        The coefficient array.
    x : ArrayLike
        The RHS vector.
    max_iterations : int | None, optional
        How many iterations to run, by default 1000.
    tolerance : float, optional
        The tolerance to use for the stopping condition, by default
        ``1e-6``.

    Returns
    -------
    ArrayLike
        The solution vector.
    """
    return _fnnls(
        np.dot(A.T, A),
        np.dot(A.T, x),
        iter_max=max_iterations,
        epsilon=tolerance,
    )


def _fnnls(AtA, Aty, epsilon=None, iter_max=None):
    """
    Given a matrix A and vector y, find x which minimizes the objective function
    f(x) = ||Ax - y||^2.
    This algorithm is similar to the widespread Lawson-Hanson method, but
    implements the optimizations described in the paper
    "A Fast Non-Negativity-Constrained Least Squares Algorithm" by
    Rasmus Bro and Sumen De Jong.

    Note that the inputs are not A and y, but are
    A^T * A and A^T * y

    This is to avoid incurring the overhead of computing these products
    many times in cases where we need to call this routine many times.

    :param AtA:       A^T * A. See above for definitions. If A is an (m x n)
                      matrix, this should be an (n x n) matrix.
    :type AtA:        numpy.ndarray
    :param Aty:       A^T * y. See above for definitions. If A is an (m x n)
                      matrix and y is an m dimensional vector, this should be an
                      n dimensional vector.
    :type Aty:        numpy.ndarray
    :param epsilon:   Anything less than this value is consider 0 in the code.
                      Use this to prevent issues with floating point precision.
                      Defaults to the machine precision for doubles.
    :type epsilon:    float
    :param iter_max:  Maximum number of inner loop iterations. Defaults to
                      30 * [number of cols in A] (the same value that is used
                      in the publication this algorithm comes from).
    :type iter_max:   int, optional

    """
    if epsilon is None:
        epsilon = np.finfo(np.float64).eps

    n = AtA.shape[0]

    if iter_max is None:
        iter_max = 3 * n

    if Aty.ndim != 1 or Aty.shape[0] != n:
        raise ValueError(
            "Invalid dimension; got Aty vector of size {}, "
            "expected {}".format(Aty.shape, n)
        )

    # Represents passive and active sets.
    # If sets[j] is 0, then index j is in the active set (R in literature).
    # Else, it is in the passive set (P).
    sets = np.zeros(n, dtype=bool)
    # The set of all possible indices. Construct P, R by using `sets` as a mask
    ind = np.arange(n, dtype=int)
    P = ind[sets]
    R = ind[~sets]

    x = np.zeros(n, dtype=np.float64)
    w = Aty
    s = np.zeros(n, dtype=np.float64)

    i = 0
    # While R not empty and max_(n \in R) w_n > epsilon
    while not np.all(sets) and np.max(w[R]) > epsilon and i < iter_max:
        # Find index of maximum element of w which is in active set.
        j = np.argmax(w[R])
        # We have the index in MASKED w.
        # The real index is stored in the j-th position of R.
        m = R[j]

        # Move index from active set to passive set.
        sets[m] = True
        P = ind[sets]
        R = ind[~sets]

        # Get the rows, cols in AtA corresponding to P
        AtA_in_p = AtA[P][:, P]
        # Do the same for Aty
        Aty_in_p = Aty[P]

        # Update s. Solve (AtA)^p * s^p = (Aty)^p
        # s[P] = np.linalg.lstsq(AtA_in_p, Aty_in_p, rcond=None)[0]
        s[R] = 0.0
        s[P] = np.linalg.inv(AtA_in_p).dot(Aty_in_p)

        while np.any(s[P] <= epsilon):
            i += 1

            mask = s[P] <= epsilon
            alpha = np.min(x[P][mask] / (x[P][mask] - s[P][mask]))
            x += alpha * (s - x)

            # Move all indices j in P such that x[j] = 0 to R
            # First get all indices where x == 0 in the MASKED x
            zero_mask = x[P] < epsilon
            # These correspond to indices in P
            zeros = P[zero_mask]
            # Finally, update the passive/active sets.
            sets[zeros] = False
            P = ind[sets]
            R = ind[~sets]

            # Get the rows, cols in AtA corresponding to P
            AtA_in_p = AtA[P][:, P]
            # Do the same for Aty
            Aty_in_p = Aty[P]

            # Update s. Solve (AtA)^p * s^p = (Aty)^p
            # s[P] = np.linalg.lstsq(AtA_in_p, Aty_in_p, rcond=None)[0]
            s[P] = np.linalg.inv(AtA_in_p).dot(Aty_in_p)
            s[R] = 0.0

        x = s.copy()
        w = Aty - AtA.dot(x)

    return x


class CDNNLS:
    """
    Coordinate descent NNLS.

    Parameters
    ----------
    X : ArrayLike
        The coefficient array.
    y : ArrayLike
        The RHS vector.
    """

    def __init__(self, X: ArrayLike, y: ArrayLike):
        self._set_objective(X, y)

    def _set_objective(self, X, y):
        # use Fortran order to compute gradient on contiguous columns
        # self.X, self.y = np.asfortranarray(X), y
        self.X, self.y = X, y

        # Make sure we cache the numba compilation.
        self.run(1)

    def run(
        self,
        method: str = "classic",
        n_iter: int | None = None,
        epsilon: float = 1e-6,
        **kwargs,
    ):
        """
        Run a non-negative least squares solver.

        Parameters
        ----------
        method : str, optional
            The method to use, either the default "classic", or "antilop".
        n_iter : int | None, optional
            How many iterations to run. If ``None``, then this is set to
            ``3*X.shape[1]``.
        epsilon : float, optional
            The tolerance to use for the stopping condition, by default
            ``1e-6``.
        **kwargs : dict, optional
            Other arguments to pass to the individual methods.
        """

        if n_iter is None:
            n_iter = 3 * self.X.shape[1]
        match method:
            case "classic":
                # print("classic")
                self._run_classic(n_iter, epsilon, **kwargs)
            case "antilop":
                # print("antilop")
                self.w = self._run_antilop(self.X, self.y, n_iter, epsilon, **kwargs)

    # @staticmethod
    # @njit('double[:,:](double[:,:], double[:,:],double,int_,double[:,:])')
    # def _solve_NQP(Q, q, epsilon, max_n_iter, x):

    #     # Initialize
    #     x_diff     = np.zeros_like(x)
    #     grad_f     = q.copy()
    #     grad_f_bar = q.copy()

    #     # Loop over iterations
    #     for i in range(max_n_iter):

    #         # Get passive set information
    #         passive_set = np.logical_or(x > 0, grad_f < 0)
    #         # print (f"{passive_set.shape=}")
    #         n_passive = np.sum(passive_set)

    #         # Calculate gradient
    #         grad_f_bar[:] = grad_f
    #         # orig_shape = grad_f_bar
    #         grad_f_bar = grad_f_bar.flatten()
    #         grad_f_bar[~passive_set.flatten()] = 0
    #         grad_f_bar = grad_f_bar.reshape(grad_f.shape)
    #         grad_norm = np.dot(grad_f_bar.flatten(), grad_f_bar.flatten())
    #         # print (grad_norm)

    #         # Abort?
    #         if (n_passive == 0 or grad_norm < epsilon):
    #             break

    #         # Exact line search
    #         # print (Q.dot(grad_f_bar).shape, grad_f_bar.shape)
    #         alpha_den = np.vdot(grad_f_bar.flatten(), Q.dot(grad_f_bar).flatten())
    #         alpha = grad_norm / alpha_den

    #         # Update x
    #         x_diff = -x.copy()
    #         x -= alpha * grad_f_bar
    #         x = np.maximum(x, 0.)
    #         x_diff += x

    #         # Update gradient
    #         grad_f += Q.dot(x_diff)
    #         # print (grad_f)

    #     # # Return
    #     return x

    @staticmethod
    @njit(
        numba.double[:, ::1](
            numba.double[:, ::1], numba.double[::1], numba.int_, numba.double
        )
    )
    def _run_antilop(X, Y, n_iter, epsilon):

        # if not n_iter:
        #     n_iter = 3*Y.shape[0]

        XTX = np.dot(X.T, X)
        XTY = np.dot(X.T, Y)

        # Normalization factors
        H_diag = np.diag(XTX).reshape(-1, 1)
        Q_den = np.sqrt(np.outer(H_diag, H_diag))
        q_den = np.sqrt(H_diag)

        # Normalize
        Q = XTX / Q_den
        q = -XTY.reshape(-1, 1) / q_den
        # q = -XTY / q_den.flatten()

        # print (XTX.shape, XTY.shape, q_den.flatten().shape)

        # Solve NQP
        y = np.zeros((X.shape[1], 1))
        # print (q.shape)
        # y = self.solveNQP_NUMBA(Q, q, epsilon, max_n_iter, y)
        y_diff = np.zeros_like(y)
        grad_f = q.copy()
        grad_f_bar = q.copy()

        # Loop over iterations
        for i in range(n_iter):

            # Get passive set information
            passive_set = np.logical_or(y > 0, grad_f < 0)
            # print (f"{passive_set.shape=}")
            n_passive = np.sum(passive_set)

            # Calculate gradient
            grad_f_bar[:] = grad_f
            # orig_shape = grad_f_bar
            grad_f_bar = grad_f_bar.flatten()
            grad_f_bar[~passive_set.flatten()] = 0
            grad_f_bar = grad_f_bar.reshape(grad_f.shape)
            grad_norm = np.dot(grad_f_bar.flatten(), grad_f_bar.flatten())
            # print (grad_norm)

            # Abort?
            if n_passive == 0 or grad_norm < epsilon:
                print("Finished, iter=", i)
                break

            # Exact line search
            # print (Q.dot(grad_f_bar).shape, grad_f_bar.shape)
            alpha_den = np.vdot(grad_f_bar.flatten(), np.dot(Q, grad_f_bar).flatten())
            alpha = grad_norm / alpha_den

            # Update y
            y_diff = -y.copy()
            y -= alpha * grad_f_bar
            y = np.maximum(y, 0.0)
            y_diff += y

            # Update gradient
            grad_f += np.dot(Q, y_diff)

        # Undo normalization
        x = y / q_den

        # Return
        return x

    def _run_classic(self, n_iter: int, epsilon: float = 1e-6):
        X = np.asfortranarray(self.X)
        L = (X**2).sum(axis=0)
        if sparse.issparse(self.X):
            self.w = self._sparse_cd_classic(
                self.X.data, self.X.indices, self.X.indptr, self.y, L, n_iter
            )
        else:
            self.w = self._cd_classic(X, self.y, L, n_iter, epsilon)

    @staticmethod
    @njit
    def _cd_classic(X, y, L, n_iter, epsilon):
        n_features = X.shape[1]
        R = np.copy(y)
        w = np.zeros(n_features)
        XTX = X.T.dot(X)
        f = -X.T.dot(y)
        for _ in range(n_iter):
            Hxf = np.dot(XTX, w) + f
            # print (_, np.sum(np.abs(Hxf) <= epsilon)/n_features)
            criterion_1 = np.all(Hxf >= -epsilon)
            criterion_2 = np.all(Hxf[w > 0] <= epsilon)
            if criterion_1 and criterion_2:
                print("Finished, iter=", _)
                break
            for j in range(n_features):
                if L[j] == 0.0:
                    continue
                old = w[j]
                w[j] = max(w[j] + X[:, j] @ R / L[j], 0)
                diff = old - w[j]
                if diff != 0:
                    R += diff * X[:, j]
        return w

    @staticmethod
    @njit
    def _sparse_cd_classic(X_data, X_indices, X_indptr, y, L, n_iter):
        n_features = len(X_indptr) - 1
        w = np.zeros(n_features)
        R = np.copy(y)
        for _ in range(n_iter):
            for j in range(n_features):
                if L[j] == 0.0:
                    continue
                old = w[j]
                start, end = X_indptr[j : j + 2]
                scal = 0.0
                for ind in range(start, end):
                    scal += X_data[ind] * R[X_indices[ind]]
                w[j] = max(w[j] + scal / L[j], 0)
                diff = old - w[j]
                if diff != 0:
                    for ind in range(start, end):
                        R[X_indices[ind]] += diff * X_data[ind]
        return w
