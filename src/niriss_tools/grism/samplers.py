"""Classes for sampling model seeds and spectral templates."""

import multiprocessing
import multiprocessing.pool
from multiprocessing.managers import SharedMemoryManager
from pathlib import Path
from types import TracebackType
from typing import Self

import numpy as np
from numpy.typing import ArrayLike, NDArray
from wfss_tools.grism import float_dtype

__all__ = [
    "TemplateSampler",
    # "GrizliTemplateSampler",
]


class TemplateSampler:
    """
    The base class for spectral template sampling.

    Parameters
    ----------
    n_regions : int
        The number of regions for which templates will be generated.
    seed : int, optional
        The base seed for all sampling.
    **kwargs : dict
        Any additional keyword parameters.
    """

    def __init__(self, n_regions: int, seed: int = 2744, **kwargs):

        self.seed = seed

    def __enter__(self) -> Self:
        return self

    def __del__(self) -> None:
        """
        Ensure that the Pool is terminated correctly.
        """

        if hasattr(self, "_process_pool"):
            self._process_pool.close()
            self._process_pool.terminate()
            del self._process_pool

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.__del__()

    def close(self) -> None:
        """
        A method to explicitly destroy the object.
        """
        self.__del__()

    def initialise_process_pool(self, cpu_count: int) -> None:
        """
        Initialise a pool of processes.

        This is stored as a class attribute, to reduce the overhead of
        creating this each time it is needed.

        Parameters
        ----------
        cpu_count : int
            The number of processes to create.
        """

        self._process_pool = multiprocessing.Pool(
            processes=cpu_count,
        )

    @property
    def process_pool(self) -> multiprocessing.pool.Pool:
        """The pool of workers for all multiprocessing (`~multiprocessing.Pool`, read-only)."""
        return self._process_pool

    @process_pool.setter
    def process_pool(self, value: None = None):  # numpydoc ignore=GL08
        raise AttributeError(
            "`self.process_pool` cannot be set directly. Initialise this "
            "attribute using `self.initialise_process_pool(cpu_count)` instead."
        )

    def gen_model_seeds_from_iter(
        self, iter_seed: int, n_samples: int, n_extra_regions: int = 0, **kwargs
    ) -> tuple[NDArray[np.int_], NDArray[np.int_]]:
        """
        Construct a list of model seeds for a given iteration.

        For a given `iter_seed`, this must return the exact same set of
        model seeds. This method must be implemented by subclasses.

        Parameters
        ----------
        iter_seed : int
            The seed for a given iteration, typically the number of
            iterations already performed.
        n_samples : int
            The number of model seeds to generate.
        n_extra_regions : int
            The number of additional regions that will be sampled, by
            default ``0``.
        **kwargs : dict
            Any additional keyword parameters.

        Returns
        -------
        model_seeds : np.NDArray[np.int_]
            The 1D array of model seeds.
        extra_region_idxs : np.NDArray[np.int_]
            The indices of the additional regions to sample.
        """

        raise NotImplementedError("Subclasses should implement this method.")

    def gen_all_spectra_from_seeds(
        self,
        model_seeds: np.ndarray[int],
        extra_region_idxs: np.ndarray[int] | None = None,
        n_extra_samples: int = 0,
        **kwargs,
    ) -> None:
        """
        Generate all model spectra given a set of model seeds.

        Optionally, generate ``n_extra_samples`` for each extra region
        enumerated in ``extra_region_idxs``.

        The implementation details are left to the subclasses.

        Parameters
        ----------
        model_seeds : np.ndarray[int]
            The set of model seeds. In `bagpipes`, these are interpreted
            as being the row indices in the 2D posterior parameter array.
        extra_region_idxs : np.ndarray[int] | None, optional
            A set of indices of extra regions to sample, by default
            ``None``. This is implemented as an offset which wraps around,
            providing a minimal description of the full set of sampled
            regions.
        n_extra_samples : int, optional
            For each additional region sampled by ``extra_region_idxs``,
            this number of additional models will be generated. This is
            implemented by indexing the array of ``model_seeds``, and is
            by default ``0``.
        **kwargs : dict, optional
            Any additional subclass-specific keyword arguments.
        """

        raise NotImplementedError("Subclasses should implement this method.")


def _test_process_affinity(task_id):

    import psutil

    print(f"{task_id} : {psutil.Process()} : {psutil.Process().cpu_affinity()}")
