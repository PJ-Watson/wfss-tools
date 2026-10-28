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


class GrizliTemplateSampler(TemplateSampler):

    def __init__(
        self,
        posterior_dir: Path,
        seed: int = 2744,
        cpu_count: int = multiprocessing.cpu_count(),
        cache_all_spectra: bool = True,
        veldisp: float = 50,
        spec_wavs: np.ndarray[float] = np.arange(10000.0, 23000.0, 22.5),
        apply_R_curve: bool = False,
        redshift: float = 0,
        apply_IGM: bool = True,
    ):

        super().__init__(seed)

        self.cpu_count = cpu_count

        self.veldisp = veldisp
        self.spec_wavs = spec_wavs

        # Temporary hack, need to actually pass the region IDs through to start with.
        # Currently order may not be exactly the same between the MultiRegionFitter
        # and this class.
        self.posterior_ids = [f.stem for f in posterior_dir.glob("*.h5")]
        try:
            self.posterior_ids.sort(key=int)
        except:
            self.posterior_ids.sort()

        import grizli.utils as grizli_utils

        all_SpectrumTemplate = grizli_utils.load_templates(
            fwhm=self.veldisp,
            line_complexes=False,
            fsps_templates=False,
            full_line_list=[
                "BrA",
                "BrB",
                "BrG",
                "PfG",
                "PfD",
                "PaA",
                "PaB",
                "PaG",
                "PaD",
                "HeI-1083",
                "SIII",
                "OII-7325",
                "ArIII-7138",
                "SII",
                "NII",
                "Ha",
                "OI-6302",
                "HeI-5877",
                "OIII",
                "Hb",
                "OIII-4363",
                "Hg",
                "Hd",
                "H7",
                "H8",
                "H9",
                "H10",
                "NeIII-3867",
                "OII",
                "NeVI-3426",
                "NeV-3346",
                "MgII",
                "CIV-1549",
                "CIII-1906",
                "CIII-1908",
                "OIII-1663",
                "HeII-1640",
                "NIII-1750",
                "NIV-1487",
                "NV-1240",
                "Lya",
            ],
        )
        broad_SpectrumTemplate = grizli_utils.load_templates(
            fwhm=1000, line_complexes=False, fsps_templates=False
        )
        agn_SpectrumTemplate = grizli_utils.load_templates(
            fwhm=2500, line_complexes=False, fsps_templates=False
        )
        for k, v in broad_SpectrumTemplate.items():
            if "line" in k:
                all_SpectrumTemplate[f"line broad {k.removeprefix("line ")}"] = v
        for k, v in agn_SpectrumTemplate.items():
            if ("line" in k) and np.isin(k, ["line Ha", "line Hb", "line Hg"]):
                all_SpectrumTemplate[f"line agn {k.removeprefix("line ")}"] = v

        # z_SpectrumTemplate = []
        # for s_i, s in all_SpectrumTemplate:
        #     z_SpectrumTemplate.append(s.zscale(z=redshift))

        z_SpectrumTemplate = {
            k: v.zscale(z=redshift) for k, v in all_SpectrumTemplate.items()
        }

        self.temp_names = np.array(list(z_SpectrumTemplate.keys()))

        wave, self.temp_flux_arr, self.temp_is_line = grizli_utils.array_templates(
            z_SpectrumTemplate, wave=self.spec_wavs, z=redshift, apply_igm=apply_IGM
        )

        ok_temp = np.nansum(self.temp_flux_arr, axis=-1) > 0

        self.temp_names = self.temp_names[ok_temp]
        self.temp_flux_arr = self.temp_flux_arr[ok_temp]
        self.temp_is_line = self.temp_is_line[ok_temp]

        self.Ntemp = self.temp_flux_arr.shape[0]

    def gen_all_spectra_from_seeds(
        self,
        model_seeds: np.ndarray[int],
        extra_region_idxs: np.ndarray[int] | None = None,
        n_extra_samples: int = 0,
        shared_memory_manger: SharedMemoryManager | None = None,
        shared_memory_name: str | None = None,
        shared_memory_shape: tuple[int] | None = None,
        **kwargs,
    ) -> tuple[str, tuple[int]] | None:
        """
        Generate all model spectra given a set of model seeds.

        Optionally, generate ``n_extra_samples`` for each extra region
        enumerated in ``extra_region_idxs``.

        Parameters
        ----------
        model_seeds : np.ndarray[int]
            The set of model seeds. In `bagpipes`, these are interpreted
            as being the row indices in the 2D posterior parameter array.
        extra_region_idxs : np.ndarray[int] | None, optional
            A set of indices of extra regions to sample, by default
            ``None``.
        n_extra_samples : int, optional
            For each additional region sampled by ``extra_region_idxs``,
            this number of additional models will be generated. This is
            implemented by indexing the array of ``model_seeds``, and is
            by default ``0``.
        shared_memory_manger : SharedMemoryManager | None, optional
            An instance of a SharedMemoryManager, by default ``None``. If
            not ``None``, ``self.model_spectra`` will be copied into
            shared memory, to allow direct access from other processes.
        shared_memory_name : str | None, optional
            The name of the shared memory block to use. If ``None`` (the
            default), a new object will be created.
        shared_memory_shape : tuple[int] | None, optional
            The shape of the shared memory block, by default ``None``.
        **kwargs : dict, optional
            Any additional keyword arguments.

        Returns
        -------
        shared_memory_name : str, optional
            The name of the shared memory block. Only returned if
            ``shared_memory_manager`` is not ``None``.
        shared_memory_shape : tuple[int], optional
            The shape of the shared memory block. Only returned if
            ``shared_memory_manager`` is not ``None``.
        """

        # all_models = np.zeros(
        #     (
        #         len(self.posterior_ids),
        #         len(model_seeds) + len(extra_region_idxs) * n_extra_samples,
        #     )
        # )
        all_models = np.tile(model_seeds, (len(self.posterior_ids), 1))

        print(all_models.shape)
        all_models_shape = all_models.shape

        unique_models, unique_models_inv = np.unique(
            all_models.ravel(), return_inverse=True
        )

        self.model_spectra = self.temp_flux_arr[unique_models_inv]

        if shared_memory_manger is not None:
            if (shared_memory_name is not None) and (shared_memory_shape is not None):
                shm_model_spectra = multiprocessing.shared_memory.SharedMemory(
                    name=shared_memory_name, create=False
                )
                model_spectra_arr = np.ndarray(
                    shared_memory_shape, dtype=float_dtype, buffer=shm_model_spectra.buf
                )
            else:
                shared_memory_shape = (*all_models_shape, self.model_spectra.shape[-1])
                shm_model_spectra = shared_memory_manger.SharedMemory(
                    size=np.dtype(float_dtype).itemsize * np.prod(shared_memory_shape),
                )
                model_spectra_arr = np.ndarray(
                    shared_memory_shape,
                    dtype=float_dtype,
                    buffer=shm_model_spectra.buf,
                )
            model_spectra_arr[:] = self.model_spectra.reshape(
                shared_memory_shape
            ).astype(float_dtype)
            return shm_model_spectra.name, model_spectra_arr.shape

    def gen_emline_spectra(
        self,
        emline: str | ArrayLike | None,
    ):
        """
        Generate emission line spectra from `self.model_params`.

        This method expects both `self.model_params` and
        `self.model_line_fluxes` to exist, so must typically be run after
        `self.gen_spectra_from_params()`, or
        `self.gen_all_spectra_from_seeds()`.

        Parameters
        ----------
        emline : str | ArrayLike | None
            The names of one or more emission lines, for which a resampled
            spectrum will be calculated on the same wavelength grid as
            `self.model_spectra`. The names are based on the
            `Cloudy <https://www.nublado.org/>`__ naming convention (see
            `here
            <https://bagpipes.readthedocs.io/en/latest/model_galaxies.html\
#getting-observables-line-fluxes>`__
            for more details). If ``None``, then all emission lines will
            be modelled.
        """

        if emline is None:
            line_mask = self.temp_is_line
        else:
            line_mask = np.zeros_like(self.temp_names, dtype=bool)
            emline = np.atleast_1d(emline)
            if "line Ha" in emline:
                emline = np.concatenate([emline, ["line NII"]])
            for e in emline:
                if e == "OIII-5007":
                    e = "OIII"
                line_mask |= np.isin(
                    self.temp_names, [f"line {e}", f"line broad {e}", f"line agn {e}"]
                )

        print(f"{emline}, {self.temp_flux_arr[line_mask].shape=}")

        emline_temps = np.zeros_like(self.temp_flux_arr)
        emline_temps[line_mask] = self.temp_flux_arr[line_mask]

        print(emline_temps.shape)

        self.model_emline_spectra = np.tile(
            emline_temps,
            (len(self.posterior_ids), 1),
        )

        return

    def gen_model_seeds_from_iter(
        self, iter_seed: int, n_samples: int, n_extra_regions: int = 0, **kwargs
    ) -> tuple[np.ndarray[int], np.ndarray[int]]:
        """
        Construct a list of model seeds for a given iteration.

        This class does not implement model variation, so each iteration
        will return identical model seeds.

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
        model_seeds : np.ndarray[int]
            The list of model seeds.
        extra_region_idxs : np.ndarray[int]
            The indices of the additional regions to sample.
        """

        return np.arange(self.temp_flux_arr.shape[0])[:n_samples].astype(int), np.zeros(
            n_extra_regions, dtype=int
        )

        # self.initialise_process_pool(self.cpu_count)


def _test_process_affinity(task_id):

    import psutil

    print(f"{task_id} : {psutil.Process()} : {psutil.Process().cpu_affinity()}")


if __name__ == "__main__":

    posterior_dir = Path(
        "/media/sharedData/data/2025_12_06_glass-a2744/glass_niriss_bcgs/"
        "reduction_v9-TEST/sed_fitting/pipes/posterior/"
        "3070_colour_3_10_jwst-nircam-f150w"
    )

    template_sampler = GrizliTemplateSampler(
        posterior_dir=posterior_dir, cache_all_spectra=False, redshift=1.33
    )

    model_seeds, extra_region_idxs = template_sampler.gen_model_seeds_from_iter(
        0, 56, 0
    )
    print(model_seeds)
    template_sampler.gen_all_spectra_from_seeds(
        model_seeds=model_seeds, extra_region_idxs=extra_region_idxs, n_extra_samples=0
    )
    print(template_sampler.model_spectra.shape)
    import matplotlib.pyplot as plt

    for k, t, u in zip(
        template_sampler.temp_names,
        template_sampler.temp_flux_arr,
        template_sampler.temp_is_line,
    ):
        if u and np.nansum(t) > 0:
            plt.plot(template_sampler.spec_wavs, t, label=k)

    plt.legend()

    # plt.imshow(np.log10(template_sampler.model_spectra[:len(model_seeds)]))
    # plt.show()
    # for t in template_sampler.model_spectra[:len(model_seeds)][template_sampler.is_line]:
    #     plt.plot(template_sampler.spec_wavs, t)
    plt.show()
    # print(template_sampler.model_params.shape)
    template_sampler.gen_emline_spectra(
        # emline=["H  1  6562.80A", "N  2  6583.45A", "N  2  6548.05A"]
        # emline=["H  1  6562.80A"]
        None
    )
    print(template_sampler.model_emline_spectra.shape)

    # import matplotlib.pyplot as plt

    # for i, (m, l) in enumerate(
    #     zip(template_sampler.model_spectra, template_sampler.model_emline_spectra)
    # ):
    #     if i > 10:
    #         continue
    #     plt.plot(template_sampler.spec_wavs, m - l)

    # plt.xlim(xmin=1e4, xmax=2.3e4)
    # plt.show()

    # exit()
