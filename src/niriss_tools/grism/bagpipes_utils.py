"""Utility functions relating to `bagpipes`."""

import ast
import multiprocessing
from functools import partial
from itertools import repeat
from multiprocessing.managers import SharedMemoryManager
from pathlib import Path

import h5py
import numpy as np
from bagpipes import config
from bagpipes_extended.sed.specgen import BagpipesSpecGenerator, air_to_vac
from grizli.utils_numba.interp import interp_conserve_c
from numpy.typing import ArrayLike
from wfss_tools.grism import float_dtype
from wfss_tools.grism.samplers import TemplateSampler

__all__ = ["BagpipesTemplateSampler", "init_bagpipes_spec_gen"]


def init_bagpipes_spec_gen(
    fit_instructions: dict, veldisp: float, spec_wavs: np.ndarray[float]
):
    """
    Initialise an instance of the bagpipes spectral generator.

    This is used to separate data within multiprocessing.

    Parameters
    ----------
    fit_instructions : dict
        A dictionary containing information about the model to be
        generated.
    veldisp : float
        The velocity dispersion of the model galaxy in km/s.
    spec_wavs : np.ndarray[float]
        The wavelengths onto which the spectrum will be sampled, in
        Angstroms.
    """

    global spec_generator
    spec_generator = BagpipesSpecGenerator(
        fit_instructions=fit_instructions, veldisp=veldisp, spec_wavs=spec_wavs
    )


class BagpipesTemplateSampler(TemplateSampler):
    """
    A subclass of TemplateSampler to be used with SED fits from `bagpipes`.

    Parameters
    ----------
    posterior_dir : Path
        The directory containing the posterior ``*.h5`` files.
    seed : int, optional
        The base seed for all sampling, by default ``2744``.
    cpu_count : int, optional
        The number of CPUs to use for multiprocessing. Defaults to the
        number returned by `~multiprocessing.cpu_count()`.
    cache_all_spectra : bool, optional
        If ``True`` (default), then on initialisation all possible spectra
        will be pre-generated. If ``False``, these will be generated
        on-the-fly during sampling. Which is more efficient depends on
        whether the total number of spectra is substantially lower than
        the number of spectra that may be sampled during fitting.

        TODO: Inlcude this explanation somewhere
        Why not cache spectra when generating the model atlas? bagpipes
        speed depends on the spectral resolution. Broad and medium band photometry
         is much faster than for a 2x oversampled NIRISS spectrum.
    veldisp : float, optional
        The velocity dispersion of the model galaxy in km/s, by default
        ``50``.
    spec_wavs : np.ndarray[float], optional
        The wavelengths onto which the spectrum will be sampled, in
        Angstroms. By default, this is set to
        ``np.arange(10000.0, 23000.0, 22.5)``, covering the full range of
        JWST/NIRISS.
    apply_R_curve : bool, optional
        Implement the variable spectral resolution of the JWST/NIRISS
        grisms when generating spectra, even if not in the original model
        components. By default ``False``.
    """

    def __init__(
        self,
        posterior_dir: Path,
        seed: int = 2744,
        cpu_count: int = multiprocessing.cpu_count(),
        cache_all_spectra: bool = True,
        veldisp: float = 50,
        spec_wavs: np.ndarray[float] = np.arange(10000.0, 23000.0, 22.5),
        apply_R_curve: bool = False,
    ):

        super().__init__(seed)

        self.base_rng = np.random.Generator(np.random.PCG64(self.seed))

        self.posterior_ids = [f.stem for f in posterior_dir.glob("*.h5")]
        try:
            self.posterior_ids.sort(key=int)
        except:
            self.posterior_ids.sort()

        self.cpu_count = cpu_count

        self.cache_all_spectra = cache_all_spectra

        self.fit_instructions = self.load_fit_instructions(
            posterior_dir / f"{self.posterior_ids[0]}.h5"
        )
        if apply_R_curve:
            self.add_niriss_R_curve(self.fit_instructions)

        self.veldisp = veldisp
        self.spec_wavs = spec_wavs

        self.dummy_spec_gen = BagpipesSpecGenerator(
            self.fit_instructions, self.veldisp, self.spec_wavs
        )
        self.dummy_spec_gen.sample(
            self.load_model_params(posterior_dir / f"{self.posterior_ids[0]}.h5")[0]
        )

        self.model_comp = self.dummy_spec_gen.model_components

        self.param_names = self.dummy_spec_gen.params

        self.initialise_process_pool(self.cpu_count)

        params_lists = self.process_pool.map(
            self.load_model_params,
            [posterior_dir / f"{i}.h5" for i in self.posterior_ids],
        )

        params_array = np.concatenate(params_lists, axis=0)

        # Normalise all masses to the median.
        # This should allow templates to be close enough to their unnormalised
        # values (thereby avoiding very large/small coefficients during the fit),
        # whilst minimising the number of templates to generate
        for p_i, p in enumerate(self.param_names):
            if "massformed" in p:
                params_array[:, p_i] = np.nanmedian(params_array[:, p_i])

        # Keep as string for now, reconsider later if np.searchsorted adds
        # support for axes
        params_array = np.array([str(r.tolist()) for r in params_array])

        u, inv = np.unique(params_array, return_inverse=True, axis=0)

        self.all_model_params = u

        self.posterior_params_map = inv.reshape(len(params_lists), -1)

        self.line_names = np.array(config.line_names)
        self.line_wavs_rf = np.array(config.line_wavs)

        self.all_model_spectra = None
        self.all_model_line_fluxes = None

        if self.cache_all_spectra:

            self.all_model_spectra, self.all_model_line_fluxes = (
                self.gen_spectra_from_params(self.all_model_params)
            )

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
            ``None``. This is implemented as an offset which wraps around,
            providing a minimal description of the full set of sampled
            regions.
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

        assert n_extra_samples <= len(model_seeds), (
            "The number of models per additional region cannot "
            "exceed the number of model seeds generated."
        )

        # Simple case: no extra samples
        # These are the indices of `self.all_model_params`, which
        # contains the actual model parameters. That array contains
        # only unique values, whilst this is not guaranteed to.
        all_models = self.posterior_params_map[:, model_seeds]

        if (
            (extra_region_idxs is not None)
            and (len(extra_region_idxs) > 0)
            and (n_extra_samples > 0)
        ):
            extra_ids = np.tile(
                np.arange(len(self.posterior_ids))[:, np.newaxis, np.newaxis],
                (1, len(extra_region_idxs), n_extra_samples),
            )

            # Ensure that the indexes don't exceed the length of the array
            extra_ids = np.remainder(
                extra_ids + extra_region_idxs[np.newaxis, :, np.newaxis],
                len(self.posterior_ids),
            )

            extra_models = self.posterior_params_map[
                extra_ids, model_seeds[:n_extra_samples]
            ].reshape((len(self.posterior_ids), -1))

            all_models = np.concatenate((all_models, extra_models), axis=-1)

        all_models_shape = all_models.shape

        unique_models, unique_models_inv = np.unique(
            all_models.ravel(), return_inverse=True
        )

        model_spectra, model_line_fluxes = self.gen_spectra_from_params(
            self.all_model_params[unique_models].ravel()
        )

        self.model_spectra = model_spectra[unique_models_inv]
        self.model_line_fluxes = model_line_fluxes[unique_models_inv]
        self.model_params = self.model_params[unique_models_inv]

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
            if self.model_spectra.dtype == float_dtype:
                model_spectra_arr.flat = self.model_spectra.flat
            else:
                model_spectra_arr[:] = self.model_spectra.reshape(
                    shared_memory_shape
                ).astype(float_dtype)
            return shm_model_spectra.name, model_spectra_arr.shape

        return

    def gen_spectra_from_params(
        self, params_array: np.ndarray[str]
    ) -> tuple[np.ndarray[float], np.ndarray[float]]:
        """
        Generate resampled spectra from a set of bagpipes parameters.

        If all models were already generated, this simply returns the
        relevant portions of those arrays.

        Parameters
        ----------
        params_array : np.ndarray[str]
            A 1D array of strings, each of which can be evaluated as a set
            of parameters for `bagpipes`, following the setup specified in
            `self.fit_instructions`.

        Returns
        -------
        model_spectra : np.ndarray[float]
            An ``(m x n)`` array of resampled spectra, for ``m`` models in
            ``params_array``. The length of the spectra ``n`` is
            determined by ``self.spec_wavs``.
        model_line_fluxes : np.ndarray[float]
            An ``(m x l)`` array of emission line fluxes, for ``m`` models
            in ``params_array`` and ``l`` emission lines in
            `bagpipes.config.line_names`.
        """

        self.model_params = params_array

        # Check if we already computed all possible spectra
        if self.all_model_spectra is not None:
            sorter = np.argsort(self.all_model_params)
            arr_idxs = sorter[
                np.searchsorted(self.all_model_params, self.model_params, sorter=sorter)
            ]

            return (
                self.all_model_spectra[arr_idxs],
                self.all_model_line_fluxes[arr_idxs],
            )

        spec_lists, line_flux_dicts = zip(
            *self.process_pool.map(
                self.worker_gen_spec_and_fluxes,
                self.model_params,
            )
        )

        model_spectra = np.array(spec_lists)
        del spec_lists

        merged_line_flux_dict = {
            k: [d.get(k, np.nan) for d in line_flux_dicts] for k in self.line_names
        }
        model_line_fluxes = np.array(list(merged_line_flux_dict.values())).T
        del merged_line_flux_dict, line_flux_dicts

        return model_spectra, model_line_fluxes

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
            emline = self.line_names.copy()

        # Ensure that emission lines will always be an array
        emline = np.atleast_1d(emline)

        if not (
            hasattr(self, "model_params")
            and hasattr(self, "model_line_fluxes")
            and (len(self.model_params) == len(self.model_line_fluxes))
        ):
            raise ValueError(
                "Either the model parameters or the line fluxes have not "
                "been initialised correctly."
            )

        self.emline = emline

        unique_params, unique_idxs, unique_inv = np.unique(
            self.model_params, return_index=True, return_inverse=True
        )
        unique_line_fluxes = self.model_line_fluxes[unique_idxs]

        model_wavs_rf = self.dummy_spec_gen.model_gal.wavelengths

        if "redshift" in self.param_names:
            z_idx = (np.array(self.param_names) == "redshift").argmax()
            model_redshifts = np.array(
                [ast.literal_eval(m)[z_idx] for m in unique_params]
            )

        # Find the exact index of each emission line name
        # (order must be preserved)
        sorter = np.argsort(self.line_names)
        emline_idxs = sorter[np.searchsorted(self.line_names, emline, sorter=sorter)]

        emline_wavs_rf = self.line_wavs_rf[emline_idxs] * (
            1 + (self.model_comp["nebular"].get("velshift", 0) / (3 * 10**5))
        )

        wav_idxs = np.abs(model_wavs_rf[:, np.newaxis] - emline_wavs_rf).argmin(axis=0)

        line_templates = np.zeros((len(unique_params), len(model_wavs_rf)))

        for wav_idx, line_idx in zip(wav_idxs, emline_idxs):
            width = (model_wavs_rf[wav_idx + 1] - model_wavs_rf[wav_idx - 1]) / 2

            line_templates[:, wav_idx] = unique_line_fluxes[:, line_idx] / width

        # Replicate the same sampling used within bagpipes
        if "veldisp" in list(self.model_comp):
            vres = 3 * 10**5 / config.R_spec / 2.0
            sigma_pix = self.model_comp["veldisp"] / vres
            k_size = 4 * int(sigma_pix + 1)
            x_kernel_pix = np.arange(-k_size, k_size + 1)

            kernel = np.exp(-(x_kernel_pix**2) / (2 * sigma_pix**2))
            kernel /= np.trapezoid(kernel)  # Explicitly normalise kernel

            model_wavs_rf = model_wavs_rf[k_size:-k_size]

            convolved_line_templates = np.apply_along_axis(
                np.convolve, -1, line_templates, kernel, mode="valid"
            )

        else:
            convolved_line_templates = line_templates

        redshifted_wavs = (1 + model_redshifts)[:, np.newaxis] * model_wavs_rf

        if "R_curve" in list(self.model_comp):
            oversample = 4  # Number of samples per FWHM at resolution R
            new_wavs = self.dummy_spec_gen.model_gal._get_R_curve_wav_sampling(
                oversample=oversample
            )

            resampled_spectra = np.array(
                self.process_pool.starmap(
                    interp_conserve_c,
                    zip(repeat(new_wavs), redshifted_wavs, convolved_line_templates),
                )
            )

            redshifted_wavs = np.tile(
                new_wavs[np.newaxis, :], (resampled_spectra.shape[0], 1)
            )

            sigma_pix = oversample / 2.35  # sigma width of kernel in pixels
            k_size = 4 * int(sigma_pix + 1)
            x_kernel_pix = np.arange(-k_size, k_size + 1)

            kernel = np.exp(-(x_kernel_pix**2) / (2 * sigma_pix**2))
            kernel /= np.trapezoid(kernel)  # Explicitly normalise kernel

            # Disperse non-uniformly sampled spectrum
            # spectrum = np.convolve(spectrum, kernel, mode="valid")
            convolved_line_templates = np.apply_along_axis(
                np.convolve, -1, resampled_spectra, kernel, mode="valid"
            )
            redshifted_wavs = redshifted_wavs[:, k_size:-k_size]

        vac_redshifted_wavs = air_to_vac(redshifted_wavs)

        self.model_emline_spectra = np.array(
            self.process_pool.starmap(
                interp_conserve_c,
                zip(
                    repeat(self.spec_wavs),
                    vac_redshifted_wavs,
                    convolved_line_templates,
                ),
            )
        )

        self.model_emline_spectra /= (1 + model_redshifts)[:, np.newaxis]

        self.model_emline_spectra = self.model_emline_spectra[unique_inv]

        if self.dummy_spec_gen.model_gal.spec_units == "mujy":
            self.model_emline_spectra /= 10**-29 * 2.9979 * 10**18 / self.spec_wavs**2

    @staticmethod
    def worker_gen_spec_and_fluxes(param_vector: str) -> tuple[np.ndarray[float], dict]:
        """
        Generate a spectrum and emission line fluxes using `bagpipes`.

        Parameters
        ----------
        param_vector : str
            The string-formatted set of input parameters, typically in the
            form of a list of floats.

        Returns
        -------
        model_spectrum : np.ndarray[float]
            The model spectrum, generated on the wavelength grid used to
            initialise `spec_generator`.
        line_fluxes : dict
            The names and fluxes of all emission lines in the model
            spectrum.
        """

        return spec_generator.sample_spec(
            ast.literal_eval(param_vector), return_line_fluxes=True
        )

    @staticmethod
    def load_model_params(posterior_path: Path) -> np.ndarray:
        """
        Convert a bagpipes posterior object to a 1D array of strings.

        Each element in the array is a string-formatted list of the input
        model parameters.

        Parameters
        ----------
        posterior_path : Path
            The path of the posterior object.

        Returns
        -------
        np.ndarray
            A 1D array of model parameters.
        """

        with h5py.File(posterior_path, "r") as post_file:
            samples2d = np.array(post_file["samples2d"])

        # return np.array([str(r.tolist()) for r in samples2d])
        return samples2d

    @staticmethod
    def load_fit_instructions(posterior_path: Path) -> dict:
        """
        Load the bagpipes fit instructions from a posterior object.

        Parameters
        ----------
        posterior_path : Path
            The path of the posterior object.

        Returns
        -------
        dict
            The fit instructions as a key/value dictionary.
        """

        with h5py.File(
            posterior_path,
            "r",
        ) as test_post:

            fit_info_str = test_post.attrs["fit_instructions"]
            fit_info_str = fit_info_str.replace("array", "np.array")
            fit_info_str = fit_info_str.replace("float", "np.float")
            fit_info_str = fit_info_str.replace("np.np.", "np.")
            fit_instructions = eval(fit_info_str)

        return fit_instructions

    @staticmethod
    def add_niriss_R_curve(
        fit_instructions: dict, wav_sampling: float = 100, wav_increment: float = 94
    ):
        """
        Add the NIRISS spectral resolution curve to the model components.

        The input is modified in place.

        Parameters
        ----------
        fit_instructions : dict
            A dictionary containing information about the model to be
            generated.
        wav_sampling : float, optional
            The wavelength sampling of the R curve in Angstroms, by
            default ``100``.
        wav_increment : float, optional
            The wavelength increment over 2 pixels, which determines the
            spectral resolution of JWST/NIRISS. By default this is set to
            ``94`` Angstroms, which is approximately the value for the 1st
            order spectra in both grisms at the centre of the detector.
        """

        if not ("R_curve" in fit_instructions.keys()):
            wavs = np.arange(0.5e4, 3e4, wav_sampling)
            fit_instructions["R_curve"] = np.c_[wavs, wavs / wav_increment]

    # Cache the model seeds in self
    # e.g. calling `self.sample_spec_from_iter(iter_seed, posterior_id)`
    # checks if self.model_seeds[iter_seed] already exists

    def gen_model_seeds_from_iter(
        self, iter_seed: int, n_samples: int, n_extra_regions: int = 0, **kwargs
    ) -> tuple[np.ndarray[int], np.ndarray[int]]:
        """
        Construct a list of model seeds for a given iteration.

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

        iter_rng = np.random.Generator(np.random.PCG64(self.seed + iter_seed))

        model_seeds = iter_rng.choice(
            np.arange(self.posterior_params_map.shape[-1]),
            size=n_samples,
            replace=False,
        ).astype(int)

        extra_region_idxs = iter_rng.choice(
            np.arange(len(self.posterior_ids)),
            size=n_extra_regions,
            replace=False if n_extra_regions <= len(self.posterior_ids) else True,
        ).astype(int)

        return model_seeds, extra_region_idxs

    def initialise_process_pool(self, cpu_count: int):
        """
        Initialise a pool of processes.

        This is stored as a class attribute, to reduce the overhead of
        creating this each time it is needed. This overrides the inherited
        base class method to initialise the `BagpipesSpecGenerator`
        instances for each process.

        Parameters
        ----------
        cpu_count : int
            The number of processes to create.
        """

        self._process_pool = multiprocessing.Pool(
            processes=cpu_count,
            initializer=init_bagpipes_spec_gen,
            initargs=(self.fit_instructions, self.veldisp, self.spec_wavs),
        )


if __name__ == "__main__":

    posterior_dir = Path(
        "/media/sharedData/data/2025_12_06_glass-a2744/glass_niriss_bcgs/"
        "reduction_v9-TEST/sed_fitting/pipes/posterior/"
        "3070_colour_3_10_jwst-nircam-f150w"
    )

    template_sampler = BagpipesTemplateSampler(
        posterior_dir=posterior_dir, cache_all_spectra=False
    )

    model_seeds, extra_region_idxs = template_sampler.gen_model_seeds_from_iter(0, 5, 5)
    template_sampler.gen_all_spectra_from_seeds(
        model_seeds=model_seeds, extra_region_idxs=extra_region_idxs, n_extra_samples=1
    )
    print(template_sampler.model_spectra.shape)
    print(template_sampler.model_params.shape)
    template_sampler.gen_emline_spectra(
        emline=["H  1  6562.80A", "N  2  6583.45A", "N  2  6548.05A"]
        # emline=["H  1  6562.80A"]
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
