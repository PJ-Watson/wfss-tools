"""Base classes for 2D spatially-resolved SED fitting."""

from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from wfss_tools.grism.samplers import GrizliTemplateSampler, TemplateSampler

__all__ = ["SEDFitResult", "SEDFitter2D", "NullSEDFitter"]


@dataclass
class SEDFitResult:
    """
    Standardised output dataclass produced by a 2D SED fitting backend.

    Attributes
    ----------
    fitter_name : str
        Identifier for the SED fitting backend (e.g. ``"bagpipes"`` or
        ``"grizli_empirical"``).
    run_name : str
        Unique name identifying the SED fitting run.
    binned_data_path : Path
        Location of the Stage 1 binned photometry and segmentation map
        FITS file used as input.
    catalogue_path : Path | None, optional
        Location of the output best-fit parameter catalogue, if generated.
    posterior_dir : Path | None, optional
        Directory containing per-region posterior samples, if generated.
    metadata : dict[str, Any], optional
        Additional key-value provenance metadata to propagate into Stage 3
        FITS headers.
    """

    fitter_name: str
    run_name: str
    binned_data_path: Path
    catalogue_path: Path | None = None
    posterior_dir: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class SEDFitter2D:
    """
    Base class for 2D spatially-resolved SED fitting backends.

    Parameters
    ----------
    obj_z : float | ArrayLike
        The redshift of the object to fit.
    config : dict | PathLike | None, optional
        Either a dictionary, or a TOML-formatted configuration file
        containing parameters for the SED fit.
    **kwargs : dict
        Additional backend-specific keyword arguments.
    """

    fitter_name: str = "base"

    def __init__(
        self,
        obj_z: float | ArrayLike,
        config: dict | PathLike | None = None,
        **kwargs,
    ):
        self.obj_z = obj_z

    def import_config(self, config: dict | PathLike) -> dict:
        """
        Load a configuration file.

        Parameters
        ----------
        config : dict or PathLike
            Either a dictionary, or the TOML-formatted configuration file
            containing the parameters to use for the fit.

        Returns
        -------
        dict
            The parsed configuration dictionary.
        """
        if isinstance(config, dict):
            return config
        elif isinstance(config, PathLike) or isinstance(config, str):
            import tomllib

            with open(config, "rb") as f:
                return tomllib.load(f)
        else:
            raise TypeError("Cannot import config.")

    def fit(self, binned_data_path: PathLike, **kwargs) -> SEDFitResult:
        """
        Perform a 2D SED fit to the binned photometric data.

        Parameters
        ----------
        binned_data_path : PathLike
            The location of the binned photometric catalogue and
            segmentation map FITS file produced by Stage 1.
        **kwargs : dict
            Additional backend-specific fitting arguments.

        Returns
        -------
        SEDFitResult
            The standardised Stage 2 output object.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def get_sampler(
        self,
        fit_result: SEDFitResult,
        region_ids: ArrayLike,
        spec_wavs: ArrayLike = np.arange(10000.0, 23000.0, 22.5),
        veldisp: float = 50.0,
        seed: int = 2744,
        cpu_count: int = -1,
        cache_all_spectra: bool = False,
        **kwargs,
    ) -> TemplateSampler:
        """
        Instantiate a `~wfss_tools.grism.samplers.TemplateSampler`.

        This can be passed directly to the resolved grism modelling
        (i.e. Stage 3 in the current workflow).

        Parameters
        ----------
        fit_result : SEDFitResult
            The output returned by `fit`.
        region_ids : ArrayLike
            The ordered array of spatial bin IDs to sample.
        spec_wavs : ArrayLike, optional
            The wavelength grid (in Angstroms) on which templates will be
            evaluated.
        veldisp : float, optional
            The velocity dispersion of the template spectra in km/s, by
            default ``50.0``.
        seed : int, optional
            Random seed for template sampling, by default ``2744``.
        cpu_count : int, optional
            Number of worker processes for template generation, by default
            ``-1``.
        cache_all_spectra : bool, optional
            Whether to pre-generate and cache spectra in memory, by default
            ``False``.
        **kwargs : dict
            Additional sampler-specific keyword arguments.

        Returns
        -------
        TemplateSampler
            An initialised template sampler ready for
            `~wfss_tools.grism.multiregion.MultiRegionFit`.
        """
        raise NotImplementedError("Subclasses should implement this method.")


class NullSEDFitter(SEDFitter2D):
    """
    A pass-through SED fitter for prior-free grism fitting.

    Currently this is designed to work with `grizli` templates.

    This bypasses photometric SED fitting while preserving the Stage 2 ->
    Stage 3 dataclass, allowing `~wfss_tools.grism.multiregion.MultiRegionFit`
    to fit spatially-resolved empirical `grizli` templates across the
    Stage 1 segmentation regions.
    """

    fitter_name: str = "grizli_empirical"

    def fit(self, binned_data_path: PathLike, **kwargs) -> SEDFitResult:
        """
        Return a pass-through `SEDFitResult` without running an SED fit.

        Parameters
        ----------
        binned_data_path : PathLike
            The location of the binned photometric catalogue and
            segmentation map FITS file produced by Stage 1.
        **kwargs : dict
            Ignored.

        Returns
        -------
        SEDFitResult
            A metadata-only dataclass pointing to ``binned_data_path``.
        """
        binned_path = Path(binned_data_path)
        return SEDFitResult(
            fitter_name=self.fitter_name,
            run_name=f"{binned_path.stem.removesuffix('_data')}_empirical",
            binned_data_path=binned_path,
            metadata={"MRSEDFIT": (self.fitter_name, "2D SED fitting backend")},
        )

    def get_sampler(
        self,
        fit_result: SEDFitResult,
        region_ids: ArrayLike,
        spec_wavs: ArrayLike = np.arange(10000.0, 23000.0, 22.5),
        veldisp: float = 50.0,
        seed: int = 2744,
        cpu_count: int = -1,
        cache_all_spectra: bool = False,
        **kwargs,
    ) -> GrizliTemplateSampler:
        """
        Instantiate a `~wfss_tools.grism.samplers.GrizliTemplateSampler`.

        Parameters
        ----------
        fit_result : SEDFitResult
            The output returned by `fit`.
        region_ids : ArrayLike
            The ordered array of spatial bin IDs to sample.
        spec_wavs : ArrayLike, optional
            The wavelength grid (in Angstroms) on which templates will be
            evaluated.
        veldisp : float, optional
            The velocity dispersion of the template spectra in km/s, by
            default ``50.0``.
        seed : int, optional
            Random seed for template sampling, by default ``2744``.
        cpu_count : int, optional
            Number of worker processes for template generation, by default
            ``-1``.
        cache_all_spectra : bool, optional
            Whether to pre-generate and cache spectra in memory, by default
            ``False``.
        **kwargs : dict
            Additional keyword arguments forwarded to `GrizliTemplateSampler`.

        Returns
        -------
        GrizliTemplateSampler
            An initialised `grizli` template sampler.
        """
        scalar_z = float(np.median(self.obj_z))
        return GrizliTemplateSampler(
            posterior_dir=fit_result.binned_data_path.parent,
            seed=seed,
            cpu_count=cpu_count,
            cache_all_spectra=cache_all_spectra,
            veldisp=veldisp,
            spec_wavs=np.asarray(spec_wavs),
            redshift=scalar_z,
            **kwargs,
        )
