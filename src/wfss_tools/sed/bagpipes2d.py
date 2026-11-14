"""A module for performing 2D SED fits with bagpipes."""

from functools import partial
from os import PathLike
from pathlib import Path
from typing import Callable

import numpy as np
from astropy.table import Table
from bagpipes_extended.pipeline import generate_fit_params, load_photom_bagpipes
from bagpipes_extended.sed.atlas_fitter import AtlasFitter, temp_chdir
from bagpipes_extended.sed.atlas_generator import AtlasGenerator
from numpy.typing import ArrayLike

import wfss_tools
from wfss_tools.grism.bagpipes_utils import BagpipesTemplateSampler
from wfss_tools.pipeline.reduction import recursive_merge
from wfss_tools.sed.core_2d import SEDFitResult, SEDFitter2D

__all__ = ["Bagpipes2D"]


class Bagpipes2D(SEDFitter2D):
    """
    A class for performing 2D SED fits with Bagpipes.

    Parameters
    ----------
    obj_z : float | ArrayLike
        The redshift of the object to fit. If a scalar value is
        passed, and ``z_range==0.0``, the object will be fit to a
        single redshift value. If ``z_range!=0.0``, this will be the
        centre of the redshift window. If an array is passed, this
        explicitly sets the redshift range to use for fitting.
    config : dict | PathLike | None, optional
        Either a dictionary, or the TOML-formatted configuration file
        containing the parameters to use for the fit. Any other
        parameters passed on initialisation will override those set in the
        configuration file.
    z_range : float | None, optional
        The maximum redshift range to search over. If ``None`` (default),
        uses the value from ``config`` or defaults to ``0.005``.
    fit_instructions : dict | None, optional
        A dictionary containing instructions on the kind of model
        which should be fitted to the data. If ``None`` (default),
        instructions are loaded from ``config`` or generated using
        ``fit_params_kwargs``.
    fit_params_kwargs : dict | None, optional
        Keyword arguments passed to generate fit parameters if
        ``fit_instructions`` is not provided.
    n_samples : int | None, optional
        The number of samples to generate for the model grid. If
        ``None`` (default), uses the value from ``config`` or defaults
        to ``1e6``.
    remake_atlas : bool | None, optional
        If ``True``, any existing model atlas with the same name will
        be recreated and overwritten. If ``None`` (default), uses the
        value from ``config`` or defaults to ``False``.
    n_cores : int | None, optional
        The number of processes to use when generating the model grid or
        fitting. If set to ``0``, the code will run on a single process.
        If set to an integer less than 0, this will run on the number of
        cores returned by `multiprocessing.cpu_count`. If ``None``
        (default), uses the value from ``config`` or defaults to ``4``.
    overwrite_fit : bool | None, optional
        If ``True``, then any existing posterior distributions and
        output catalogues will be overwritten. If ``None`` (default),
        uses the value from ``config`` or defaults to ``False``.
    root_dir : PathLike | None, optional
        The root directory for all input and output files. If ``None``
        (default), uses the value from ``config`` or defaults to
        ``"."``.
    out_dir : PathLike | None, optional
        The output directory relative to ``root_dir`` where output
        files will be saved. If ``None`` (default), uses the value from
        ``config`` or defaults to ``"."``.
    atlas_dir : PathLike | None, optional
        The directory where model atlases are stored. If ``None``
        (default), uses the value from ``config`` or defaults to
        ``<out_dir>/sed_fitting/pipes/atlases``.
    seed : int, optional
        Random seed for sample generation, by default ``2744``.
    """

    fitter_name: str = "bagpipes"

    def __init__(
        self,
        obj_z: float | ArrayLike,
        config: dict | PathLike | None = None,
        z_range: float | None = None,
        fit_instructions: dict | None = None,
        fit_params_kwargs: dict | None = None,
        n_samples: int | None = None,
        remake_atlas: bool | None = None,
        n_cores: int | None = None,
        overwrite_fit: bool | None = None,
        root_dir: PathLike | None = None,
        out_dir: PathLike | None = None,
        atlas_dir: PathLike | None = None,
        seed: int = 2744,
    ):
        super().__init__(obj_z=obj_z, config=config)

        args = dict(locals())
        args.pop("self", None)
        args.pop("config", None)

        for k, v in args.copy().items():
            if v is None:
                args.pop(k)

        self.obj_z = obj_z

        if config is not None:
            config = self.import_config(config)
        else:
            config = {}

        # Merge legacy [SED] / [fit_params] / [atlas] tables with unified [sed]
        # Of course, these can be kept separate if desired
        config_sed = recursive_merge(
            config.get("SED", {}),
            config.get("sed", {}),
        )
        config_fit_params = recursive_merge(
            config.get("fit_params", {}),
            config_sed,
        )
        config_atlas = recursive_merge(
            config.get("atlas", {}),
            config_sed,
        )

        self.z_range = args.get("z_range", config_fit_params.get("z_range", 0.005))

        self.fit_instructions = args.get(
            "fit_instructions", config.get("fit_instructions")
        )

        if self.fit_instructions is None:

            merged_fit_params = {
                "obj_z": config_fit_params.get("obj_z", self.obj_z),
                "z_range": self.z_range,
                "sfh_type": config_fit_params.get("sfh_type", "continuity"),
                "num_age_bins": config_fit_params.get("num_age_bins", 5),
                "min_age_bin": config_fit_params.get("min_age_bin", 20),
            }
            if fit_params_kwargs is not None:
                merged_fit_params.update(fit_params_kwargs)

            for k in list(merged_fit_params.keys()):
                if args.get(k) is not None:
                    merged_fit_params[k] = args[k]

            self.fit_instructions = generate_fit_params(**merged_fit_params)

        self.n_samples = args.get("n_samples", config_atlas.get("n_samples", 1e6))
        self.remake_atlas = args.get(
            "remake_atlas", config_atlas.get("remake_atlas", False)
        )
        self.n_cores = args.get(
            "n_cores",
            config_atlas.get("n_cores", config_atlas.get("n_cores_atlas", 4)),
        )
        self.n_cores_fit = config_atlas.get("n_cores_fit", self.n_cores)

        self.overwrite_fit = args.get(
            "overwrite_fit", config_atlas.get("overwrite_fit", False)
        )

        config_files = config.get("files", {})
        self.root_dir = Path(args.get("root_dir", config_files.get("root_dir", ".")))
        self.out_dir = self.root_dir / args.get(
            "out_dir", config_files.get("out_dir", ".")
        )
        self.out_dir.mkdir(exist_ok=True, parents=True)

        self.pipes_dir = self.out_dir / "sed_fitting" / "pipes"
        self.pipes_dir.mkdir(exist_ok=True, parents=True)
        atlas_dir = args.get("atlas_dir", config_files.get("atlas_dir", ""))
        if not atlas_dir:
            self.atlas_dir = self.pipes_dir / "atlases"
        else:
            self.atlas_dir = self.root_dir / atlas_dir
        self.atlas_dir.mkdir(exist_ok=True, parents=True)

        self.seed = seed

        # Blank parameters until first run
        self.atlas_name = None
        self.atlas_path = None
        self.filter_list = None
        self.run_name = None
        self.sed_fit_cat_path = None
        self.posterior_dir = None

        return

    def load_filters_from_names(self, filter_names: ArrayLike) -> list[str]:
        """
        Load filter transmission curves given a set of filter names.

        Parameters
        ----------
        filter_names : ArrayLike
            A set of filter curve names, matching the internal
            nomenclature.

        Returns
        -------
        list[str]
            A list of filepaths for each of the filter curves.

        Notes
        -----
        In future the internal naming convention will be updated to match
        that used by `cigale` (Boquien+19).
        """

        import shutil

        default_filter_dir = (
            Path(wfss_tools.__file__).parent / "data" / "filter_throughputs"
        )

        # Create the filter directory; populate as needed
        # Better to copy filter curves so it's explicit which files were
        # used in the fit
        filter_dir = self.pipes_dir / "filter_throughputs"
        filter_dir.mkdir(exist_ok=True, parents=True)

        # Create a list of the filters used in our data
        self.filter_list = []
        for name in filter_names:
            new_loc = filter_dir / f"{name}.txt"
            shutil.copy(default_filter_dir / f"{name}.txt", new_loc)
            self.filter_list.append(str(new_loc))

        return self.filter_list

    def gen_atlas(
        self,
        atlas_name: str | None = None,
        obj_z: float | ArrayLike | None = None,
        n_samples: int | float | None = None,
        remake_atlas: bool | None = None,
        n_cores_atlas: int | None = None,
    ) -> Path:
        """
        Generate the fit parameters and model grid.

        This method generates a dictionary of fit parameters for Bagpipes,
        as well as a large model grid to speed up the SED fitting.

        Parameters
        ----------
        atlas_name : str | None, optional
            If not ``None``, override the default naming convention for
            the model atlas.
        obj_z : float | ArrayLike | None, optional
            The redshift of the object to fit. If a scalar value is
            passed, and ``z_range==0.0``, the object will be fit to a
            single redshift value. If ``z_range!=0.0``, this will be the
            centre of the redshift window. If an array is passed, this
            explicitly sets the redshift range to use for fitting. If
            ``None`` (default), this will be set to ``self.obj_z``.
        n_samples : int | float | None, optional
            The number of samples to generate. If ``None`` (default), uses
            ``self.n_samples``. A useful number will typically be ``>10^5``.
        remake_atlas : bool | None, optional
            If ``True``, any existing model atlas with the same name will
            be recreated and overwritten. If ``None`` (default), uses
            ``self.remake_atlas``.
        n_cores_atlas : int | None, optional
            The number of processes to use when generating the model grid.
            If set to ``0``, the code will run on a single process. If set
            to an integer less than 0, this will run on the number of
            cores returned by `multiprocessing.cpu_count`. If ``None``
            (default), uses ``self.n_cores``.

        Returns
        -------
        Path
            The location of the model atlas.
        """

        if self.filter_list is None:
            raise RuntimeError(
                "Filter curves have not been loaded. Call "
                "`Bagpipes2D.load_filters_from_names()` before `gen_atlas()`."
            )

        # Allow for method override of existing attributes
        eff_n_samples = self.n_samples if n_samples is None else n_samples
        eff_remake = self.remake_atlas if remake_atlas is None else remake_atlas
        eff_n_cores = self.n_cores if n_cores_atlas is None else n_cores_atlas

        if obj_z is not None:
            self.obj_z = obj_z

        if atlas_name is None:
            self.atlas_name = (
                f"z_{self.fit_instructions['redshift'][0]}_"
                f"{self.fit_instructions['redshift'][-1]}_"
                f"{eff_n_samples:.2E}"
            )
        else:
            self.atlas_name = atlas_name
        self.atlas_path = self.atlas_dir / f"{self.atlas_name}.hdf5"

        if not self.atlas_path.is_file() or eff_remake:

            with AtlasGenerator(
                fit_instructions=self.fit_instructions,
                filt_list=self.filter_list,
                phot_units="ergscma",
            ) as atlas_gen:

                atlas_gen.gen_samples(
                    n_samples=eff_n_samples,
                    seed=self.seed,
                    parallel=eff_n_cores,
                )

                atlas_gen.write_samples(filepath=self.atlas_path)

        return self.atlas_path

    def fit_atlas(
        self,
        binned_data_path: PathLike,
        binned_data_hdu: str | int | None = "PHOT_CAT",
        bagpipes_atlas_params: dict | None = None,
        load_fn: Callable | None = None,
        overwrite_fit: bool | None = None,
        id_colname: str = "bin_id",
        n_cores: int | None = None,
        obj_z: float | ArrayLike | None = None,
        z_range: float | None = None,
        run_name: str | None = None,
    ) -> SEDFitResult:
        """
        Perform a 2D SED fit to the binned photometric data.

        Parameters
        ----------
        binned_data_path : PathLike
            The location of the binned photometric catalogue and
            segmentation map used as input for Bagpipes.
        binned_data_hdu : str | int | None, optional
            The identifier of the HDU within ``binned_data_path``
            containing the photometric catalogue, by default
            ``"PHOT_CAT"``.
        bagpipes_atlas_params : dict | None, optional
            A dictionary containing instructions on the kind of model
            which should be fitted to the data. This should match the
            previously generated model grid. If ``None`` (default),
            ``self.fit_instructions`` will be used.
        load_fn : Callable | None, optional
            A function which takes the ID as an argument and returns the
            model photometry. This should be in the form of an array with
            a column of fluxes in microjanskys and a column of flux errors
            in the same units. If ``None`` (default),
            `~bagpipes_extended.pipeline.load_photom_bagpipes` will be
            used.
        overwrite_fit : bool | None, optional
            If ``True``, then any existing posterior distributions and
            output catalogues will be overwritten. If ``None`` (default),
            uses ``self.overwrite_fit``.
        id_colname : str, optional
            The name of the column in the photometric catalogue containing
            the bin ID, by default ``"bin_id"``.
        n_cores : int | None, optional
            The number of processes to use when fitting the catalogue.
            If set to ``0``, the code will run on a single process. If set
            to an integer less than 0, this will run on the number of
            cores returned by `multiprocessing.cpu_count`. If ``None``
            (default), uses ``self.n_cores_fit``.
        obj_z : float | ArrayLike | None, optional
            This can be used to override the redshift used for fitting, in
            case of a mismatch between the model atlas and the object of
            interest. See `~wfss_tools.sed.bagpipes2d.Bagpipes2D.gen_atlas`
            for more details.
        z_range : float | None, optional
            As above. If ``None`` (default), uses ``self.z_range``.
        run_name : str | None, optional
            Override the output run name. If ``None`` (default), defaults
            to ``"<binned_data_stem>_SED_fit"``.

        Returns
        -------
        SEDFitResult
            The standardised Stage 2 output dataclass containing paths to
            the output catalogue and posterior directory.
        """

        binned_path = Path(binned_data_path)
        self.run_name = (
            run_name if run_name is not None else f"{binned_path.stem}_SED_fit"
        )
        catalogue_out_path = self.pipes_dir.parent / f"{self.run_name}.fits"
        posterior_out_dir = self.pipes_dir / "posterior" / self.run_name

        # Method-level overrides, as in `gen_atlas`. We should always allow
        # this as an option (not least because I use it a lot for testing)
        eff_overwrite = self.overwrite_fit if overwrite_fit is None else overwrite_fit
        eff_n_cores = self.n_cores_fit if n_cores is None else n_cores
        eff_z_range = self.z_range if z_range is None else z_range
        eff_instructions = (
            self.fit_instructions
            if bagpipes_atlas_params is None
            else bagpipes_atlas_params
        )

        if (not catalogue_out_path.is_file()) or eff_overwrite:

            with temp_chdir(self.pipes_dir):

                if load_fn is None:

                    load_fn = partial(
                        load_photom_bagpipes,
                        phot_cat=binned_path,
                        cat_hdu_index=binned_data_hdu,
                    )

                obs_table = Table.read(binned_path, hdu=binned_data_hdu)
                cat_IDs = np.array(obs_table[id_colname])

                assert (self.atlas_path is not None) and (self.atlas_path.is_file()), (
                    "The model atlas has not been generated. Please run "
                    "`Bagpipes2D.gen_atlas()` before attempting to fit observations."
                )

                with AtlasFitter(
                    fit_instructions=eff_instructions,
                    atlas_path=self.atlas_path,
                    out_path=self.pipes_dir.parent,
                    overwrite=eff_overwrite,
                ) as atlas_fitter:

                    atlas_fitter.fit_catalogue(
                        IDs=cat_IDs,
                        load_data=load_fn,
                        spectrum_exists=False,
                        make_plots=False,
                        cat_filt_list=self.filter_list,
                        run=self.run_name,
                        parallel=eff_n_cores,
                        redshifts=self.obj_z if obj_z is None else obj_z,
                        redshift_range=eff_z_range,
                        n_posterior=500,
                    )

        self.sed_fit_cat_path = catalogue_out_path
        self.posterior_dir = posterior_out_dir

        return SEDFitResult(
            fitter_name=self.fitter_name,
            run_name=self.run_name,
            binned_data_path=binned_path,
            catalogue_path=self.sed_fit_cat_path,
            posterior_dir=self.posterior_dir,
            metadata={
                "MRSEDFIT": (self.fitter_name, "2D SED fitting backend"),
                "MRBPRUN": (
                    str(self.run_name),
                    "The name of the bagpipes run used to generate the prior templates.",
                ),
                "MRBPFCAT": (
                    str(self.sed_fit_cat_path),
                    "The bagpipes output fit catalogue.",
                ),
            },
        )

    def fit(
        self,
        binned_data_path: PathLike,
        binned_data_hdu: str | int | None = "PHOT_CAT",
        **kwargs,
    ) -> SEDFitResult:
        """
        Execute the full Stage 2 Bagpipes workflow (filters, atlas, and fit).

        Parameters
        ----------
        binned_data_path : PathLike
            The location of the binned photometric catalogue and
            segmentation map FITS file produced by Stage 1.
        binned_data_hdu : str | int | None, optional
            The identifier of the HDU within ``binned_data_path``
            containing the photometric catalogue, by default
            ``"PHOT_CAT"``.
        **kwargs : dict
            Additional keyword arguments forwarded to `fit_atlas`.

        Returns
        -------
        SEDFitResult
            The standardised Stage 2 output dataclass.
        """
        if self.filter_list is None:
            phot_cat = Table.read(binned_data_path, hdu=binned_data_hdu)
            filt_names = [
                c.removesuffix("_var") for c in phot_cat.colnames if c.endswith("_var")
            ]
            self.load_filters_from_names(filter_names=filt_names)

        self.gen_atlas()
        return self.fit_atlas(
            binned_data_path=binned_data_path,
            binned_data_hdu=binned_data_hdu,
            **kwargs,
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
    ) -> BagpipesTemplateSampler:
        """
        Instantiate a `BagpipesTemplateSampler` from Stage 2 posterior outputs.

        Parameters
        ----------
        fit_result : SEDFitResult
            The output dataclass returned by `fit` or `fit_atlas`.
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
            Additional keyword arguments forwarded to
            `BagpipesTemplateSampler`.

        Returns
        -------
        BagpipesTemplateSampler
            An initialised Bagpipes template sampler for Stage 3.
        """
        if fit_result.posterior_dir is None:
            raise ValueError(
                "`fit_result.posterior_dir` is None; cannot initialise "
                "`BagpipesTemplateSampler`."
            )

        return BagpipesTemplateSampler(
            posterior_ids=region_ids,
            posterior_dir=fit_result.posterior_dir,
            seed=seed,
            cpu_count=cpu_count,
            cache_all_spectra=cache_all_spectra,
            veldisp=veldisp,
            spec_wavs=np.asarray(spec_wavs),
            **kwargs,
        )
