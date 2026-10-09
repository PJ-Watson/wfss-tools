"""A module for performing 2D SED fits with bagpipes."""

import inspect
from functools import partial
from os import PathLike
from pathlib import Path
from typing import Callable

import numpy as np
from bagpipes_extended.pipeline import generate_fit_params, load_photom_bagpipes
from bagpipes_extended.sed.atlas_fitter import AtlasFitter, temp_chdir
from bagpipes_extended.sed.atlas_generator import AtlasGenerator
from numpy.typing import ArrayLike


class Bagpipes2D:

    def __init__(
        self,
        obj_z: float | ArrayLike,
        config: dict | PathLike | None = None,
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
        args = dict(locals())
        args.pop("self", "config")

        for k, v in args.copy().items():
            if v is None:
                args.pop(k)

        self.obj_z = obj_z

        if config is not None:
            config = self.import_config(config)
        else:
            config = {}

        self.fit_instructions = args.get(
            "fit_instructions", config.get("fit_instructions")
        )

        if self.fit_instructions is None:

            config_fit_params = config.get("fit_params", {})

            fit_params_kwargs = {
                "obj_z": config_fit_params.get("obj_z", self.obj_z),
                "z_range": config_fit_params.get("z_range", 0.005),
                "sfh_type": config_fit_params.get("sfh_type", "continuity"),
                "num_age_bins": config_fit_params.get("num_age_bins", 5),
                "min_age_bin": config_fit_params.get("min_age_bin", 20),
            }

            for k, v in fit_params_kwargs.items():
                if args.get(k) is not None:
                    fit_params_kwargs[k] = args[k]

            self.fit_instructions = generate_fit_params(**fit_params_kwargs)

        config_atlas = config.get("atlas", {})
        self.n_samples = args.get("n_samples", config_atlas.get("n_samples", 1e5))
        self.remake_atlas = args.get(
            "remake_atlas", config_atlas.get("remake_atlas", False)
        )
        self.n_cores = args.get("n_cores", config_atlas.get("n_cores", 4))

        self.overwrite_fit = args.get(
            "overwrite_fit", config_atlas.get("overwrite_fit", False)
        )

        config_files = config.get("files", {})
        self.root_dir = Path(args.get("root_dir", config_files.get("root_dir", ".")))
        self.out_dir = self.root_dir / args.get(
            "out_dir", config_files.get("out_dir", ".")
        )
        self.out_dir.mkdir(exist_ok=True, parents=True)

        self.pipes_dir = self.out_dir / "pipes"
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

        return

    def import_config(self, config: dict | PathLike):
        """
        Load a configuration file.

        Parameters
        ----------
        config : dict or PathLike
            Either a dictionary, ot the TOML-formatted configuration file
            containing the parameters to use for the fit.
        """

        if isinstance(config, dict):
            pass
        elif isinstance(config, PathLike) or isinstance(config, str):
            import tomllib

            with open(config, "rb") as f:
                config = tomllib.load(f)
        else:
            raise TypeError("Cannot import config.")

        return config

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
            Path(__file__).parent.parent / "data" / "filter_throughputs"
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
        n_samples: int | float = 1e6,
        remake_atlas: bool = False,
        n_cores_atlas: int | None = None,
    ) -> PathLike:
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
            explicity sets the redshift range to use for fitting. If
            ``None`` (default), this will be set to ``self.obj_z``.
        n_samples : int | float, optional
            The number of samples to generate. By default ``1e6``. A
            useful number will typically be ``>10^5``.
        remake_atlas : bool, optional
            If ``True``, any existing model atlas with the same name will
            be recreated and overwritten. By default ``False``.
        n_cores_atlas : int, optional
            The number of processes to use when generating the model grid.
            If set to ``0``, the code will run on a single process. If set
            to an integer less than 0, this will run on the number of
            cores returned by  `multiprocessing.cpu_count`. By default,
            ``4`` processes will be used.

        Returns
        -------
        PathLike
            The location of the model atlas.
        """

        if atlas_name is None:
            self.atlas_name = (
                f"z_{self.fit_instructions["redshift"][0]}_"
                f"{self.fit_instructions["redshift"][-1]}_"
                f"{self.n_samples:.2E}"
            )
        else:
            self.atlas_name = atlas_name
        self.atlas_path = self.atlas_dir / f"{self.atlas_name}.hdf5"

        if not self.atlas_path.is_file() or self.remake_atlas:

            with AtlasGenerator(
                fit_instructions=self.fit_instructions,
                filt_list=self.filter_list,
                phot_units="ergscma",
            ) as atlas_gen:

                atlas_gen.gen_samples(
                    n_samples=self.n_samples,
                    seed=self.seed,
                    parallel=self.n_cores,
                )

                atlas_gen.write_samples(filepath=self.atlas_path)

        return self.atlas_path

    # def run(self):

    #     self.atlas_path = self.gen_atlas(**self.sed_fit_kwargs)
    #     self.fit_atlas(
    #         self.atlas_path,
    #         self.binned_data_path,
    #         overwrite_fit=self.overwrite_atlas_fit,
    #         n_cores=self.n_cores_atlas_fit,
    #         z_range=self.sed_fit_kwargs["z_range"],
    #     )

    def fit_atlas(
        self,
        # atlas_path: PathLike,
        binned_data_path: PathLike,
        binned_data_hdu: str | int | None = "PHOT_CAT",
        bagpipes_atlas_params: dict | None = None,
        load_fn: Callable | None = None,
        overwrite_fit: bool = False,
        id_colname: str = "bin_id",
        n_cores: int = 4,
        obj_z: float | ArrayLike | None = None,
        z_range: float = 0.005,
    ):
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
            ``self.bagpipes_atlas_params`` will be used.
        load_fn : Callable | None, optional
            A function which takes the ID as an argument and returns the
            model photometry. This should be in the form of an array with
            a column of fluxes in microjanskys and a column of flux errors
            in the same units. If ``None`` (default),
            `~bagpipes_extended.pipeline.load_photom_bagpipes` will be
            used.
        overwrite_fit : bool, optional
            If ``True``, then any existing posterior distributions and
            output catalogues will be overwritten. By default ``False``.
        id_colname : str, optional
            The name of the column in the photometric catalogue containing
            the bin ID, by default ``"bin_id"``.
        n_cores : int, optional
            The number of processes to use when fitting the catalogue.
            If set to ``0``, the code will run on a single process. If set
            to an integer less than 0, this will run on the number of
            cores returned by  `multiprocessing.cpu_count`. By default,
            ``4`` processes will be used.
        obj_z : float | ArrayLike | None, optional
            This can be used to override the redshift used for fitting, in
            case of a mismatch between the model atlas and the object of
            interest. See `~wfss_tools.grism.MultiRegionFit.gen_atlas`
            for more details.
        z_range : float, optional
            As above.
        """

        self.run_name = f"{str(Path(binned_data_path).stem)}_SED_fit"
        catalogue_out_path = self.pipes_dir.parent / f"{self.run_name}.fits"

        if (not catalogue_out_path.is_file()) or overwrite_fit:

            with temp_chdir(self.pipes_dir):

                if load_fn is None:

                    load_fn = partial(
                        load_photom_bagpipes,
                        phot_cat=binned_data_path,
                        cat_hdu_index=binned_data_hdu,
                    )

                obs_table = Table.read(binned_data_path, hdu=binned_data_hdu)
                cat_IDs = np.array(obs_table[id_colname])

                assert (self.atlas_path is not None) and (self.atlas_path.is_file()), (
                    "The model atlas has not been generated. Please run "
                    "`Bagpipes2D.gen_atlas()` before attempting to fit observations."
                )

                with AtlasFitter(
                    fit_instructions=self.fit_instructions,
                    atlas_path=self.atlas_path,
                    out_path=self.pipes_dir.parent,
                    overwrite=self.overwrite_fit,
                ) as atlas_fitter:

                    atlas_fitter.fit_catalogue(
                        IDs=cat_IDs,
                        load_data=load_fn,
                        spectrum_exists=False,
                        make_plots=False,
                        cat_filt_list=self.filter_list,
                        run=self.run_name,
                        parallel=self.n_cores,
                        redshifts=self.obj_z if obj_z is None else obj_z,
                        # redshift_range=self.z_range,
                        n_posterior=500,
                    )
            # else:
            #     fit.cat = Table.read(catalogue_out_path)

            self.sed_fit_cat_path = self.pipes_dir.parent / f"{self.run_name}.fits"


if __name__ == "__main__":

    bagpipes_2d = Bagpipes2D(
        1.34,
        config="/media/sharedData/python/py3.13_PIE/code/wfss-tools/src/wfss_tools/sed/config_v9-TEST.toml",
        out_dir="glass_niriss_bcgs/reduction_v9-TEST_v2",
    )

    phot_cat_path = Path(
        "/media/sharedData/data/2025_12_06_glass-a2744/glass_niriss_bcgs/"
        "reduction_v9-TEST/binned_data/3070_colour_3_10_jwst-nircam-f150w_data.fits"
    )
    from astropy.table import Table

    phot_cat = Table.read(phot_cat_path)

    filt_names = [c.removesuffix("_var") for c in phot_cat.colnames if "_var" in c]

    bagpipes_2d.load_filters_from_names(filter_names=filt_names)
    bagpipes_2d.gen_atlas()

    bagpipes_2d.fit_atlas(binned_data_path=phot_cat_path)
