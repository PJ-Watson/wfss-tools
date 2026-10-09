"""
Functions and classes related to multi-region grism fitting.
"""

import multiprocessing
import os
import shutil
import sys
import traceback
from collections.abc import Callable
from copy import deepcopy
from functools import partial
from multiprocessing import Lock, Manager, Pool, cpu_count, shared_memory
from multiprocessing.managers import SharedMemoryManager
from os import PathLike
from pathlib import Path
from time import time

import h5py
import numpy as np
import scipy.optimize
import wfss_tools
from astropy.io import fits
from astropy.nddata import block_reduce
from astropy.table import Table
from astropy.wcs import WCS
from bagpipes_extended.pipeline import generate_fit_params, load_photom_bagpipes
from bagpipes_extended.sed.atlas_fitter import AtlasFitter
from bagpipes_extended.sed.atlas_generator import AtlasGenerator
from grizli import utils as grizli_utils
from grizli.multifit import MultiBeam, drizzle_to_wavelength
from numpy.typing import ArrayLike
from reproject import reproject_interp
from tqdm import tqdm
from wfss_tools.grism.bagpipes_utils import BagpipesTemplateSampler
from wfss_tools.grism.fitting_tools import CDNNLS, fennls, fnnls

# from wfss_tools.grism.samplers import GrizliTemplateSampler
from wfss_tools.grism.specgen import (
    CLOUDY_LINE_MAP,
    check_coverage,
)
from wfss_tools.grism.utils import (
    LINE_UP,
    align_direct_images,
    gen_stacked_beams,
    log_with_offset,
)
from wfss_tools.pipeline.reduction import recursive_merge
from wfss_tools.sed.binning import bin_and_save

"""
TODO: remove logic from _gen functions. Instead create template IDs as
`{seg_id}-{model_id}`.
Simplify model generation and combine output table columns into "tempID".
"""

from wfss_tools.grism import float_dtype

__all__ = ["MultiRegionFit", "DEFAULT_PLINE"]

DEFAULT_PLINE = {
    "pixscale": 0.06,
    "pixfrac": 1.0,
    "size": 5,
    "kernel": "lanczos3",
}


class MultiRegionFit:
    """
    A multi-region version of `grizli.multifit.MultiBeam`.

    Parameters
    ----------
    config_path : PathLike
        The TOML-formatted configuration file containing the parameters to
        use for the fit.
    obj_id : int
        The object ID number to fit.
    obj_z : float
        The redshift at which the object should be fitted.
    run_all : bool, optional
        If ``True`` (default), the fit will proceed automatically based on
        the setup described in the configuration file. If ``False``, the
        configuration file will be parsed, but the individual fitting
        methods must be called manually.
    """

    def __init__(
        self,
        config_path: PathLike,
        obj_id: int,
        obj_z: float,
        run_all: bool = True,
    ):

        self.obj_id = obj_id
        self.obj_z = obj_z

        self.smm = SharedMemoryManager()
        self.smm.start()

        self.import_config(config_path)

        if run_all:
            self.run_all()

    def run_all(self):
        """
        Run the full fit based on the supplied configuration file.
        """

        self.beam_path, self.binned_data_path = self.gen_aligned_photometry(
            self.binning_kwargs,
            use_stacks=self.use_stacks,
            **self.multibeam_kwargs,
        )
        self.atlas_path = self.gen_atlas(**self.sed_fit_kwargs)
        self.fit_atlas(
            self.atlas_path,
            self.binned_data_path,
            overwrite_fit=self.overwrite_atlas_fit,
            n_cores=self.n_cores_atlas_fit,
            z_range=self.sed_fit_kwargs["z_range"],
        )

        self.MB = MultiBeam(beams=str(self.beam_path), **self.multibeam_kwargs)
        self.ra, self.dec = self.MB.ra, self.MB.dec

        self.regions_phot_cat = Table.read(self.binned_data_path, "PHOT_CAT")
        self.n_regions = len(self.regions_phot_cat)

        with fits.open(self.binned_data_path) as hdul:
            self.regions_seg_map = hdul["SEG_MAP"].data.copy()
            self.regions_seg_hdr = hdul["SEG_MAP"].header.copy()
            self.regions_seg_wcs = WCS(self.regions_seg_hdr)

        self.regions_seg_ids = np.asarray(self.regions_phot_cat["bin_id"], dtype=int)

        self.fit_at_z(self.obj_z, **self.grism_fit_kwargs)

    def import_config(self, config_path: PathLike):
        """
        Parse a configuration file and set class attributes accordingly.

        Parameters
        ----------
        config_path : PathLike
            The TOML-formatted configuration file containing the parameters to
            use for the fit.
        """

        import tomllib
        import warnings

        import yaml

        with open(config_path, "rb") as f:
            config = tomllib.load(f)

        self.root_dir = Path(config["files"].get("root_dir", "/"))
        self.out_dir = self.root_dir / config["files"].get("out_dir", ".")
        self.out_dir.mkdir(exist_ok=True, parents=True)
        self.extractions_dir = self.root_dir / config["files"].get(
            "extractions_dir", "."
        )

        _info_dict = config["files"].get("info_dict", "")
        if not _info_dict:
            raise ValueError("`info_dict` is not present in the supplied config file.")

        with open(self.root_dir / _info_dict, "r") as file:
            self.info_dict = yaml.safe_load(file)

        self.pipes_dir = self.out_dir / "sed_fitting" / "pipes"
        self.pipes_dir.mkdir(exist_ok=True, parents=True)
        if not config["files"].get("atlas_dir", ""):
            self.atlas_dir = self.pipes_dir / "atlases"
        else:
            self.atlas_dir = self.root_dir / config["files"]["atlas_dir"]
        self.atlas_dir.mkdir(exist_ok=True, parents=True)

        self.binning_kwargs = {
            "bin_scheme": config["SED"].get("bin_scheme", "colour"),
            "target_sn": config["SED"].get("target_sn", 10),
            "bin_diameter": config["SED"].get("bin_diameter", 3),
            "sn_filter": config["SED"]["sn_filter"],
            **config["SED"].get("bin_kwargs", {}),
        }

        self.sed_fit_kwargs = {
            "z_range": config["SED"].get("z_range", 0.005),
            "sfh_type": config["SED"].get("sfh_type", "continuity"),
            "min_age_bin": config["SED"].get("min_age_bin", 20),
            "num_age_bins": config["SED"].get("num_age_bins", 5),
            "n_samples": config["SED"].get("n_samples", 1e6),
            "remake_atlas": config["SED"].get("remake_atlas", False),
            "n_cores_atlas": config["SED"].get("n_cores_atlas", 4),
        }

        self.n_cores_atlas_fit = config["SED"].get("n_cores_fit", 4)
        self.overwrite_atlas_fit = config["SED"].get("overwrite_fit", False)

        self.field_name = config["grism"]["field_name"]

        self.use_stacks = config["grism"].get("use_stacks", True)

        self.stack_beam_kwargs = config["grism"].get("stack_beam_kwargs", {})

        multibeam_kwargs = config["grism"].get("multibeam_kwargs", {})

        default_multib_kwargs = {
            "min_mask": 0.0,
            "min_sens": 0.0,
            "mask_resid": False,
            "verbose": False,
            "fcontam": 0.2,
            "group_name": self.field_name,
        }
        self.multibeam_kwargs = recursive_merge(default_multib_kwargs, multibeam_kwargs)

        self.grism_fit_kwargs = {
            "fit_background": config["grism"].get("fit_background", True),
            "poly_order": config["grism"].get("poly_order", 0),
            "n_samples": config["grism"].get("n_region_samples", 3),
            "n_iters": config["grism"].get("n_iters", 10),
            "bad_pa_threshold": config["grism"].get("bad_pa_threshold", 1.6),
            "spec_wavs": config["grism"].get("spec_wavs", None),
            "oversamp_factor": config["grism"].get("oversamp_factor", 1),
            "veldisp": config["grism"].get("veldisp", 50),
            "out_dir": config["grism"].get("out_dir", "multiregion"),
            "temp_dir": config["grism"].get("temp_dir", None),
            "memmap": config["grism"].get("memmap", False),
            "cpu_count": config["grism"].get("cpu_count", -1),
            "overwrite": config["grism"].get("overwrite", False),
            "use_lines": config["grism"].get("use_lines", CLOUDY_LINE_MAP),
            "save_lines": config["grism"].get("save_lines", True),
            "save_stacks": config["grism"].get("save_stacks", True),
            "pline": config["grism"].get("pline", DEFAULT_PLINE),
            "seed": config["grism"].get("seed", 2744),
            "nnls_method": config["grism"].get("nnls_method", "scipy"),
            "nnls_iters": config["grism"].get("nnls_iters", 10),
            "nnls_tol": config["grism"].get("nnls_tol", 1e-5),
            "n_shifted": config["grism"].get("n_shifted", 2),
            "n_shifted_samples": config["grism"].get("n_shifted_samples", 1),
            "cache_spec": config["grism"].get("cache_spec", False),
        }

    def fit_atlas(
        self,
        atlas_path: PathLike,
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
        atlas_path : PathLike
            The location of the previously generated model grid.
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

        self.run_name = str(Path(binned_data_path).stem).removesuffix("_data")
        catalogue_out_path = self.pipes_dir.parent / f"{self.run_name}.fits"

        if (not catalogue_out_path.is_file()) or overwrite_fit:

            if bagpipes_atlas_params is None:
                bagpipes_atlas_params = self.bagpipes_atlas_params

            os.chdir(self.pipes_dir)

            if load_fn is None:

                load_fn = partial(
                    load_photom_bagpipes,
                    phot_cat=binned_data_path,
                    cat_hdu_index=binned_data_hdu,
                )

            obs_table = Table.read(binned_data_path, hdu=binned_data_hdu)
            cat_IDs = np.array(obs_table[id_colname])

            with AtlasFitter(
                fit_instructions=bagpipes_atlas_params,
                atlas_path=atlas_path,
                out_path=self.pipes_dir.parent,
                overwrite=overwrite_fit,
            ) as atlas_fitter:

                atlas_fitter.fit_catalogue(
                    IDs=cat_IDs,
                    load_data=load_fn,
                    spectrum_exists=False,
                    make_plots=False,
                    cat_filt_list=self.filter_list,
                    run=self.run_name,
                    parallel=n_cores,
                    redshifts=self.obj_z if obj_z is None else obj_z,
                    redshift_range=z_range,
                    n_posterior=500,
                )
        # else:
        #     fit.cat = Table.read(catalogue_out_path)

        self.sed_fit_cat_path = self.pipes_dir.parent / f"{self.run_name}.fits"

    def gen_atlas(
        self,
        obj_z: float | ArrayLike | None = None,
        z_range: float = 0.005,
        num_age_bins: int = 5,
        min_age_bin: float = 20,
        sfh_type: str = "continuity",
        n_samples: int | float = 1e6,
        remake_atlas: bool = False,
        n_cores_atlas: int = 4,
    ) -> PathLike:
        """
        Generate the fit parameters and model grid.

        This method generates a dictionary of fit parameters for Bagpipes,
        as well as a large model grid to speed up the SED fitting.

        Parameters
        ----------
        obj_z : float | ArrayLike | None, optional
            The redshift of the object to fit. If a scalar value is
            passed, and ``z_range==0.0``, the object will be fit to a
            single redshift value. If ``z_range!=0.0``, this will be the
            centre of the redshift window. If an array is passed, this
            explicity sets the redshift range to use for fitting. If
            ``None`` (default), this will be set to ``self.obj_z``.
        z_range : float, optional
            The maximum redshift range to search over, by default 0.005.
            To fit to a single redshift, pass a single value for
            ``obj_z``, and set ``z_range=0.0``. If ``obj_z`` is
            ``ArrayLike``, this parameter is ignored.
        num_age_bins : int, optional
            The number of age bins to fit, each of which will have a
            constant star formation rate following Leja+19. By default,
            ``5`` bins are generated.
        min_age_bin : float, optional
            The minimum age to use for the continuity SFH in Myr, i.e. the
            first bin will range from ``(0,min_age_bin)``. By default 20.
        sfh_type : str, optional
            The type of SFH prior to generate. Currently supports
            ``"continuity"`` (Leja+19, fixed age bins),
            ``"continuity_varied_z"`` (Leja+19, only the youngest age bin
            is fixed), and ``"dblplaw"``.
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

        default_filter_dir = (
            Path(wfss_tools.__file__).parent / "data" / "filter_throughputs"
        )

        # Create the filter directory; populate as needed
        filter_dir = self.pipes_dir / "filter_throughputs"
        filter_dir.mkdir(exist_ok=True, parents=True)
        for file in default_filter_dir.glob("*.txt"):
            shutil.copy(file, filter_dir)

        # Create a list of the filters used in our data
        self.filter_list = []
        for key in self.info_dict.keys():
            self.filter_list.append(str(filter_dir / f"{key}.txt"))

        self.bagpipes_atlas_params = generate_fit_params(
            obj_z=self.obj_z if obj_z is None else obj_z,
            z_range=z_range,
            num_age_bins=num_age_bins,
            sfh_type=sfh_type,
            min_age_bin=min_age_bin,
        )

        self.atlas_run_name = (
            f"z_{self.bagpipes_atlas_params["redshift"][0]}_"
            f"{self.bagpipes_atlas_params["redshift"][1]}_"
            f"{n_samples:.2E}"
        )
        atlas_path = self.atlas_dir / f"{self.atlas_run_name}.hdf5"

        if not atlas_path.is_file() or remake_atlas:

            with AtlasGenerator(
                fit_instructions=self.bagpipes_atlas_params,
                filt_list=self.filter_list,
                phot_units="ergscma",
            ) as atlas_gen:

                atlas_gen.gen_samples(n_samples=n_samples, parallel=n_cores_atlas)

                atlas_gen.write_samples(filepath=atlas_path)

        return atlas_path

    def gen_aligned_photometry(
        self,
        binning_kwargs: dict,
        use_stacks: bool = True,
        beams_path: PathLike | None = None,
        img_cutout: int = 500,
        stack_beam_kwargs: dict = {},
        **multibeam_kwargs,
    ) -> tuple[PathLike, PathLike]:
        """
        Align photometric data to the direct image in an extracted beam.

        Parameters
        ----------
        binning_kwargs : dict
            Any arguments to pass to
            `~wfss_tools.sed.binning.bin_and_save`.
        use_stacks : bool, optional
            Whether to fit to individual beams, or beams stacked by filter
            and grism. By default ``True``.
        beams_path : PathLike | None, optional
            The location of a ``*beams.fits`` file to use for fitting. If
            ``None`` (default), this will be selected automatically based
            on the ``obj_id`` and directory structure specified in the
            configuration file.
        img_cutout : int, optional
            Make a slice of the original image with size in pixels
            ``[-cutout,+cutout]`` around the centre of the object, before
            alignment. By default, ``cutout=500``.
        stack_beam_kwargs : dict, optional
            Any additional parameters to pass through to
            `~wfss_tools.grism.utils.gen_stacked_beams`.
        **multibeam_kwargs : dict, optional
            Any additional parameters to pass through to
            `grizli.multifit.MultiBeam`.

        Returns
        -------
        new_beam_path : PathLike
            The location of the ``*beams.fits`` used for alignment. If
            ``use_stacks==True``, this will be a stacked version of the
            input file.
        binned_data_path : PathLike
            The location of the binned data in FITS format. This contains
            both the segmentation map and the binned photometric
            catalogue.

        Raises
        ------
        IOError
            If no ``*beams.fits`` file can be found, this method will
            raise an error.
        """

        binned_data_dir = self.out_dir / "binned_data"
        binned_data_dir.mkdir(exist_ok=True, parents=True)

        new_beam_loc = (
            binned_data_dir / f"{self.field_name}_{self.obj_id:0>5}.beams.fits"
        )

        # Give a descriptive name for the binned data
        binned_name = (
            f"{self.obj_id}_{binning_kwargs["bin_scheme"]}_"
            f"{binning_kwargs["bin_diameter"]}_{binning_kwargs["target_sn"]}"
            f"_{binning_kwargs["sn_filter"]}"
        )
        binned_data_path = binned_data_dir / f"{binned_name}_data.fits"

        if not binned_data_path.is_file():
            try:
                multib = MultiBeam(
                    str(new_beam_loc),
                    **multibeam_kwargs,
                )
            except:

                if beams_path is None:

                    beams_path = [
                        *self.extractions_dir.glob(f"**/*{self.obj_id:0>5}.beams.fits")
                    ]
                    if len(beams_path) >= 1:
                        beams_path = [str(b) for b in beams_path]
                    else:
                        raise IOError(
                            f"Original beams file does not exist in {self.extractions_dir}"
                        )

                if multibeam_kwargs is None:
                    multibeam_kwargs = self.multibeam_kwargs

                if len(stack_beam_kwargs) == 0:
                    stack_beam_kwargs = self.stack_beam_kwargs

                if use_stacks:
                    multib = gen_stacked_beams(
                        beams_path,
                        **stack_beam_kwargs,
                        **multibeam_kwargs,
                    )
                else:
                    multib = MultiBeam(
                        beams_path,
                        **multibeam_kwargs,
                    )

                # Write the realigned and (stacked?) beam to a file
                beam_hdul = multib.write_master_fits(get_hdu=True)
                beam_hdul.writeto(new_beam_loc, overwrite=True)

            # Align all images to the new beam
            aligned_info_dict = align_direct_images(
                multib.beams[0],
                info_dict=self.info_dict,
                out_dir=binned_data_dir / f"{self.obj_id:0>5}",
                overwrite=False,
                cutout=img_cutout,
            )

            _seg = fits.getdata(new_beam_loc, "SEG")
            bin_and_save(
                obj_id=self.obj_id,
                out_dir=binned_data_dir,
                seg_map=_seg,
                info_dict=aligned_info_dict,
                binned_name=binned_name,
                **binning_kwargs,
            )

        return new_beam_loc, binned_data_path

    def add_pipes_info(self, header: fits.Header) -> fits.Header:
        """
        Update a header with information about the 2D SED fitting.

        Parameters
        ----------
        header : fits.Header
            The original header.

        Returns
        -------
        fits.Header
            The updated header.
        """
        header["MRBPRUN"] = (
            str(self.run_name),
            "The name of the bagpipes run used to generate the prior templates.",
        )
        header["MRBPPCAT"] = (
            str(self.binned_data_path),
            "The binned photometric catalogue and segmentation map used as input for bagpipes.",
        )
        header["MRBPFCAT"] = (
            str(self.sed_fit_cat_path),
            "The bagpipes output fit catalogue.",
        )
        return header

    def fit_at_z(
        self,
        z: float = 0.0,
        fit_background: bool = True,
        poly_order: int = 0,
        n_samples: int = 3,
        n_iters: int = 10,
        bad_pa_threshold: float | None = 1.6,
        spec_wavs: ArrayLike | None = None,
        oversamp_factor: int = 1,
        veldisp: float = 50,
        direct_images: None = None,
        out_dir: PathLike | None = None,
        temp_dir: PathLike | None = None,
        memmap: bool = False,
        cpu_count: int = -1,
        overwrite: bool = False,
        use_lines: dict = CLOUDY_LINE_MAP,
        save_lines: bool = True,
        save_stacks: bool = True,
        pline: dict = DEFAULT_PLINE,
        seed: int = 2744,
        nnls_method: str = "scipy",
        nnls_iters: int = 100,
        nnls_tol: float = 1e-5,
        n_shifted: int = 2,
        n_shifted_samples: int = 1,
        cache_spec: bool = False,
    ):
        """
        Fit the object at a specified redshift.

        Parameters
        ----------
        z : float, optional
            The redshift at which the object will be fitted, by default 0.
        fit_background : bool, optional
            Fit a constant background level, by default ``True``.
        poly_order : int, optional
            Fit a polynomial function to the spectrum, with a default
            order of ``0``.
        n_samples : int, optional
            The number of samples to draw from the joint posterior
            distributions in each region, by default ``3``.
        n_iters : int, optional
            The number of iterations to perform when fitting, by default
            ``10``.
        bad_pa_threshold : float | None, optional
            The threshold for identifying bad PAs before fitting. By
            default ``1.6``, if ``None`` all beams will be used.
        spec_wavs : ArrayLike | None, optional
            The wavelength sampling to use when generating the template
            spectra from the `bagpipes` posterior distributions. The
            default value of ``None`` sets this to the
            :math:`0.96 - 2.3\\mu\\rm{m}` range in :math:`45\\mathring{A}`
            steps.
        oversamp_factor : int, optional
            The factor by which the region segmentation map is oversampled
            before reprojecting to the beam coordinate system. This
            significantly slows down the model generation, but is
            essential to ensure that the template spectra correspond to
            the correct pixels in the NIRISS detector frame, and defaults
            to a factor of ``1``.
        veldisp : float, optional
            The velocity dispersion of the template spectra in km/s, by
            default ``500``.
        direct_images : _type_, optional
            WIP, may allow for changing beam direct images at some point.
            By default ``None``.
        out_dir : PathLike | None, optional
            Where the output files will be written. If ``None`` (default),
            files will be written to ``self.out_dir/multiregion``.
        temp_dir : PathLike | None, optional
            The temporary directory to use for memmapped files (if
            ``memmap==True``). If ``None`` (default), the current working
            directory will be used.
        memmap : bool, optional
            Whether to use a memmap to store large files. By default
            ``False``. If ``True``, the large model array will be written
            to a temporary array on disk.
        cpu_count : int, optional
            The number of CPUs to use for multiprocessing, by default -1.
        overwrite : bool, optional
            If ``True``, overwrite any existing fit. By default ``False``,
            and will attempt to load a previous fit.
        use_lines : ArrayLike, optional
            A list of lines, for which a 2D map will be generated based on
            the multi-region fit. Each item in the list should be a
            ``dict``, containing the following keys:

            * ``"cloudy"`` : The name of one or more lines following the
              ``Cloudy`` nomenclature
              (`Ferland+17 <https://ui.adsabs.harvard.edu/abs/2017RMxAA..53..385F/abstract>`__).
            * ``"grizli"`` : The name of the emission line in `grizli`, or
              any other desired name.
            * ``"wave"`` : The rest-frame vacuum wavelength of the line.

        save_lines : bool, optional
            Save the drizzled emission line maps, matching the `grizli`
            output format. By default ``True``.
        save_stacks : bool, optional
            Save the stacked beams, full models, and continuum models. The
            output format differs from grizli in that these stacks are not
            drizzled to account for the subpixel shifts. By default
            ``True``.
        pline : dict, optional
            Parameters for generating the drizzled emission line maps.
            Defaults to `~wfss_tools.grism.DEFAULT_PLINE`.
        seed : int | None, optional
            The seed for the random sampling, by default 2744. If None,
            then a new seed will be generated each time this method is
            called.
        nnls_method : str, optional
            The method to use for finding the best-fit coefficients for
            the set of templates. Must be one of "scipy", "numba", or
            "adelie" (ordered in increasing speed). By default, "scipy"
            will be used.
        nnls_iters : int or ArrayLike, optional
            The maximum number of iterations to attempt if the NNLS
            solution has not converged to the specified tolerance. If more
            than one value is passed, a two-stage fit will be run, whereby
            the best-fit solution from the first ``n_iters`` attempts will
            be re-fit with ``nnls_iters[0]`` iterations. This is only
            relevant if ``nnls_method != "scipy"``.
        nnls_tol : float or ArrayLike, optional
            The desired tolerance for the NNLS solver. As with
            ``nnls_iters``, the second value can be used to perform a more
            precise fit after the initial ``n_iters`` attempts.
        n_shifted : int or None, optional
            This allows for drawing additional posterior samples from
            other regions of the object. Regions will be selected
            randomly, with no preference as to spatial or spectral
            coherence (they are selected by shifting each segmentation map
            id when generating the models, hence the name). This reduces
            the chance of template mismatch based on the SED fit. By
            default, 2 additional regions will be used.
        n_shifted_samples : int or None, optional
            This determines the number of samples drawn from each of the
            additional regions (i.e. the total number of samples is given
            by ``n_samples + n_shifted * n_shifted_samples``). By default,
            only 1 sample is drawn from each extra region.
        cache_spec : bool, optional
            Pre-generate the spectra before forward modelling. Can give a
            large speedup if running for many iterations, at the cost of
            additional disk space.

        Returns
        -------
        tuple
            Exact form of return still WIP.
        """

        if nnls_method == "adelie":
            try:
                # Reset core binding, interferes with python process affinity
                os.environ["OMP_PROC_BIND"] = "FALSE"

                import adelie

                log_with_offset("Using `adelie` solver.", blank_lines=-1)
                HAS_ADELIE = True
            except:
                HAS_ADELIE = False

        nnls_iters = np.atleast_1d(nnls_iters).astype(int)
        nnls_tol = np.atleast_1d(nnls_tol)
        TWO_STAGE = (len(nnls_iters) > 1) | (len(nnls_tol) > 1)

        if spec_wavs is None:
            spec_wavs = np.arange(10000.0, 23000.0, 22.5)

        self.spec_wavs = spec_wavs

        self.seed = seed

        if memmap:
            if temp_dir is None:
                temp_dir = Path.cwd()
            else:
                temp_dir = Path(temp_dir)
                temp_dir.mkdir(exist_ok=True, parents=True)

        if not out_dir:
            multireg_out_dir = self.out_dir / "multiregion"
        else:
            multireg_out_dir = self.out_dir / Path(out_dir)

        multireg_out_dir.mkdir(exist_ok=True, parents=True)

        if bad_pa_threshold is not None:
            out = self.MB.check_for_bad_PAs(
                chi2_threshold=bad_pa_threshold,
                poly_order=1,
                reinit=True,
                fit_background=True,
            )
            fit_log, keep_dict, has_bad = out
            if has_bad:
                print(f"Has bad PA!  Final list: {keep_dict}\n{fit_log}")

        self.MB.init_poly_coeffs(poly_order=poly_order)

        if fit_background:
            self.fit_bg = True
            A = np.vstack((self.MB.A_bg, self.MB.A_poly))
        else:
            self.fit_bg = False
            A = self.MB.A_poly * 1

        self.template_sampler = BagpipesTemplateSampler(
            posterior_dir=(self.pipes_dir / "posterior" / self.run_name),
            seed=seed,
            cpu_count=cpu_count,
            cache_all_spectra=cache_spec,
            veldisp=veldisp,
            spec_wavs=self.spec_wavs,
        )
        # self.template_sampler = GrizliTemplateSampler(
        #     posterior_dir=(self.pipes_dir / "posterior" / self.run_name),
        #     seed=seed,
        #     cpu_count=cpu_count,
        #     cache_all_spectra=cache_spec,
        #     veldisp=veldisp,
        #     spec_wavs=self.spec_wavs,
        #     redshift=self.obj_z,
        # )
        # n_samples = self.template_sampler.Ntemp
        # n_shifted = 0
        # n_shifted_samples = 0

        self.n_samples = n_samples
        self.n_shifted = n_shifted
        self.n_shifted_samples = n_shifted_samples

        # Try to allow for both memory and file-backed multiprocessing of
        # large arrays

        for ib, beam in enumerate(self.MB.beams):
            repr_seg_map = reproject_interp(
                (self.regions_seg_map, self.regions_seg_wcs),
                beam.direct.wcs,
                beam.beam.sh,
                return_footprint=False,
                order=0,
            )
            repr_seg_map[~np.isfinite(repr_seg_map)] = 0
            self.MB.beams[ib].regions_seg_map = repr_seg_map

        # The total number of templates
        NTEMP = self.n_regions * n_samples
        if self.n_shifted > 0:
            NTEMP += self.n_regions * self.n_shifted * self.n_shifted_samples

        # Non-template rows in A (background and polynomial components)
        self.temp_offset = A.shape[0]

        # Set up both the shared memory and process pool
        self.initialise_shared_memory(
            self.n_samples + (self.n_shifted * self.n_shifted_samples)
        )

        self.initialise_process_pool(cpu_count)

        self.stacked_A[: self.temp_offset] = A[:, self.MB.fit_mask]

        # Allow for background fitting by including an offset
        if fit_background:
            pedestal = 0.04
        else:
            pedestal = 0.0

        y = self.MB.scif[self.MB.fit_mask] + pedestal
        y *= self.MB.sivarf[self.MB.fit_mask]

        y = y.astype(float_dtype)

        sivarf_masked = self.MB.sivarf[self.MB.fit_mask]

        # TODO: make the output name a parameter?
        self.output_table_path = multireg_out_dir / (
            f"{self.obj_id}_{len(np.unique(self.regions_phot_cat["bin_id"]))}"
            f"bins_{n_iters}iters_{n_samples}samples_z_{self.obj_z}_sig_{veldisp}.ecsv"
        )

        # The function to produce the templates - only a couple of
        # parameters change on each iteration.
        fwd_model_fn = partial(
            worker_forward_model,
            spec_wavs=self.spec_wavs,
            temp_offset=self.temp_offset,
            memmap=memmap,
        )

        # These column names should be fixed for all objects
        init_col_names = [
            "iteration",
            "chi2",
            "max_nnls_iters",
            "solve_nnls_iters",
            "nnls_tol",
            "model_seeds",
            "id_shifts",
            "n_shifted_samples",
            "fitting_time",
            "total_time",
            "unique_temp",
        ]

        total_iters = n_iters + int(TWO_STAGE)

        # Construct the table if it doesn't already exist
        try:
            output_table = Table.read(self.output_table_path, format="ascii.ecsv")
            assert np.logical_not(overwrite)
        except:
            output_table = Table(
                [
                    np.zeros(total_iters),
                    np.full(total_iters, np.nan),
                    *np.zeros((3, total_iters)),
                    np.zeros((total_iters, n_samples), dtype=int),
                    np.zeros((total_iters, self.n_shifted), dtype=int),
                    *np.zeros((4, total_iters)),
                    *np.zeros((self.temp_offset, total_iters)),
                    *np.zeros(
                        (
                            self.n_regions,
                            total_iters,
                            self.n_samples + (self.n_shifted * self.n_shifted_samples),
                        )
                    ),
                ],
                names=init_col_names
                + [f"base_coeffs_{b}" for b in np.arange(self.temp_offset)]
                + [f"bin_{p}" for p in self.regions_phot_cat["bin_id"]],
                dtype=[int, float, int, int, float, int, int, int, float, float, int]
                + [float] * self.temp_offset
                + [float] * self.n_regions,
            )

        # Check if seed was previously set, write to table if not
        seed = output_table.meta.get("RNGSEED", [self.seed])[0]
        output_table.meta["RNGSEED"] = (self.seed, "Random seed")

        output_table.meta["ID"] = (self.obj_id, "Object ID")
        output_table.meta["RA"] = (self.ra, "Right Ascension")
        output_table.meta["DEC"] = (self.dec, "Declination")
        output_table.meta["Z"] = (self.obj_z, "Best-fit redshift")
        output_table.meta["DOF"] = (self.MB.DoF, "Degrees of freedom (active pixels)")

        # Check if the table length matches the expected number of iterations
        n_prev_iters = np.max(
            (output_table["iteration"] + 1)[np.isfinite(output_table["chi2"])],
            initial=0,
        )

        if n_prev_iters < total_iters:

            remaining_iters = total_iters - n_prev_iters

            output_table.write(
                self.output_table_path, overwrite=True, format="ascii.ecsv"
            )

            iterations = np.arange(n_prev_iters, total_iters)

            # Pass `verbose` parameter as 0, 1, 2:
            # 0 : No info printed anywhere
            # 1 : tqdm progress bars
            # 2 : all info printed
            # Look into using logging module as well
            # for iteration in tqdm(iterations):
            #     import time
            #     time.sleep(2)
            #     continue
            for iteration in iterations:
                try:
                    curr_line = (
                        f"Minimum chi2: {np.nanmin(output_table["chi2"]):.3f}"
                        f"\t\t(Iteration {np.nanargmin(output_table["chi2"])})"
                    )
                except:
                    curr_line = "Minimum chi2: ---"
                log_with_offset("", curr_line=curr_line)

                # On the final iteration, reuse the samples from the current
                # best-fit solution
                if TWO_STAGE and (iteration == iterations[-1]):

                    best_iter = np.nanargmin(output_table["chi2"])

                    model_seeds = np.array(
                        [int(s) for s in output_table["model_seeds"][best_iter]]
                    )
                    id_shifts = np.array(
                        [int(s) for s in output_table["id_shifts"][best_iter]]
                    )
                else:
                    model_seeds, id_shifts = (
                        self.template_sampler.gen_model_seeds_from_iter(
                            iteration, self.n_samples, self.n_shifted
                        )
                    )

                log_with_offset(
                    f"Iteration {iteration}, {model_seeds=}, {id_shifts=}",
                    curr_line=curr_line,
                )
                t0 = time()

                # Generate the template spectra
                log_with_offset(f"Generating models...", curr_line=curr_line)

                self.template_sampler.gen_all_spectra_from_seeds(
                    model_seeds=model_seeds,
                    extra_region_idxs=id_shifts,
                    n_extra_samples=self.n_shifted_samples,
                    shared_memory_manger=self.smm,
                    shared_memory_name=self.shm_model_spectra.name,
                    shared_memory_shape=self.model_spectra_arr.shape,
                )

                t0p5 = time()
                log_with_offset(
                    LINE_UP + f"Generating models...    DONE in {t0p5-t0:.3f}s",
                    curr_line=curr_line,
                )

                # Generate the forward-modelled spectra
                log_with_offset(f"Dispersing spectra...", curr_line=curr_line)
                self.process_pool.starmap(fwd_model_fn, enumerate(self.regions_seg_ids))
                t1 = time()
                log_with_offset(
                    LINE_UP + f"Dispersing spectra...    DONE in {t1-t0p5:.3f}s",
                    curr_line=curr_line,
                )

                # Remove any negative or zero templates
                summed_temps = np.sum(self.stacked_A, axis=1)
                ok_temp = summed_temps > 0

                # TODO: finding which of the forward modelled templates are unique is
                # the main bottleneck here. We make the assumption that the sum is
                # unlikely to be identical between two different models.
                # print (np.isnan(self.stacked_A).sum())
                # _, unique_idxs = np.unique(self.stacked_A, axis=0, return_index=True)
                _, unique_idxs = np.unique(summed_temps, return_index=True)
                unique_temp_mask = np.isin(
                    np.arange(self.stacked_A.shape[0]), unique_idxs
                )
                ok_temp &= unique_temp_mask

                out_coeffs = np.zeros(self.stacked_A.shape[0])

                # Transpose the template array
                stacked_Ax = self.stacked_A[ok_temp].T

                stacked_Ax *= sivarf_masked[:, np.newaxis]

                # Change the max iters and tolerance for the final iteration
                if TWO_STAGE and (iteration == iterations[-1]):
                    log_with_offset("Final iteration", curr_line=curr_line)
                    _nnls_i = nnls_iters[1]
                    _nnls_t = nnls_tol[1]
                else:
                    _nnls_i = nnls_iters[0]
                    _nnls_t = nnls_tol[0]

                log_with_offset("NNLS fitting...         ", curr_line=curr_line)

                # Three different methods of fitting, each with different call
                # signatures and return values
                if nnls_method == "adelie" and HAS_ADELIE:

                    stacked_Ax_ad = adelie.matrix.dense(stacked_Ax, n_threads=cpu_count)
                    state = adelie.solver.bvls(
                        stacked_Ax_ad,
                        y,
                        lower=np.zeros(stacked_Ax.shape[-1], dtype=float_dtype),
                        upper=np.full(stacked_Ax.shape[-1], np.inf, dtype=float_dtype),
                        max_iters=_nnls_i,
                        tol=_nnls_t,
                        n_threads=cpu_count,
                    )
                    state.solve()
                    state_iters = deepcopy(state.iters)
                    coeffs = deepcopy(state.beta)
                    coeffs[: self.MB.N] -= pedestal
                    del state, stacked_Ax_ad

                elif nnls_method == "numba":

                    nnls_solver = CDNNLS(stacked_Ax, y)
                    nnls_solver.run(n_iter=_nnls_i, epsilon=_nnls_t)
                    coeffs = nnls_solver.w
                    coeffs[: self.MB.N] -= pedestal

                elif nnls_method == "fnnls":
                    coeffs = fnnls(
                        stacked_Ax,
                        y,
                        tolerance=_nnls_t,
                        max_iterations=_nnls_i,
                    )
                    coeffs[: self.MB.N] -= pedestal

                elif nnls_method == "fennls":
                    coeffs = fennls(
                        stacked_Ax,
                        y,
                        tolerance=_nnls_t,
                        max_iterations=_nnls_i,
                    )
                    coeffs[: self.MB.N] -= pedestal
                else:

                    coeffs, rnorm, info = scipy.optimize._nnls._nnls(
                        stacked_Ax, y, _nnls_i
                    )
                    coeffs[: self.MB.N] -= pedestal

                t2 = time()
                log_with_offset(
                    LINE_UP + f"NNLS fitting...         DONE in {t2-t1:.3f}s",
                    curr_line=curr_line,
                )

                out_coeffs[ok_temp] = coeffs
                stacked_modelf = np.dot(out_coeffs, self.stacked_A)
                chi2 = np.nansum(
                    (
                        self.MB.weightf[self.MB.fit_mask]
                        * (self.MB.scif[self.MB.fit_mask] - stacked_modelf) ** 2
                        * self.MB.ivarf[self.MB.fit_mask]
                    )
                )
                output_table[iteration] = [
                    iteration,
                    chi2,
                    _nnls_i,
                    state_iters if (nnls_method == "adelie" and HAS_ADELIE) else 0,
                    _nnls_t,
                    model_seeds,
                    id_shifts,
                    self.n_shifted_samples,
                    t2 - t1,
                    time() - t0,
                    ok_temp.sum(),
                    *out_coeffs[: self.temp_offset],
                    *out_coeffs[self.temp_offset :].reshape(self.n_regions, -1),
                ]

                output_table.write(self.output_table_path, overwrite=True)
                log_with_offset(
                    f"Iteration {iteration}: chi2={chi2:.3f}", curr_line=curr_line
                )

                # Reset the template arrays
                self.stacked_A[self.temp_offset :].fill(0.0)
                self.model_spectra_arr.fill(0.0)

            del stacked_Ax

        # There must be a better way to obtain the coefficients, but
        # slicing tables is not entirely straightforward
        best_iter = np.argmin(output_table["chi2"])
        out_coeffs = np.asarray(
            [
                i
                for d in output_table[best_iter][len(init_col_names) :]
                for i in np.atleast_1d(d)
            ]
        ).ravel()

        # Repopulate the background parameters for MultiBeam
        if fit_background:
            for ib, beam in enumerate(self.MB.beams):
                # for k_i, (k, v) in enumerate(beam_info.items()):
                # for ib in v["list_idx"]:
                self.MB.beams[ib].background = out_coeffs[ib]

        self.best_model_seeds = np.array(
            [int(s) for s in output_table["model_seeds"][best_iter]]
        )
        self.best_id_shifts = np.array(
            [int(s) for s in output_table["id_shifts"][best_iter]]
        )

        # Refill array with best model
        print("Calculating covariance array...")
        # shm_model_spectra_name, model_spectra_arr.shape = (
        self.template_sampler.gen_all_spectra_from_seeds(
            model_seeds=self.best_model_seeds,
            extra_region_idxs=self.best_id_shifts,
            n_extra_samples=self.n_shifted_samples,
            shared_memory_manger=self.smm,
            shared_memory_name=self.shm_model_spectra.name,
            shared_memory_shape=self.model_spectra_arr.shape,
        )
        # )

        self.process_pool.starmap(fwd_model_fn, enumerate(self.regions_seg_ids))

        ok_temp = (np.sum(self.stacked_A, axis=1) > 0) & (out_coeffs != 0)
        stacked_Ax = self.stacked_A[ok_temp, :].T * 1
        stacked_Ax *= self.MB.sivarf[self.MB.fit_mask][:, np.newaxis]

        try:
            covar = grizli_utils.safe_invert(np.dot(stacked_Ax.T, stacked_Ax))
        except:
            N = ok_temp.sum()
            covar = np.zeros((N, N))

        covard = np.sqrt(covar.diagonal())

        coeffs_errs = out_coeffs * 0.0
        coeffs_errs[ok_temp] = covard

        chi2nu = output_table["chi2"][best_iter] / (
            self.MB.DoF - output_table["unique_temp"][best_iter]
        )

        # Ensure that the array is cleaned before repopulating
        self.stacked_A[self.temp_offset :].fill(0.0)

        # Largely unmodified from the original grizli code. Included within
        # this particular class method to avoid dealing with SharedMemory
        # if save_stacks:
        print("Generating models...")

        self.template_sampler.gen_all_spectra_from_seeds(
            model_seeds=self.best_model_seeds,
            extra_region_idxs=self.best_id_shifts,
            n_extra_samples=self.n_shifted_samples,
            shared_memory_manger=self.smm,
            shared_memory_name=self.shm_model_spectra.name,
            shared_memory_shape=self.model_spectra_arr.shape,
        )

        self.process_pool.starmap(fwd_model_fn, enumerate(self.regions_seg_ids))

        masked_modelf = np.dot(out_coeffs, self.stacked_A)

        self.stacked_A[self.temp_offset :].fill(0.0)

        print("Generating nebular lines...")
        self.template_sampler.gen_emline_spectra(emline=None)

        full_temp_arr = deepcopy(self.model_spectra_arr)
        self.model_spectra_arr[:] = self.template_sampler.model_emline_spectra.reshape(
            self.model_spectra_arr.shape
        )

        self.process_pool.starmap(fwd_model_fn, enumerate(self.regions_seg_ids))

        masked_nebularf = np.dot(out_coeffs, self.stacked_A)

        masked_contf = masked_modelf - masked_nebularf

        self.model_spectra_arr[:] = full_temp_arr[:]
        del full_temp_arr

        # Reset the forward model array
        self.stacked_A[self.temp_offset :].fill(0.0)

        # Reconstruct the full flattened arrays without the fit mask
        full_modelf = np.zeros_like(self.MB.scif)
        full_modelf[self.MB.fit_mask] = masked_modelf

        full_nebularf = np.zeros_like(self.MB.scif)
        full_nebularf[self.MB.fit_mask] = masked_nebularf

        full_contf = np.zeros_like(self.MB.scif)
        full_contf[self.MB.fit_mask] = masked_contf

        # Create the FITS file
        stacked_hdul = fits.HDUList(fits.PrimaryHDU())

        for ib, shape in enumerate(self.MB.shapes):

            slice_beam = self.MB.idf == ib

            hdus = [
                fits.ImageHDU(
                    data=self.MB.scif[slice_beam].reshape(shape),
                    name="SCI",
                ),
                fits.ImageHDU(
                    data=self.MB.weightf[slice_beam].reshape(shape),
                    name="WHT",
                ),
                fits.ImageHDU(
                    data=self.MB.ivarf[slice_beam].reshape(shape),
                    name="IVAR",
                ),
                fits.ImageHDU(
                    data=self.MB.fit_mask[slice_beam].reshape(shape) * 1.0,
                    name="MASK",
                ),
                fits.ImageHDU(
                    data=full_modelf[slice_beam].reshape(shape),
                    name="MODEL",
                ),
                fits.ImageHDU(
                    data=full_contf[slice_beam].reshape(shape),
                    name="CONT",
                ),
                fits.ImageHDU(
                    data=full_nebularf[slice_beam].reshape(shape),
                    name="NEB",
                ),
            ]
            for h in hdus:
                k = f"{self.MB.beams[0].grism.pupil}-{self.MB.beams[0].grism.filter}"
                h.header["EXTVER"] = ib
                h.header["RA"] = (self.ra, "Right ascension")
                h.header["DEC"] = (self.dec, "Declination")
                h.header["GRISM"] = (k.split("-")[0], "Grism")
                h.header["CONF"] = (
                    self.MB.beams[0].beam.conf.conf_file,
                    "Configuration file",
                )
                h.header["REDSHIFT"] = (z, "Redshift used")
                h.header["CHI2"] = (
                    output_table["chi2"][best_iter],
                    "Chi^2 statistic",
                )
                h.header["DOF"] = (
                    self.MB.DoF,
                    "Degrees of freedom (active pixels)",
                )
                h.header["NTEMP"] = (
                    output_table["unique_temp"][best_iter],
                    "Number of unique templates",
                )
                h.header["CHI2NU"] = (chi2nu, "Reduced chi^2 statistic")
                h.header = self.add_pipes_info(h.header)
            stacked_hdul.extend(hdus)

        stacked_hdul.writeto(
            multireg_out_dir / f"regions_{self.obj_id:05d}_z_{self.obj_z}_stacked.fits",
            output_verify="silentfix",
            overwrite=True,
        )

        # def gen_line_maps(
        #     self,
        # ):

        if save_lines:
            line_hdu = None
            saved_lines = []

            # shm_model_spectra_name, model_spectra_arr.shape = (
            self.template_sampler.gen_all_spectra_from_seeds(
                model_seeds=self.best_model_seeds,
                extra_region_idxs=self.best_id_shifts,
                n_extra_samples=self.n_shifted_samples,
                shared_memory_manger=self.smm,
                shared_memory_name=self.shm_model_spectra.name,
                shared_memory_shape=self.model_spectra_arr.shape,
            )
            # )

            self.process_pool.starmap(fwd_model_fn, enumerate(self.regions_seg_ids))

            masked_modelf = np.dot(
                out_coeffs[self.temp_offset :], self.stacked_A[self.temp_offset :]
            )

            full_modelf = np.zeros_like(self.MB.scif)
            full_modelf[self.MB.fit_mask] = masked_modelf

            self.stacked_A[self.temp_offset :].fill(0.0)

            # for l_i, l_v in enumerate(use_lines):
            for l_i, l_v in tqdm(
                enumerate(use_lines),
                desc="Generating emission line maps",
                total=len(use_lines),
            ):

                if not check_coverage(l_v["wave"] * (1 + self.obj_z)):
                    continue

                # log_with_offset(f"Generating map for {l_v["grizli"]}")
                # print("Generating nebular lines...")
                self.template_sampler.gen_emline_spectra(emline=l_v["cloudy"])
                # self.template_sampler.gen_emline_spectra(emline=l_v["grizli"])

                self.model_spectra_arr[:].fill(0.0)

                # full_temp_arr = deepcopy(self.model_spectra_arr)
                self.model_spectra_arr[:] = (
                    self.template_sampler.model_emline_spectra.reshape(
                        self.model_spectra_arr.shape
                    )
                )

                self.process_pool.starmap(fwd_model_fn, enumerate(self.regions_seg_ids))

                # Nebular without background fitting
                masked_nebularf = np.dot(
                    out_coeffs[self.temp_offset :],
                    self.stacked_A[self.temp_offset :],
                )

                line_sn = np.nansum(
                    np.dot(
                        out_coeffs[self.temp_offset :],
                        self.stacked_A[self.temp_offset :],
                    )
                ) / np.sqrt(
                    np.nansum(
                        np.dot(
                            coeffs_errs[self.temp_offset :],
                            self.stacked_A[self.temp_offset :],
                        )
                        ** 2
                    )
                )

                masked_contf = masked_modelf - masked_nebularf

                # Reset the forward model array
                self.stacked_A[self.temp_offset :].fill(0.0)

                # Reconstruct the full flattened arrays without the fit mask
                full_nebularf = np.zeros_like(self.MB.scif)
                full_nebularf[self.MB.fit_mask] = masked_nebularf

                full_contf = np.zeros_like(self.MB.scif)
                full_contf[self.MB.fit_mask] = masked_contf

                add_hdu = None
                for continuum_temp in [True, False]:

                    for ib, shape in enumerate(self.MB.shapes):

                        slice_beam = self.MB.idf == ib
                        self.MB.beams[ib].beam.model = (
                            full_contf if continuum_temp else full_nebularf
                        )[slice_beam].reshape(shape)

                    hdu = drizzle_to_wavelength(
                        self.MB.beams,
                        ra=self.ra,
                        dec=self.dec,
                        wave=l_v["wave"] * (1 + self.obj_z),
                        fcontam=self.MB.fcontam,
                        **pline,
                    )

                    hdu[0].header["REDSHIFT"] = (self.obj_z, "Redshift used")
                    hdu[0].header["CHI2"] = (
                        output_table["chi2"][best_iter],
                        "Chi^2 statistic",
                    )
                    hdu[0].header["DOF"] = (
                        self.MB.DoF,
                        "Degrees of freedom (active pixels)",
                    )
                    hdu[0].header["NTEMP"] = (
                        output_table["unique_temp"][best_iter],
                        "Number of unique templates",
                    )
                    hdu[0].header["CHI2NU"] = (chi2nu, "Reduced chi^2 statistic")
                    hdu[0].header = self.add_pipes_info(hdu[0].header)
                    for e in [-4, -3, -2, -1]:
                        hdu[e].header["EXTVER"] = l_v["grizli"]
                        hdu[e].header["REDSHIFT"] = (self.obj_z, "Redshift used")
                        hdu[e].header["RESTWAVE"] = (
                            l_v["wave"],
                            "Line rest wavelength",
                        )

                    if add_hdu is None:
                        add_hdu = hdu
                    else:
                        hdu[-3].header["EXTNAME"] = "MODEL"
                        add_hdu.append(hdu[-3])
                        line_flux_i = np.nansum(hdu[-3].data) * 1e-17
                        line_err_i = line_flux_i / line_sn

                saved_lines.append(l_v["grizli"])

                if line_hdu is None:
                    line_hdu = add_hdu
                    line_hdu[0].header["NUMLINES"] = (
                        1,
                        "Number of lines in this file",
                    )
                else:
                    line_hdu.extend(add_hdu[-5:])
                    line_hdu[0].header["NUMLINES"] += 1

                    # Make sure DSCI extension is filled.  Can be empty for
                    # lines at the edge of the grism throughput
                    for f_i in range(hdu[0].header["NDFILT"]):
                        filt_i = hdu[0].header["DFILT{0:02d}".format(f_i + 1)]
                        if hdu["DWHT", filt_i].data.max() != 0:
                            line_hdu["DSCI", filt_i] = hdu["DSCI", filt_i]
                            line_hdu["DWHT", filt_i] = hdu["DWHT", filt_i]

                li = line_hdu[0].header["NUMLINES"]
                line_hdu[0].header["LINE{0:03d}".format(li)] = l_v["grizli"]
                line_hdu[0].header["FLUX{0:03d}".format(li)] = (
                    line_flux_i,
                    "Line flux, erg/s/cm2",
                )
                line_hdu[0].header["ERR{0:03d}".format(li)] = (
                    line_err_i,
                    "Line flux err, erg/s/cm2",
                )

            if line_hdu is not None:
                line_hdu[0].header["HASLINES"] = (
                    " ".join(saved_lines),
                    "Lines in this file",
                )

                line_wcs = WCS(line_hdu[1].header)
                segm = self.MB.drizzle_segmentation(wcsobj=line_wcs)
                seg_hdu = fits.ImageHDU(data=segm.astype(np.int32), name="SEG")
                line_hdu.insert(1, seg_hdu)

                line_hdu.writeto(
                    multireg_out_dir
                    / f"regions_{self.obj_id:05d}_z_{self.obj_z}_{pline.get("pixscale", 0.06)}arcsec.line.fits",
                    output_verify="silentfix",
                    overwrite=True,
                )

                if "DSCI" in line_hdu:

                    from grizli.fitting import show_drizzled_lines

                    # s, si = 1, line_size
                    s = 4.0e-19 / np.max(
                        [beam.beam.total_flux for beam in self.MB.beams]
                    )
                    s = np.clip(s, 0.25, 4)

                    s /= (pline.get("pixscale", 0.06) / 0.1) ** 2

                    scale_linemap = 1
                    if scale_linemap < 0:
                        s = -1

                    dscale = 1.0 / 4

                    fig = show_drizzled_lines(
                        line_hdu,
                        size_arcsec=1.6,
                        cmap="plasma_r",
                        scale=s * scale_linemap,
                        dscale=s * dscale * scale_linemap,
                        full_line_list=[
                            "Lya",
                            "OII",
                            "Hb",
                            "OIII-5007",
                            "Ha",
                            "SII",
                            "SIII-9068",
                            "SIII-9531",
                        ],
                    )
                    fig.savefig(
                        multireg_out_dir
                        / f"regions_{self.obj_id:05d}_z_{self.obj_z}_{pline.get("pixscale", 0.06)}arcsec.line.png",
                    )

        return

    def __enter__(self):
        return self

    def __del__(self):
        """
        Ensure that all attributes are correctly destroyed.
        """

        if hasattr(self, "template_sampler"):
            self.template_sampler.close()
            del self.template_sampler

        if hasattr(self, "_process_pool"):
            self._process_pool.close()
            self._process_pool.terminate()
            del self._process_pool

        if hasattr(self, "smm"):
            self.smm.shutdown()
            del self.smm

    def __exit__(self, exc_type, exc_value, traceback):
        self.__del__()

    def initialise_process_pool(self, cpu_count: int):
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
            initializer=init_forward_model,
            initargs=(
                self.shm_model_spectra.name,
                self.model_spectra_arr.shape,
                self.shm_stacked_A.name,
                self.stacked_A.shape,
                self.MB,
            ),
        )

    @property
    def process_pool(self) -> multiprocessing.Pool():
        """The pool of workers for all multiprocessing (`~multiprocessing.Pool`, read-only)."""
        return self._process_pool

    @process_pool.setter
    def process_pool(self, value: None = None):  # numpydoc ignore=GL08
        raise AttributeError(
            "`self.process_pool` cannot be set directly. Initialise this "
            "attribute using `self.initialise_process_pool(cpu_count)` instead."
        )

    def initialise_shared_memory(self, n_spec_per_region: int, memmap: bool = False):
        """
        Intitialise the shared memory arrays for multiprocessing.

        Currently constructs one array for the sampled spectra, and one
        to hold the dispersed spectra and any polynomial or background
        templates.

        Parameters
        ----------
        n_spec_per_region : int
            The number of template spectra that will be used per region.
        memmap : bool, optional
            If ``True``, the forward-modelled spectra will be placed in
            an array stored in a binary file on disk using `numpy.memmap`.
            By default ``False``, as this is much slower to access than
            arrays stored in RAM, but can be used to work around
            out of memory errors.

        Raises
        ------
        MemoryError
            If the amount of memory requested exceeds the current amount available.
        """

        # Initialise the shared memory for the sampled spectra
        model_spectra_arr_shape = (
            len(self.regions_seg_ids),
            n_spec_per_region,
            len(self.spec_wavs),
        )

        # This is the large array of models. Each row corresponds to a
        # (probably) unique template, forward-modelled across all beams,
        # and flattened.
        stacked_A_shape = (
            self.temp_offset + n_spec_per_region * self.n_regions,
            self.MB.Nmask,
        )

        if not memmap:
            # Check that there is enough free memory before trying to allocate it
            import psutil

            total_req = (
                np.prod(model_spectra_arr_shape) + np.prod(stacked_A_shape)
            ) * np.dtype(float_dtype).itemsize
            total_avail = psutil.virtual_memory().available

            if total_req > total_avail:
                raise MemoryError(
                    "The required memory would exceed the amount available. "
                    "Try passing `memmap=True`, or reducing the number of "
                    "samples and regions."
                )

        self.shm_model_spectra = self.smm.SharedMemory(
            size=np.dtype(float_dtype).itemsize * np.prod(model_spectra_arr_shape),
        )
        self.model_spectra_arr = np.ndarray(
            model_spectra_arr_shape,
            dtype=float_dtype,
            buffer=self.shm_model_spectra.buf,
        )

        # Ensure the array is blank on first run
        self.model_spectra_arr.fill(0.0)

        if memmap:
            self.stacked_A = np.memmap(
                self.temp_dir / "memmap_stacked_A.dat",
                dtype=float_dtype,
                mode="w+",
                shape=stacked_A_shape,
            )
        else:
            self.shm_stacked_A = self.smm.SharedMemory(
                size=np.dtype(float_dtype).itemsize * np.prod(stacked_A_shape)
            )
            self.stacked_A = np.ndarray(
                stacked_A_shape,
                dtype=float_dtype,
                buffer=self.shm_stacked_A.buf,
            )

        self.stacked_A.fill(0.0)


def init_forward_model(
    shared_model_spectra_name: str,
    shared_model_spectra_shape: tuple[int],
    shared_temp_arr_name: str,
    shared_temp_arr_shape: tuple[int],
    beams: MultiBeam | None = None,
):

    global shm_model_spectra, shared_model_spectra
    shm_model_spectra = shared_memory.SharedMemory(
        name=shared_model_spectra_name, create=False
    )
    shared_model_spectra = np.ndarray(
        shared_model_spectra_shape, dtype=float_dtype, buffer=shm_model_spectra.buf
    )

    global shm_temp_arr, shared_temp_arr
    shm_temp_arr = shared_memory.SharedMemory(name=shared_temp_arr_name, create=False)
    shared_temp_arr = np.ndarray(
        shared_temp_arr_shape, dtype=float_dtype, buffer=shm_temp_arr.buf
    )

    if beams is not None:
        global multibeam_object
        multibeam_object = deepcopy(beams)

    return shared_model_spectra, shared_temp_arr


def worker_forward_model(
    seg_idx: int,
    seg_id: int,
    spec_wavs: np.ndarray[float],
    temp_offset: int = 0,
    memmap: bool = False,
):
    """
    Forward model all template spectra for a given region.

    We do not assume that segmentation IDs are 0-indexed, nor that they
    are contiguous, hence the requirement for both the index and value in
    the function parameters.

    Parameters
    ----------
    seg_idx : int
        The index of the region in an ordered array of IDs.
    seg_id : int
        The actual ID of the region.
    spec_wavs : np.ndarray[float]
        A 1D array of wavelengths.
    temp_offset : int, optional
        The fixed offset into the shared template array, to allow for
        polynomial or background templates. By default ``0``.
    memmap : bool, optional
        Whether the shared forward-modelled array is backed by a binary
        file on disk using `numpy.memmap`. By default, ``False``.
    """

    direct_masked = [
        beam.beam.direct * (beam.regions_seg_map == seg_id)
        for beam in multibeam_object.beams
    ]

    temp_resamp_1d = np.zeros((2, len(spec_wavs)))
    temp_resamp_1d[0, :] = spec_wavs

    row_offset = (shared_model_spectra[seg_idx].shape[0] * seg_idx) + temp_offset

    for sample_i, temp_spec in enumerate(shared_model_spectra[seg_idx]):
        temp_resamp_1d[1, :] = temp_spec
        tmodel = np.hstack(
            [
                beam.compute_model(
                    spectrum_1d=temp_resamp_1d,
                    thumb=direct,
                    in_place=False,
                    is_cgs=True,
                )[beam.fit_mask]
                for beam, direct in zip(multibeam_object.beams, direct_masked)
            ]
        )

        shared_temp_arr[
            row_offset + sample_i,
            :,
        ] += tmodel

    if memmap:
        shared_temp_arr.flush()
