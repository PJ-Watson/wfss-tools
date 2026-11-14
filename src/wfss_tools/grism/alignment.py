"""A module for stacking grism beams, aligning imaging, and spatial binning."""

from os import PathLike
from pathlib import Path

from astropy.io import fits
from grizli.multifit import MultiBeam
from numpy.typing import ArrayLike

from wfss_tools.grism.utils import align_direct_images, gen_stacked_beams
from wfss_tools.pipeline.reduction import recursive_merge
from wfss_tools.sed.binning import bin_and_save

__all__ = ["BeamAligner"]


class BeamAligner:
    """
    A class for stacking grism beams, aligning imaging, and binning.

    In general, this class is best initialised via a configuration file
    (i.e. ``config``), rather than manually setting all of the parameters
    listed below. However, these can be used to override any of the values
    in the configuration, either for more manual control, or quick
    iteration for testing purposes.

    Parameters
    ----------
    obj_id : int
        The object ID number to align and bin.
    config : dict | PathLike | None, optional
        Either a dictionary, or the TOML-formatted configuration file
        containing the parameters to use for alignment and binning. Any
        other parameters passed on initialisation will override those set
        in the configuration file.
    info_dict : dict | PathLike | None, optional
        Either a nested dictionary, or the path to a YAML file, where
        each top-level key corresponds to a filter name and maps to a
        dictionary containing ``"sci"`` and ``"var"`` image paths. If
        ``None`` (default), this is loaded from ``config``.
    field_name : str | None, optional
        The name of the field, used for naming output beam files and as
        the default ``group_name`` in `grizli.multifit.MultiBeam`. If
        ``None`` (default), uses the value from ``config``.
    use_stacks : bool | None, optional
        Whether to fit to individual beams, or beams stacked by filter
        and grism. If ``None`` (default), uses the value from ``config``
        or defaults to ``True``.
    stack_beam_kwargs : dict | None, optional
        Additional parameters to pass through to
        `~wfss_tools.grism.utils.gen_stacked_beams`.
    multibeam_kwargs : dict | None, optional
        Additional parameters to pass through to
        `grizli.multifit.MultiBeam`.
    binning_kwargs : dict | None, optional
        Additional parameters to pass through to
        `~wfss_tools.sed.binning.bin_and_save`.
    reproject_method : str | None, optional
        The method used to reproject direct images to the beam WCS. Must
        be one of ``"exact"`` (`reproject.reproject_exact`) or
        ``"subpixel"`` (`~wfss_tools.grism.utils.reproject_subpixel`). If
        ``None`` (default), uses the value from ``config`` or defaults to
        ``"exact"``.
    subpixel_sampling : int | None, optional
        The subpixel sampling factor to use when
        ``reproject_method="subpixel"``. If ``None`` (default), uses the
        value from ``config`` or defaults to ``5``.
    root_dir : PathLike | None, optional
        The root directory for all input and output files. If ``None``
        (default), uses the value from ``config`` or defaults to
        ``"."``.
    out_dir : PathLike | None, optional
        The output directory relative to ``root_dir`` where output
        files will be saved. If ``None`` (default), uses the value from
        ``config`` or defaults to ``"."``.
    extractions_dir : PathLike | None, optional
        The directory relative to ``root_dir`` containing the original
        ``*beams.fits`` extractions. If ``None`` (default), uses the
        value from ``config`` or defaults to ``"."``.
    binned_data_dir : PathLike | None, optional
        The directory where aligned beams, cutout images, and binned
        photometry will be written. If ``None`` (default), uses the value
        from ``config`` or defaults to ``<out_dir>/binned_data``.
    run_all : bool, optional
        If ``True``, the full alignment and binning workflow will run
        automatically on initialisation. By default ``False``.
    """

    def __init__(
        self,
        obj_id: int,
        config: dict | PathLike | None = None,
        info_dict: dict | PathLike | None = None,
        field_name: str | None = None,
        use_stacks: bool | None = None,
        stack_beam_kwargs: dict | None = None,
        multibeam_kwargs: dict | None = None,
        binning_kwargs: dict | None = None,
        reproject_method: str | None = None,
        subpixel_sampling: int | None = None,
        root_dir: PathLike | None = None,
        out_dir: PathLike | None = None,
        extractions_dir: PathLike | None = None,
        binned_data_dir: PathLike | None = None,
        run_all: bool = False,
    ):
        args = dict(locals())
        args.pop("self", None)
        args.pop("config", None)
        args.pop("run_all", None)

        for k, v in args.copy().items():
            if v is None:
                args.pop(k)

        self.obj_id = obj_id

        if config is not None:
            config = self.import_config(config)
        else:
            config = {}

        config_files = config.get("files", {})
        config_grism = config.get("grism", {})
        config_sed = config.get("SED", {})
        config_align = config.get("alignment", {})

        self.root_dir = Path(args.get("root_dir", config_files.get("root_dir", ".")))
        self.out_dir = self.root_dir / args.get(
            "out_dir", config_files.get("out_dir", ".")
        )
        self.out_dir.mkdir(exist_ok=True, parents=True)

        self.extractions_dir = self.root_dir / args.get(
            "extractions_dir", config_files.get("extractions_dir", ".")
        )

        _binned_data_dir = args.get(
            "binned_data_dir", config_files.get("binned_data_dir", "")
        )
        if not _binned_data_dir:
            self.binned_data_dir = self.out_dir / "binned_data"
        else:
            self.binned_data_dir = self.root_dir / _binned_data_dir
        self.binned_data_dir.mkdir(exist_ok=True, parents=True)

        _info_dict = args.get("info_dict", config_files.get("info_dict", ""))
        if not _info_dict:
            raise ValueError("`info_dict` is not present in the supplied config file.")
        self.info_dict = self.import_info_dict(_info_dict)

        self.field_name = args.get(
            "field_name",
            config_align.get("field_name", config_grism.get("field_name")),
        )
        if not self.field_name:
            raise ValueError(
                "`field_name` must be supplied either directly or in `config`."
            )

        self.use_stacks = args.get(
            "use_stacks",
            config_align.get("use_stacks", config_grism.get("use_stacks", True)),
        )

        cfg_stack_kwargs = recursive_merge(
            config_grism.get("stack_beam_kwargs", {}),
            config_align.get("stack_beam_kwargs", {}),
        )
        self.stack_beam_kwargs = recursive_merge(
            cfg_stack_kwargs, args.get("stack_beam_kwargs", {})
        )

        default_multib_kwargs = {
            "min_mask": 0.0,
            "min_sens": 0.0,
            "mask_resid": False,
            "verbose": False,
            "fcontam": 0.2,
            "group_name": self.field_name,
        }
        cfg_multib_kwargs = recursive_merge(
            config_grism.get("multibeam_kwargs", {}),
            config_align.get("multibeam_kwargs", {}),
        )
        self.multibeam_kwargs = recursive_merge(
            default_multib_kwargs,
            recursive_merge(cfg_multib_kwargs, args.get("multibeam_kwargs", {})),
        )

        default_binning_kwargs = {
            "bin_scheme": config_align.get(
                "bin_scheme", config_sed.get("bin_scheme", "colour")
            ),
            "target_sn": config_align.get("target_sn", config_sed.get("target_sn", 10)),
            "bin_diameter": config_align.get(
                "bin_diameter", config_sed.get("bin_diameter", 3)
            ),
            "sn_filter": config_align.get("sn_filter", config_sed.get("sn_filter")),
            **config_sed.get("bin_kwargs", {}),
            **config_align.get("bin_kwargs", {}),
        }
        self.binning_kwargs = recursive_merge(
            default_binning_kwargs, args.get("binning_kwargs", {})
        )
        if self.binning_kwargs.get("sn_filter") is None:
            raise ValueError(
                "`sn_filter` must be specified in `binning_kwargs` or `config`."
            )

        self.reproject_method = args.get(
            "reproject_method", config_align.get("reproject_method", "exact")
        )
        self.subpixel_sampling = args.get(
            "subpixel_sampling", config_align.get("subpixel_sampling", 5)
        )

        # Blank parameters until first run
        self.beam_path = None
        self.aligned_info_dict = None
        self.binned_data_path = None

        if run_all:
            self.run_all()

        return

    def import_config(self, config: dict | PathLike) -> dict:
        """
        Load a configuration file.

        Parameters
        ----------
        config : dict | PathLike
            Either a dictionary, or the TOML-formatted configuration file
            containing the parameters to use for alignment and binning.

        Returns
        -------
        dict
            The parsed configuration dictionary.
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

    def import_info_dict(self, info_dict: dict | PathLike) -> dict:
        """
        Load the imaging file lookup dictionary.

        Parameters
        ----------
        info_dict : dict | PathLike
            Either a dictionary, or the YAML-formatted file mapping filter
            names to ``"sci"`` and ``"var"`` FITS file locations.

        Returns
        -------
        dict
            The parsed imaging dictionary.
        """

        if isinstance(info_dict, dict):
            return info_dict
        elif isinstance(info_dict, PathLike) or isinstance(info_dict, str):
            import yaml

            info_path = Path(info_dict)
            if not info_path.is_absolute():
                info_path = self.root_dir / info_path

            with open(info_path, "r") as file:
                return yaml.safe_load(file)
        else:
            raise TypeError("Cannot import `info_dict`.")

    @property
    def default_beam_path(self) -> Path:
        """The default path for the aligned/stacked beams FITS file (`~pathlib.Path`, read-only)."""
        return self.binned_data_dir / f"{self.field_name}_{self.obj_id:0>5}.beams.fits"

    @property
    def binned_name(self) -> str:
        """The descriptive base filename for the binned photometry (`str`, read-only)."""
        # Give a descriptive name for the binned data
        return (
            f"{self.obj_id}_{self.binning_kwargs["bin_scheme"]}_"
            f"{self.binning_kwargs["bin_diameter"]}_{self.binning_kwargs["target_sn"]}"
            f"_{self.binning_kwargs["sn_filter"]}"
        )

    @property
    def default_binned_data_path(self) -> Path:
        """The default path for the binned photometry FITS file (`~pathlib.Path`, read-only)."""
        return self.binned_data_dir / f"{self.binned_name}_data.fits"

    def prepare_beams(
        self,
        beams_path: PathLike | ArrayLike | None = None,
        use_stacks: bool | None = None,
        stack_beam_kwargs: dict | None = None,
        overwrite: bool = False,
        **multibeam_kwargs,
    ) -> tuple[Path, MultiBeam]:
        """
        Load, optionally stack, and save the aligned grism beams.

        Parameters
        ----------
        beams_path : PathLike | ArrayLike | None, optional
            The location of one or more ``*beams.fits`` files to use. If
            ``None`` (default), this will be selected automatically based
            on ``obj_id`` and ``extractions_dir``.
        use_stacks : bool | None, optional
            Whether to fit to individual beams, or beams stacked by filter
            and grism. If ``None`` (default), uses ``self.use_stacks``.
        stack_beam_kwargs : dict | None, optional
            Any additional parameters to pass through to
            `~wfss_tools.grism.utils.gen_stacked_beams`.
        overwrite : bool, optional
            If ``True``, regenerate and overwrite any existing aligned or
            stacked ``*beams.fits`` file. By default ``False``.
        **multibeam_kwargs : dict, optional
            Any additional parameters to pass through to
            `grizli.multifit.MultiBeam`.

        Returns
        -------
        new_beam_loc : Path
            The location of the ``*beams.fits`` file used for alignment.
            If ``use_stacks==True``, this will be a stacked version of the
            input file.
        multib : `grizli.multifit.MultiBeam`
            The loaded or stacked `~grizli.multifit.MultiBeam` object.

        Raises
        ------
        IOError
            If no ``*beams.fits`` file can be found, this method will
            raise an error.
        """

        new_beam_loc = self.default_beam_path

        mb_kwargs = recursive_merge(self.multibeam_kwargs, multibeam_kwargs)
        sb_kwargs = recursive_merge(self.stack_beam_kwargs, stack_beam_kwargs or {})
        do_stack = self.use_stacks if use_stacks is None else use_stacks

        multib = None
        if not overwrite:
            try:
                multib = MultiBeam(
                    str(new_beam_loc),
                    **mb_kwargs,
                )
            except Exception:
                multib = None

        if multib is None:
            if beams_path is None:
                found_beams = [
                    *self.extractions_dir.glob(f"**/*{self.obj_id:0>5}.beams.fits")
                ]
                if len(found_beams) >= 1:
                    beams_path = [str(b) for b in found_beams]
                else:
                    raise IOError(
                        f"Original beams file does not exist in {self.extractions_dir}"
                    )

            if do_stack:
                multib = gen_stacked_beams(
                    beams_path,
                    **sb_kwargs,
                    **mb_kwargs,
                )
            else:
                multib = MultiBeam(
                    beams_path,
                    **mb_kwargs,
                )

            # Write the realigned and (stacked?) beam to a file
            beam_hdul = multib.write_master_fits(get_hdu=True)
            beam_hdul.writeto(new_beam_loc, overwrite=True)

        self.beam_path = new_beam_loc

        return new_beam_loc, multib

    def align_images(
        self,
        multib: MultiBeam | None = None,
        bounding_cutout_kwargs: dict | None = None,
        overwrite: bool = False,
        reproject_method: str | None = None,
        subpixel_sampling: int | None = None,
    ) -> dict:
        """
        Align photometric images to the direct image in an extracted beam.

        Parameters
        ----------
        multib : `grizli.multifit.MultiBeam` | None, optional
            The `~grizli.multifit.MultiBeam` object whose first beam is
            used as the astrometric reference. If ``None`` (default),
            loads from ``self.beam_path`` (or ``self.default_beam_path``).
        bounding_cutout_kwargs : dict | None, optional
            Any additional parameters to pass through to
            `~wfss_tools.grism.utils.get_bounding_cutout_slice`.
        overwrite : bool, optional
            If ``True``, overwrite existing realigned filter cutouts. By
            default ``False``.
        reproject_method : str | None, optional
            The reprojection method to use (``"exact"`` or
            ``"subpixel"``). If ``None`` (default), uses
            ``self.reproject_method``.
        subpixel_sampling : int | None, optional
            The subpixel sampling factor when
            ``reproject_method="subpixel"``. If ``None`` (default), uses
            ``self.subpixel_sampling``.

        Returns
        -------
        dict
            An updated version of ``self.info_dict`` pointing to the
            realigned images.
        """

        if multib is None:
            beam_loc = self.beam_path or self.default_beam_path
            multib = MultiBeam(
                str(beam_loc),
                **self.multibeam_kwargs,
            )

        # Align all images to the new beam
        self.aligned_info_dict = align_direct_images(
            multib.beams[0],
            info_dict=self.info_dict,
            out_dir=self.binned_data_dir / f"{self.obj_id:0>5}",
            bounding_cutout_kwargs=bounding_cutout_kwargs,
            overwrite=overwrite,
            reproject_method=(
                self.reproject_method if reproject_method is None else reproject_method
            ),
            subpixel_sampling=(
                self.subpixel_sampling
                if subpixel_sampling is None
                else subpixel_sampling
            ),
        )

        return self.aligned_info_dict

    def bin_photometry(
        self,
        aligned_info_dict: dict | None = None,
        beam_path: PathLike | None = None,
        binning_kwargs: dict | None = None,
        overwrite: bool = False,
    ) -> Path:
        """
        Spatially bin the aligned photometry using the beam segmentation map.

        Parameters
        ----------
        aligned_info_dict : dict | None, optional
            The dictionary pointing to the realigned ``"sci"`` and
            ``"var"`` images. If ``None`` (default), uses
            ``self.aligned_info_dict``.
        beam_path : PathLike | None, optional
            The path to the aligned/stacked ``*beams.fits`` file
            containing the ``"SEG"`` extension. If ``None`` (default),
            uses ``self.beam_path`` or ``self.default_beam_path``.
        binning_kwargs : dict | None, optional
            Any additional parameters to pass through to
            `~wfss_tools.sed.binning.bin_and_save`.
        overwrite : bool, optional
            If ``True``, regenerate and overwrite any existing binned
            photometry file. By default ``False``.

        Returns
        -------
        Path
            The location of the binned data in FITS format, containing
            both the segmentation map and the binned photometric
            catalogue.
        """

        eff_binning_kwargs = recursive_merge(self.binning_kwargs, binning_kwargs or {})
        binned_name = (
            f"{self.obj_id}_{eff_binning_kwargs["bin_scheme"]}_"
            f"{eff_binning_kwargs["bin_diameter"]}_{eff_binning_kwargs["target_sn"]}"
            f"_{eff_binning_kwargs["sn_filter"]}"
        )
        binned_data_path = self.binned_data_dir / f"{binned_name}_data.fits"

        if not binned_data_path.is_file() or overwrite:
            info = aligned_info_dict or self.aligned_info_dict
            if info is None:
                raise RuntimeError(
                    "Aligned images not found. Run `BeamAligner.align_images()` "
                    "before calling `BeamAligner.bin_photometry()`."
                )

            new_beam_loc = Path(beam_path or self.beam_path or self.default_beam_path)
            _seg = fits.getdata(new_beam_loc, "SEG")

            bin_and_save(
                obj_id=self.obj_id,
                out_dir=self.binned_data_dir,
                seg_map=_seg,
                info_dict=info,
                binned_name=binned_name,
                **eff_binning_kwargs,
            )

            with fits.open(binned_data_path, mode="update") as hdul:
                hdul[0].header["OBJID"] = (self.obj_id, "Target object ID")
                hdul[0].header["BEAMFILE"] = (
                    str(new_beam_loc),
                    "Aligned/stacked beams FITS file",
                )

        self.binned_data_path = binned_data_path

        return binned_data_path

    def run_all(
        self,
        beams_path: PathLike | ArrayLike | None = None,
        bounding_cutout_kwargs: dict | None = None,
        overwrite_beams: bool = False,
        overwrite_images: bool = False,
        overwrite_bins: bool = False,
    ) -> tuple[Path, Path]:
        """
        Run the full beam stacking, image alignment, and spatial binning workflow.

        Parameters
        ----------
        beams_path : PathLike | ArrayLike | None, optional
            The location of one or more ``*beams.fits`` files to use. If
            ``None`` (default), this will be selected automatically based
            on ``obj_id`` and ``extractions_dir``.
        bounding_cutout_kwargs : dict | None, optional
            Any additional parameters to pass through to
            `~wfss_tools.grism.utils.get_bounding_cutout_slice`.
        overwrite_beams : bool, optional
            If ``True``, overwrite any existing aligned/stacked beams
            file. By default ``False``.
        overwrite_images : bool, optional
            If ``True``, overwrite any existing realigned filter images.
            By default ``False``.
        overwrite_bins : bool, optional
            If ``True``, overwrite any existing binned photometry FITS
            file. By default ``False``.

        Returns
        -------
        new_beam_path : Path
            The location of the ``*beams.fits`` used for alignment. If
            ``use_stacks==True``, this will be a stacked version of the
            input file.
        binned_data_path : Path
            The location of the binned data in FITS format. This contains
            both the segmentation map and the binned photometric
            catalogue.
        """

        new_beam_loc, multib = self.prepare_beams(
            beams_path=beams_path,
            overwrite=overwrite_beams,
        )
        aligned_info_dict = self.align_images(
            multib=multib,
            bounding_cutout_kwargs=bounding_cutout_kwargs,
            overwrite=overwrite_images,
        )
        binned_data_path = self.bin_photometry(
            aligned_info_dict=aligned_info_dict,
            beam_path=new_beam_loc,
            overwrite=overwrite_bins,
        )

        return new_beam_loc, binned_data_path


if __name__ == "__main__":

    # Test the current implementation against the existing functions
    aligner = BeamAligner(
        obj_id=3070,
        config="/media/sharedData/python/py3.13_PIE/code/wfss-tools/src/wfss_tools/sed/config_v9-TEST.toml",
        out_dir="glass_niriss_bcgs/reduction_v9-TEST_v2",
    )

    beam_path, binned_data_path = aligner.run_all()
