"""Additional functions used in the pipeline."""

from os import PathLike
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from grizli import prep
from numpy.typing import ArrayLike

__all__ = ["parse_files_grizli_aws", "seg_slice", "regen_catalogue"]


def split(delimiters, string: str, maxsplit: int = 0):
    """
    Some code.

    Parameters
    ----------
    delimiters : _type_
        _description_.
    string : str
        _description_.
    maxsplit : int, optional
        _description_, by default 0.

    Returns
    -------
    str
        String with something removed.
    """
    import re

    regex_pattern = "|".join(map(re.escape, delimiters))
    return re.split(regex_pattern, string, maxsplit)


def parse_files_grizli_aws(
    data_dir: PathLike,
    root: str = "abell2744clu",
    out_path: PathLike | None = None,
) -> dict:
    """
    Parse files processed using grizli aws/DJA into a ``dict``.

    Only files in the correct format will be parsed, so it is safe to have
    other files in the same directory. The image mosaics are separated by
    telescope, instrument, and filter, in an attempt to avoid any
    namespace collisions resulting from the same filter names being used
    in multiple locations. Images are also separated by type.
    The output can optionally be saved to a YAML file to avoid reading all
    of the headers again (this can be extremely slow if working with
    large, compressed mosaics).

    Parameters
    ----------
    data_dir : PathLike
        The directory to scan for image files.
    root : str, optional
        The expected root of the filenames, by default ``"abell2744clu"``.
    out_path : PathLike | None, optional
        The path to which the output will be written (in YAML format).
        By default ``None``; the output will not be written to disk.

    Returns
    -------
    dict
        A dictionary containing the relevant files in the directory,
        separated by telescope, instrument, and filter.
    """
    obs_dict = {}

    files = []
    for ext in ["*.fits", "*.fits.gz"]:
        files.extend(Path(data_dir).glob(f"{root}{ext}"))
    files.sort()

    for f in files[:]:

        if "grizli-v" not in f.name:
            continue

        name = f.name
        v_start = name.index("grizli-v") + 8
        v_end = name.index("-", v_start)
        version = name[v_start:v_end]

        # filt, pupil =
        flag_incomplete = False
        try:
            filt_details = split(["_drz", "_drc"], name[v_end + 1 :])[0].split("-")
            if len(filt_details) > 2:
                flag_incomplete = True
        except:
            print(f"File `{name}` not in expected format. Continuing.")
            continue

        hdr = fits.getheader(f)
        filt = hdr.get(
            "FILTER",
            (
                hdr.get("FILTER2", "UNKNOWN")
                if not hdr.get("FILTER1", "UNKNOWN").lower().startswith("f")
                else hdr.get("FILTER1", "UNKNOWN")
            ),
        )
        pupil = hdr.get("PUPIL", "UNKNOWN")
        detector = hdr.get("DETECTOR", "UNKNOWN")
        instrument = hdr.get("INSTRUMENT", hdr.get("INSTRUME", "UNKNOWN"))
        telescope = hdr.get("TELESCOPE", hdr.get("TELESCOP", "UNKNOWN"))

        file_types = split(
            [
                ".fits",
                ".gz",
                f"{filt.lower()}-{pupil.lower()}",
                filt.lower(),
                pupil.lower(),
                "_drz_",
                "_drc_",
            ],
            name[v_end + 1 :],
        )
        file_match_idx = np.isin(file_types, ["sci", "exp", "var", "wht"])

        if not np.any(file_match_idx):
            print(f"Unrecognised file type for file name `{name}`.")
            continue
        else:
            file_type = str(np.asarray(file_types)[file_match_idx][0])

        if instrument.lower() == "niriss":
            id_key = f"{telescope}-{instrument}-{pupil}".lower()
        else:
            id_key = f"{telescope}-{instrument}-{filt}".lower()

        if id_key not in obs_dict:
            obs_dict[id_key] = {
                "filt": filt,
                "pupil": pupil,
                "detector": detector,
                "instrument": instrument,
                "telescope": telescope,
                "psf": "",
            }

        if file_type in obs_dict[id_key]:
            if obs_dict[id_key][file_type]["version"] > float(version):
                continue
            if not obs_dict[id_key][file_type]["flag_incomplete"] and flag_incomplete:
                continue

        obs_dict[id_key][file_type] = {
            "path": str(f),
            "version": float(version),
            "flag_incomplete": flag_incomplete,
        }
        print(f"Added file {name}")

    obs_dict = dict(sorted(obs_dict.items()))

    if out_path is not None:
        import yaml

        with open(out_path, "w") as outfile:
            yaml.dump(obs_dict, outfile, default_flow_style=False, sort_keys=False)

    # Need to know photfnu/photflam/zeropoint
    # Pixelscale
    # _sci, _var
    return obs_dict


def seg_slice(
    seg_map: ArrayLike,
    seg_id: int,
    padding: int = 50,
    equal_ratio: bool = False,
) -> tuple:
    """
    Find the location of an object in a segmentation map.

    Parameters
    ----------
    seg_map : ArrayLike
        The segmentation map, where each pixel value indicates the ID of
        the object to which that pixel belongs.
    seg_id : int
        The ID of the object to find.
    padding : int, optional
        The extra padding to add to the object boundaries, by default 50.
    equal_ratio : bool, optiona
        Force the slice to be a square.

    Returns
    -------
    tuple
        The indices of the object in the array.
    """

    x_idxs, y_idxs = (seg_map == seg_id).nonzero()

    x_min = np.nanmin(x_idxs)
    x_max = np.nanmax(x_idxs)
    y_min = np.nanmin(y_idxs)
    y_max = np.nanmax(y_idxs)

    x_pad = y_pad = padding
    if equal_ratio:
        x_width = x_max - x_min
        y_width = y_max - y_min
        if x_width > y_width:
            y_pad = (x_width - y_width) // 2 + padding
            x_pad = padding
        elif x_width > y_width:
            x_pad = (x_width - y_width) // 2 + padding
            y_pad = padding
        else:
            pass

    new_idxs = (
        slice(
            int(np.nanmax([x_min - x_pad, 0])),
            int(np.nanmin([x_max + x_pad, seg_map.shape[0]])),
        ),
        slice(
            int(np.nanmax([y_min - y_pad, 0])),
            int(np.nanmin([y_max + y_pad, seg_map.shape[1]])),
        ),
    )
    return new_idxs


def regen_catalogue(
    new_seg_map,
    root="",
    sci=None,
    wht=None,
    threshold=2.0,
    get_background=True,
    bkg_only=False,
    bkg_params={"bw": 64, "bh": 64, "fw": 3, "fh": 3, "pixel_scale": 0.06},
    verbose=True,
    phot_apertures=prep.SEXTRACTOR_PHOT_APERTURES_ARCSEC,
    aper_segmask=False,
    prefer_var_image=True,
    rescale_weight=False,
    err_scale=-np.inf,
    use_bkg_err=False,
    column_case=str.upper,
    save_to_fits=True,
    include_wcs_extension=True,
    source_xy=None,
    compute_auto_quantities=True,
    autoparams=[2.5, 0.35 * u.arcsec, 2.4, 3.8],
    flux_radii=[0.2, 0.5, 0.9],
    subpix=0,
    mask_kron=False,
    max_total_corr=2,
    detection_params=prep.SEP_DETECT_PARAMS,
    bkg_mask=None,
    pixel_scale=0.06,
    log=False,
    gain=2000.0,
    extract_pixstack=int(3e7),
    sub_object_limit=4096,
    exposure_footprints=None,
    suffix="",
    full_mask=None,
    **kwargs,
):
    """
    Make a catalog from drizzle products using SEP.

    Parameters
    ----------
    new_seg_map : ArrayLike
        The segmentation map from which the catalogue will be regenerated.

    root : str
        Rootname of the FITS images to use for source extraction.  This
        function is designed to work with the single-image products from
        `drizzlepac`, so the default data/science image is searched by

        >>> drz_file = glob.glob(f'{root}_dr[zc]_sci.fits*')[0]

        Note that this will find and use gzipped versions of the images,
        if necessary.

        The associated weight image filename is then assumed to be

        >>> weight_file = drz_file.replace('_sci.fits', '_wht.fits')
        >>> weight_file = weight_file.replace('_drz.fits', '_wht.fits')
        .

    sci, wht : str
        Filenames to override `drz_file` and `weight_file` derived from the
        ``root`` parameter.

    threshold : float
        Detection threshold for `sep.extract`.

    get_background : bool
        Compute the background with `sep.Background`.

    bkg_only : bool
        If `True`, then just return the background data array and don't run
        the source detection.

    bkg_params : dict
        Keyword arguments for `sep.Background`.  Note that this can include
        a separate optional keyword ``pixel_scale`` that indicates that the
        background sizes `bw`, `bh` are set for a paraticular pixel size.
        They will be scaled to the pixel dimensions of the target images using
        the pixel scale derived from the image WCS.

    verbose : bool
        Print status messages.

    phot_apertures : str or array-like
        Photometric aperture *diameters*. If given as a string then assume
        units of pixels. If an array or list, can have units, e.g.,
        `astropy.units.arcsec`.

    aper_segmask : bool
        If true, then run SEP photometry with segmentation masking.  This
        requires the sep fork at https://github.com/gbrammer/sep.git,
        or `sep >= 1.10.0`.

    prefer_var_image : bool
        Use a variance image ``_wht.fits > _var.fits`` if found.

    rescale_weight : bool
        If true, then a scale factor is calculated from the ratio of the
        weight image to the variance estimated by `sep.Background`.

    err_scale : float
        Explicit value to use for the weight scaling, rather than calculating
        with `rescale_weight`.  Only used if ``err_scale > 0``.

    use_bkg_err : bool
        If true, then use the full error array derived by `sep.Background`.
        This is turned off by default in order to preserve the pixel-to-pixel
        variation in the drizzled weight maps.

    column_case : func
        Function to apply to the catalog column names.  E.g., the default
        `str.upper` results in uppercase column names.

    save_to_fits : bool
        Save catalog FITS file ``{root}.cat.fits``.

    include_wcs_extension : bool
        An extension will be added to the FITS catalog with the detection
        image WCS.

    source_xy : (x, y) or (ra, dec) arrays
        Force extraction positions.  If the arrays have units, then pass them
        through the header WCS.  If no units, positions are *zero indexed*
        array coordinates.

        To run with segmentation masking (`1sep > 1.10``), also provide
        `aseg` and `aseg_id` arrays with `source_xy`, like

        >>> source_xy = ra, dec, aseg, aseg_id
        .

    compute_auto_quantities : bool
        Compute Kron/auto-like quantities with
        `~grizli.prep.compute_SEP_auto_params`.

    autoparams : list
        Parameters of Kron/AUTO calculations with
        `~grizli.prep.compute_SEP_auto_params`.

    flux_radii : list
        Light fraction radii to compute with
        `~grizli.prep.compute_SEP_auto_params`, e.g., ``[0.5]`` will calculate
        the half-light radius (``FLUX_RADIUS``).

    subpix : int
        Pixel oversampling.

    mask_kron : bool
        Not used.

    max_total_corr : float
        Not used.

    detection_params : dict
        Parameters passed to `sep.extract`.

    bkg_mask : array
        Additional mask to apply to `sep.Background` calculation.

    pixel_scale : float
        Not used.

    log : bool
        Send log message to `grizli.utils.LOGFILE`.

    gain : float
        Gain value passed to `sep.sum_circle`.

    extract_pixstack : int
        See `sep.set_extract_pixstack`.

    sub_object_limit : int
        See `sep.set_sub_object_limit`.

    exposure_footprints : list, None
        An optional list of objects that can be parsed with `sregion.SRegion`.  If
        specified, add a column ``nexp`` to the catalog corresponding to the number
        of entries in the list that overlap with a particular source position.

    suffix : str
        Additional suffix to add to the catalogue name.

    full_mask : ArrayLike
        Parts of the image to mask out.

    **kwargs : dict, optional
        Any additional keyword arguments.

    Returns
    -------
    `~astropy.table.Table`
        Source catalogue.
    """

    if log:
        frame = inspect.currentframe()
        utils.log_function_arguments(
            utils.LOGFILE, frame, "prep.make_SEP_catalog", verbose=True
        )

    import copy
    import glob
    import os

    #     )
    import sep
    from grizli import utils

    # try:
    #     import sep_pjw as sep
    # except ImportError:
    #     print(
    #         """
    # Couldn't import `sep_pjw`. SEP is no longer maintained; install the
    # fork with `python -m pip install sep-pjw`.
    # """

    sep.set_extract_pixstack(extract_pixstack)
    sep.set_sub_object_limit(sub_object_limit)

    logstr = "make_SEP_catalog: sep version = {0}".format(sep.__version__)
    utils.log_comment(utils.LOGFILE, logstr, verbose=verbose)

    if sci is not None:
        drz_file = sci
    else:
        drz_file = glob.glob(f"{root}_dr[zc]_sci.fits*")[0]

    im = fits.open(drz_file)

    # Filter
    drz_filter = utils.parse_filter_from_header(im[0].header)
    if "PHOTPLAM" in im[0].header:
        drz_photplam = im[0].header["PHOTPLAM"]
    else:
        drz_photplam = None

    # Get AB zeropoint
    ZP = utils.calc_header_zeropoint(im, ext=0)

    # Scale fluxes to mico-Jy
    uJy_to_dn = 1 / (3631 * 1e6 * 10 ** (-0.4 * ZP))

    if wht is not None:
        weight_file = wht
    else:
        weight_file = str(drz_file).replace("_sci.fits", "_wht.fits")
        weight_file = weight_file.replace("_drz.fits", "_wht.fits")

    if (weight_file == drz_file) | (not os.path.exists(weight_file)):
        WEIGHT_TYPE = "NONE"
        weight_file = None
    else:
        WEIGHT_TYPE = "MAP_WEIGHT"

    if (WEIGHT_TYPE == "MAP_WEIGHT") & (prefer_var_image):
        var_file = str(weight_file).replace("wht.fits", "var.fits")
        if os.path.exists(var_file) & (var_file != weight_file):
            weight_file = var_file
            WEIGHT_TYPE = "VARIANCE"

    drz_im = fits.open(drz_file)
    # data = drz_im[0].data.byteswap().newbyteorder()
    data = drz_im[0].data.view(drz_im[0].data.dtype.newbyteorder()).byteswap()

    logstr = f"make_SEP_catalog: {drz_file} weight={weight_file} ({WEIGHT_TYPE})"
    utils.log_comment(utils.LOGFILE, logstr, verbose=verbose, show_date=True)

    logstr = "make_SEP_catalog: Image AB zeropoint =  {0:.3f}".format(ZP)
    utils.log_comment(utils.LOGFILE, logstr, verbose=verbose, show_date=False)

    try:
        wcs = WCS(drz_im[0].header)
        wcs_header = utils.to_header(wcs)
        pixel_scale = utils.get_wcs_pscale(wcs)  # arcsec
    except:
        wcs = None
        wcs_header = drz_im[0].header.copy()
        pixel_scale = np.sqrt(wcs_header["CD1_1"] ** 2 + wcs_header["CD1_2"] ** 2)
        pixel_scale *= 3600.0  # arcsec

    # Add some header keywords to the wcs header
    for k in ["EXPSTART", "EXPEND", "EXPTIME"]:
        if k in drz_im[0].header:
            wcs_header[k] = drz_im[0].header[k]

    if isinstance(phot_apertures, str):
        apertures = np.asarray(phot_apertures.replace(",", "").split(), dtype=float)
    else:
        apertures = []
        for ap in phot_apertures:
            if hasattr(ap, "unit"):
                apertures.append(ap.to(u.arcsec).value / pixel_scale)
            else:
                apertures.append(ap)

    # Do we need to compute the error from the wht image?
    need_err = (not use_bkg_err) | (not get_background)
    if (weight_file is not None) & need_err:
        wht_im = fits.open(weight_file)
        # wht_data = wht_im[0].data.byteswap().newbyteorder()
        wht_data = wht_im[0].data.view(wht_im[0].data.dtype.newbyteorder()).byteswap()

        if WEIGHT_TYPE == "VARIANCE":
            err_data = np.sqrt(wht_data)
        else:
            err_data = 1 / np.sqrt(wht_data)

        del wht_data

        # True mask pixels are masked with sep
        mask = (~np.isfinite(err_data)) | (err_data == 0) | (~np.isfinite(data))
        err_data[mask] = 0

        wht_im.close()
        del wht_im

    else:
        # True mask pixels are masked with sep
        mask = (data == 0) | (~np.isfinite(data))
        err_data = None

    try:
        drz_im.close()
        del drz_im
    except:
        pass

    if full_mask is not None:
        mask |= full_mask

    data_mask = np.asarray(mask, dtype=data.dtype)

    if get_background | (err_scale < 0) | (use_bkg_err):

        # Account for pixel scale in bkg_params
        bkg_input = {}
        if "pixel_scale" in bkg_params:
            bkg_pscale = bkg_params["pixel_scale"]
        else:
            bkg_pscale = pixel_scale

        for k in bkg_params:
            if k in ["pixel_scale"]:
                continue

            if k in ["bw", "bh"]:
                bkg_input[k] = bkg_params[k] * bkg_pscale / pixel_scale
            else:
                bkg_input[k] = bkg_params[k]

        logstr = "SEP: Get background {0}".format(bkg_input)
        utils.log_comment(utils.LOGFILE, logstr, verbose=verbose, show_date=True)

        if bkg_mask is not None:
            bkg = sep.Background(data, mask=mask | bkg_mask, **bkg_input)
        else:
            bkg = sep.Background(data, mask=mask, **bkg_input)

        bkg_data = bkg.back()
        if bkg_only:
            return bkg_data

        if get_background == 2:
            bkg_file = "{0}_bkg.fits".format(root)
            if os.path.exists(bkg_file):
                logstr = "SEP: use background file {0}".format(bkg_file)
                utils.log_comment(
                    utils.LOGFILE, logstr, verbose=verbose, show_date=True
                )

                bkg_im = fits.open("{0}_bkg.fits".format(root))
                bkg_data = bkg_im[0].data * 1
        else:
            fits.writeto(
                "{0}_bkg.fits".format(root),
                data=bkg_data,
                header=wcs_header,
                overwrite=True,
            )

        if (err_data is None) | use_bkg_err:
            logstr = "sep: Use bkg.rms() for error array"
            utils.log_comment(utils.LOGFILE, logstr, verbose=verbose, show_date=True)

            err_data = bkg.rms()

        if err_scale == -np.inf:
            ratio = bkg.rms() / err_data
            err_scale = np.median(ratio[(~mask) & np.isfinite(ratio)])
        else:
            # Just return the error scale
            if err_scale < 0:
                ratio = bkg.rms() / err_data
                xerr_scale = np.median(ratio[(~mask) & np.isfinite(ratio)])
                del bkg
                return xerr_scale

        del bkg

    else:
        if err_scale is None:
            err_scale = 1.0

    if not get_background:
        bkg_data = 0.0
        data_bkg = data
    else:
        data_bkg = data - bkg_data

    if rescale_weight:
        if verbose:
            print("SEP: err_scale={:.3f}".format(err_scale))

        err_data *= err_scale

    if source_xy is None:
        # Run the detection
        if verbose:
            print("   SEP: Extract...")

        from photutils.segmentation import SegmentationImage, SourceCatalog

        if new_seg_map is None:
            objects, seg = sep.extract(
                data_bkg,
                threshold,
                err=err_data,
                mask=mask,
                segmentation_map=True,
                **detection_params,
            )

            objects = Table(objects)

            objects["number"] = np.arange(len(objects), dtype=np.int32) + 1
            print(len(objects))
        else:

            print("Using old segmentation map")

            seg_img = SegmentationImage(new_seg_map)

            if detection_params.get("filter_kernel", None) is not None:
                from astropy.convolution import convolve_fft as convolve

                conv_data = convolve(data_bkg, detection_params["filter_kernel"])
            else:
                conv_data = data_bkg

            print("Convolved")

            source_cat = SourceCatalog(
                data=data_bkg,
                segment_img=seg_img,
                convolved_data=conv_data,
                error=err_data,
                mask=mask,
                progress_bar=True,
            )

            print(source_cat.to_table())
            print(source_cat.to_table().colnames)
            print(source_cat.moments)
            print(source_cat.moments[0])
            print(source_cat.to_table()[0])
            print(source_cat.bbox_xmin[0])
            print(source_cat.bbox_xmax[0])

            # for p in source_cat.properties:
            #     print(p, getattr(source_cat[0], p))

            rename_cols = {
                "label": "number",
                "area": "npix",
                "bbox_xmin": "xmin",
                "bbox_xmax": "xmax",
                "bbox_ymin": "ymin",
                "bbox_ymax": "ymax",
                "x_centroid": "x",
                "y_centroid": "y",
                "covariance_xx": "x2",
                "covariance_yy": "y2",
                "covariance_xy": "xy",
                "semimajor_axis": "a",
                "semiminor_axis": "b",
                "orientation": "theta",
                "ellipse_cxx": "cxx",
                "ellipse_cyy": "cyy",
                "ellipse_cxy": "cxy",
                "segment_flux": "flux",
                "max_value": "peak",
                "maxval_xindex": "xpeak",
                "maxval_yindex": "ypeak",
            }
            remove_cols = [
                "skycentroid",
            ]

            objects = Table()
            for orig_phot, sex_name in rename_cols.items():
                objects[sex_name] = getattr(source_cat, orig_phot)

            extra_cnames = {
                "segment_flux": "cflux",
                "max_value": "cpeak",
                "maxval_xindex": "xcpeak",
                "maxval_yindex": "ycpeak",
            }
            for orig_phot, sex_name in extra_cnames.items():
                objects[sex_name] = getattr(source_cat, orig_phot)

            objects["theta"] = np.deg2rad(objects["theta"])
            seg = new_seg_map
            # objects["id"] = objects["number"]
            # print (objects.convdata_ma)
            # print (objects.convdata_ma.shape)
            # print(objects)

        # objects, seg = sep.extract(
        #     data_bkg,
        #     threshold,
        #     err=err_data,
        #     mask=mask,
        #     segmentation_map=new_seg_map,
        #     **detection_params,
        # )

        # if verbose:
        #     print("    Done.")

        tab = utils.GTable(objects)
        tab.meta["VERSION"] = (sep.__version__, "SEP version")

        # make unit-indexed like SExtractor
        tab["x_image"] = tab["x"] + 1
        tab["y_image"] = tab["y"] + 1

        # # ID
        # # tab["number"] = np.arange(len(tab), dtype=np.int32) + 1
        # tab["number"] = np.unique(new_seg_map).astype(np.int32)[1:]
        tab["theta"] = np.clip(tab["theta"], -np.pi / 2, np.pi / 2)
        for row in tab:
            test = [
                np.isfinite(row[c])
                for c in ["a", "b", "x", "y", "x_image", "y_image", "theta"]
            ]
            if not np.all(test):
                # if np.any(~np.isfinite(row["a", "b", "x", "y", "x_image", "y_image", "theta"])):
                print(row)
        for c in ["a", "b", "x", "y", "x_image", "y_image", "theta"]:
            tab = tab[np.isfinite(tab[c])]
        # for c in ["x", "y", "x_image", "y_image", "theta"]:
        #     tab = tab[np.isfinite(tab[c])]
        # Segmentation
        seg[mask] = 0

        fits.writeto(
            f"{root}_seg{suffix}.fits", data=seg, header=wcs_header, overwrite=True
        )

        # WCS coordinates
        if wcs is not None:
            tab["ra"], tab["dec"] = wcs.all_pix2world(tab["x"], tab["y"], 0)
            tab["ra"].unit = u.deg
            tab["dec"].unit = u.deg
            tab["x_world"], tab["y_world"] = tab["ra"], tab["dec"]

        if "minarea" in detection_params:
            tab.meta["MINAREA"] = (
                detection_params["minarea"],
                "Minimum source area in pixels",
            )
        else:
            tab.meta["MINAREA"] = (5, "Minimum source area in pixels")

        if "clean" in detection_params:
            tab.meta["CLEAN"] = (detection_params["clean"], "Detection cleaning")
        else:
            tab.meta["CLEAN"] = (True, "Detection cleaning")

        if "deblend_cont" in detection_params:
            tab.meta["DEBCONT"] = (
                detection_params["deblend_cont"],
                "Deblending contrast ratio",
            )
        else:
            tab.meta["DEBCONT"] = (0.005, "Deblending contrast ratio")

        if "deblend_nthresh" in detection_params:
            tab.meta["DEBTHRSH"] = (
                detection_params["deblend_nthresh"],
                "Number of deblending thresholds",
            )
        else:
            tab.meta["DEBTHRSH"] = (32, "Number of deblending thresholds")

        if "filter_type" in detection_params:
            tab.meta["FILTER_TYPE"] = (
                detection_params["filter_type"],
                "Type of filter applied, conv or weight",
            )
        else:
            tab.meta["FILTER_TYPE"] = ("conv", "Type of filter applied, conv or weight")

        tab.meta["THRESHOLD"] = (threshold, "Detection threshold")

        # ISO fluxes (flux within segments)
        iso_flux, iso_fluxerr, iso_area = prep.get_seg_iso_flux(
            data_bkg, seg, tab, err=err_data, verbose=1
        )

        tab["flux_iso"] = iso_flux / uJy_to_dn * u.uJy
        tab["fluxerr_iso"] = iso_fluxerr / uJy_to_dn * u.uJy
        tab["area_iso"] = iso_area
        tab["mag_iso"] = 23.9 - 2.5 * np.log10(tab["flux_iso"])

        # Compute FLUX_AUTO, FLUX_RADIUS
        if compute_auto_quantities:
            auto = prep.compute_SEP_auto_params(
                data,
                data_bkg,
                mask,
                pixel_scale=pixel_scale,
                err=err_data,
                segmap=seg,
                tab=tab,
                autoparams=autoparams,
                flux_radii=flux_radii,
                subpix=subpix,
                verbose=verbose,
            )

            for k in auto.meta:
                tab.meta[k] = auto.meta[k]

            auto_flux_cols = ["flux_auto", "fluxerr_auto", "bkg_auto"]
            for c in auto.colnames:
                if c in auto_flux_cols:
                    tab[c] = auto[c] / uJy_to_dn * u.uJy
                else:
                    tab[c] = auto[c]

            # Problematic sources
            # bad = (tab['flux_auto'] <= 0) | (tab['flux_radius'] <= 0)
            # bad |= (tab['kron_radius'] <= 0)
            # tab = tab[~bad]

            # Correction for flux outside Kron aperture
            tot_corr = prep.get_kron_tot_corr(
                tab, drz_filter, pixel_scale=pixel_scale, photplam=drz_photplam
            )

            tab["tot_corr"] = tot_corr
            tab.meta["TOTCFILT"] = (drz_filter, "Filter for tot_corr")
            tab.meta["TOTCWAVE"] = (drz_photplam, "PLAM for tot_corr")

            total_flux = tab["flux_auto"] * tot_corr
            tab["mag_auto"] = 23.9 - 2.5 * np.log10(total_flux)
            tab["magerr_auto"] = (
                2.5 / np.log(10) * (tab["fluxerr_auto"] / tab["flux_auto"])
            )

        # More flux columns
        for c in ["cflux", "flux", "peak", "cpeak"]:
            tab[c] *= 1.0 / uJy_to_dn
            tab[c].unit = u.uJy

        source_x, source_y = tab["x"], tab["y"]

        # Use segmentation image to mask aperture fluxes
        if aper_segmask:
            aseg = seg
            aseg_id = tab["number"]
        else:
            aseg = aseg_id = None

        # Rename some columns to look like SExtractor
        for c in ["a", "b", "theta", "cxx", "cxy", "cyy", "x2", "y2", "xy"]:
            tab.rename_column(c, c + "_image")

    else:
        if len(source_xy) == 2:
            source_x, source_y = source_xy
            aseg, aseg_id = None, None
            aper_segmask = False
        else:
            source_x, source_y, aseg, aseg_id = source_xy
            aper_segmask = True

        if hasattr(source_x, "unit"):
            if source_x.unit == u.deg:
                # Input positions are ra/dec, convert with WCS
                ra, dec = source_x, source_y
                source_x, source_y = wcs.all_world2pix(ra, dec, 0)

        tab = utils.GTable()
        tab.meta["VERSION"] = (sep.__version__, "SEP version")

    # Exposure footprints
    # --------------------
    if (exposure_footprints is not None) & ("ra" in tab.colnames):
        tab["nexp"] = catalog_exposure_overlaps(
            tab["ra"], tab["dec"], exposure_footprints=exposure_footprints
        )

        tab["nexp"].description = "Number of overlapping exposures"

    # Info
    tab.meta["ZP"] = (ZP, "AB zeropoint")
    if "PHOTPLAM" in im[0].header:
        tab.meta["PLAM"] = (im[0].header["PHOTPLAM"], "Filter pivot wave")
        if "PHOTFNU" in im[0].header:
            tab.meta["FNU"] = (im[0].header["PHOTFNU"], "Scale to Jy")

        tab.meta["FLAM"] = (im[0].header["PHOTFLAM"], "Scale to flam")

    tab.meta["uJy2dn"] = (uJy_to_dn, "Convert uJy fluxes to image DN")

    tab.meta["DRZ_FILE"] = (drz_file[:36], "SCI file")
    tab.meta["WHT_FILE"] = (weight_file[:36], "WHT file")

    tab.meta["GET_BACK"] = (get_background, "Background computed")
    for k in bkg_params:
        tab.meta[f"BACK_{k.upper()}"] = (bkg_params[k], f"Background param {k}")

    tab.meta["ERR_SCALE"] = (
        err_scale,
        "Scale factor applied to weight image (like MAP_WEIGHT)",
    )
    tab.meta["RESCALEW"] = (rescale_weight, "Was the weight applied?")

    tab.meta["APERMASK"] = (aper_segmask, "Mask apertures with seg image")

    # Photometry
    for iap, aper in enumerate(apertures):
        if sep.__version__ > "1.03":
            # Should work with the sep fork at gbrammer/sep and latest sep
            flux, fluxerr, flag = sep.sum_circle(
                data_bkg,
                source_x,
                source_y,
                aper / 2,
                err=err_data,
                gain=gain,
                subpix=subpix,
                segmap=aseg,
                seg_id=aseg_id,
                mask=mask,
            )
        else:
            tab.meta["APERMASK"] = (False, "Mask apertures with seg image - Failed")
            flux, fluxerr, flag = sep.sum_circle(
                data_bkg,
                source_x,
                source_y,
                aper / 2,
                err=err_data,
                gain=gain,
                subpix=subpix,
                mask=mask,
            )

        tab.meta["GAIN"] = gain

        tab["flux_aper_{0}".format(iap)] = flux / uJy_to_dn * u.uJy
        tab["fluxerr_aper_{0}".format(iap)] = fluxerr / uJy_to_dn * u.uJy
        tab["flag_aper_{0}".format(iap)] = flag

        if get_background:
            try:
                flux, fluxerr, flag = sep.sum_circle(
                    bkg_data,
                    source_x,
                    source_y,
                    aper / 2,
                    err=None,
                    gain=1.0,
                    segmap=aseg,
                    seg_id=aseg_id,
                    mask=mask,
                )
            except:
                flux, fluxerr, flag = sep.sum_circle(
                    bkg_data,
                    source_x,
                    source_y,
                    aper / 2,
                    err=None,
                    gain=1.0,
                    mask=mask,
                )

            tab["bkg_aper_{0}".format(iap)] = flux / uJy_to_dn * u.uJy
        else:
            tab["bkg_aper_{0}".format(iap)] = 0.0 * u.uJy

        # Count masked pixels in the aperture, not including segmask
        flux, fluxerr, flag = sep.sum_circle(
            data_mask,
            source_x,
            source_y,
            aper / 2,
            err=err_data,
            gain=gain,
            subpix=subpix,
        )

        tab["mask_aper_{0}".format(iap)] = flux

        tab.meta["aper_{0}".format(iap)] = (aper, "Aperture diameter, pix")
        tab.meta["asec_{0}".format(iap)] = (
            aper * pixel_scale,
            "Aperture diameter, arcsec",
        )

    try:
        # Free memory objects explicitly
        del data_mask
        del data
        del err_data
    except:
        pass

    # if uppercase_columns:
    for c in tab.colnames:
        tab.rename_column(c, column_case(c))

    if save_to_fits:
        tab.write(f"{root}.cat{suffix}.fits", format="fits", overwrite=True)

        if include_wcs_extension:
            try:
                hdul = fits.open(f"{root}.cat.fits", mode="update")
                wcs_hdu = fits.ImageHDU(header=wcs_header, data=None, name="WCS")
                hdul.append(wcs_hdu)
                hdul.flush()
            except:
                pass

    logstr = "# SEP {0}.cat.fits: {1:d} objects".format(root, len(tab))
    utils.log_comment(utils.LOGFILE, logstr, verbose=verbose)

    return tab
