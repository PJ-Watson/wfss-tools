"""
A module to handle alignment and reprojection of ancillary images.
"""

from copy import deepcopy
from os import PathLike
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs import utils as wcs_utils
from numpy.typing import ArrayLike
from reproject import reproject_adaptive, reproject_exact, reproject_interp
from reproject.mosaicking import find_optimal_celestial_wcs
from scipy import ndimage

__all__ = [
    "gen_new_wcs",
    "pc_to_cd",
    "pad_wcs",
    "calc_full_var",
    "reproject_image",
]


def gen_new_wcs(
    ref_path: PathLike,
    ref_hdu_index: int = 0,
    resolution: float | u.Quantity = 0.04,
    padding: int | tuple[int, int] = 100,
) -> tuple[WCS, tuple[int, int]]:
    """
    Generate a new WCS for a given reference image.

    The new WCS is North-aligned, and can optionally take on a specified
    pixel scale. Padding is applied relative to the minimal extent of the
    reference image (ignoring NaNs).

    Parameters
    ----------
    ref_path : PathLike
        The path of the reference image, to which all other images will be
        cropped.
    ref_hdu_index : int, optional
        The index of the HDU containing the reference image, by default 0.
    resolution : float | `~astropy.units.Quantity`, optional
        The pixel scale used in the new WCS. If a float is supplied, the
        units are assumed to be arcseconds. By default 0.04.
    padding : int | tuple[int, int], optional
        The padding to apply around the reference image in pixels, by
        default 100. If a tuple is supplied, the values are applied in the
        order (axis2, axis1).

    Returns
    -------
    new_wcs : `~astropy.wcs.WCS`
        The new WCS, as calculated from the input parameters.
    new_shape : (int, int)
        The shape of the array required to cover the reproject reference
        image, and the associated padding.
    """

    if isinstance(resolution, u.Quantity):
        resolution = resolution.to(u.arcsec)
    else:
        resolution *= u.arcsec

    with fits.open(ref_path) as ref_hdul:
        ref_hdu = ref_hdul[ref_hdu_index]

        min_wcs, min_shape = find_optimal_celestial_wcs(
            ref_hdu, auto_rotate=True, resolution=resolution
        )

        if len(np.atleast_1d(padding)) == 1:
            padding = (padding, padding)

        padded_wcs = pad_wcs(min_wcs, padding)
        padded_shape = (min_shape[0] + padding[1] * 2, min_shape[1] + padding[0] * 2)

        new_wcs, new_shape = find_optimal_celestial_wcs(
            (padded_shape, padded_wcs), resolution=resolution
        )

        del ref_hdu, ref_hdul

    return new_wcs, new_shape


def pc_to_cd(
    hdr_in: fits.Header,
) -> fits.Header:
    """
    Convert a PC matrix to a CD matrix.

    `astropy.wcs.WCS` recognises CD keywords as input but converts them
    and works internally with the PC matrix.
    `~astropy.wcs.WCS.to_header` returns the PC matrix even if the input
    was a CD matrix. This can cause problems if used to update an existing
    header with a CD matrix, so we convert between the two.

    Adapated from `stwcs.wcsutil.altwcs.pc2cd`.

    Parameters
    ----------
    hdr_in : `~astropy.io.fits.Header`
        The input header, containing a PC matrix.

    Returns
    -------
    `~astropy.io.fits.Header`
        The output header, containing a CD matrix.
    """
    hdr = deepcopy(hdr_in)
    cdelt1 = hdr.pop("CDELT1", 1)
    cdelt2 = hdr.pop("CDELT2", 1)
    hdr["CD1_1"] = (
        cdelt1 * hdr.pop("PC1_1", 1),
        "partial of first axis coordinate w.r.t. x",
    )
    hdr["CD1_2"] = (
        cdelt1 * hdr.pop("PC1_2", 0),
        "partial of first axis coordinate w.r.t. y",
    )
    hdr["CD2_1"] = (
        cdelt2 * hdr.pop("PC2_1", 0),
        "partial of second axis coordinate w.r.t. x",
    )
    hdr["CD2_2"] = (
        cdelt2 * hdr.pop("PC2_2", 1),
        "partial of second axis coordinate w.r.t. y",
    )
    return hdr


def pad_wcs(
    wcs_in: WCS,
    padding: int | tuple[int, int],
) -> WCS:
    """
    Add padding to all relevant WCS keywords.

    Adapted from `grizli.model.ImageData.add_padding_to_wcs`, with
    minor modifications to avoid slicing attributes of
    `~astropy.wcs.Wcsprm`, and allowing for symmetrical padding.

    Parameters
    ----------
    wcs_in : `~astropy.wcs.WCS`
        The input WCS.
    padding : int | tuple[int, int]
        The number of pixels with which to pad the input WCS.

    Returns
    -------
    `~astropy.wcs.WCS`
        The output (padded) WCS.
    """

    wcs_out = wcs_in.deepcopy()

    is_new = True
    for attr in ["naxis1", "_naxis1"]:
        if hasattr(wcs_out, attr):
            is_new = False
            value = wcs_out.__getattribute__(attr)
            if value is not None:
                wcs_out.__setattr__(attr, value + 2 * padding[1])

    for attr in ["naxis2", "_naxis2"]:
        if hasattr(wcs_out, attr):
            is_new = False
            value = wcs_out.__getattribute__(attr)
            if value is not None:
                wcs_out.__setattr__(attr, value + 2 * padding[0])

    # Handle changing astropy.wcs.WCS attributes
    if is_new:
        wcs_out._naxis[0] += 2 * padding[1]
        wcs_out._naxis[1] += 2 * padding[0]

        wcs_out.naxis1, wcs_out.naxis2 = wcs_out._naxis
    else:
        wcs_out.naxis1 = wcs_out._naxis1
        wcs_out.naxis2 = wcs_out._naxis2

    # Mutable attribute of `astropy.wcs.wcsprm`
    wcs_out.wcs.crpix = np.array(
        [wcs_out.wcs.crpix[0] + padding[1], wcs_out.wcs.crpix[1] + padding[0]]
    )

    # Pad CRPIX for SIP
    for wcs_ext in [wcs_out.sip]:
        if wcs_ext is not None:
            wcs_ext.crpix[0] += np.array(
                [wcs_ext.crpix[0] + padding[1], wcs_ext.crpix[1] + padding[0]]
            )

    # Pad CRVAL for Lookup Table, if necessary (e.g., ACS)
    for wcs_ext in [wcs_out.cpdis1, wcs_out.cpdis2, wcs_out.det2im1, wcs_out.det2im2]:
        if wcs_ext is not None:
            wcs_ext.crval = np.array(
                [wcs_ext.crval[0] + padding[1], wcs_ext.crval[1] + padding[0]]
            )

    return wcs_out


def calc_full_var(
    sci_path: PathLike,
    wht_path: PathLike,
    exp_path: PathLike,
    sci_hdu_index: int = 0,
    wht_hdu_index: int = 0,
    exp_hdu_index: int = 0,
    compress: bool = True,
    overwrite: bool = False,
) -> PathLike:
    """
    Calculate the full variance array from "sci", "wht", and "exp" images.

    These images are assumed to have been processed by `grizli`. The "exp"
    image can be reconstructed from a downsampled array if necessary. The
    output array will be saved to the same directory as `exp_path`.

    Parameters
    ----------
    sci_path : PathLike
        The path of the science image, containing the measured fluxes.
    wht_path : PathLike
        The path of the weight image, containing the inverse variance from
        the sky and read noise.
    exp_path : PathLike
        The path of the exposure image, containing the exposure time map.
    sci_hdu_index : int, optional
        The index of the HDU containing the science image, by default 0.
    wht_hdu_index : int, optional
        The index of the HDU containing the weight image, by default 0.
    exp_hdu_index : int, optional
        The index of the HDU containing the exposure time map, by
        default 0.
    compress : bool, optional
        If True (default), the output file will be compressed and saved
        with the extension ".fits.gz".
    overwrite : bool, optional
        Overwrite the file if it exists already, by default False.

    Returns
    -------
    PathLike
        The path of the new variance array.
    """

    sci_path = Path(sci_path)
    exp_path = Path(exp_path)
    wht_path = Path(wht_path)

    output_path = deepcopy(exp_path)
    if output_path.suffix == ".gz":
        output_path = output_path.with_suffix("")

    output_path = output_path.with_name(
        f"{output_path.name}{".gz" if compress else ""}".replace("_exp", "_var")
    )

    if output_path.is_file() and (not overwrite):
        print(
            f"'{output_path.name}' already exists. Set `overwrite=True` "
            "if you wish to overwrite this file."
        )
        return output_path

    else:

        with fits.open(exp_path) as exp_hdul:

            exp_hdr = exp_hdul[exp_hdu_index].header
            sample = int(exp_hdr.get("SAMPLE", 1))

            with fits.open(sci_path) as sci_hdul:
                if sample == 1:
                    full_exp = exp_hdul[exp_hdu_index].data
                else:
                    full_exp = np.memmap(
                        output_path.parent / "temp_exp.dat",
                        dtype=float,
                        mode="w+",
                        shape=(sci_hdul[sci_hdu_index].data.shape),
                    )
                    # Grow the exposure map to the original frame
                    # full_exp = np.zeros(sci_hdul[sci_hdu_index].data.shape, dtype=int)
                    full_exp[int(sample / 2) :: sample, int(sample / 2) :: sample] += (
                        exp_hdul[exp_hdu_index].data * 1.0
                    )
                    full_exp = ndimage.maximum_filter(full_exp, sample, output=full_exp)

                del exp_hdul[exp_hdu_index].data

                print("Grown full exp map")

                # Multiplicative factors that have been applied since the original count-rate images
                phot_scale = 1.0

                for k in ["PHOTMJSR", "PHOTSCAL"]:
                    # print(f"{k} {exp_hdr[k]:.3f}")
                    phot_scale /= exp_hdr.get(k, 1)

                # Unit and pixel area scale factors
                if "OPHOTFNU" in exp_hdr:
                    phot_scale *= exp_hdr.get("PHOTFNU", 1) / exp_hdr.get("OPHOTFNU", 1)

                # "effective_gain" = electrons per DN of the mosaic
                # effective_gain = phot_scale * full_exp
                full_exp *= phot_scale
                print("Calculated effective gain")

                # Poisson variance in mosaic DN
                # var_poisson_dn = (
                #     np.maximum(sci_hdul[sci_hdu_index].data, 0) / effective_gain
                # )
                full_exp /= np.maximum(sci_hdul[sci_hdu_index].data, 0)
                full_exp = full_exp ** (-1)
                del sci_hdul[sci_hdu_index].data

                print("About to open wht")

                with fits.open(wht_path) as wht_hdul:
                    # Original variance from the `wht` image = RNOISE + BACKGROUND
                    # var_wht = 1 / wht_hdul[wht_hdu_index].data
                    full_exp += 1 / wht_hdul[wht_hdu_index].data

                    del wht_hdul[wht_hdu_index].data

                    new_hdr = sci_hdul[0].header.copy()
                    # new_hdr[]

                    # New total variance
                    # var_total = var_wht + var_poisson_dn

                    print("Making HDUList")
                    var_hdul = fits.HDUList(
                        [fits.PrimaryHDU(data=full_exp, header=new_hdr)]
                    )
                    print("Writing output")

                    var_hdul.writeto(output_path, overwrite=True)

        return output_path


def reproject_image(
    input_path: PathLike,
    output_path: PathLike,
    output_wcs: WCS,
    output_shape: tuple[int, int],
    input_hdu_index: int = 0,
    method: str = "adaptive",
    conserve_flux: bool = True,
    compress: bool = True,
    overwrite: bool = False,
    prefix: str = "repr_",
    preliminary_buffer: int | None = 250,
    **reproject_kwargs,
) -> PathLike:
    """
    Reproject an image to a new WCS.

    A wrapper around `reproject`, for file handling and default
    parameters. If `~reproject.reproject_adaptive` is used as the
    method, the flux is conserved by default (``conserve_flux=True``). For
    other methods, the input is assumed to be in surface brightness units.

    Parameters
    ----------
    input_path : PathLike
        The path of the original image.
    output_path : PathLike
        The path of the output file. If a directory is passed, the output
        filename will be the same as the input, with the addition of
        ``prefix``.
    output_wcs : `~astropy.wcs.WCS`
        The new WCS onto which the image will be reprojected.
    output_shape : tuple[int,int]
        The shape of the reprojected image.
    input_hdu_index : int, optional
        The index of the HDU containing the input image, by default 0.
    method : {"interp", "adaptive", "exact"}
        The reprojection method to use, by default ``"adaptive"``.
    conserve_flux : bool, optional
        If True (default), the input flux will be conserved. This will
        only work if ``method="adaptive"``.
    compress : bool, optional
        If True (default), the output file will be compressed and saved
        with the extension ".fits.gz".
    overwrite : bool, optional
        Overwrite the file if it exists already, by default False.
    prefix : str, optional
        The string to prepend to the output filename, by default
        ``"repr_"``. Only used if ``output_path`` is a directory.
    preliminary_buffer : int | None, optional
        For very large images, perform a rough initial crop, padded by
        ``preliminary_buffer``, before the full reprojection. By default,
        this is set to 250 pixels.
    **reproject_kwargs : dict, optional
        Additional keyword arguments to pass to the reproject function.

    Returns
    -------
    PathLike
        The path of the reprojected image.

    Raises
    ------
    ValueError
        An error will be raised if the ``method`` argument is not valid.

    Notes
    -----
    To reproject a segmentation map, call this function with
    ``method=interp`` and ``order="nearest-neighbour"``.
    """

    if output_path.is_dir():
        _path = deepcopy(Path(input_path))
        if _path.suffix == ".gz":
            _path = _path.with_suffix("")

        output_path = (
            Path(output_path) / f"{prefix}{_path.name}{".gz" if compress else ""}"
        )
    # else:
    #     output_path = output_path.parent / output_path.name.with_name

    if output_path.is_file() and (not overwrite):
        print(
            f"'{output_path.name}' already exists. Set `overwrite=True` "
            "if you wish to overwrite this file."
        )
        return output_path

    else:

        match method:
            case "interp":
                reproject_fn = reproject_interp
            case "adaptive":
                reproject_fn = reproject_adaptive
                if "conserve_flux" not in reproject_kwargs.keys():
                    reproject_kwargs["conserve_flux"] = conserve_flux
            case "exact":
                reproject_fn = reproject_exact
            case _:
                raise ValueError(
                    f'"{method}" is not a valid reproject function. '
                    'Please supply one of "interp", "adaptive", or '
                    '"exact".'
                )

        if "parallel" not in reproject_kwargs.keys():
            reproject_kwargs["parallel"] = True
        if "block_size" not in reproject_kwargs.keys():
            reproject_kwargs["block_size"] = "auto"
        if "return_footprint" not in reproject_kwargs.keys():
            reproject_kwargs["return_footprint"] = False

        with fits.open(input_path) as orig_hdul:

            print("Opened")

            out_hdul = fits.HDUList()
            out_hdr = deepcopy(orig_hdul[input_hdu_index].header)

            hdr_partial = output_wcs.to_header()

            # Necessary if working with cd matrices,
            # else hdr not updated properly
            if WCS(out_hdr).wcs.has_cd:
                hdr_partial = pc_to_cd(hdr_partial)

            out_hdr.update(hdr_partial)
            print("Updated header")

            # print ()
            if preliminary_buffer is not None:
                print("Using buffer")
                corners = np.array(
                    [
                        [-preliminary_buffer, -preliminary_buffer],
                        [-preliminary_buffer, output_shape[1] + preliminary_buffer],
                        [
                            preliminary_buffer + output_shape[0],
                            output_shape[1] + preliminary_buffer,
                        ],
                        [preliminary_buffer + output_shape[0], -preliminary_buffer],
                    ]
                )

                orig_wcs = WCS(orig_hdul[input_hdu_index].header.copy())
                # # orig_data = orig_hdul[input_hdu_index].data
                # print (dir(orig_wcs))
                # print (orig_wcs.array_shape)
                # # print (getattr(orig_wcs, "naxis1",getattr(orig_wcs, "_naxis1")))
                # print (orig_hdul[input_hdu_index].data.shape)

                proj_corners = wcs_utils.pixel_to_pixel(
                    output_wcs, orig_wcs, *corners.T
                )
                crop_slice = (
                    slice(
                        int(np.nanmax([np.nanmin(proj_corners[1]), 0])),
                        int(
                            np.nanmin(
                                [np.nanmax(proj_corners[1]), orig_wcs.array_shape[1]]
                            )
                        ),
                    ),
                    slice(
                        int(np.nanmax([np.nanmin(proj_corners[0]), 0])),
                        int(
                            np.nanmin(
                                [np.nanmax(proj_corners[0]), orig_wcs.array_shape[0]]
                            )
                        ),
                    ),
                )
            print("About to reproject")
            out_dtype = orig_hdul[input_hdu_index].data.dtype

            repr_output = reproject_fn(
                # orig_hdul[input_hdu_index],
                (
                    (orig_hdul[input_hdu_index].data[crop_slice], orig_wcs[crop_slice])
                    if preliminary_buffer is not None
                    else orig_hdul[input_hdu_index]
                ),
                output_wcs,
                shape_out=output_shape,
                **reproject_kwargs,
            )

            # from drizzlepac import adrizzle
            # from grizli import utils as grizli_utils

            # adrizzle.log.setLevel("ERROR")
            # drizzler = adrizzle.do_driz

            # orig_data = orig_hdul[input_hdu_index].data
            # orig_wcs = WCS(orig_hdul[input_hdu_index])

            # orig_wcs.pscale = np.abs(grizli_utils.get_wcs_pscale(orig_wcs))
            # output_wcs.pscale = np.abs(grizli_utils.get_wcs_pscale(output_wcs))

            # print (orig_wcs.pscale, output_wcs.pscale)
            # # exit()

            # sh = output_shape
            # outsci = np.zeros(sh, dtype=np.float32)
            # outwht = np.zeros(sh, dtype=np.float32)
            # outctx = np.zeros(sh, dtype=np.int32)

            # drizzler(
            #     # orig_data.astype(np.float32),
            #     # orig_wcs,
            #     # np.ones_like(orig_data).astype(np.float32),
            #     orig_data[crop_slice].astype(np.float32),
            #     orig_wcs[crop_slice],
            #     np.ones_like(orig_data[crop_slice]).astype(np.float32),
            #     output_wcs,
            #     outsci,
            #     outwht,
            #     outctx,
            #     1.0,
            #     "cps",
            #     1,
            #     wcslin_pscale=orig_wcs.pscale,
            #     uniqid=1,
            #     pixfrac=1.0,
            #     kernel="square",
            #     fillval="0",
            #     wcsmap=grizli_utils.WCSMapAll,
            # )

            # area_ratio = (orig_wcs.pscale / output_wcs.pscale)**2

            # spatial_scale = 1.0
            # flux_density_scale = spatial_scale**2

            # outsci *= area_ratio * flux_density_scale

            # repr_output = outsci

            # Check if footprint array returned
            if type(repr_output) == tuple:
                out_hdul.append(
                    fits.ImageHDU(
                        data=repr_output[0].astype(out_dtype),
                        header=out_hdr,
                    )
                )
                out_hdul.append(
                    fits.ImageHDU(
                        data=repr_output[1],
                        header=out_hdr,
                        name="FOOTPRINT",
                    )
                )
            else:
                out_hdul.append(
                    fits.ImageHDU(
                        data=repr_output.astype(out_dtype),
                        header=out_hdr,
                    )
                )

            if reproject_kwargs.get("order") in ["nearest-neighbor", 0]:
                out_hdul[0].data[~(out_hdul[0].data >= 0)] = 0

            out_hdul.writeto(output_path, overwrite=True)

        return output_path
