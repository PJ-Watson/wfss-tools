"""
A module to handle PSF matching, and a wrapper to batch process images.
"""

import os
from os import PathLike
from pathlib import Path
from typing import Callable

import numpy as np
from astropy.io import fits
from astropy.nddata import block_reduce
from numpy.typing import ArrayLike

# Check for photutils deprecation
try:
    from photutils.psf_matching import (
        CosineBellWindow,
        TopHatWindow,
        TukeyWindow,
        create_matching_kernel,
    )
except:

    from photutils.psf.matching import (
        CosineBellWindow,
        TopHatWindow,
        TukeyWindow,
        create_matching_kernel,
    )

from scipy.ndimage import zoom
from wfss_tools.isophotal import align

__all__ = ["match_photutils", "match_pypher", "reproject_and_convolve"]


def match_photutils(
    source_psf: ArrayLike | PathLike,
    target_psf: ArrayLike | PathLike,
    out_path: PathLike | None = None,
    window: Callable | None = CosineBellWindow(0.5),
    oversample: int = 3,
) -> ArrayLike:
    """
    Create a PSF-matching kernel using `photutils`.

    Parameters
    ----------
    source_psf : ArrayLike | PathLike
        The source point spread function (PSF), passed as either an array
        or a path to a FITS file.
    target_psf : ArrayLike | PathLike
        The target PSF, which should be broader than the source. The shape
        and pixel scale of ``source_psf`` and ``target_psf`` must match.
    out_path : PathLike | None, optional
        The path to which the output kernel should be saved. If ``None``
        (default), the kernel will not be written to a file.
    window : Callable | None, optional
        The window function to filter high-frequency noise, by default
        `~photutils.psf_matching.CosineBellWindow(0.5)`. For further
        details see `~photutils.psf_matching.create_matching_kernel`.
    oversample : int, optional
        The factor by which the source and target PSF should be
        oversampled before computing the matching kernel, using
        `scipy.ndimage.zoom`. By default, ``oversample=3``.

    Returns
    -------
    ArrayLike
        The homogenisation kernel. When convolved with ``source_psf``,
        this should reproduce ``target_psf``.
    """

    if isinstance(source_psf, PathLike):
        source_psf = fits.getdata(source_psf)
    if oversample != 1:
        source_psf = zoom(source_psf, oversample)
    source_psf /= source_psf.sum()

    if isinstance(target_psf, PathLike):
        target_psf = fits.getdata(target_psf)
    if oversample != 1:
        target_psf = zoom(target_psf, oversample)
    target_psf /= target_psf.sum()

    phot_kernel = create_matching_kernel(source_psf, target_psf, window=window)

    if oversample != 1:
        phot_kernel = block_reduce(phot_kernel, oversample)

    phot_kernel /= phot_kernel.sum()
    if out_path is not None:
        fits.writeto(out_path, data=phot_kernel)
    return phot_kernel


def match_pypher(
    source_psf: ArrayLike | PathLike,
    target_psf: ArrayLike | PathLike,
    out_path: PathLike | None = None,
    pypher_r: float = 3e-3,
    oversample: int = 3,
    pixscale: float = 0.04,
    tmp_dir: PathLike | None = None,
) -> ArrayLike:
    """
    Create a PSF-matching kernel using `pypher`.

    Parameters
    ----------
    source_psf : ArrayLike | PathLike
        The source point spread function (PSF), passed as either an array
        or a path to a FITS file.
    target_psf : ArrayLike | PathLike
        The target PSF, which should be broader than the source. The shape
        and pixel scale of ``source_psf`` and ``target_psf`` must match.
    out_path : PathLike | None, optional
        The path to which the output kernel should be saved. If ``None``
        (default), the kernel will not be written to a file.
    pypher_r : float, optional
        The regularisation parameter for the Wiener filter. By default,
        ``pypher_r=3e-3``; for further details see `pypher`.
    oversample : int, optional
        The factor by which the source and target PSF should be
        oversampled before computing the matching kernel, using
        `scipy.ndimage.zoom`. By default, ``oversample=3``.
    pixscale : float, optional
        The pixel scale of the input PSFs, by default 0.04.
    tmp_dir : PathLike | None, optional
        The temporary directory to which the intermediate files will be
        written. By default None, in which case the current working
        directory will be used.

    Returns
    -------
    ArrayLike
        The homogenisation kernel. When convolved with ``source_psf``,
        this should reproduce ``target_psf``.
    """

    if tmp_dir is None:
        tmp_dir = Path.cwd()
    if (
        isinstance(source_psf, PathLike) or isinstance(source_psf, str)
    ) and oversample == 1:
        source_path = source_psf
    else:
        if isinstance(source_psf, PathLike) or isinstance(source_psf, str):
            source_psf = fits.getdata(source_psf)

        source_path = tmp_dir / "psf_a.fits"
        header = fits.Header()
        header["PIXSCALE"] = pixscale / oversample
        if oversample != 1:
            source_psf = zoom(source_psf, oversample)
        source_psf /= source_psf.sum()
        fits.writeto(source_path, data=source_psf, header=header, overwrite=True)

    if (
        isinstance(target_psf, PathLike) or isinstance(target_psf, str)
    ) and oversample == 1:
        target_path = target_psf
    else:
        if isinstance(target_psf, PathLike) or isinstance(target_psf, str):
            target_psf = fits.getdata(target_psf)
        target_path = tmp_dir / "psf_b.fits"
        header = fits.Header()
        header["PIXSCALE"] = pixscale / oversample
        if oversample != 1:
            target_psf = zoom(target_psf, oversample)
        target_psf /= target_psf.sum()
        fits.writeto(target_path, data=target_psf, header=header, overwrite=True)

        # fits.writeto(DIR_KERNELS+'psf_b.fits',target_psf,header=hdr,overwrite=True)
    if out_path is None:
        out_path = tmp_dir / "kernel_a_to_b.fits"

    out_path.unlink(missing_ok=True)

    os.system(f"pypher {source_path} {target_path} {out_path} -r {pypher_r:.3g}")
    with fits.open(out_path, mode="update") as kern_hdul:
        # kernel = kern_hdul[0].data
        if oversample != 1:
            kern_hdul[0].data = block_reduce(kern_hdul[0].data, oversample)

        kern_hdul[0].data /= kern_hdul[0].data.sum()
        kern_hdul.flush()

        kernel = kern_hdul[0].data.copy()

    for p in ["psf_a.fits", "psf_b.fits", "kernel_a_to_b.fits"]:
        (tmp_dir / p).unlink(missing_ok=True)

    return kernel


def reproject_and_convolve(
    ref_path: PathLike,
    orig_images: PathLike | list[PathLike],
    psfs: ArrayLike | PathLike | None | list[ArrayLike | PathLike | None],
    out_dir: PathLike,
    psf_target: ArrayLike | PathLike,
    oversample: int = 3,
    save_unconvolved: bool = True,
    new_wcs_kw: dict | None = None,
    reproject_image_kw: dict | None = None,
    psf_method: str = "pypher",
    psf_match_kw: dict | None = None,
    convolve_method: str = "fft",
    new_names: list[str] | None = None,
) -> list[PathLike]:
    """
    Align, reproject, and PSF match one or more images to a reference.

    Parameters
    ----------
    ref_path : PathLike
        The path of the image to use as a reference. All other images will
        be matched to the North-aligned reprojection of this image, at the
        same pixel scale.
    orig_images : PathLike | list[PathLike]
        The locations of one or more images, to be matched to the
        reference image.
    psfs : ArrayLike | PathLike | None | list[ArrayLike | PathLike | None]
        The corresponding PSF for each image in ``orig_image``. If any PSF
        is ``None``, the convolution step will be skipped for that image.
    out_dir : PathLike
        The directory to which the output (reprojected and convolved)
        images will be written.
    psf_target : ArrayLike | PathLike
        The PSF to which all of the input images will be matched. In most
        circumstances, this will be the PSF of ``ref_path``, but it does
        not need to be (e.g. if matching to the resolution of a ground-based
        IFU).
    oversample : int, optional
        The factor by which the source and target PSF should be
        oversampled before computing the matching kernel,
        using `scipy.ndimage.zoom`. By default, ``oversample=3``.
    save_unconvolved : bool, optional
        Whether the reprojected (but unconvolved) images should also be
        saved, by default True.
    new_wcs_kw : dict | None, optional
        Any additional keyword arguments to pass through to
        `~wfss_tools.isophotal.gen_new_wcs`, by default ``None``.
    reproject_image_kw : dict | None, optional
        Any additional keyword arguments to pass through to
        `~wfss_tools.isophotal.reproject_image`, by default ``None``.
    psf_method : {"pypher" or "photutils"}, optional
        The method used to generate the PSF-matching homogenisation
        kernel, by default ``"pypher"``.
    psf_match_kw : dict | None, optional
        Any additional keyword arguments to pass through to
        `wfss_tools.isophotal.match_photutils` or
        `wfss_tools.isophotal.match_pypher`, depending on
        ``psf_method``. By default None.
    convolve_method : {"fft", "direct"}, optional
        The method used to convolve the reprojected images with the
        homogenisation kernel. By default ``"fft"``.
    new_names : str | list[str] | None, optional
        The new base names for each original image, by default ``None``.

    Returns
    -------
    list[PathLike]
        The paths corresponding to the convolved images.
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    if new_wcs_kw is not None:
        new_wcs_kwargs = dict(new_wcs_kw or {})
        new_wcs, new_shape = align.gen_new_wcs(ref_path=ref_path, **new_wcs_kwargs)
    else:
        from astropy.wcs import WCS

        new_shape = fits.getdata(ref_path).shape
        new_wcs = WCS(fits.getheader(ref_path))

    reproject_image_kwargs = dict(reproject_image_kw or {})

    if "method" not in reproject_image_kwargs:
        reproject_image_kwargs["method"] = "adaptive"

    if convolve_method == "fft":
        from astropy.convolution import convolve_fft as conv_fn
    elif convolve_method == "direct":
        from astropy.convolution import convolve as conv_fn

    orig_images = np.atleast_1d(orig_images).ravel()
    psfs = np.atleast_1d(psfs).ravel()
    if new_names is None:
        new_names = [None] * len(orig_images)
    else:
        new_names = np.atleast_1d(new_names).ravel()

    return_paths = []

    for i, (o, p, n) in enumerate(zip(orig_images, psfs, new_names)):
        out_path = out_dir
        if n is not None:
            out_path = out_path / n
        print(out_path)
        # if out_path.is_file()
        # exit()
        repr_path = align.reproject_image(
            o, out_path, new_wcs, new_shape, **reproject_image_kwargs
        )
        # exit()

        if p is None:
            return_paths.append(None)
            continue

        p = Path(p)
        psf_target = Path(psf_target)

        if p.stem == psf_target.stem:
            conv_path = repr_path.with_name(repr_path.name.replace("repr_", "conv_"))
            if conv_path.is_file():
                print(f"'{conv_path.name}' already exists. Skipping.")
            else:
                print(
                    "Target PSF is the same as source PSF. Copying file without convolving."
                )
                import shutil

                shutil.copy(repr_path, conv_path)

            return_paths.append(conv_path)
            continue

        if (out_path.parent / f"{p.stem}_matched_{psf_target.stem}.fits").is_file():
            kernel = fits.getdata(
                out_path.parent / f"{p.stem}_matched_{psf_target.stem}.fits"
            )
        else:
            psf_match_kwargs = dict(psf_match_kw or {})
            if psf_method == "pypher":
                kernel = match_pypher(
                    p,
                    psf_target,
                    out_path=out_path.parent
                    / f"{p.stem}_matched_{psf_target.stem}.fits",
                    **psf_match_kwargs,
                )
            # import pypher
            else:
                kernel = match_photutils(p, psf_target, **psf_match_kwargs)

        kernel /= kernel.sum()

        conv_path = repr_path.with_name(repr_path.name.replace("repr_", "conv_"))
        if conv_path.is_file():
            print(f"'{conv_path.name}' already exists. Skipping.")
        else:
            with fits.open(repr_path) as repr_hdul:

                # import time
                # t0 = time.time()
                # print (t0)
                # conv_data = convolve(repr_hdul[0].data, kernel)
                # t1 = time.time()
                # print (t1-t0)
                conv_data = conv_fn(repr_hdul[0].data, kernel)
                # print (time.time()-t1)
                fits.writeto(
                    conv_path,
                    data=conv_data,
                    header=repr_hdul[0].header,
                )
        return_paths.append(conv_path)

    return return_paths
