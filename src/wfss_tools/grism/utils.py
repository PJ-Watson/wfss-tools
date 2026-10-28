"""
General utility functions related to handling grism data.
"""

import datetime
from copy import deepcopy
from os import PathLike
from pathlib import Path

import astropy
import grizli.utils as grizli_utils
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from grizli import utils as grizli_utils
from grizli.model import BeamCutout, GrismDisperser
from grizli.multifit import MultiBeam
from numpy.typing import ArrayLike
from tqdm import tqdm

__all__ = [
    "gen_stacked_beams",
    "align_direct_images",
    "gen_psf",
    "log_with_offset",
    "LINE_UP",
    "LINE_CLEAR",
]

LINE_UP = "\033[1A"
LINE_CLEAR = "\x1b[2K"


def log_with_offset(
    s: str,
    blank_lines: int = 1,
    curr_line: str = "",
    logfile: Path | None = None,
    verbose: bool = True,
    timestamp: bool = True,
) -> None:
    """
    Print to previous lines on the console.

    Parameters
    ----------
    s : str
        The string to print.
    blank_lines : int, optional
        How many blank lines to leave between the current one and the
        previous, by default ``1``.
    curr_line : str, optional
        The string to print to the current line, by default ``""``.
    logfile : Path | None, optional
        The file to write all information to. If ``None`` (default),
        logs will only be printed to the console.
    verbose : bool, optional
        If ``True`` (default), logs will be printed to the console.
    timestamp : bool, optional
        If ``True`` (default), messages in ``logfile`` will include the
        current timestamp.
    """

    if verbose:
        print(
            (blank_lines + 1) * LINE_UP
            + LINE_CLEAR
            + s
            + (blank_lines + 1) * ("\n" + LINE_CLEAR)
            + curr_line,
            flush=True,
        )

    if timestamp:
        s = f"[{datetime.datetime.now()}] {s}"

    if logfile is not None:
        with open(logfile, "a") as f:
            print(s, file=f, flush=True)

    return


def _cluster_beams_centres(
    mb: MultiBeam, beam_idxs: ArrayLike, dbscan_kwargs: dict = {}
) -> ArrayLike:
    """
    Perform DBSCAN clustering.
    """

    from sklearn.cluster import DBSCAN

    # Detector coordinate centres
    detector_coords = np.array(
        [[mb.beams[b].beam.xc, mb.beams[b].beam.yc] for b in beam_idxs]
    )

    db = DBSCAN(**dbscan_kwargs).fit(detector_coords)
    labels = np.array(db.labels_)

    # print (labels)

    grouped_beam_idxs = [beam_idxs[labels == u] for u in np.unique(labels)]

    return grouped_beam_idxs


def gen_stacked_beams(
    mb: str | MultiBeam,
    pixfrac: float = 1.0,
    kernel: str = "square",
    dfillval: float = 0,
    fit_trace_shift: bool = False,
    trace_shift_kwargs: dict = {},
    cluster_beams: bool = False,
    dbscan_kwargs: dict = {"eps": 5, "min_samples": 3},
    split_by_fwcpos: bool = True,
    **multibeam_kwargs,
):
    """
    Stack individual beams with the same grism and blocking filter.

    This returns a "master" `~grizli.multifit.MultiBeam` object, with a
    single beam for each combination of grism orientation and blocking
    filter.

    Parameters
    ----------
    mb : str | `grizli.multifit.MultiBeam`
        The original MultiBeam object, or the location of the
        ``*beams.fits.`` file.
    pixfrac : float, optional
        The fraction by which input pixels are “shrunk” before being
        drizzled onto the output image grid, given as a real number
        between 0 and 1. This specifies the size of the footprint, or
        “dropsize”, of a pixel in units of the input pixel size. By
        default ``pixfrac=1.0``.
    kernel : str, optional
        The form of the kernel function used to distribute flux onto the
        separate output images, by default ``"square"``. The current
        options are ``"square"``, ``"point"``, ``"turbo"``,
        ``"gaussian"``, and ``"lanczos3"``.
    dfillval : float, optional
        The value to be assigned to output pixels that have zero weight,
        or that do not receive flux from any input pixels during
        drizzling. By default this is 0.
    fit_trace_shift : bool, optional
        Fit for a cross-dispersion offset before stacking the beams, using
        `~grizli.multifit.MultiBeam.fit_trace_shift()`. By default
        ``False``.
    trace_shift_kwargs : dict, optional
        Additional keyword arguments to pass through to
        `~grizli.multifit.MultiBeam.fit_trace_shift()` if used.
    cluster_beams : bool, optional
        Cluster the beams based on their detector position before
        stacking, using the DBSCAN algorithm. This can minimise residuals
        arising from the trace variation across the detector, at the cost
        of an increased number of stacked beams. By default ``False``.
    dbscan_kwargs : dict, optional
        Any additional parameters to pass through to
        `sklean.cluster.DBSCAN`. By default ``dbscan_kwargs={"eps": 5,
        "min_samples": 3}``.
    split_by_fwcpos : bool, optional
        Split beams by the NIRISS filter wheel position angle, in addition
        to the blocking filter and grism. By default ``True``.
    **multibeam_kwargs : dict, optional
        Any additional parameters to pass through to
        `~grizli.multifit.MultiBeam` when loading the original object.

    Returns
    -------
    `~grizli.multifit.MultiBeam`
        The stacked multibeam object.
    """

    # if type(mb) is str:
    if not isinstance(mb, MultiBeam):
        mb = MultiBeam(beams=mb, **multibeam_kwargs)

    if fit_trace_shift:
        mb.fit_trace_shift(**trace_shift_kwargs)

    from drizzlepac import adrizzle

    adrizzle.log.setLevel("ERROR")
    drizzler = adrizzle.do_driz

    new_beam_list = []

    for filt, pa_info in tqdm(mb.PA.items(), desc="Stacking beams"):
        for pa, pa_beam_idxs in pa_info.items():

            pa_beam_idxs = np.array(pa_beam_idxs)

            if split_by_fwcpos:
                # Round to avoid numerical noise
                fwcpos_arr = np.round(
                    np.asarray([mb.beams[b].beam.fwcpos for b in pa_beam_idxs]), 3
                )
                grouped_beam_idxs = []

                unique, u_inv = np.unique(fwcpos_arr, return_inverse=True)

                for i, u in enumerate(unique):
                    if cluster_beams:
                        grouped_beam_idxs.extend(
                            _cluster_beams_centres(
                                mb, pa_beam_idxs[u_inv == i], dbscan_kwargs
                            )
                        )
                    else:
                        grouped_beam_idxs.extend(np.asarray([pa_beam_idxs[u_inv == i]]))
            elif cluster_beams:
                grouped_beam_idxs = _cluster_beams_centres(
                    mb, pa_beam_idxs, dbscan_kwargs
                )
            else:
                grouped_beam_idxs = [pa_beam_idxs]

            for beam_idxs in grouped_beam_idxs:

                # As a reference beam, we use the one with the smallest shift from the centre
                # along the x-axis
                # This minimises the chance of trace pixel errors due to integer rounding
                # in the grizli and grismconf code
                direct_cen = (
                    np.asarray(
                        mb.beams[beam_idxs[0]].direct.data["REF"].shape[::-1]
                    )  # + 1
                ) / 2

                shift_dx = np.zeros((len(beam_idxs), 2))
                for i, b_i in enumerate(beam_idxs):
                    shift_dx[i] = direct_cen - np.array(
                        mb.beams[b_i]
                        .direct.wcs.all_world2pix(
                            [[mb.ra, mb.dec]],
                            1,
                            ra_dec_order=True,
                        )
                        .flatten()
                    )

                new_beam = deepcopy(
                    mb.beams[
                        beam_idxs[
                            np.argmin(np.abs(shift_dx[:, 0] - np.round(shift_dx[:, 0])))
                        ]
                    ]
                )

                # Set centre of direct image to the actual coordinates
                shift_crpix = direct_cen - np.array(
                    new_beam.direct.wcs.all_world2pix(
                        [[mb.ra, mb.dec]],
                        1,
                        ra_dec_order=True,
                    ).flatten()
                )

                new_beam.grism.wcs = grizli_utils.transform_wcs(
                    new_beam.grism.wcs,
                    translation=[
                        shift_crpix[0] - new_beam.beam.xoffset,
                        shift_crpix[1] - new_beam.beam.yoffset,
                    ],
                )
                new_beam.direct.wcs = grizli_utils.transform_wcs(
                    new_beam.direct.wcs, translation=shift_crpix
                )

                sh = new_beam.sh
                outsci = np.zeros(sh, dtype=np.float32)
                outwht = np.zeros(sh, dtype=np.float32)
                outctx = np.zeros(sh, dtype=np.int32)

                outvar = np.zeros(sh, dtype=np.float32)
                outwv = np.zeros(sh, dtype=np.float32)
                outcv = np.zeros(sh, dtype=np.int32)

                outcon = np.zeros(sh, dtype=np.float32)
                outwc = np.zeros(sh, dtype=np.float32)
                outcc = np.zeros(sh, dtype=np.int32)

                outdir = np.zeros(new_beam.direct.data["REF"].shape, dtype=np.float32)
                outwd = np.zeros(new_beam.direct.data["REF"].shape, dtype=np.float32)
                outcd = np.zeros(new_beam.direct.data["REF"].shape, dtype=np.int32)

                grism_data = [mb.beams[i].grism.data["SCI"] for i in beam_idxs]
                direct_data = [mb.beams[i].direct.data["REF"] for i in beam_idxs]

                dir_scale = np.nanmedian(
                    new_beam.direct.data["REF"] / new_beam.beam.direct
                )

                new_seg = grizli_utils.blot_nearest_exact(
                    mb.beams[beam_idxs[0]].beam.seg,
                    mb.beams[beam_idxs[0]].direct.wcs,
                    new_beam.direct.wcs,
                    verbose=False,
                    stepsize=-1,
                    scale_by_pixel_area=False,
                )

                for i, idx in enumerate(beam_idxs):

                    beam = mb.beams[idx]
                    direct_wcs_i = beam.direct.wcs.copy()
                    grism_wcs_i = grizli_utils.transform_wcs(
                        beam.grism.wcs.copy(),
                        translation=[-beam.beam.xoffset, -beam.beam.yoffset],
                    )

                    # contam_weight = np.exp(
                    #     -(mb.fcontam * np.abs(beam.contam) * np.sqrt(beam.ivar))
                    # )
                    contam_weight = np.ones_like(beam.ivar)
                    # grism_wht = beam.ivar * contam_weight
                    grism_wht = beam.ivar
                    grism_wht[~np.isfinite(grism_wht)] = 0.0
                    contam_wht = beam.ivar
                    contam_wht[~np.isfinite(contam_wht)] = 0.0

                    drizzler(
                        direct_data[i],
                        direct_wcs_i,
                        np.ones_like(direct_data[i]),
                        new_beam.direct.wcs,
                        outdir,
                        outwd,
                        outcd,
                        1.0,
                        "cps",
                        1,
                        wcslin_pscale=1.0,
                        uniqid=1,
                        pixfrac=pixfrac,
                        kernel=kernel,
                        fillval=dfillval,
                        wcsmap=grizli_utils.WCSMapAll,
                    )
                    drizzler(
                        grism_data[i],
                        grism_wcs_i,
                        grism_wht,
                        new_beam.grism.wcs,
                        outsci,
                        outwht,
                        outctx,
                        1.0,
                        "cps",
                        1,
                        wcslin_pscale=1.0,
                        uniqid=1,
                        pixfrac=pixfrac,
                        kernel=kernel,
                        fillval=dfillval,
                        wcsmap=grizli_utils.WCSMapAll,
                    )
                    drizzler(
                        beam.contam,
                        grism_wcs_i,
                        contam_wht,
                        new_beam.grism.wcs,
                        outcon,
                        outwc,
                        outcc,
                        1.0,
                        "cps",
                        1,
                        wcslin_pscale=1.0,
                        uniqid=1,
                        pixfrac=pixfrac,
                        kernel=kernel,
                        fillval=dfillval,
                        wcsmap=grizli_utils.WCSMapAll,
                    )

                    drizzler(
                        contam_weight,
                        grism_wcs_i,
                        grism_wht,
                        new_beam.grism.wcs,
                        outvar,
                        outwv,
                        outcv,
                        1.0,
                        "cps",
                        1,
                        wcslin_pscale=1.0,
                        uniqid=1,
                        pixfrac=pixfrac,
                        kernel=kernel,
                        fillval=dfillval,
                        wcsmap=grizli_utils.WCSMapAll,
                    )

                # Correct for drizzle scaling
                area_ratio = 1.0 / new_beam.grism.wcs.pscale**2

                # preserve flux density
                spatial_scale = 1.0
                flux_density_scale = spatial_scale**2

                # science
                outsci *= area_ratio * flux_density_scale
                # Direct
                outdir *= area_ratio * flux_density_scale
                # Variance
                outvar *= area_ratio / outwv * flux_density_scale**2
                outwht = 1 / outvar
                outwht[(outvar == 0) | (~np.isfinite(outwht))] = 0
                # Contam
                outcon *= area_ratio * flux_density_scale

                new_beam.grism.data["SCI"] = outsci
                new_beam.grism.data["ERR"] = np.sqrt(outvar)
                new_beam.grism.data["DQ"] = np.zeros_like(outsci)
                new_beam.contam = outcon
                new_beam.direct.data["REF"] = outdir
                new_beam.direct.header.update(
                    grizli_utils.to_header(new_beam.direct.wcs)
                )
                new_beam.grism.header.update(grizli_utils.to_header(new_beam.grism.wcs))

                new_beam.beam = GrismDisperser(
                    id=mb.id,
                    direct=outdir,
                    segmentation=new_seg,
                    origin=np.nanmedian(
                        np.asarray([mb.beams[i].direct.origin for i in beam_idxs]),
                        axis=0,
                    ),
                    pad=np.nanmedian(
                        np.asarray([mb.beams[i].direct.pad for i in beam_idxs]),
                        axis=0,
                    ),
                    grow=np.nanmedian(
                        np.asarray([mb.beams[i].direct.grow for i in beam_idxs]),
                        axis=0,
                    ),
                    beam=mb.beams[beam_idxs[0]].beam.beam,
                    xcenter=0,
                    ycenter=0,
                    conf=mb.beams[beam_idxs[0]].beam.conf,
                    fwcpos=mb.beams[beam_idxs[0]].beam.fwcpos,
                    MW_EBV=mb.beams[beam_idxs[0]].beam.MW_EBV,
                    xoffset=0.0,
                    yoffset=0.0,
                )
                new_beam.beam.compute_model()

                new_beam.modelf = new_beam.beam.modelf
                new_beam.model = new_beam.beam.modelf.reshape(new_beam.beam.sh_beam)
                # new_beam.compute_model()
                new_beam._parse_from_data(
                    isJWST=True,
                    contam_sn_mask=[10, 3],
                    min_mask=mb.min_mask,
                    min_sens=mb.min_sens,
                    mask_resid=mb.mask_resid,
                )

                new_beam.direct.data["REF"] /= dir_scale
                new_beam.direct.ref_photflam = new_beam.direct.photflam

                new_beam_list.append(new_beam)

    new_multibeam = MultiBeam(
        new_beam_list,
        group_name=mb.group_name,
        fcontam=mb.fcontam,
        min_mask=mb.min_mask,
        min_sens=mb.min_sens,
        mask_resid=mb.mask_resid,
    )

    return new_multibeam


def align_direct_images(
    ref_beam: BeamCutout,
    info_dict: dict,
    out_dir: PathLike = None,
    cutout=200,
    overwrite: bool = False,
) -> dict:
    """
    Align a set of images to the orientation of a dispersed beam.

    Given a nested dictionary, containing both ``"sci"`` and ``"var"``
    keys pointing to the location of the images, this blots the images
    to the same coordinate system used in the direct image of a
    `grizli.model.BeamCutout`.

    Parameters
    ----------
    ref_beam : BeamCutout
        The dispersed beam to be used as a reference. All images will be
        aligned to the direct image in this beam.
    info_dict : dict
        A nested dictionary, where each value is a dictionary containing
        ``"sci"`` and ``"var"`` keys. The values for these should point
        to the location of the original FITS images to be blotted.
    out_dir : PathLike, optional
        The location in which the realigned images will be saved. This
        will default to the current working directory.
    cutout : int, optional
        Make a slice of the original image with size ``[-cutout,+cutout]``
        around the centre position of the desired object, before passing
        to blot. By default, ``cutout=200``.
    overwrite : bool, optional
        Overwrite existing images if they exist already. By default
        ``False``.

    Returns
    -------
    dict
        An updated version of ``info_dict``, now with the locations of the
        realigned images.
    """
    from drizzlepac.astrodrizzle import ablot

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(exist_ok=True, parents=True)
    else:
        out_dir = Path.cwd()

    beam_wcs = ref_beam.direct.wcs

    beam_ra, beam_dec = beam_wcs.all_pix2world(
        [(np.asarray(ref_beam.direct.sh) + 1) / 2],
        1,
    ).flatten()

    new_info_dict = info_dict.copy()
    for k, v in info_dict.items():
        for img_type in ["sci", "var"]:
            if (not (out_dir / Path(v[img_type]).name).is_file()) or overwrite:
                with fits.open(v[img_type]) as orig_hdul:
                    orig_data = orig_hdul[0].data
                    orig_header = orig_hdul[0].header
                    orig_image_filename = Path(orig_hdul.filename()).name

                    if orig_data.dtype not in [np.float32, np.dtype(">f4")]:
                        orig_data = orig_data.astype(np.float32)

                    orig_wcs = WCS(orig_header, relax=True)
                    orig_wcs.pscale = grizli_utils.get_wcs_pscale(orig_wcs)

                    if not hasattr(orig_wcs, "_naxis1") & hasattr(orig_wcs, "_naxis"):
                        orig_wcs._naxis1, orig_wcs._naxis2 = orig_wcs._naxis

                    if "PHOTPLAM" in orig_header:
                        orig_photplam = orig_header["PHOTPLAM"]
                    else:
                        orig_photplam = 1.0

                    if "PHOTFLAM" in orig_header:
                        orig_photflam = orig_header["PHOTFLAM"]
                    else:
                        orig_photflam = 1.0

                    try:
                        orig_filter = grizli_utils.parse_filter_from_header(orig_header)
                    except:
                        orig_filter = "N/A"

                    xy = np.asarray(
                        np.round(orig_wcs.all_world2pix([beam_ra], [beam_dec], 0)),
                        dtype=int,
                    ).flatten()

                    sh = orig_data.shape
                    slx = slice(
                        np.maximum(xy[0] - cutout, 0), np.minimum(xy[0] + cutout, sh[1])
                    )
                    sly = slice(
                        np.maximum(xy[1] - cutout, 0), np.minimum(xy[1] + cutout, sh[0])
                    )

                    if hasattr(beam_wcs, "idcscale"):
                        if beam_wcs.idcscale is None:
                            delattr(beam_wcs, "idcscale")

                    if not hasattr(beam_wcs, "_naxis1") & hasattr(beam_wcs, "_naxis"):
                        beam_wcs._naxis1, beam_wcs._naxis2 = beam_wcs._naxis

                    blotted = ablot.do_blot(
                        orig_data[sly, slx],
                        orig_wcs.slice([sly, slx]),
                        beam_wcs,
                        1,
                        coeffs=True,
                        interp="sinc",
                        sinscl=1.0,
                        stepsize=1,
                        wcsmap=grizli_utils.WCSMapAll,
                    )

                    orig_header.update(beam_wcs.to_header())

                    new_hdul = fits.HDUList()
                    new_hdul.append(
                        fits.ImageHDU(
                            data=blotted.astype(np.float32),
                            header=orig_header,
                        )
                    )
                    new_hdul.writeto(
                        (out_dir / Path(v[img_type]).name), overwrite=overwrite
                    )
            new_info_dict[k][img_type] = str(out_dir / Path(v[img_type]).name)

    return new_info_dict


def gen_psf(multibeam: MultiBeam) -> dict:
    """
    Generate a PSF aligned with the direct image in extracted beams.

    The PSF matches the rotation of the direct imaging using the
    ``"PA_APER"`` header keyword.

    Parameters
    ----------
    multibeam : MultiBeam
        The grizli-extracted multiple beams object.

    Returns
    -------
    dict
        A dictionary with keys corresponding to each unique grism and filter
        combination, and values of the PSF image.
    """

    import stpsf
    from drizzlepac.astrodrizzle import ablot

    psf_aligned_images = {}

    for i, (beam_cutout, cutout_shape) in enumerate(
        zip(multibeam.beams, multibeam.Nflat)
    ):
        beam_name = f"{beam_cutout.grism.pupil}-{beam_cutout.grism.filter}"
        if not beam_name in psf_aligned_images:
            header = beam_cutout.direct.header
            beam_wcs = beam_cutout.direct.wcs
            inst = stpsf.instrument(header["INSTRUME"])
            inst.set_position_from_aperture_name("NIS_CEN")
            inst.filter = header["PUPIL"]
            dateobs = astropy.time.Time(
                header["DATE-BEG"]
            )  # + 'T' + header['TIME-OBS'])
            inst.load_wss_opd_by_date(
                dateobs, verbose=False, plot=False, choice="closest"
            )
            psf = inst.calc_psf(
                fov_pixels=np.nanmax(beam_wcs._naxis) * 2 + 1,
            )
            psf_data = psf["DET_DIST"].data
            psf_wcs = WCS(psf["DET_DIST"])
            psf_wcs.wcs.crpix = (np.asarray(psf_data.shape) + 1) / 2
            psf_wcs.wcs.crval = [multibeam.ra, multibeam.dec]
            rotation_angle_rad = np.radians(header["PA_APER"] - 360)
            psf_wcs.wcs.cd = (
                np.array(
                    [
                        [np.cos(rotation_angle_rad), -np.sin(rotation_angle_rad)],
                        [np.sin(rotation_angle_rad), np.cos(rotation_angle_rad)],
                    ]
                )
                * (inst.pixelscale * u.arcsec).to(u.deg).value
            )
            psf_wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]

            psf_wcs.pscale = grizli_utils.get_wcs_pscale(psf_wcs)

            blotted = ablot.do_blot(
                psf_data.astype(np.float32),
                psf_wcs,
                beam_wcs,
                1,
                coeffs=True,
                sinscl=1.0,
                stepsize=1,
                wcsmap=grizli_utils.WCSMapAll,
            )
            psf_aligned_images[beam_name] = blotted

    return psf_aligned_images
