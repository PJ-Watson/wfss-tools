"""
General utility functions related to handling grism data.
"""

import datetime
from copy import deepcopy
from os import PathLike
from pathlib import Path

import astropy
import astropy.units as u
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from grizli import utils as grizli_utils
from grizli.model import BeamCutout, GrismDisperser
from grizli.multifit import MultiBeam
from numba import njit
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


def cluster_beams_centres(
    mb: MultiBeam, beam_idxs: ArrayLike, dbscan_kwargs: dict | None = None
) -> list:
    """
    Perform DBSCAN clustering on beam detector center coordinates.

    Noisy (or unclustered) points are grouped into a single cluster,
    rather than being split into individual points (as the entire point of
    this function is to reduce the number of individual beams!).

    Parameters
    ----------
    mb : `grizli.multifit.MultiBeam`
        The original MultiBeam object containing the beam data.
    beam_idxs : ArrayLike
        List or array of beam indices to be clustered.
    dbscan_kwargs : dict | None, optional
        Keyword arguments to pass through to `sklearn.cluster.DBSCAN`.
        If ``None`` (default), uses ``{"eps": 5, "min_samples": 3}``.

    Returns
    -------
    list of numpy.ndarray
        A list of numpy arrays, where each array contains the beam indices
        grouped into a single spatial cluster.
    """

    from sklearn.cluster import DBSCAN

    if dbscan_kwargs is None:
        dbscan_kwargs = {"eps": 5, "min_samples": 3}

    beam_idxs = np.asarray(beam_idxs)

    # Detector coordinate centres
    detector_coords = np.array(
        [[mb.beams[b].beam.xc, mb.beams[b].beam.yc] for b in beam_idxs]
    )

    db = DBSCAN(**dbscan_kwargs).fit(detector_coords)
    labels = np.array(db.labels_)

    grouped_beam_idxs = [beam_idxs[labels == u] for u in np.unique(labels)]

    return grouped_beam_idxs


def gen_stacked_beams(
    mb: str | MultiBeam,
    pixfrac: float = 1.0,
    kernel: str = "square",
    dfillval: float = 0,
    fit_trace_shift: bool = False,
    trace_shift_kwargs: dict | None = None,
    cluster_beams: bool = False,
    dbscan_kwargs: dict | None = None,
    split_by_fwcpos: bool = True,
    **multibeam_kwargs,
) -> MultiBeam:
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
        `sklearn.cluster.DBSCAN`. By default ``dbscan_kwargs={"eps": 5,
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

    if dbscan_kwargs is None:
        dbscan_kwargs = {"eps": 5, "min_samples": 3}

    if trace_shift_kwargs is None:
        trace_shift_kwargs = {}

    if not isinstance(mb, MultiBeam):
        mb = MultiBeam(beams=mb, **multibeam_kwargs)

    if fit_trace_shift:
        mb.fit_trace_shift(**trace_shift_kwargs)

    new_beam_list = []

    for filt, pa_info in tqdm(mb.PA.items(), desc="Stacking beams"):
        for pa, pa_beam_idxs in pa_info.items():

            pa_beam_idxs = np.array(pa_beam_idxs)

            if split_by_fwcpos:
                # Round to avoid numerical noise
                fwcpos_arr = np.round(
                    np.asarray(
                        [
                            getattr(mb.beams[b].beam, "fwcpos", 0) or 0
                            for b in pa_beam_idxs
                        ]
                    ),
                    3,
                )
                grouped_beam_idxs = []

                unique, u_inv = np.unique(fwcpos_arr, return_inverse=True)

                for i, u in enumerate(unique):
                    sub_idxs = pa_beam_idxs[u_inv == i]
                    if cluster_beams:
                        grouped_beam_idxs.extend(
                            cluster_beams_centres(mb, sub_idxs, dbscan_kwargs)
                        )
                    else:
                        grouped_beam_idxs.append(sub_idxs)
            elif cluster_beams:
                grouped_beam_idxs = cluster_beams_centres(
                    mb, pa_beam_idxs, dbscan_kwargs
                )
            else:
                grouped_beam_idxs = [pa_beam_idxs]

            for beam_idxs in grouped_beam_idxs:

                ref_beam_0 = mb.beams[beam_idxs[0]]

                # As a reference beam, we use the one with the smallest shift from the centre
                # along the x-axis
                # This minimises the chance of trace pixel errors due to integer rounding
                # in the grizli and grismconf code
                direct_cen = (
                    np.asarray(ref_beam_0.direct.data["REF"].shape[::-1])  # + 1
                ) / 2

                shift_dx = np.zeros((len(beam_idxs), 2))
                for i, b_i in enumerate(beam_idxs):
                    pix = (
                        mb.beams[b_i]
                        .direct.wcs.all_world2pix(
                            [[mb.ra, mb.dec]], 1, ra_dec_order=True
                        )
                        .flatten()
                    )
                    shift_dx[i] = direct_cen - pix

                best_idx = beam_idxs[
                    np.argmin(np.abs(shift_dx[:, 0] - np.round(shift_dx[:, 0])))
                ]
                new_beam = deepcopy(mb.beams[best_idx])

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

                # Initialise output arrays (and get shapes)
                sh = new_beam.sh
                outsci_sum = np.zeros(sh, dtype=np.float32)
                outwht = np.zeros(sh, dtype=np.float32)

                outcon_sum = np.zeros(sh, dtype=np.float32)
                outwc = np.zeros(sh, dtype=np.float32)

                dir_sh = new_beam.direct.data["REF"].shape
                outdir_sum = np.zeros(dir_sh, dtype=np.float32)
                outwd = np.zeros(dir_sh, dtype=np.float32)

                dir_scale = np.nanmedian(
                    new_beam.direct.data["REF"] / new_beam.beam.direct
                )

                new_seg = grizli_utils.blot_nearest_exact(
                    ref_beam_0.beam.seg,
                    ref_beam_0.direct.wcs,
                    new_beam.direct.wcs,
                    verbose=False,
                    stepsize=-1,
                    scale_by_pixel_area=False,
                )

                for idx in beam_idxs:

                    beam = mb.beams[idx]
                    direct_wcs_i = beam.direct.wcs.copy()

                    in_pix_d = direct_wcs_i.all_world2pix(
                        [[mb.ra, mb.dec]], 1, ra_dec_order=True
                    ).flatten()
                    dx_d = float(direct_cen[0] - in_pix_d[0])
                    dy_d = float(direct_cen[1] - in_pix_d[1])

                    grism_wht = beam.ivar.copy().astype(np.float32)
                    grism_wht[~np.isfinite(grism_wht)] = 0.0

                    direct_wht = np.ones_like(beam.direct.data["REF"], dtype=np.float32)
                    direct_wht[~np.isfinite(beam.direct.data["REF"])] = 0.0

                    # Direct image
                    shift_and_accumulate_2d(
                        beam.direct.data["REF"].astype(np.float32),
                        direct_wht,
                        dx_d,
                        dy_d,
                        outdir_sum,
                        outwd,
                    )

                    # Dispersed science spectrum
                    shift_and_accumulate_2d(
                        beam.grism.data["SCI"].astype(np.float32),
                        grism_wht,
                        dx_d,
                        dy_d,
                        outsci_sum,
                        outwht,
                    )

                    # Dispersed contamination model
                    shift_and_accumulate_2d(
                        beam.contam.astype(np.float32),
                        grism_wht,
                        dx_d,
                        dy_d,
                        outcon_sum,
                        outwc,
                    )

                # Correct for weighting
                outsci = np.zeros(sh, dtype=np.float32)
                mask_sci = outwht > 0.0
                outsci[mask_sci] = outsci_sum[mask_sci] / outwht[mask_sci]

                outdir = np.zeros(dir_sh, dtype=np.float32)
                mask_dir = outwd > 0.0
                outdir[mask_dir] = outdir_sum[mask_dir] / outwd[mask_dir]

                outcon = np.zeros(sh, dtype=np.float32)
                mask_con = outwc > 0.0
                outcon[mask_con] = outcon_sum[mask_con] / outwc[mask_con]

                # Variance from accumulated inverse-variance weights
                outvar = np.zeros(sh, dtype=np.float32)
                outvar[mask_sci] = 1.0 / outwht[mask_sci]

                new_beam.grism.data["SCI"] = outsci
                new_beam.grism.data["ERR"] = np.sqrt(outvar)
                new_beam.grism.data["DQ"] = np.zeros_like(outsci, dtype=np.int32)
                new_beam.contam = outcon
                new_beam.direct.data["REF"] = outdir
                new_beam.direct.header.update(
                    grizli_utils.to_header(new_beam.direct.wcs)
                )
                new_beam.grism.header.update(grizli_utils.to_header(new_beam.grism.wcs))

                origin_med = np.round(
                    np.nanmedian([mb.beams[i].direct.origin for i in beam_idxs], axis=0)
                ).astype(int)
                pad_med = int(
                    np.round(np.nanmedian([mb.beams[i].direct.pad for i in beam_idxs]))
                )
                grow_med = int(
                    np.round(np.nanmedian([mb.beams[i].direct.grow for i in beam_idxs]))
                )

                new_beam.beam = GrismDisperser(
                    id=mb.id,
                    direct=outdir,
                    segmentation=new_seg,
                    origin=origin_med,
                    pad=pad_med,
                    grow=grow_med,
                    beam=ref_beam_0.beam.beam,
                    xcenter=0,
                    ycenter=0,
                    conf=ref_beam_0.beam.conf,
                    fwcpos=ref_beam_0.beam.fwcpos,
                    MW_EBV=ref_beam_0.beam.MW_EBV,
                    xoffset=0.0,
                    yoffset=0.0,
                )
                new_beam.beam.compute_model()

                new_beam.modelf = new_beam.beam.modelf
                new_beam.model = new_beam.beam.modelf.reshape(new_beam.beam.sh_beam)

                is_jwst = getattr(ref_beam_0, "isJWST", True)
                new_beam._parse_from_data(
                    isJWST=is_jwst,
                    contam_sn_mask=[10, 3],
                    min_mask=mb.min_mask,
                    min_sens=mb.min_sens,
                    mask_resid=mb.mask_resid,
                )

                if np.isfinite(dir_scale) and dir_scale != 0:
                    new_beam.direct.data["REF"] /= dir_scale
                new_beam.direct.ref_photflam = new_beam.direct.photflam

                new_beam_list.append(new_beam)

    return MultiBeam(
        new_beam_list,
        group_name=mb.group_name,
        fcontam=mb.fcontam,
        min_mask=mb.min_mask,
        min_sens=mb.min_sens,
        mask_resid=mb.mask_resid,
    )


@njit(fastmath=False, parallel=False, cache=True, boundscheck=False)
def shift_and_accumulate_2d(
    in_sci: np.ndarray,
    in_wht: np.ndarray,
    dx: float,
    dy: float,
    out_sci: np.ndarray,
    out_wht: np.ndarray,
) -> None:
    """
    Exact 2D box-overlap shift-and-add kernel with strict flux conservation.

    Parameters
    ----------
    in_sci : numpy.ndarray
        2D array of input science values.
    in_wht : numpy.ndarray
        2D array of input inverse variance weights.
    dx : float
        Subpixel shift along the x-axis in pixel units.
    dy : float
        Subpixel shift along the y-axis in pixel units.
    out_sci : numpy.ndarray
        2D accumulator image, modified in-place with weighted science
        values.
    out_wht : numpy.ndarray
        2D accumulator image, modified in-place with accumulated weights.
    """
    ny, nx = in_sci.shape
    out_ny, out_nx = out_sci.shape

    # Decompose uniform translation into integer offsets and subpixel fractions
    iy0 = int(np.floor(dy))
    fy = dy - iy0
    fy_inv = 1.0 - fy

    ix0 = int(np.floor(dx))
    fx = dx - ix0
    fx_inv = 1.0 - fx

    # Precompute the 4 bilinear overlap area fractions
    w00 = fy_inv * fx_inv
    w01 = fy_inv * fx
    w10 = fy * fx_inv
    w11 = fy * fx

    # Clip input iteration bounds to rows/columns that overlap the output frame
    y_start = max(0, -iy0 - 1)
    y_end = min(ny, out_ny - iy0)
    x_start = max(0, -ix0 - 1)
    x_end = min(nx, out_nx - ix0)

    for y in range(y_start, y_end):
        oy0 = y + iy0
        oy1 = oy0 + 1
        y0_valid = 0 <= oy0 < out_ny
        y1_valid = (fy > 0.0) and (0 <= oy1 < out_ny)

        if not (y0_valid or y1_valid):
            continue

        for x in range(x_start, x_end):
            w_in = in_wht[y, x]
            if w_in <= 0.0 or not np.isfinite(w_in):
                continue

            s_in = in_sci[y, x]
            if not np.isfinite(s_in):
                continue

            sw = s_in * w_in
            ox0 = x + ix0
            ox1 = ox0 + 1
            x0_valid = 0 <= ox0 < out_nx
            x1_valid = (fx > 0.0) and (0 <= ox1 < out_nx)

            if y0_valid:
                if x0_valid:
                    out_sci[oy0, ox0] += sw * w00
                    out_wht[oy0, ox0] += w_in * w00
                if x1_valid:
                    out_sci[oy0, ox1] += sw * w01
                    out_wht[oy0, ox1] += w_in * w01

            if y1_valid:
                if x0_valid:
                    out_sci[oy1, ox0] += sw * w10
                    out_wht[oy1, ox0] += w_in * w10
                if x1_valid:
                    out_sci[oy1, ox1] += sw * w11
                    out_wht[oy1, ox1] += w_in * w11


def get_bounding_cutout_slice(
    out_wcs: WCS,
    out_shape: tuple,
    in_wcs: WCS,
    in_shape: tuple,
    buffer_frac: float = 0.05,
    n_samples: int = 2,
) -> tuple[slice, slice]:
    """
    Find the minimal slice (sly, slx) covering the target output frame.

    This finds the slice in the input image frame, accounting for
    WCS distortions. For extremely distorted frames, the number of samples
    should be increased, rather than relying on the corner reprojection.

    Parameters
    ----------
    out_wcs : WCS
        The target image WCS.
    out_shape : tuple
        The target image shape, in the order ``(ny_out, nx_out)``.
    in_wcs : WCS
        The original image WCS object.
    in_shape : tuple
        The shape of the original image, ``(ny_in, nx_in)``.
    buffer_frac : float, optional
        The fractional padding to add to the bounding box extent, by
        default ``0.05``.
    n_samples : int, optional
        The number of perimeter sample points per side, by default ``2``
        (i.e. only the corners are considered).

    Returns
    -------
    tuple[slice, slice]
        The ``(sly, slx)`` slices to extract from ``orig_data``.
    """
    ny_out, nx_out = out_shape
    ny_in, nx_in = in_shape

    # Sample the perimeter of the target output frame, in pixel coordinates
    x_edge = np.linspace(-0.5, nx_out - 0.5, n_samples)
    y_edge = np.linspace(-0.5, ny_out - 0.5, n_samples)

    top = np.column_stack([x_edge, np.full(n_samples, -0.5)])
    bottom = np.column_stack([x_edge, np.full(n_samples, ny_out - 0.5)])
    left = np.column_stack([np.full(n_samples, -0.5), y_edge])
    right = np.column_stack([np.full(n_samples, nx_out - 0.5), y_edge])

    perimeter_out = np.vstack([top, bottom, left, right])

    # Map the perimeter from the target pixel space -> world (RA, Dec),
    # and again from world (RA, Dec) -> input pixel space
    ra, dec = out_wcs.all_pix2world(perimeter_out[:, 0], perimeter_out[:, 1], 0)
    x_in, y_in = in_wcs.all_world2pix(ra, dec, 0)

    # Find the minimal bounding box in the input pixel space
    x_min, x_max = np.min(x_in), np.max(x_in)
    y_min, y_max = np.min(y_in), np.max(y_in)

    # Add a fractional buffer (to account for pixelation effects)
    w = x_max - x_min
    h = y_max - y_min
    buf_x = max(2, int(np.ceil(w * buffer_frac)))
    buf_y = max(2, int(np.ceil(h * buffer_frac)))

    x0 = int(np.floor(x_min)) - buf_x
    x1 = int(np.ceil(x_max)) + buf_x
    y0 = int(np.floor(y_min)) - buf_y
    y1 = int(np.ceil(y_max)) + buf_y

    # Clip the slice indices to the extent of the input image
    x0 = max(0, x0)
    x1 = min(nx_in, x1)
    y0 = max(0, y0)
    y1 = min(ny_in, y1)

    if x0 >= x1 or y0 >= y1:
        raise ValueError(
            "No spatial overlap detected between input image and reference beam footprint."
        )

    return slice(y0, y1), slice(x0, x1)


def align_direct_images(
    ref_beam: BeamCutout,
    info_dict: dict,
    out_dir: PathLike = None,
    bounding_cutout_kwargs: dict | None = None,
    overwrite: bool = False,
    reproject_method: str = "exact",
    subpixel_sampling: int = 5,
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
    bounding_cutout_kwargs : dict, optional
        Keyword arguments to pass through to `get_bounding_cutout_slice`,
        by default ``None``.
        Make a slice of the original image with size ``[-cutout,+cutout]``
        around the centre position of the desired object, before passing
        to blot. By default, ``cutout=200``.
    overwrite : bool, optional
        Overwrite existing images if they exist already. By default
        ``False``.
    reproject_method : str, optional
        The function to use for reprojecting the images to the same WCS as
        the beam. Must be one of ``"exact"`` (`reproject.reproject_exact`)
        or ``"subpixel"`` (`wfss_tools.grism.utils.reproject_subpixel`).
        By default, ``"exact"``.
    subpixel_sampling : int, optional
        The subpixel sampling factor to use for reprojection. Only used if
        `reproject_method="subpixel"`.

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

    if bounding_cutout_kwargs is None:
        bounding_cutout_kwargs = {}

    beam_ra, beam_dec = beam_wcs.all_pix2world(
        [(np.asarray(ref_beam.direct.sh[::-1]) + 1) / 2],
        1,
    ).flatten()

    new_info_dict = deepcopy(info_dict)

    beam_area = beam_wcs.proj_plane_pixel_area().value

    from reproject import reproject_adaptive, reproject_exact

    match reproject_method:
        case "exact":
            from reproject import reproject_exact as reproject_fn

            reproject_kwargs = dict(
                parallel=True,
                block_size="auto",
                return_footprint=False,
            )
        case "subpixel":
            reproject_fn = reproject_subpixel
            reproject_kwargs = dict(subsample=subpixel_sampling)

    for k, v in info_dict.items():
        for img_type in ["sci", "var"]:
            out_file = out_dir / Path(v[img_type]).name

            if not out_file.is_file() or overwrite:
                with fits.open(v[img_type]) as orig_hdul:
                    orig_data = orig_hdul[0].data
                    orig_header = orig_hdul[0].header.copy()

                    if orig_data.dtype not in [np.float32, np.dtype(">f4")]:
                        orig_data = orig_data.astype(np.float32)

                    orig_wcs = WCS(orig_header, relax=True)

                    sly, slx = get_bounding_cutout_slice(
                        out_wcs=beam_wcs,
                        out_shape=ref_beam.direct.sh,
                        in_wcs=orig_wcs,
                        in_shape=orig_data.shape,
                        **bounding_cutout_kwargs,
                    )

                    sub_wcs = orig_wcs.slice([sly, slx])

                    blotted = reproject_fn(
                        (orig_data[sly, slx], sub_wcs),
                        beam_wcs,
                        shape_out=ref_beam.direct.sh,
                        **reproject_kwargs,
                    )

                    if reproject_method == "exact":
                        blotted *= beam_area / (sub_wcs.proj_plane_pixel_area().value)

                    # Replace non-overlapping NaN values with zero
                    blotted = np.nan_to_num(blotted, nan=0.0)

                    out_header = beam_wcs.to_header()
                    out_header["EXTEND"] = True

                    new_hdul = fits.HDUList()
                    new_hdul.append(
                        fits.PrimaryHDU(
                            data=blotted.astype(np.float32),
                            header=out_header,
                        )
                    )
                    new_hdul.writeto(out_file, overwrite=overwrite)

            new_info_dict[k][img_type] = str(out_file)

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
