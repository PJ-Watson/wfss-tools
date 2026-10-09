"""
Functions to bin data to a specified signal/noise.
"""

from os import PathLike
from typing import Any

import astropy.visualization as astrovis
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from numpy.typing import ArrayLike

try:
    POWERBIN_AVAIL = True
    from powerbin import PowerBin
except:
    POWERBIN_AVAIL = False

from wfss_tools.sed import colour_binning

__all__ = [
    "hexbin",
    "constrained_adaptive",
    "save_binned_data_arr",
    "save_binned_data_fits",
    "bin_and_save",
]


def hexbin(
    x: ArrayLike,
    y: ArrayLike,
    signal: ArrayLike,
    noise: ArrayLike,
    bin_diameter: float = 10,
    centre: tuple[float, float] | None = None,
) -> tuple[ArrayLike, int, ArrayLike, ArrayLike]:
    """
    Hexagonally bin data, returning the binned S/N.

    Parameters
    ----------
    x : ArrayLike
        A 1D array containing the x-positions of the data.
    y : ArrayLike
        A 1D array containing the y-positions of the data.
    signal : ArrayLike
        A 1D array of the signal.
    noise : ArrayLike
        A 1D array of the noise. All of ``signal``, ``noise``, ``x``, and
        ``y`` must have the same shape.
    bin_diameter : float, optional
        The minimum diameter of each bin (subject to pixel discretisation
        effects), i.e. the long diameter of each hexagon. By default 10.
    centre : tuple[float, float] | None, optional
        If given, all bins will be aligned such that a hexagon is centred
        on these coordinates. If not given, the highest S/N pixel will be
        used instead.

    Returns
    -------
    bin_labels : ArrayLike
        The bin label assigned to each element of the input arrays.
    nbins : int
        The number of bins.
    binned_s_n : ArrayLike
        The S/N in each bin.
    bin_inv : ArrayLike
        A 1D array performing the inverse binning operation, i.e.
        ``binned_s_n[inv]`` gives an array of the same shape as ``x``.
    """

    if centre is None:
        centre_idx = np.argmax(signal / noise)
        centre = np.array([x[centre_idx], y[centre_idx]])

    xmin = np.min(x)
    xmax = np.max(x)
    ymin = np.min(y)
    ymax = np.max(y)

    x = np.asarray(x, float)
    y = np.asarray(y, float)

    sx = bin_diameter * 0.5 * np.sqrt(3)
    sy = bin_diameter * 3 / 2

    ix = ((x - xmin) + ((centre[0] - xmin) % sx)) / sx - 0.5
    iy = ((y - ymin) + ((centre[1] - ymin) % sy)) / sy

    nx = (np.max(ix) - np.min(ix)) + 1
    ny = (np.max(iy) - np.min(iy)) + 1
    nx1 = nx + 1
    ny1 = ny + 1
    nx2 = nx
    ny2 = ny

    ix1 = np.round(ix).astype(int)
    iy1 = np.round(iy).astype(int)
    ix2 = np.floor(ix).astype(int)
    iy2 = np.floor(iy).astype(int)

    # flat indices, plus one so that out-of-range points go to position 0.
    i1 = np.where(
        (0 <= ix1) & (ix1 < nx1) & (0 <= iy1) & (iy1 < ny1), ix1 * ny1 + iy1 + 1, 0
    )
    i2 = np.where(
        (0 <= ix2) & (ix2 < nx2) & (0 <= iy2) & (iy2 < ny2), ix2 * ny2 + iy2 + 1, 0
    )

    d1 = (ix - ix1) ** 2 + 3.0 * (iy - iy1) ** 2
    d2 = (ix - ix2 - 0.5) ** 2 + 3.0 * (iy - iy2 - 0.5) ** 2
    bdist = d1 < d2

    idxs = bdist * i1 + (~bdist * (i2 + np.nanmax(i1)))
    u, bin_inv = np.unique(idxs, return_inverse=True)
    nbins = len(u)
    bin_labels = np.arange(len(u), dtype=int)[bin_inv]

    binned_signal = np.bincount(bin_labels, weights=signal)
    binned_noise = np.sqrt(np.bincount(bin_labels, weights=noise**2))

    binned_s_n = binned_signal / binned_noise

    return bin_labels, nbins, binned_s_n, bin_inv


def constrained_adaptive(
    signal: ArrayLike,
    noise: ArrayLike,
    bin_diameter: float = 4,
    target_sn: float = 100,
    plot: bool = True,
    quiet: bool = False,
    cvt: bool = True,
    wvt: bool = True,
    mask: ArrayLike | None = None,
    use_hex: bool = True,
    use_powerbin: bool = True,
) -> tuple[ArrayLike, int, ArrayLike, ArrayLike]:
    """
    Perform a constrained adaptive binning method to reach a target S/N.

    This function performs a 2-stage binning procedure, subject to a
    minimum bin diameter. The initial stage is a hexagonal binning scheme,
    which selects any bins that achieve the target S/N. This is followed
    by a Voronoi tesselation on all remaining pixels, using the method of
    `Cappellari & Copin 2003\
        <https://ui.adsabs.harvard.edu/abs/2003MNRAS.342..345C>`_.

    Parameters
    ----------
    signal : ArrayLike
        A 2D array containing the signal to be binned.
    noise : ArrayLike
        A 2D array containing the associated noise.
    bin_diameter : float, optional
        The minimum diameter of each bin (subject to pixel discretisation
        effects), by default 10.
    target_sn : float, optional
        The desired S/N in each output bin, by default 100. This is only
        guaranteed to be achieved for the hexagonal bins, as there is some
        scatter on the S/N achieved through Voronoi binning.
    plot : bool, optional
        Show a comparison of the binned and unbinned data, alongside the
        S/N and bin maps. By default ``False``.
    quiet : bool, optional
        Print the output of the Voronoi binning procedure, by default
        ``False``.
    cvt : bool, optional
        Attempt to achieve a centroidal Voronoi tessellation, using the
        method of `Cappellari & Copin 2003 \
            <https://ui.adsabs.harvard.edu/abs/2003MNRAS.342..345C>`_,
        by default ``True``.
    wvt : bool, optional
        Use the weighted Voronoi tessellation method by
        `Diehl & Statler 2006 \
            <https://ui.adsabs.harvard.edu/abs/2006MNRAS.368..497D>`_.
        By default True.
    mask : ArrayLike | None, optional
        Values in the input signal and noise to mask out, i.e. where
        ``mask==True`` will not be included in the binned data.
    use_hex : bool, optional
        Whether to perform the first stage of the binning procedure. If
        ``False``, then this will be skipped, and only the Voronoi binning
        procedure will be attempted. By default ``True``.
    use_powerbin : bool, optional
        Use the PowerBin algorithm instead.

    Returns
    -------
    bin_labels : ArrayLike
        A 2D ``int`` array, containing the bin label assigned to each
        element of the input arrays.
    nbins : int
        The number of bins.
    binned_s_n : ArrayLike
        A 1D array of length ``nbins``, containing the S/N in
        each bin.
    bin_inv : ArrayLike
        A 2D array performing the inverse binning operation, i.e.
        ``binned_s_n[inv]`` gives an array of the same shape as ``x``.
    """

    Y, X = np.mgrid[0 : signal.shape[0], 0 : signal.shape[1]]

    f_signal = signal.ravel()
    f_noise = noise.ravel()
    if mask is None:
        f_mask = np.zeros_like(f_noise)
    else:
        f_mask = mask.ravel()
    # plt.imshow(mask)
    # plt.show()

    hex_idxs, hex_bin_n, hex_s_n, hex_inv = hexbin(
        X.ravel(),
        Y.ravel(),
        f_signal,
        f_noise,
        bin_diameter=bin_diameter,
    )

    if not use_hex:
        print("Skipping hex bin")
        hex_s_n = np.zeros_like(hex_s_n)
    good_hex_data = (hex_s_n >= target_sn)[hex_inv]
    if mask is not None:
        good_hex_data &= ~f_mask

    num_hex_bins = len(np.unique(hex_idxs[good_hex_data]))

    vorbin_idx = (
        ~good_hex_data & np.isfinite(f_noise) & np.isfinite(f_signal) & (f_noise > 0)
    )  # & (f_signal > 0)

    if mask is not None:
        vorbin_idx &= ~mask.ravel()

    if POWERBIN_AVAIL and use_powerbin:

        def capacity_spec(index):
            """Calculates (S/N)^2 for a bin from its pixel indices."""
            # Standard S/N formula for uncorrelated noise
            sn = np.sum(f_signal[vorbin_idx][index]) / np.sqrt(
                np.sum(f_noise[vorbin_idx][index] ** 2)
            )
            # Example for correlated noise (see full example file for details):
            # sn /= 1 + 1.07 * np.log10(len(index))
            return sn**2

        # print(
        #     f_signal[vorbin_idx],
        #     f_noise[vorbin_idx],
        #     (np.any(f_signal[vorbin_idx] <= 0)),
        #     (np.any(f_noise[vorbin_idx] <= 0)),
        # )
        pow_obj = PowerBin(
            np.column_stack([X.ravel()[vorbin_idx], Y.ravel()[vorbin_idx]]),
            capacity_spec,
            target_capacity=target_sn**2,
            # verbose=3,
            pixelsize=1.0,
        )

        # pow_obj.plot(capacity_scale='sqrt', ylabel='S/N')
        # plt.show()

        vor_idxs, vor_inv = np.unique(pow_obj.bin_num, return_inverse=True)

    else:

        from vorbin.voronoi_2d_binning import voronoi_2d_binning

        vorbin_output = voronoi_2d_binning(
            x=X.ravel()[vorbin_idx],
            y=Y.ravel()[vorbin_idx],
            signal=f_signal[vorbin_idx],
            noise=f_noise[vorbin_idx],
            target_sn=target_sn,
            pixelsize=1,
            plot=False,
            quiet=quiet,
            cvt=cvt,
            wvt=wvt,
        )

        vor_idxs, vor_inv = np.unique(vorbin_output[0], return_inverse=True)

    vor_idxs += np.max(hex_idxs) + 1

    num_vor_bins = len(np.unique(vor_idxs))

    hex_idxs[vorbin_idx != ~good_hex_data] = -1
    hex_idxs[vorbin_idx] = vor_idxs[vor_inv]

    _all_unq, all_inv = np.unique(hex_idxs, return_inverse=True)
    nbins = len(_all_unq)
    all_idxs = np.arange(len(_all_unq))[all_inv]

    binned_signal = np.bincount(all_idxs, weights=f_signal)
    binned_noise = np.sqrt(np.bincount(all_idxs, weights=f_noise**2))

    binned_s_n = binned_signal / binned_noise  # [all_inv].reshape(signal.shape)
    bin_labels = all_idxs.reshape(signal.shape)
    bin_inv = all_inv.reshape(signal.shape)
    print(f"Total bins: {len(_all_unq)} (Hex: {num_hex_bins}, Voronoi: {num_vor_bins})")

    if plot:
        fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, constrained_layout=True)

        img_norm = astrovis.ImageNormalize(
            data=signal,
            interval=astrovis.ManualInterval(
                vmin=0.0, vmax=np.nanpercentile(signal, q=99.9)
            ),
            stretch=astrovis.LogStretch(),
        )
        axs[0, 0].imshow(signal, norm=img_norm, origin="lower", cmap="plasma")
        bin_sig_plot = (
            np.bincount(all_idxs, weights=f_signal) / np.bincount(all_idxs)
        )[all_inv]
        bin_sig_plot[all_idxs == 0] = np.nan
        axs[0, 1].imshow(
            bin_sig_plot.reshape(signal.shape),
            origin="lower",
            norm=img_norm,
            cmap="plasma",
        )

        sn_plot = (
            np.bincount(all_idxs, weights=f_signal)
            / np.sqrt(np.bincount(all_idxs, weights=f_noise**2))
        )[all_inv]
        sn_plot[all_idxs == 0] = np.nan
        axs[1, 0].imshow(
            sn_plot.reshape(signal.shape),
            origin="lower",
            # norm=img_norm,
            cmap="plasma",
        )

        rng = np.random.default_rng()
        bin_plot = rng.random(size=len(_all_unq))[all_inv]
        bin_plot[all_idxs == 0] = np.nan

        axs[1, 1].imshow(
            bin_plot.reshape(signal.shape),
            origin="lower",
            # norm=img_norm,
            cmap="jet",
            interpolation="none",
        )

        for a in axs.flatten():
            a.set_facecolor("k")

        # plt.imshow(hex_idxs.reshape(signal.shape), origin="lower")

        plt.show()

    return bin_labels, nbins, binned_s_n, bin_inv


def save_binned_data_arr(
    bin_labels: ArrayLike,
    data_path: PathLike,
    out_path: PathLike,
    signal_hdu_idxs: ArrayLike,
    noise_hdu_idxs: ArrayLike | None = None,
    var_hdu_idxs: ArrayLike | None = None,
    crop: tuple | None = None,
):
    """
    Create a binned photometric catalogue from a segmentation map.

    Realistically, this needs to be better thought out. Unlikely that
    the input will be MEF, highly likely that it'll be combined from
    multiple single FITS files.

    Parameters
    ----------
    bin_labels : ArrayLike
        A 2D ``int`` array, containing the bin label assigned to each
        element of the input arrays.
    data_path : PathLike
        The multi-extension fits file containing the data to be binned.
    out_path : PathLike
        The path to save the output.
    signal_hdu_idxs : ArrayLike
        The indices of the HDUs containing the direct images.
    noise_hdu_idxs : ArrayLike | None, optional
        The indices of the HDUs containing the noise images. One of
        ``noise_hdu_idxs`` and ``var_hdu_idxs`` must be given.
    var_hdu_idxs : ArrayLike | None, optional
        The indices of the HDUs containing the variance images.
    crop : tuple | None, optional
        A crop to be applied to the images before applying the binning
        scheme, by default None.
    """

    phot_cat = Table()
    bin_ids, bin_inv = np.unique(bin_labels, return_inverse=True)
    phot_cat["bin_id"] = bin_ids.astype(str)
    phot_cat["bin_area"] = np.bincount(bin_labels.ravel())

    if crop is None:
        crop = (slice(0, None), slice(0, None))

    with fits.open(data_path) as hdul:

        for s in signal_hdu_idxs:
            try:
                phot_cat[s] = np.bincount(
                    bin_labels.ravel(), weights=hdul[s].data[crop].ravel()
                )
            except Exception as e:
                print(e)
                phot_cat[s] = np.nan
        if noise_hdu_idxs is not None:
            for s in noise_hdu_idxs:
                try:
                    phot_cat[s] = np.bincount(
                        bin_labels.ravel(), weights=hdul[s].data[crop].ravel() ** 2
                    )
                except Exception as e:
                    print(e)
                    phot_cat[s] = np.nan
        elif var_hdu_idxs is not None:
            for s in var_hdu_idxs:
                try:
                    phot_cat[s] = np.bincount(
                        bin_labels.ravel(), weights=hdul[s].data[crop].ravel()
                    )
                except Exception as e:
                    print(e)
                    phot_cat[s] = np.nan

    binned_data = fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.ImageHDU(
                data=bin_labels,
                # header=signal.header.copy(),
                name="SEG_MAP",
            ),
            fits.BinTableHDU(data=phot_cat, name="PHOT_CAT"),
        ]
    )

    binned_data.writeto(out_path)


def save_binned_data_fits(
    bin_labels: ArrayLike,
    out_path: PathLike,
    filter_keys: list,
    signal_paths: list,
    var_paths: list,
    crop: tuple | None = None,
    header: fits.Header | None = None,
    overwrite: bool = True,
) -> None:
    """
    Create a binned photometric catalogue from a segmentation map.

    Unlike `~wfss_tools.sed.save_binned_data_arr`, this uses as input
    multiple FITS files containing the signal and variance arrays.

    Parameters
    ----------
    bin_labels : ArrayLike
        A 2D ``int`` array, containing the bin label assigned to each
        element of the input arrays.
    out_path : PathLike
        The path to save the output.
    filter_keys : list
        The names of the filters corresponding to each of the signal and
        variance image pairs. These will be used as column names in the
        photometric catalogue.
    signal_paths : list
        A list containing the locations of the direct images.
    var_paths : list
        A list containing the locations of the associated variance images.
    crop : tuple | None, optional
        A crop to be applied to the images before applying the binning
        scheme, by default None.
    header : fits.Header | None = None
        If supplied, this will be used to generate a header for the
        segmentation map in the output file. If ``None``, a header will be
        taken from the first image in ``signal_paths``.
    overwrite : bool, optional
        If a catalogue already exists at ``out_path``, this determines if
        it should be written over. By default ``True``.
    """

    phot_cat = Table()
    bin_ids, bin_inv = np.unique(bin_labels, return_inverse=True)
    phot_cat["bin_id"] = bin_ids.astype(str)
    phot_cat["bin_area"] = np.bincount(bin_labels.ravel())

    if crop is None:
        crop = (slice(0, None), slice(0, None))

    for i, (f, s, v) in enumerate(zip(filter_keys, signal_paths, var_paths)):
        with fits.open(s) as sci_hdul:
            try:
                phot_cat[f"{f}_sci".lower()] = np.bincount(
                    bin_labels.ravel(), weights=sci_hdul[0].data[crop].ravel()
                )
                if header is None:
                    header = sci_hdul[0].header
                    wcs = WCS(header)[crop]
                    header.update(wcs.to_header())

            except Exception as e:
                print(e, "blah", f)
                phot_cat[f"{f}_sci".lower()] = [np.nan] * len(bin_ids)
        with fits.open(v) as var_hdul:
            try:
                phot_cat[f"{f}_var".lower()] = np.bincount(
                    bin_labels.ravel(), weights=var_hdul[0].data[crop].ravel()
                )
            except Exception as e:
                print(e, "whhy")
                phot_cat[f"{f}_var".lower()] = np.nan

    binned_data = fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.ImageHDU(
                data=bin_labels.astype(float),
                header=header,
                name="SEG_MAP",
            ),
            fits.BinTableHDU(data=phot_cat, name="PHOT_CAT"),
        ]
    )

    binned_data.writeto(out_path, overwrite=overwrite)


def bin_and_save(
    obj_id: int,
    out_dir: PathLike,
    seg_map: ArrayLike | PathLike,
    info_dict: dict,
    sn_filter: str,
    target_sn: float = 100,
    bin_scheme: str = "hexbin",
    bin_diameter: float = 4,
    seg_hdu_index: int | str = 0,
    padding: int = 50,
    overwrite: bool = False,
    binned_name: str | None = None,
    **bin_kwargs,
) -> PathLike:
    """
    Rebin and save one object in a segmentation map.

    This is a thin wrapper around other functions in this module, with a
    slightly opinionated default set of parameters. If the user is
    familiar with the lower-level functions, those will provide an
    equivalent output.

    Parameters
    ----------
    obj_id : int
        The unique identifier of an object in the full segmentation map.
    out_dir : PathLike
        The directory to which the output will be written, consisting of a
        segmentation image for the binning scheme, and a photometric
        catalogue.
    seg_map : ArrayLike | PathLike
        The segmentation map identifying the location of objects in the
        individual images, where pixels with a given unique integer
        correspond to the extent of a single object. This should have the
        same alignment and shape as images in ``info_dict``.
    info_dict : dict
        A dictionary containing all information on the images to be used
        when binning and generating the photometric catalogue. The keys of
        the dictionary should correspond to the names of the filters (or
        any desired column name in the output catalogue), and each entry
        should itself contain a ``dict``, with the keys ``sci`` and
        ``var`` describing the locations of the PSF-matched science and
        variance images.
    sn_filter : str
        The ``info_dict`` key to use when calculating the S/N for binning.
    target_sn : float, optional
        The desired S/N in each output bin, by default 100. This is only
        guaranteed to be achieved for the hexagonal bins (if using this
        binning scheme), as there is some scatter on the S/N achieved
        through Voronoi binning.
    bin_scheme : str, optional
        Which binning scheme to use, by default ``"hexbin"``. In this
        case, the first stage of the binning procedure will be performed.
        If ``"vorbin"``, then this will be skipped, and only the Voronoi
        binning procedure will be attempted. If ``"colour"``, then
        pixels will be aggregated based on their colour.
    bin_diameter : float, optional
        The minimum diameter of each bin (subject to pixel discretisation
        effects), by default 4.
    seg_hdu_index : int | str, optional
        The index or name of the HDU containing the segmentation map,
        by default ``0``.
    padding : int, optional
        The extra padding to add to the object boundaries, by default 50.
    overwrite : bool, optional
        If a catalogue already exists in ``out_dir``, this determines if
        it should be written over. By default ``False``.
    binned_name : str | None, optional
        Specify the exact name used for writing the output file. If
        ``None`` (default), this will be automatically generated from
        the chosen binning parameters.
    **bin_kwargs : dict, optional
        Any additional parameters to be passed through to
        `~wfss_tools.sed.constrained_adaptive`.

    Returns
    -------
    PathLike
        The path to the binned data, saved as a multi-extension FITS file.
    """

    if isinstance(seg_map, PathLike):
        seg_map = fits.getdata(seg_map, seg_hdu_index)

    from wfss_tools.pipeline import seg_slice

    obj_img_idxs = seg_slice(seg_map, obj_id, padding=padding)

    signal = fits.getdata(info_dict[sn_filter]["sci"])[obj_img_idxs]
    signal_hdr = fits.getheader(info_dict[sn_filter]["sci"])
    signal_wcs = WCS(signal_hdr)[obj_img_idxs]
    signal_hdr.update(signal_wcs.to_header())

    # Don't forget that the noise is the square root of the variance array
    noise = np.sqrt(fits.getdata(info_dict[sn_filter]["var"]))[obj_img_idxs]

    if bin_scheme == "colour":
        orig_images = []
        for k, v in info_dict.items():
            orig_images.append(fits.getdata(v["sci"])[obj_img_idxs])
        bin_labels, nbins, bin_sn, bin_inv = colour_binning.colour_aggregate(
            orig_images=orig_images,
            signal=signal,
            noise=noise,
            target_sn=target_sn,
            mask=seg_map[obj_img_idxs] != obj_id,
            # crop=obj_img_idxs,
            **bin_kwargs,
        )
    else:

        bin_labels, nbins, bin_sn, bin_inv = constrained_adaptive(
            signal=signal,
            noise=noise,
            target_sn=target_sn,
            mask=seg_map[obj_img_idxs] != obj_id,
            use_hex=use_hex,
            bin_diameter=bin_diameter,
            **bin_kwargs,
        )

    if binned_name is None:
        # Give it a meaningful name - this avoids confusion if rerunning with multiple configurations
        binned_name = f"{obj_id}_{bin_scheme}_{bin_diameter}_{target_sn}_{sn_filter}"

    save_path = out_dir / f"{binned_name}_data.fits"
    if save_path.is_file() and not overwrite:
        print(
            f"'{save_path.name}' already exists. Set `overwrite=True` "
            "if you wish to overwrite this file."
        )
    else:
        save_binned_data_fits(
            bin_labels=bin_labels,
            out_path=out_dir / f"{binned_name}_data.fits",
            filter_keys=[*info_dict.keys()],
            signal_paths=[info_dict[f]["sci"] for f in info_dict.keys()],
            var_paths=[info_dict[f]["var"] for f in info_dict.keys()],
            crop=obj_img_idxs,
            header=signal_hdr,
        )

    return save_path
