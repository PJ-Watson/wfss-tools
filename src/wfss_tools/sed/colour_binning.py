"""
Functions to bin data in colour space.
"""

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from numpy.typing import ArrayLike

__all__ = ["permute_axes_subtract", "colour_aggregate"]


import astropy.visualization as astrovis
from scipy.spatial import KDTree
from tqdm import tqdm


def colour_aggregate(
    orig_images: ArrayLike,
    signal: ArrayLike,
    noise: ArrayLike,
    target_sn: float = 100,
    plot: bool = False,
    quiet: bool = False,
    mask: ArrayLike | None = None,
    **kwargs,
) -> tuple[ArrayLike, int, ArrayLike, ArrayLike]:
    """
    Bin pixels to a specified signal/noise ratio.

    Pixels are binned based on their separation in colour space,
    accounting for all combinations of single-band images supplied.

    Parameters
    ----------
    orig_images : ArrayLike
        A 3D array, or list of arrays. Each slice of the array along
        ``axis=0`` should be a different photometric band, with the same
        shape and alignment as the ``signal`` and ``noise`` arrays.
    signal : ArrayLike
        A 2D array containing the signal to be binned.
    noise : ArrayLike
        A 2D array containing the associated noise.
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
    mask : ArrayLike | None, optional
        Values in the input signal and noise to mask out, i.e. where
        ``mask==True`` will not be included in the binned data.
    **kwargs : dict, optional
        A catch-all for additional parameters not relevant for this
        binning scheme.

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
        f_mask = np.zeros_like(f_noise, dtype=bool)
    else:
        f_mask = mask.ravel()

    f_mask |= ~np.isfinite(f_signal)
    f_mask |= ~np.isfinite(f_noise)
    f_mask |= f_noise <= 0

    f_orig_images = np.array(orig_images).reshape(len(orig_images), -1)

    f_mask |= np.any(np.isnan(f_orig_images), axis=0)

    all_colours = permute_axes_subtract(f_orig_images)
    f_colours = all_colours[np.triu_indices(all_colours.shape[0], k=1)]
    m_X = X.ravel()[~f_mask]
    m_Y = Y.ravel()[~f_mask]
    m_S = f_signal[~f_mask]
    m_N = f_noise[~f_mask]
    m_C = f_colours[:, ~f_mask].T

    bin_map = np.zeros_like(signal)
    avail_idxs = np.ones_like(m_S, dtype=bool)

    curr_bin_idxs = [np.argmax((m_S / m_N)[avail_idxs])]
    avail_idxs[curr_bin_idxs] = False

    kd = KDTree(m_C)

    with tqdm(unit=" pixels binned") as progress:

        while avail_idxs.any():

            curr_sn = np.nansum(m_S[curr_bin_idxs]) / np.sqrt(
                np.nansum(m_N[curr_bin_idxs] ** 2)
            )
            if curr_sn >= target_sn:
                new_bin_id = np.nanmax(bin_map) + 1
                for c in curr_bin_idxs:
                    bin_map[m_Y[c], m_X[c]] = new_bin_id

                try:
                    sn_masked = np.ma.masked_where(~avail_idxs, (m_S / m_N))
                    curr_bin_idxs = [np.argmax(sn_masked)]
                    avail_idxs[curr_bin_idxs] = False
                except Exception as e:
                    print(e)
                    break
            if not avail_idxs.any():
                break

            d, poss_idxs = kd.query(
                m_C[curr_bin_idxs[-1]], k=len(avail_idxs), workers=-1
            )

            filtered = avail_idxs[poss_idxs]
            filtered_idxs = poss_idxs[filtered]

            curr_bin_idxs.append(filtered_idxs[0])

            avail_idxs[filtered_idxs[0]] = False

            progress.update()

    new_bin_id = np.nanmax(bin_map) + 1
    for c in curr_bin_idxs:
        bin_map[m_Y[c], m_X[c]] = new_bin_id

    u, bin_inv = np.unique(bin_map.ravel(), return_inverse=True)
    nbins = len(u)
    bin_labels = np.arange(len(u), dtype=int)[bin_inv]

    binned_signal = np.bincount(bin_labels, weights=f_signal)
    binned_noise = np.sqrt(np.bincount(bin_labels, weights=f_noise**2))

    binned_s_n = binned_signal / binned_noise

    if plot:

        fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, constrained_layout=True)

        img_norm = astrovis.ImageNormalize(
            data=signal,
            interval=astrovis.ManualInterval(
                vmin=0.0, vmax=np.nanpercentile(signal, q=99.9)
            ),
            stretch=astrovis.LogStretch(),
        )
        plot_orig_sig = signal.copy()
        plot_orig_sig[mask] = np.nan
        axs[0, 0].imshow(signal, norm=img_norm, origin="lower", cmap="plasma")

        bin_sig_plot = (
            np.bincount(bin_labels, weights=f_signal) / np.bincount(bin_labels)
        )[bin_inv]
        bin_sig_plot[bin_labels == 0] = np.nan

        axs[0, 1].imshow(
            bin_sig_plot.reshape(signal.shape),
            origin="lower",
            norm=img_norm,
            cmap="plasma",
        )

        sn_plot = (
            np.bincount(bin_labels, weights=f_signal)
            / np.sqrt(np.bincount(bin_labels, weights=f_noise**2))
        )[bin_inv]
        sn_plot[bin_labels == 0] = np.nan
        axs[1, 0].imshow(
            sn_plot.reshape(signal.shape),
            origin="lower",
            cmap="plasma",
        )

        rng = np.random.default_rng()
        bin_plot = rng.random(size=len(u))[bin_inv]
        bin_plot[bin_labels == 0] = np.nan

        axs[1, 1].imshow(
            bin_plot.reshape(signal.shape),
            origin="lower",
            cmap="jet",
            interpolation="none",
        )

        for a in axs.flatten():
            a.set_facecolor("k")

        plt.show()

    return bin_labels.reshape(signal.shape), nbins, binned_s_n, bin_inv


def permute_axes_subtract(arr: ArrayLike, axis: int = 0) -> ArrayLike:
    """
    Find the difference between all pairs of points.

    Original solution taken from
    `https://stackoverflow.com/questions/55353703`.

    Parameters
    ----------
    arr : ArrayLike
        The (n_0 x ... n_i) array containing the data.
    axis : int, optional
        The axis along which to find all combinations of differences, by
        default 0.

    Returns
    -------
    ArrayLike
        The colour combination array. The additional axis will be inserted
        after `axis`, e.g. for an (m x n) array and `axis=0`, an
        (m x m x n) array will be returned.
    """
    # Get array shape
    s = arr.shape

    # Get broadcastable shapes by introducing singleton dimensions
    s1 = np.insert(s, axis, 1)
    s2 = np.insert(s, axis + 1, 1)

    # Perform subtraction after reshaping input array to
    # broadcastable ones against each other
    return arr.reshape(s1) - arr.reshape(s2)
