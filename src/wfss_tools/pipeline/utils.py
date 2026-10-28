"""Pipeline utility functions."""

import os
import zipfile
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.table import Table
from grizli import utils as grizli_utils
from grizli.multifit import MultiBeam
from numpy.typing import ArrayLike
from tqdm import tqdm

__all__ = [
    "parse_images_from_pattern",
    "find_matches",
    "getObsIdFromQuery",
    "getExpIdFromQuery",
    "queryMAST",
    "separate_oned_spectra",
    "gen_linefinding_outputs",
    "gen_pygcg_outputs",
]


def parse_images_from_pattern(img_dir: os.PathLike, pattern: str = "*.fits") -> dict:
    """
    Populate a dict with info and locations of images matching a pattern.

    Parameters
    ----------
    img_dir : os.PathLike
        The directory to search.
    pattern : str, optional
        The pattern to search with, by default ``"*.fits"``.

    Returns
    -------
    dict
        The keys of each entry are lower case, and in the format
        ``"{telescope}-{instrument}-{filter}"``. Each entry is itself a
        dictionary, containg basic information on the image and its
        location.
    """

    out_dict = {}
    for i, filepath in enumerate(img_dir.glob(f"{pattern}")):
        print(filepath.name)
        hdr = fits.getheader(filepath)

        key = f"{hdr["TELESCOP"]}-{hdr["INSTRUME"]}-"
        filt = grizli_utils.parse_filter_from_header(hdr)
        key += f"{filt.removeprefix("F150W2-").removesuffix("-CLEAR")}"
        key = key.lower()

        out_dict[key] = {
            "filt": grizli_utils.parse_filter_from_header(hdr, filter_only=True),
            "pupil": hdr.get("PUPIL", "UNKNOWN"),
            "detector": hdr["DETECTOR"],
            "instrument": hdr["INSTRUME"],
            "telescope": hdr["TELESCOP"],
            "sci": str(filepath),
        }

    return out_dict


def find_matches(
    img_dir: os.PathLike,
    info_dict: dict,
    pattern: str = "*{filt}-*_drc_var.fits*",
    key_name: str = "var",
    case_sensitive: bool = False,
) -> dict:
    """
    Update an info_dict with matches from a specified directory.

    Parameters
    ----------
    img_dir : os.PathLike
        The directory to search.
    info_dict : dict
        A nested dictionary, where each entry contains information about
        observed images.
    pattern : str, optional
        The filename pattern to match, by default
        ``"*{filt}-*_drc_var.fits*"``.
    key_name : str, optional
        The name to add to each dictionary item, by default ``"var"``.
    case_sensitive : bool, optional
        Whether to match on case, by default ``False``.

    Returns
    -------
    dict
        The modified input ``info_dict``.
    """

    # I can't wait for Python 3.14 to introduce template strings and clear this mess up
    import re

    kw_names = re.findall(r"\{(.*?)\}", pattern)
    print(kw_names)

    for key, info in info_dict.items():
        eval_kws = {}
        for kw in kw_names:
            eval_kws[kw] = info[kw]
        print(pattern.format(**eval_kws))
        for i, filepath in enumerate(
            img_dir.glob(
                pattern.format(**eval_kws),
                case_sensitive=case_sensitive,
            )
        ):
            info_dict[key][key_name] = str(filepath)

    return info_dict


@np.vectorize
def getObsIdFromQuery(obsName: str) -> int:
    """
    Cutout the obs ID from the long, jumbled MAST obs ID.

    The original version of this function was written by VM and ZS for
    `passagepipe.utils`.

    Parameters
    ----------
    obsName : str
        The MAST observation name.

    Returns
    -------
    int
        The observation ID.
    """

    return int(obsName.split("_")[0][7:-3])


@np.vectorize
def getExpIdFromQuery(obsName: str) -> int:
    """
    Cutout the exp ID from the long, jumbled MAST obs ID.

    The original version of this function was written by VM and ZS for
    `passagepipe.utils`.

    Parameters
    ----------
    obsName : str
        The MAST observation name.

    Returns
    -------
    int
        The exposure ID.
    """

    return int(obsName.split("_")[1])


def queryMAST(
    pid: int, instrument: str = "NIRISS", use_filter: ArrayLike | None = None
) -> Table:
    """
    Query MAST for the full list of observations for specific PID.

    The original version of this function was written by VM and ZS for
    `passagepipe.utils`.

    Parameters
    ----------
    pid : int
        The JWST Proposal ID.
    instrument : str, optional
        Select the instrument to query observations. By default only NIRISS
        observations will be returned.
    use_filter : ArrayLike | None, optional
        Return observations only in a specific set of filters, by default
        `None`.

    Returns
    -------
    Table
        The set of observations requested.
    """

    from astropy.table import vstack
    from astroquery.mast import Observations
    from mastquery import query

    query.DEFAULT_QUERY["project"] = ["JWST"]
    query.DEFAULT_QUERY["obs_collection"] = ["JWST"]
    query.DEFAULT_QUERY["instrument_name"] = [f"{instrument.upper()}*"]

    queryList = query.run_query(
        box=None,
        proposal_id=[pid],
        base_query=query.DEFAULT_QUERY,
    )
    if use_filter is not None:
        queryList = queryList[np.isin(queryList["filter"], use_filter)]

    if "target_name" not in queryList.columns:
        queryList["target_name"] = queryList["target"]
    subqueryList = Observations.get_product_list(queryList)

    cond = (
        (subqueryList["calib_level"] == 1)
        # & (subqueryList["productType"] == "SCIENCE")
        & (subqueryList["productSubGroupDescription"] == "UNCAL")
    )

    uncalList = subqueryList[cond]
    _, idx = np.unique(uncalList["obs_id"], return_index=True)
    uncalList = uncalList[idx]

    uncalList["obs_id_num"] = getObsIdFromQuery(obsName=np.asarray(uncalList["obs_id"]))
    uncalList["exp_id_num"] = getExpIdFromQuery(obsName=np.asarray(uncalList["obs_id"]))
    return uncalList


def separate_oned_spectra(mb: MultiBeam, tfit: dict | None = None) -> fits.HDUList:
    """
    Extract 1D spectra for each grism, e.g. GR150R/GR150C.

    The standard 1D spectra derived by grizli combine all position angles
    and grisms. This allows the output to be used directly with codes such
    as `jwstwfss/line-finding <https://github.com/jwstwfss/line-finding>`_
    (`Nedkova+26 <doi.org/10.5281/zenodo.19228845>`_).

    Parameters
    ----------
    mb : MultiBeam
        The grizli-extracted multiple beams object.
    tfit : dict or None
        Dictionary of fit results (templates, coefficients, etc) from
        `~grizli.fitting.GroupFitter.template_at_z`.

    Returns
    -------
    fits.HDUList
        FITS version of the 1D spectrum tables.
    """

    from copy import deepcopy

    new_hdul = mb.oned_spectrum_to_hdu(tfit=tfit)

    for k, v in mb.PA.items():
        for pa, beam_idx in v.items():
            try:
                _mb = deepcopy(mb)
                _mb.beams = [_mb.beams[i] for i in beam_idx]
                _mb._parse_beams(psf=_mb.psf_param_dict is not None)
                _mb.initialize_masked_arrays()
                if tfit is not None:
                    _tfit = tfit.copy()
                    _tfit["coeffs"] = np.asarray([tfit["coeffs"][i] for i in beam_idx])
                    _tfit["coeffs"] = np.concatenate(
                        [_tfit["coeffs"], tfit["coeffs"][mb.N :]]
                    )
                    out = _mb.oned_spectrum_to_hdu(tfit=_tfit)
                else:
                    _mb.oned_spectrum_to_hdu()
                out[-1].header["EXTVER"] = pa
                out[-1].header["FILTER"] = _mb.beams[0].grism.filter
                new_hdul.append(out[-1])
            except:
                continue

    return new_hdul


def gen_linefinding_outputs(
    grizli_home_dir: os.PathLike,
    field_name: str = "passage-par682",
    new_field_name: str = "Par682",
    out_dir: os.PathLike | None = None,
    zipfile_kwargs: dict = {"compression": zipfile.ZIP_DEFLATED},
) -> Path:
    """
    Create an archive with the files required for `jwstwfss/linefinding`.

    Parameters
    ----------
    grizli_home_dir : os.PathLike
        Directory containing the usual grizli folders, e.g. ``"Prep"``,
        ``"visits"``.
    field_name : str, optional
        The name of the field, by default ``"passage-par682"``.
    new_field_name : str, optional
        The linefinding-compatible field name, by default ``"Par682"``.
    out_dir : os.PathLike | None, optional
        The output directory. If ``None`` (default), the archive will be
        saved to ``grizli_home_dir``.
    zipfile_kwargs : dict, optional
        Additional keyword arguments to pass through to `zipfile.ZipFile`,
        by default ``{"compression": zipfile.ZIP_DEFLATED}``.

    Returns
    -------
    Path
        The path to the zipped archive.
    """

    grizli_home_dir = Path(grizli_home_dir)

    if out_dir is not None:
        out_dir = Path(out_dir)
    else:
        out_dir = grizli_home_dir

    lf_archive = out_dir / f"linefinding_data.zip"

    with zipfile.ZipFile(lf_archive, "a", **zipfile_kwargs) as myzip:
        zip_path = zipfile.Path(myzip)

        if not (zip_path / "linelist").is_dir():
            myzip.mkdir("linelist")

        new_photcat_path = (
            zip_path
            / new_field_name
            / "DATA"
            / "DIRECT_GRISM"
            / f"{new_field_name}_photcat.fits"
        )
        if not (new_photcat_path).is_file():
            myzip.write(
                grizli_home_dir / "Prep" / f"{field_name}_phot.fits",
                new_photcat_path.relative_to(zip_path),
            )

        for f in tqdm(
            list((grizli_home_dir / "Prep").glob(f"{field_name}-*.fits")),
            desc="Compressing aligned images",
        ):
            new_filepath = (
                zip_path
                / f"{new_field_name}"
                / "DATA"
                / f"{f.name}".replace(f"{field_name}-", f"{new_field_name}_")
            )
            if not new_filepath.is_file():
                myzip.write(f, new_filepath.relative_to(zip_path))

        detector_drz_files = list(
            (grizli_home_dir / "visits").glob(f"*/Prep/*_drz_*.fits")
        )
        detector_drz_files.sort()
        for f in tqdm(
            list((grizli_home_dir / "visits").glob(f"*/Prep/*_drz_*.fits")),
            desc="Compressing detector images",
        ):
            new_filepath = (
                zip_path
                / f"{new_field_name}"
                / "DATA"
                / f"{new_field_name}_{f.name[f.name.index("-f")+1:]}".replace(
                    "-clear", ""
                )
            )
            if not new_filepath.is_file():
                myzip.write(f, new_filepath.relative_to(zip_path))

        for oned_dir in ["1D_RC", "1D"]:
            for f in tqdm(
                list((grizli_home_dir / "Extractions" / oned_dir).glob("*.fits")),
                desc=f"Compressing {oned_dir} spectra",
            ):
                new_filepath = (
                    zip_path
                    / f"{new_field_name}"
                    / "spec1D"
                    / f"{f.name}".replace(f"{field_name}_", f"{new_field_name}_")
                    .replace("1D_RC", "1D")
                    .replace("1D", "spec1D")
                )

                if not new_filepath.is_file():
                    myzip.write(f, new_filepath.relative_to(zip_path))

        for f in tqdm(
            list((grizli_home_dir / "Extractions" / "stack").glob("*.fits")),
            desc="Compressing 2D spectra",
        ):
            new_filepath = (
                zip_path
                / f"{new_field_name}"
                / "spec2D"
                / f"{f.name}".replace(f"{field_name}_", f"{new_field_name}_").replace(
                    "stack", "spec2D"
                )
            )

            if not new_filepath.is_file():
                myzip.write(f, new_filepath.relative_to(zip_path))

    return lf_archive


def gen_pygcg_outputs(
    grizli_home_dir: os.PathLike,
    field_name: str = "passage-par682",
    out_dir: os.PathLike | None = None,
    zipfile_kwargs: dict = {"compression": zipfile.ZIP_DEFLATED},
) -> Path:
    """
    Create an archive with the files required for `PJ-Watson/pyGCG`.

    Parameters
    ----------
    grizli_home_dir : os.PathLike
        Directory containing the usual grizli folders, e.g. ``"Prep"``,
        ``"visits"``.
    field_name : str, optional
        The name of the field, by default ``"passage-par682"``.
    out_dir : os.PathLike | None, optional
        The output directory. If ``None`` (default), the archive will be
        saved to ``grizli_home_dir``.
    zipfile_kwargs : dict, optional
        Additional keyword arguments to pass through to `zipfile.ZipFile`,
        by default ``{"compression": zipfile.ZIP_DEFLATED}``.

    Returns
    -------
    Path
        The path to the zipped archive.
    """

    import tomlkit

    grizli_home_dir = Path(grizli_home_dir)

    if out_dir is not None:
        out_dir = Path(out_dir)
    else:
        out_dir = grizli_home_dir

    pygcg_archive = out_dir / f"pygcg_data.zip"

    with zipfile.ZipFile(pygcg_archive, "a", **zipfile_kwargs) as myzip:
        zip_path = zipfile.Path(myzip)

        new_cat_path = (
            zip_path / field_name / "Extractions" / f"{field_name}-ir.cat.fits"
        )
        if not (new_cat_path).is_file():
            myzip.write(
                grizli_home_dir / "Prep" / new_cat_path.name,
                new_cat_path.relative_to(zip_path),
            )

        for f in tqdm(
            list((grizli_home_dir / "Prep").glob(f"{field_name}-*.fits")),
            desc="Compressing aligned images",
        ):
            new_filepath = zip_path / field_name / "Prep" / f.name
            if not new_filepath.is_file():
                myzip.write(f, new_filepath.relative_to(zip_path))

        for oned_dir in ["1D"]:
            for f in tqdm(
                list((grizli_home_dir / "Extractions" / oned_dir).glob("*.fits")),
                desc=f"Compressing {oned_dir} spectra",
            ):
                new_filepath = zip_path / field_name / "Extractions" / oned_dir / f.name

                if not new_filepath.is_file():
                    myzip.write(f, new_filepath.relative_to(zip_path))

        for f in tqdm(
            list((grizli_home_dir / "Extractions" / "stack").glob("*.fits")),
            desc="Compressing 2D spectra",
        ):
            new_filepath = zip_path / field_name / "Extractions" / "stack" / f.name

            if not new_filepath.is_file():
                myzip.write(f, new_filepath.relative_to(zip_path))

        zinfo_dir = grizli_home_dir / "Extractions" / "zinfo"
        zinfo_dir.mkdir(exist_ok=True, parents=True)

        for f in tqdm(
            list((grizli_home_dir / "Extractions" / "full").glob("*.fits")),
            desc="Creating zinfo files",
        ):
            zinfo_filepath = zinfo_dir / f.name.replace("full", "zinfo")

            if not zinfo_filepath.is_file():

                with fits.open(f) as full_hdul:
                    zinfo_hdul = full_hdul[:2]
                    zinfo_hdul[1].data = fits.BinTableHDU(
                        Table(zinfo_hdul[1].data)["zgrid", "chi2"]
                    ).data
                    zinfo_hdul.writeto(zinfo_filepath)

            new_filepath = (
                zip_path / field_name / "Extractions" / "zinfo" / zinfo_filepath.name
            )

            if not new_filepath.is_file():
                myzip.write(zinfo_filepath, new_filepath.relative_to(zip_path))

        doc = tomlkit.document()

        import datetime

        curr_time = (
            datetime.datetime.now().astimezone().replace(microsecond=0).isoformat()
        )

        doc.add(tomlkit.comment(f"{field_name} config generated on {curr_time}"))
        doc.add(tomlkit.nl())
        doc.add("title", f"pyGCG config: {field_name}")

        files = tomlkit.table()
        files.add("root_dir", "")
        files.add("out_dir", f"{field_name}/pyGCG_outputs")
        files.add("extractions_dir", f"{field_name}/Extractions")
        files.add("cat_path", f"{field_name}/Extractions/{field_name}-ir.cat.fits")
        files.add("prep_dir", f"{field_name}/Prep")
        files.add("out_cat_name", f"pyGCG_class_{field_name}.fits")

        # Adding the table to the document
        doc.add("files", files)

        grisms = tomlkit.table()
        grisms.add("root_dir", "")

        all_PAs = []
        for f in tqdm(
            list((grizli_home_dir / "Extractions" / "zinfo").glob("*.fits")),
            desc="Checking zinfo files",
        ):
            hdr = fits.getheader(f)
            total_n = (
                hdr.get("N_F115W", 0) + hdr.get("N_F150W", 0) + hdr.get("N_F200W", 0)
            )
            all_PAs.extend(
                np.unique(
                    np.array([hdr.get(f"PA{i+1:0>4}") for i in np.arange(total_n)])
                )
            )

        all_PAs = np.unique(all_PAs)
        for i, p in enumerate(all_PAs):
            grisms.add(f"PA{i}", p)

        doc.add("grisms", grisms)

        with open(grizli_home_dir / f"pyGCG_config_{field_name}.toml", "w") as fp:
            tomlkit.dump(doc, fp)

        new_filepath = zip_path / f"pyGCG_config_{field_name}.toml"

        if not (new_filepath).is_file():
            myzip.write(
                grizli_home_dir / f"pyGCG_config_{field_name}.toml",
                f"pyGCG_config_{field_name}.toml",
            )

    return pygcg_archive
