"""An example workflow for reducing NIRISS/WFSS data from PASSAGE Par028."""

import os
from pathlib import Path

import numpy as np

try:
    from passagepipe import utils
except:
    import utils

from astropy.table import Table

# Latest context
os.environ["CRDS_CONTEXT"] = "jwst_1413.pmap"
# Set to "NGDEEP" to use those calibrations
os.environ["NIRISS_CALIB"] = "GRIZLI"

from wfss_tools import pipeline

root_dir = os.getenv("ROOT_DIR")
field_name = "par028"
field_obs_IDs = [228, 229, 230, 231, 232, 233, 234, 235, 236]
# field_name = "par017"
# field_obs_IDs = [172, 173]
passage_dir = Path(root_dir) / f"2025_07_28_{field_name}"
passage_dir.mkdir(exist_ok=True, parents=True)


if __name__ == "__main__":

    # Find the correct observations (utils.py is from passagepipe, I couldn't figure out
    # how to access the raw files on MAST until checking that)
    if not (passage_dir / "MAST_summary.csv").is_file():
        all_obs_tab = utils.queryMAST(1571)
        all_obs_tab.write(passage_dir / "MAST_summary.csv", overwrite=True)
    else:
        all_obs_tab = Table.read(passage_dir / "MAST_summary.csv")

    field_obs_tab = all_obs_tab[np.isin(all_obs_tab["obs_id_num"], field_obs_IDs)]

    from mastquery import utils as mastutils

    MAST_dir = passage_dir / "MAST_downloads"
    MAST_dir.mkdir(exist_ok=True, parents=True)

    # Download if not already existing
    uncal_files = list(MAST_dir.glob("*_uncal.fits"))
    if len(uncal_files) != len(field_obs_tab):
        mastutils.download_from_mast(field_obs_tab, path=MAST_dir)

    level_1_dir = passage_dir / "Level1"
    level_1_dir.mkdir(exist_ok=True, parents=True)

    # Create the _rate.fits files
    pipeline.stsci_det1(MAST_dir, level_1_dir, cpu_count=2)

    import logging

    import grizli
    from astropy.io import fits
    from grizli import fitting, jwst_utils, multifit, prep, utils
    from grizli.pipeline import auto_script

    print("Grizli version: ", grizli.__version__)

    # Quiet JWST log warnings
    jwst_utils.QUIET_LEVEL = logging.INFO
    jwst_utils.set_quiet_logging(jwst_utils.QUIET_LEVEL)

    # Setup the grizli directory structure
    grizli_home_dir = passage_dir / "grizli_home"

    grizli_home_dir.mkdir(exist_ok=True, parents=True)
    (grizli_home_dir / "Prep").mkdir(exist_ok=True)
    (grizli_home_dir / "RAW").mkdir(exist_ok=True)
    (grizli_home_dir / "visits").mkdir(exist_ok=True)

    # As PASSAGE was not ingested into the DJA in the same way as
    # other fields (e.g. GLASS), we have to create an association
    # table ourselves. This contains info on the instrument, filters,
    # footprints, and filenames per group
    assoc_dict = pipeline.gen_associations(level_1_dir, field_name)
    # for assoc, tab in assoc_dict.items():
    #     tab.pprint_all()

    if not (grizli_home_dir / "Prep" / f"{field_name}-ir_drc_sci.fits").is_file():
        pipeline.process_using_aws(
            grizli_home_dir, level_1_dir, assoc_dict, field_name=field_name
        )

    # Set up the grizli extraction directory structure
    (grizli_home_dir / "Extractions").mkdir(exist_ok=True)

    os.chdir(grizli_home_dir / "Prep")

    if not (Path.cwd() / f"{field_name}_phot.fits").is_file():

        if not (Path.cwd() / f"{field_name}-ir.cat.fits").is_file():

            from astropy.wcs import WCS
            from wfss_tools.isophotal import reproject_image
            from wfss_tools.pipeline import regen_catalogue

            # Or whatever name you came up with during the previous reduction
            old_seg_name = passage_dir / f"{field_name.capitalize()}_det_drz_seg.fits"

            aligned_seg_name = grizli_home_dir / "Prep" / f"aligned_{old_seg_name.name}"

            reproject_image(
                old_seg_name,
                aligned_seg_name,
                WCS(fits.getheader(f"{field_name}-ir_drc_sci.fits")),
                fits.getdata(f"{field_name}-ir_drc_sci.fits").shape,
                method="interp",
                order="nearest-neighbor",
            )

            segment_map = fits.getdata(aligned_seg_name)

            use_regen_seg = np.asarray(segment_map).astype(np.int32)
            new_cat = regen_catalogue(
                use_regen_seg,
                root=f"{field_name}-ir",
            )

        exist_cat_name = f"{field_name}-ir.cat.fits"

        multiband_catalog_args = auto_script.get_yml_parameters()[
            "multiband_catalog_args"
        ]
        multiband_catalog_args["run_detection"] = False
        multiband_catalog_args["filters"] = [
            "f115wn-clear",
            "f150wn-clear",
            "f200wn-clear",
        ]

        phot_cat = auto_script.multiband_catalog(
            field_root=field_name,
            master_catalog=exist_cat_name,
            **multiband_catalog_args,
        )

        # multiband_catalog_args = auto_script.get_yml_parameters()[
        #     "multiband_catalog_args"
        # ]
        # multiband_catalog_args["run_detection"] = True
        # multiband_catalog_args["filters"] = [
        #     "f115wn-clear",
        #     "f150wn-clear",
        #     "f200wn-clear",
        # ]
        # # print (multiband_catalog_args)
        # multiband_catalog_args["threshold"] = 2

        # phot_cat = auto_script.multiband_catalog(
        #     field_root=field_name,
        #     **multiband_catalog_args,
        # )

    kwargs = auto_script.get_yml_parameters()

    # The number of processes to use
    cpu_count = 4

    # We use one set of calibrations for the 1st order models, and combine
    # them with different models for the other orders.
    # Until such time as someone writes a new class for grizli.grismconf to
    # combine these properly, this means that the second set of models to
    # be generated are the only order(s) that can be extracted properly.
    # This shouldn't be a problem for F115W, F150W, and F200W, since we
    # are rarely interested in anything other than 1st order for these
    # filters (F090W may be more problematic).

    os.chdir(grizli_home_dir / "Prep")

    rate_files = [str(s) for s in Path.cwd().glob("*_rate.fits")][:]
    grism_files = [str(s) for s in Path.cwd().glob("*GrismFLT.fits")][:]

    if len(grism_files) == 0:
        # rate_files = []
        # for rate in Path.cwd().glob("*_rate.fits"):
        #     if fits.getheader(rate)["PUPIL"] != "F115W":
        #         rate_files.append(str(rate))

        # if len(rate_files) > 0:

        grism_prep_args = kwargs["grism_prep_args"]

        # For now, turn off refining contamination model with polynomial fits
        grism_prep_args["refine_niter"] = 0

        # Flat-flambda spectra
        grism_prep_args["init_coeffs"] = [1.0]

        grism_prep_args["mask_mosaic_edges"] = False

        # Here we use all of the detected objects.
        # These can be adjusted based on how deep the spectra/visits are
        grism_prep_args["refine_mag_limits"] = [14.0, 50.0]
        grism_prep_args["prelim_mag_limit"] = 50.0

        # The grism reference filters for direct images
        grism_prep_args["gris_ref_filters"] = {
            "GR150R": ["F115W", "F150W", "F200W"],
            "GR150C": ["F115W", "F150W", "F200W"],
        }

        grism_prep_args["use_jwst_crds"] = False
        grism_prep_args["files"] = rate_files[:]

        args_MB_conf = grism_prep_args.copy()

        # Calculate the non-1st order contamination using the 221215.conf files (Matharu & Brammer)
        args_MB_conf["model_kwargs"] = {
            "compute_size": True,
            "get_beams": ["B", "C", "D", "E"],
            # "get_beams": ["B"],
            "force_orders": True,
        }

        grp = auto_script.grism_prep(
            field_root=field_name, pad=800, cpu_count=cpu_count, **args_MB_conf
        )

        os.chdir(grizli_home_dir / "Prep")

        # Move the non-1st order contamination models to a different directory
        MB_conf_dir = grizli_home_dir / "Prep" / "MB_conf"
        MB_conf_dir.mkdir(exist_ok=True)
        for s in (grizli_home_dir / "Prep").glob("*GrismFLT.fits"):
            s.rename(MB_conf_dir / s.name)
            s.with_suffix(".pkl").unlink()

        # Calculate the 1st order models with the most up-to-date STScI calibrations
        args_CRDS_conf = grism_prep_args.copy()
        args_CRDS_conf["use_jwst_crds"] = True
        args_CRDS_conf["model_kwargs"] = {
            "compute_size": True,
            # "get_beams" : ["A", "C", "E"],
            "get_beams": ["A"],
            "force_orders": True,
        }
        grp = auto_script.grism_prep(
            field_root=field_name, pad=800, cpu_count=cpu_count, **args_CRDS_conf
        )

        # Add the two models together
        os.chdir(grizli_home_dir / "Prep")
        for s in (grizli_home_dir / "Prep").glob("*GrismFLT.fits"):
            with fits.open(s, mode="update") as crds_hdul:
                MB_cont_model = fits.getdata(
                    grizli_home_dir / "Prep" / "MB_conf" / s.name, "MODEL"
                )
                crds_hdul["MODEL"].data += MB_cont_model
                crds_hdul.flush()

        # Testing with only the NGDEEP calibrations
        # os.environ["NIRISS_CALIB"] = "NGDEEP"

        # args_NP_conf = grism_prep_args.copy()

        # args_NP_conf["model_kwargs"] = {
        #     "compute_size": True,
        #     "min_size": 32, # Any lower, and the 3rd order in F200W will throw errors
        # }

        # grp = auto_script.grism_prep(
        #     field_root=field_name, pad=800, cpu_count=cpu_count, **args_NP_conf
        # )

    # The usual extraction code follows

    os.chdir(grizli_home_dir / "Extractions")
    flt_files = [str(s) for s in Path.cwd().glob("*GrismFLT.fits")][:]

    grp = multifit.GroupFLT(
        grism_files=flt_files,
        catalog=f"{field_name}-ir.cat.fits",
        cpu_count=-1,
        sci_extn=1,
        pad=800,
    )

    pline = {
        "kernel": "square",
        "pixfrac": 1.0,
        "pixscale": 0.04,
        "size": 5,
        "wcs": None,
    }
    args = auto_script.generate_fit_params(
        pline=pline,
        field_root=field_name,
        min_sens=0.0,
        min_mask=0.0,
        # Set both of these to True to include photometry in fitting
        include_photometry=False,
        use_phot_obj=False,
    )

    # The galaxies in Ayan's paper
    galaxies = {1849: 3.08, 1303: 1.87, 300: 1.87, 2867: 1.97}

    for obj_id, obj_z in galaxies.items():

        if not (Path.cwd() / f"{field_name}_{obj_id:0>5}.full.fits").is_file():

            beams = grp.get_beams(
                obj_id,
                size=30,
                min_mask=0,
                min_sens=0,
                show_exception=True,
                beam_id="A",
            )
            mb = multifit.MultiBeam(
                beams, fcontam=0.2, min_sens=0.0, min_mask=0, group_name=field_name
            )
            mb.fit_trace_shift()
            mb.write_master_fits()

            _ = fitting.run_all_parallel(
                obj_id,
                zr=[obj_z - 0.05, obj_z + 0.05],
                dz=[0.001, 0.0001],
                verbose=True,
                get_output_data=True,
                skip_complete=False,
                save_figures=True,
            )
