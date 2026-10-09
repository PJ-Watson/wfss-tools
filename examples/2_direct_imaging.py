"""An example workflow for aligning and reprojecting imaging data."""

import os
from pathlib import Path

import yaml

root_dir = Path(os.getenv("ROOT_DIR"))
field_name = "glass-a2744"
proposal_ID = 1324
reduction_dir = Path(root_dir) / f"2025_08_05_{field_name}"

if __name__ == "__main__":

    new_folder = reduction_dir / "glass_niriss_bcgs"
    new_folder.mkdir(exist_ok=True, parents=True)

    bcgs_dict_path = new_folder / "existing_photometry.yaml"

    try:
        # Reload the full obs dict with PSF info
        with open(bcgs_dict_path, "r") as file:
            bcgs_dict = yaml.safe_load(file)

    except:

        from wfss_tools.pipeline import find_matches, parse_images_from_pattern

        megascience_dir = root_dir / "archival" / "grizli-v2" / "JwstMosaics" / "v7"
        bcgs_dict = parse_images_from_pattern(
            megascience_dir, pattern="*bcgs_out_sci.fits.gz"
        )

        print(bcgs_dict)

        # Find variance images
        bcgs_dict = find_matches(megascience_dir, bcgs_dict)

        # HST name format is different
        bcgs_dict = find_matches(
            megascience_dir, bcgs_dict, pattern="*{filt}_drc_var.fits*"
        )

        # Find PSFs
        bcgs_dict = find_matches(
            megascience_dir,
            bcgs_dict,
            pattern="UNCOVER_DR3_PSFs/{filt}_psf_norm.fits",
            key_name="psf",
        )

        print(bcgs_dict)

        with open(bcgs_dict_path, "w") as file:
            yaml.dump(bcgs_dict, file, sort_keys=False)

    # Where to save the details of the PSF-matched images
    conv_dict_path = new_folder / "PSF_matched_photometry.yaml"

    # Where to save the reprojected and convolved images
    conv_out_dir = new_folder / "PSF_matched_photometry"

    # The reference mosaic to align the images to
    ref_mosaic = (
        reduction_dir / "grizli_home" / "Prep" / f"{field_name}-ir_drc_sci.fits"
    )

    # Avoid parsing files again if the details already exist
    if conv_dict_path.is_file():

        with open(conv_dict_path, "r") as file:
            conv_dict = yaml.safe_load(file)

    else:

        from wfss_tools.isophotal import reproject_and_convolve

        conv_dict = {}
        for filt_key, old_details in bcgs_dict.items():
            conv_dict[filt_key] = {
                k: v
                for k, v in old_details.items()
                if k not in ["exp", "sci", "var", "wht"]
            }

            for t in ["sci", "var"]:
                print(t, bcgs_dict[filt_key][t])
                _conv_out_path = reproject_and_convolve(
                    ref_path=ref_mosaic,
                    orig_images=Path(bcgs_dict[filt_key][t]),
                    psfs=Path(bcgs_dict[filt_key]["psf"]),
                    psf_target=bcgs_dict["jwst-nircam-f200w"]["psf"],
                    out_dir=conv_out_dir,
                    new_names=f"repr_{filt_key}_{t}.fits",
                    reproject_image_kw={
                        "method": "adaptive",
                        "compress": False,
                    },
                    new_wcs_kw={"resolution": 0.04},
                )

                conv_dict[filt_key][t] = str(_conv_out_path[0])

        with open(conv_dict_path, "w") as outfile:
            yaml.dump(conv_dict, outfile, default_flow_style=False, sort_keys=False)

    print(conv_dict)
