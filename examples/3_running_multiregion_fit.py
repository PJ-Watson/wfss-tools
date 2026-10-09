"""An example workflow for performing a multi-region fit to NIRISS data."""

import os
import shutil
import tomllib
from functools import partial
from pathlib import Path

from astropy.io import fits

# Latest context
os.environ["CRDS_CONTEXT"] = "jwst_1413.pmap"

conf_type = "NGDEEP_conf_A"

from wfss_tools.grism import MultiRegionFit

# Or similar
root_dir = Path(os.getenv("ROOT_DIR"))
field_name = "glass-a2744"
proposal_ID = 1324
reduction_dir = Path(root_dir) / f"2025_08_05_{field_name}"

# The location of the config file
config_path = Path(__file__).parent / "example_config.toml"

if __name__ == "__main__":

    # Load the config
    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    # The grizli-extracted files are only used to select the redshift for
    # fitting. Catalogues are also a valid method to use here.
    full_fits = list(
        (reduction_dir / "grizli_home" / f"Extractions-{conf_type}").glob(
            "**/*full.fits"
        )
    )
    full_fits.sort()

    # Loop over all the sources extracted by grizli.
    for file in full_fits[:]:

        hdr = fits.getheader(file, 0)
        obj_id = hdr["ID"]
        obj_z = round(hdr["REDSHIFT"], 5)

        # Set `run_all=True` to run everything automatically
        multiregion = MultiRegionFit(config_path, obj_id, obj_z, run_all=False)

        # Depending on the parameters set in the config file, this path
        # may differ slightly
        out_path = (
            multiregion.out_dir
            / "multiregion"
            / f"regions_{obj_id:0>5}_z_{obj_z}_0.06arcsec.line.fits"
        )

        # Check if the file exists, and has the correct header values from
        # the current version of the code. If not, either run the full
        # process, or regenerate the existing file if a fit has already
        # been performed.
        try:
            hdr = fits.getheader(out_path, 0)
            hdr["MRBPFCAT"]

        except:
            multiregion.run_all()
