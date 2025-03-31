"""
Mappings of categorical variables.

The jsons in data/mappings should be named accordingly:
    XXX_suffix
where XXX is the name of the variable, and suffix is common for all mappings.

As for now, the suffix is "_mapping.json".

"""

import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Get absolute path
mapping_dir = os.path.join(BASE_DIR, "../../data/mappings")  # Adjust path
mapping_dir = os.path.abspath(mapping_dir)  # Normalize


mappings = {}
suffix = "_mapping.json"
keys = [f[: -len(suffix)] for f in os.listdir(mapping_dir) if f.endswith(suffix)]

# dynamically fetch the mapping jsons in the mapping directory
keys = [f[: -len(suffix)] for f in os.listdir(mapping_dir) if f.endswith(suffix)]

for key in keys:
    file = os.path.join(mapping_dir, key + suffix)
    with open(file, "r") as f:
        mappings[key] = json.load(f)

mappings["APE_NIV5"] = mappings["apet_finale"]

SURFACE_COLS = ["activ_surf_et", "SRF"]
