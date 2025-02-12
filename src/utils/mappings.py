"""
Mappings of categorical variables.
"""

import json

mappings = {}
for key in ["apet_finale", "CJ", "CRT", "NAT", "TYP"]:
    file = f"../../data/mappings/{key}_mapping.json"
    with open(file, "r") as f:
        mappings[key] = json.load(f)

mappings["APE_NIV5"] = mappings["apet_finale"]

SURFACE_COLS = ["activ_surf_et", "SRF"]
