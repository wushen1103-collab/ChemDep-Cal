#!/usr/bin/env python3
"""Download the twelve precommitted MF-PCBA assay chains with source checksums."""

import download_pubchem_four_untouched_20261004 as source


source.CAMPAIGNS = {
    "AID1117319": (1117319, (1117362,)),
    "AID1224905": (1224905, (1259350,)),
    "AID488899": (488899, (493073,)),
    "AID485317": (485317, (493248,)),
    "AID488895": (488895, (504941,)),
    "AID504621": (504621, (540268,)),
    "AID504582": (504582, (540271,)),
    "AID493091": (493091, (540297,)),
    "AID588549": (588549, (624273,)),
    "AID651658": (651658, (687022,)),
    "AID652115": (652115, (720591,)),
    "AID1979": (1979, (2423,)),
}


if __name__ == "__main__":
    source.main()
