import importlib
import sys
from pathlib import Path

HOME_DIR = Path(__file__).parent.parent.parent.parent.parent
sys.path.append(str(HOME_DIR))

import abmvisum.model_contribs.sbb_mobi_plans.simulator as simulator
from abmvisum.tools.utilities import start_logging

importlib.reload(simulator)

if __name__ == '__main__':
    """
        This script simulates mobility tool ownership choices. It contains the following choices:
        1. driving license per adult
        2. number of cars per household
        3. public transport subscription for each person 

        Author:
            Patrick Manser
    """

    start_logging()

    abm_simulation = simulator.MOBiPlansSimulator(Visum=Visum, add_skims_to_cache=False)

    abm_simulation.ownership_models()
