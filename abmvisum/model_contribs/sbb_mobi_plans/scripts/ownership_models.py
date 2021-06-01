import importlib
import sys
from pathlib import Path

HOME_DIR = Path(__file__).parent.parent.parent.parent.parent
sys.path.append(str(HOME_DIR))

import abmvisum.model_contribs.sbb_mobi_plans.simulator as simulator
from abmvisum.tools.utilities import start_logging

importlib.reload(simulator)

if __name__ == '__main__':
    start_logging()

    abm_simulation = simulator.MOBiPlansSimulator(Visum=Visum)

    abm_simulation.ownership_models()
