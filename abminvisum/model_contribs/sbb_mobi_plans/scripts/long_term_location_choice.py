"""
This scripts performs the long term location choice model within a Visum procedure.

It contains the following steps of MOBi.plans:
1. location choice with an IPF-algorithm

Output are long-term location keys (pointing to exact coordinates) for persons who are either employed (work place)
or in any kind of education (school)
"""
import importlib
import sys
from pathlib import Path

from dask.distributed import Client

HOME_DIR = Path(__file__).parent.parent.parent.parent.parent
sys.path.append(str(HOME_DIR))

import abminvisum.model_contribs.sbb_mobi_plans.simulator as simulator
from abminvisum.tools.utilities import start_logging

importlib.reload(simulator)

if __name__ == '__main__':
    start_logging()

    # powerful machine is necessary to run this client, e.g. NALA at SBB
    client = Client(processes=False, threads_per_worker=42, n_workers=1, memory_limit='200GB')

    # possible configuration for a smaller machine
    # client = Client(processes=False, threads_per_worker=8, n_workers=1, memory_limit='40GB')

    abm_simulation = simulator.MOBiPlansSimulator(Visum=Visum)

    abm_simulation.long_term_location_choice()
