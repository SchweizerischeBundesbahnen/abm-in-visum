import importlib
import sys
from pathlib import Path

from dask.distributed import Client

HOME_DIR = Path(__file__).parent.parent.parent.parent.parent
sys.path.append(str(HOME_DIR))

import abminvisum.model_contribs.sbb_mobi_plans.simulator as simulator
from abminvisum.tools.utilities import start_logging

time_budget_dict = {0: {'performing_budget': 14.0,
                        'travel_time_budget': 12.0,
                        'out_of_home_time_budget': 13.5},
                    1: {'performing_budget': 12.0,
                        'travel_time_budget': 5.0,
                        'out_of_home_time_budget': 14.0},
                    2: {'performing_budget': 11.0,
                        'travel_time_budget': 4.0,
                        'out_of_home_time_budget': 15.0},
                    3: {'performing_budget': 10.0,
                        'travel_time_budget': 3.0,
                        'out_of_home_time_budget': 16.5}}

importlib.reload(simulator)

if __name__ == '__main__':
    start_logging()

    # powerful machine is necessary to run this client, e.g. NALA at SBB
    client = Client(processes=False, threads_per_worker=42, n_workers=1, memory_limit='200GB')

    # possible configuration for a smaller machine
    # client = Client(processes=False, threads_per_worker=8, n_workers=1, memory_limit='40GB')

    abm_simulation = simulator.MOBiPlansSimulator(Visum=Visum)

    abm_simulation.long_term_location_choice()

    abm_simulation.plan_generation(time_budget_dict=time_budget_dict, out_of_home_budget=18.0)

    abm_simulation.mode_choice()

    abm_simulation.start_time_choice()
