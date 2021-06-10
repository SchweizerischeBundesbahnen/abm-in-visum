import importlib
import sys
from pathlib import Path

from dask.distributed import Client

HOME_DIR = Path(__file__).parent.parent.parent.parent.parent
sys.path.append(str(HOME_DIR))

import abmvisum.model_contribs.sbb_mobi_plans.simulator as simulator
from abmvisum.tools.utilities import start_logging

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
    """
        This script performs the plan-building algorithm within a Visum procedure.
    
        It contains the following steps of MOBi.plans:
        1. generation models (tour, stop, subtour frequencies)
        2. activity type
        3. destinations of the secondary activities
        4. durations of all activities
        5. iterative process based on time budgets (travel, performing, total out-of-home)
    
        Output are mobility plans containing tours and trips (incl. network distance) as well as
        activity executions (incl. durations and exact coordinates)
        
        Author:
            Patrick Manser
    """

    start_logging()

    # powerful machine is necessary to run this client, e.g. NALA at SBB
    client = Client(processes=False, threads_per_worker=42, n_workers=1, memory_limit='200GB')

    # possible configuration for a smaller machine
    # client = Client(processes=False, threads_per_worker=8, n_workers=1, memory_limit='40GB')

    abm_simulation = simulator.MOBiPlansSimulator(Visum=Visum)

    abm_simulation.plan_generation(time_budget_dict=time_budget_dict, out_of_home_budget=18.0)
