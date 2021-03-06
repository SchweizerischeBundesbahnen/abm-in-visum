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
        This script performs the start time choice model within a Visum procedure.
    
        It contains the following steps of MOBi.plans:
        1. start times for each activity
    
        Output are start- (and end-) times assigned to each activity execution and trip
    
        !Important: This scripts resets some important information in the case no valid start times have been found
            -> You cannot run it twice, the code will fail!
                
        Author:
            Patrick Manser
    """
    start_logging()

    abm_simulation = simulator.MOBiPlansSimulator(Visum=Visum, add_skims_to_cache=False)

    abm_simulation.start_time_choice()
