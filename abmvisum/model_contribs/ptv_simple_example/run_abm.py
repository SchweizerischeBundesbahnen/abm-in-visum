import os
import logging
import importlib

from src import abm_utilities, visum_utilities

import abm
importlib.reload(abm)  # helps prevent python from caching files


def run_abm(abm_class, model_dir, project_name=""):
    # option 1: run full abm with trip chain choice
    abm_class.run_full_abm()

    # option 2: use this to run full abm using synthetic population
    # abm_class.run_full_abm_with_syn_pop()

    # option 3/4: same as above, but generate step-by-step debug output
    # abm_class.run_full_abm(lambda step_name, only_demand : visum_utilities.write_net_and_demand(abm_class.Visum, model_dir, project_name, step_name, only_demand))
    # abm_class.run_full_abm_with_syn_pop(lambda step_name, only_demand : visum_utilities.write_net_and_demand(abm_class.Visum, model_dir, project_name, step_name, only_demand))


def run_abm_from_current_visum(Visum):
    model_dir = r'.\\'
    my_abm = abm.ABM(Visum=Visum, model_dir=model_dir)
    run_abm(my_abm, model_dir)


def run_abm_from_version_file(project_name, visum_version_no):
    model_dir = os.path.dirname(os.path.realpath(__file__))

    input_version_file  = os.path.join(model_dir, f"{project_name}.ver")
    result_version_file = os.path.join(model_dir, f"{project_name}_Result.ver")

    logging.info('--- Load version file ---')

    # run ABM
    my_abm = abm.ABM.from_version_file(version_filename=input_version_file, visum_version_no=visum_version_no, model_dir=model_dir)

    run_abm(my_abm, model_dir, project_name)

    logging.info('--- Save version file ---')
    my_abm.Visum.SaveVersion(result_version_file)

    logging.info('--- Finished ---')


if __name__ == "__main__":
    # start logging engine
    abm_utilities.start_logging()

    if 'Visum' in globals():
        run_abm_from_current_visum(Visum=globals()['Visum'])
    else:
        # Halle_ABM.ver needs to contain demand data and model specification: 
        # 1_UDA.net, 2_Model_specifications.net, 3_Base_Demand_Model.dmd, 4_Base_Demand_Model_Data.att, 5_Structural_Data.dmd
        # and results calculated in procedures steps 4-31
        run_abm_from_version_file(project_name=r'Halle_ABM', visum_version_no=22)
