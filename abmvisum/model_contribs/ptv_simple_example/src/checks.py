import logging

import VisumPy.helpers as VPH

from src import abm_utilities, config

def _check_attr_and_matrix_expr(helper_matrix, small_container, segments, check_matrix_expr):
    checked_tables = []
    for segment in segments:
        if segment['Specification'] in checked_tables:
            continue
        for attr_expr in segment['AttrExpr']:
            try:
                VPH.GetMultiByFormula(small_container, attr_expr)
            except:
                logging.warning("AttrExpr '%s' in '%s' is corrupt" , attr_expr, segment['Specification'])
        if check_matrix_expr: 
            for matrix_expr in segment['MatrixExpr']:
                try:
                    success = helper_matrix.SetValuesToResultOfFormula(matrix_expr)
                    if not success:
                        logging.warning("MatrixExpr '%s' in '%s' is corrupt", matrix_expr, segment['Specification'])
                except:
                    logging.warning("MatrixExpr '%s' in '%s' is corrupt", matrix_expr, segment['Specification'])
        checked_tables.append(segment['Specification'])


def _check_mode_impedance_definition(Visum, segments):
    for segment in segments:
        table_name = segment['Specification']
        mode_impedance_types = segment["ModeImpedanceTypes"]
        mode_impedance_definitions = segment["ModeImpedanceDefinitions"]
        for mode_impedance_type, mode_impedance_definition in zip(mode_impedance_types, mode_impedance_definitions):
            if mode_impedance_type == config.ModeImpedanceType.PrTShortestPathImpedance:
                # get part of prt impedance attribute before the first ()
                attr = mode_impedance_definition.partition('(')[0].strip()
                if not Visum.Net.Links.AttrExists(attr):
                    logging.warning(f"Table {table_name}: {attr} is not a link attribute.")
                if not Visum.Net.Turns.AttrExists(attr):
                    logging.warning(f"Table {table_name}: {attr} is not a turn attribute.")
                if not Visum.Net.MainTurns.AttrExists(attr):
                    logging.warning(f"Table {table_name}: {attr} is not a main turn attribute.")
                if not Visum.Net.Connectors.AttrExists(attr):
                    logging.warning(f"Table {table_name}: {attr} is not a connector attribute.")


def _check_time_intervals(time_interval_start_times : list, time_interval_end_times : list) -> None:
    time_gap_present = any(time_interval_start_times[i] != time_interval_end_times[i-1]
                           for i in range(1, len(time_interval_start_times)))
    if time_gap_present or time_interval_start_times[0] > 0 or time_interval_end_times[-1] != 60*60*24:
        logging.warning("The analyse time intervals do not cover the hole day. This causes errors if trips or activity executions take place outside the covered time span.")


def _create_dummy_objects_if_missing(Visum):
    """ For testing model step filters add dummy objects for first person if neccessary """
    personNos = VPH.GetMultiByFormula(Visum.Net.Persons,'[No]')
    minNo = int(min(personNos))
    person1 = abm_utilities.get_filtered_subjects(Visum.Net.Persons, f'[No] <= {minNo}')
    
    # [schedule inserted, tour inserted, acivity execution inserted]
    dummy_objs_inserted = [False, False, False]

    tours_of_person1 = abm_utilities.get_filtered_subjects(Visum.Net.Tours, f'[PersonNo] = {minNo}')
    if tours_of_person1.Count == 0:
        schedule = abm_utilities.get_filtered_subjects(Visum.Net.Schedules, f'[PersonNo] = {minNo} & [No] = 1')
        if schedule.Count == 0:
            dummy_objs_inserted[0] = True
            Visum.Net.AddMultiSchedules([(1, minNo)])
        tour_keys = [(0, minNo, 1)]
        dummy_objs_inserted[1] = True
        tours_of_person1 = Visum.Net.AddMultiTours(tour_keys)

    actEx_of_person1 = abm_utilities.get_filtered_subjects(Visum.Net.ActivityExecutions, f'[PersonNo] = {minNo}')
    if actEx_of_person1.Count == 0:
        dummy_objs_inserted[2] = True
        actEx_of_person1 = Visum.Net.AddMultiActivityExecutions([(0, minNo, 1)])
    
    return (person1, tours_of_person1, actEx_of_person1, dummy_objs_inserted)


def _create_helper_matrix_if_missing(Visum):
    helper_matrix_no = 99999 # temporary result matrix for matrix formulas
    helper_matrix_exists = helper_matrix_no in VPH.GetMulti(Visum.Net.Matrices, "NO")
    if helper_matrix_exists:
        helper_matrix = Visum.Net.Matrices.ItemByKey(helper_matrix_no)
    else:
        helper_matrix = Visum.Net.AddMatrix(helper_matrix_no, 2, 4)
    return helper_matrix


def _check_filter_definitions(Visum, conf, check_generation_steps = True):
    # test filter definitions in model steps

    person_based_steps = ['PrimLoc', 'TourFreqPrim', 'TourFreqSec'] if check_generation_steps else ['PrimLoc']
    for person_based_step in person_based_steps:
        for step in conf.choice_models[person_based_step]:
            try:
                abm_utilities.get_filtered_subjects(Visum.Net.Persons, step['Filter'])
            except:
                logging.warning("Filter '%s' for Persons in '%s' is corrupt", step['Filter'], person_based_step)

    if check_generation_steps:
        for tour_based_step in ['StopFreqInb', 'StopFreqOutb', 'SubtourFreq']:
            for step in conf.choice_models[tour_based_step]:
                try:
                    abm_utilities.get_filtered_subjects(Visum.Net.Tours, step['Filter'])
                except:
                    logging.warning("Filter '%s' for Tours in '%s' is corrupt", step['Filter'],tour_based_step)

    actEx_based_steps = ['ActDur', 'ActType', 'StartTime', 'DestMajor', 'DestMinor', 'ModeMajor', 'ModeMinor'] if check_generation_steps else [
        'DestMajor', 'DestMinor', 'ModeMajor', 'ModeMinor']
    for actEx_based_step in actEx_based_steps:
        for step in conf.choice_models[actEx_based_step]:
            try:
                abm_utilities.get_filtered_subjects(Visum.Net.ActivityExecutions, step['Filter'])
            except:
                logging.warning("Filter '%s' for Activity Executions in '%s' is corrupt", step['Filter'], actEx_based_step)


def check_specifications(Visum, conf, time_interval_start_times : list, time_interval_end_times : list, check_generation_steps = True):
    _check_time_intervals(time_interval_start_times, time_interval_end_times)

    person1, tours_of_person1, actEx_of_person1, dummy_objs_inserted = _create_dummy_objects_if_missing(Visum)
    helper_matrix = _create_helper_matrix_if_missing(Visum)

    _check_filter_definitions(Visum, conf, check_generation_steps)
    
    # check tables based on persons
    _check_attr_and_matrix_expr(helper_matrix, person1, conf.load_choice_para('PrimLoc'),      check_matrix_expr=True)
    if check_generation_steps :
        _check_attr_and_matrix_expr(helper_matrix, person1, conf.load_choice_para('TourFreqPrim'), check_matrix_expr=False)
        _check_attr_and_matrix_expr(helper_matrix, person1, conf.load_choice_para('TourFreqSec'),  check_matrix_expr=False)

        # check tables based on tours
        _check_attr_and_matrix_expr(helper_matrix, tours_of_person1, conf.load_choice_para('StopFreqInb'),  check_matrix_expr=False)
        _check_attr_and_matrix_expr(helper_matrix, tours_of_person1, conf.load_choice_para('StopFreqOutb'), check_matrix_expr=False)
        _check_attr_and_matrix_expr(helper_matrix, tours_of_person1, conf.load_choice_para('SubtourFreq'),  check_matrix_expr=False)

        # check tables based on activity executions
        _check_attr_and_matrix_expr(helper_matrix, actEx_of_person1, conf.load_act_dur_para('ActDur'),        check_matrix_expr=False)
        _check_attr_and_matrix_expr(helper_matrix, actEx_of_person1, conf.load_choice_para('ActType'),        check_matrix_expr=False)
        _check_attr_and_matrix_expr(helper_matrix, actEx_of_person1, conf.load_start_times_para('StartTime'), check_matrix_expr=False)

    _check_attr_and_matrix_expr(helper_matrix, actEx_of_person1, conf.load_choice_para('DestMajor'),      check_matrix_expr=True)
    _check_attr_and_matrix_expr(helper_matrix, actEx_of_person1, conf.load_choice_para('ModeMajor'),      check_matrix_expr=True)

    _check_attr_and_matrix_expr(helper_matrix, actEx_of_person1, conf.load_choice_para('DestAndModeMinor', 'Specification'),  check_matrix_expr=True)
    _check_attr_and_matrix_expr(helper_matrix, actEx_of_person1, conf.load_choice_para('DestAndModeMinor', 'Specification2'), check_matrix_expr=True)

    _check_mode_impedance_definition(Visum, conf.load_choice_para('ModeMajor'))
    
    # dummy_objs_inserted = [schedule inserted, tour inserted, acivity execution inserted]
    if dummy_objs_inserted[2]:
        actEx_of_person1.RemoveAll()
    if dummy_objs_inserted[1]:
        tours_of_person1.RemoveAll()
    if dummy_objs_inserted[0]:
        person1.Iterator.Item.Schedules.RemoveAll()
