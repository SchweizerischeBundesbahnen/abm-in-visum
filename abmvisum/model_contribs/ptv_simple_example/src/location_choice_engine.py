import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

import dask.array as da

import src.config

from src import abm_utilities, visum_utilities, choice_engine, mode_choice_engine, debug_helper
from settings import abm_settings

dask_factory = choice_engine.DaskFactory(abm_settings.chunk_size_zones)


def calc_dest_zone_utility_matrix_for_long_term_choice(obj_values, origin_zones, utility_matrices):
    assert obj_values.shape[0] == len(origin_zones)           # number of subjects
    assert obj_values.shape[1] == utility_matrices.shape[0]   # number of terms

    # if all subjects' attrExprs are 1, utility depends only on origin and dest zone
    # so memory can be saved by only calculating a matrix of dimension (#zones x #zones) instead of (#subjects x #zones) in this case
    # could be further optimized: first calculate all terms with AttrExpr == 1 based on zones and then only calculate
    # the remaining terms with AttrExpr != 1 based on subjects
    calculate_zone_based = np.all(obj_values == 1)

    if calculate_zone_based:
        utility_matrix = dask_factory.from_array((utility_matrices).sum(axis=0))  
    else:
        #  we need to use dask here to reduce memory usage
        # as we are dealing with matrices of size (num_subjects x num_zones x num_terms)
        utility_matrices = da.from_array(utility_matrices, chunks=('auto', abm_settings.chunk_size_zones, 'auto'))
        origin_zones = dask_factory.from_array(origin_zones)
        utility_per_destZone = utility_matrices[:, origin_zones]

        #assert utility_per_destZone.shape[0] == obj_values.shape[0] # number of subjects
        #assert utility_per_destZone.shape[1] == utility_matrices.shape[0] == utility_matrices.shape[1] # number of zones

        obj_values = da.from_array(obj_values.T[:, :, np.newaxis], chunks=('auto', abm_settings.chunk_size_zones, 'auto'))
        utility_matrix = (obj_values * utility_per_destZone).sum(axis=0)
    
    # debug_helper.warn_NaN_in_dask_matrix(utility_matrix)

    return utility_matrix, calculate_zone_based


def calc_dest_zone_probs_from_utility_matrix(utility_matrix, attraction_per_dest_zone):
    """
    Build probability matrix for subjects of current segment.

    Computes ``probs = exp_util / sum_exp_util_dest_zones`` for positive utility sums (else uniform distribution),
    where ``exp_util = exp(utility_matrix) * attraction_per_dest_zone``.

    Parameters:
    - `utility_matrix` -- Utility matrix. ``da.array(shape=(num_subjects, num_dest_zones))``
    - `attraction_per_dest_zone` -- Attraction per dest zone. ``array type shape=(num_dest_zones,)``

    Returns `da.array(shape=(num_subjects, num_dest_zones))`.
    """

    #debug_helper.warn_NaN_in_dask_matrix(utility_matrix)
    exp_util = da.exp(utility_matrix)
    #debug_helper.warn_NaN_in_dask_matrix(exp_util)
    exp_util = da.multiply(exp_util, attraction_per_dest_zone)
    #debug_helper.warn_NaN_in_dask_matrix(exp_util)

    exp_util_sum = exp_util.sum(axis=1)
    exp_util_sum_T = da.asarray(exp_util_sum[:, np.newaxis])

    # dest zones with attraction 0 must not be chosen (they may not even have a location for the current activity)
    num_dest_zones = len(attraction_per_dest_zone)
    num_available_dest_zones = np.count_nonzero(attraction_per_dest_zone)
    if num_available_dest_zones == num_dest_zones:
        uniform_dist = np.full(shape=(num_available_dest_zones), fill_value=(1 / num_available_dest_zones))
    else:
        uniform_dist = [1 / num_available_dest_zones if attraction_per_dest_zone[zone_index] != 0
                        else 0 for zone_index in range(num_dest_zones)]
        # to find persons with uniform distribution choices, uncomment this and set a breakpoint
        # if np.count_nonzero(exp_util_sum_T) != len(exp_util_sum_T):
        #     index_0 = np.where(exp_util_sum_T == 0)

    # for exp_util_sum_T == 0, set uniform distribution (which excludes dest zones with attraction = 0)
    probs = da.where(exp_util_sum_T > 0, exp_util / exp_util_sum_T, uniform_dist)
    probs = da.maximum(probs, 0)
    #debug_helper.warn_NaN_in_dask_matrix(probs)

    #assert np.allclose(np.sum(probs, axis=1), 1.0)
    return probs


def calc_dest_zone_probs_for_long_term_choice(filtered_subjects, origin_zones, segment, attraction_per_dest_zone, Visum):
    matrix_expr = segment['MatrixExpr']
    attr_expr = segment['AttrExpr']
        
    obj_values = abm_utilities.eval_attrexpr(filtered_subjects, attr_expr)
    utility_matrices = abm_utilities.eval_matexprs(Visum, matrix_expr)

    utility_matrix, is_utility_matrix_still_zone_based = calc_dest_zone_utility_matrix_for_long_term_choice(obj_values, origin_zones,
                                                                                                            utility_matrices)
    probs = calc_dest_zone_probs_from_utility_matrix(utility_matrix, attraction_per_dest_zone)
    return probs, is_utility_matrix_still_zone_based


def calc_dest_zone_probs_with_logsums(act_ex_parameters,
                                      config,
                                      time_interval_start_times,
                                      time_interval_end_times):

    mode_choice_log_sums, mode_choice_log_sums_for_rubberbanding = calc_mode_choice_log_sums(
        act_ex_parameters, config, time_interval_start_times, time_interval_end_times)

    matrix_expr = act_ex_parameters.segment['MatrixExpr']
    attr_expr = act_ex_parameters.segment['AttrExpr']
    mode_choice_table = act_ex_parameters.segment['ModeChoiceTable']

    obj_values = abm_utilities.eval_attrexpr(act_ex_parameters.act_ex_set, attr_expr)

    utility_matices_per_term, utility_matrices_per_term_for_rubberbanding = abm_utilities.evaluate_matrix_expr_with_context_time_for_location_choice(
            act_ex_parameters, config, time_interval_start_times, time_interval_end_times, matrix_exprs_list=[matrix_expr])
    utility_matices_per_term = utility_matices_per_term[0]
    utility_matrices_per_term_for_rubberbanding = utility_matrices_per_term_for_rubberbanding[0]

    num_subjects = len(act_ex_parameters.origin_zones)
    num_terms = len(matrix_expr)
    num_zones = utility_matices_per_term.shape[2]

    assert obj_values.shape == (num_subjects, num_terms)

    utility_per_zone = []
    utility_per_zone_for_rubberbanding = []
    for term_index in range(num_terms):
        term_utility_per_destZone = utility_matices_per_term[term_index]
        term_utility_per_destZone_for_rubberbanding = utility_matrices_per_term_for_rubberbanding[term_index]
        if mode_choice_table[term_index] != "":    
            term_utility_per_destZone *= mode_choice_log_sums[mode_choice_table[term_index]]
            term_utility_per_destZone_for_rubberbanding *= mode_choice_log_sums_for_rubberbanding[mode_choice_table[term_index]]
        utility_per_zone.append(term_utility_per_destZone)
        utility_per_zone_for_rubberbanding.append(term_utility_per_destZone_for_rubberbanding)

    utility_per_zone = da.stack(utility_per_zone).compute()
    utility_per_zone_for_rubberbanding = da.stack(utility_per_zone_for_rubberbanding).compute()

    assert utility_per_zone.shape == (num_terms, num_subjects, num_zones)

    # assert not np.any(np.isnan(utility_per_zone))
    # assert not np.any(np.isinf(utility_per_zone))
    # assert not np.any(np.isnan(obj_values))
    # assert not np.any(np.isinf(obj_values))

    obj_values = da.from_array(obj_values.T[:, :, np.newaxis], chunks=('auto', abm_settings.chunk_size_zones, 'auto')).compute()
    utility_matrix = (obj_values * (utility_per_zone + utility_per_zone_for_rubberbanding)).sum(axis=0)
    
    # debug_helper.warn_NaN_in_dask_matrix(utility_matrix)

    probs = calc_dest_zone_probs_from_utility_matrix(utility_matrix, act_ex_parameters.attraction_per_zone)
    return probs


def calc_mode_choice_log_sums(act_ex_parameters, config, time_interval_start_times, time_interval_end_times) -> tuple[dict[str, np.ndarray]]:
    """
    Calculate mode utility log sums for all mode choice tables of segment.

    Returns:
    - `mode_choice_log_sums` -- Mode utility log sums for each mode choice table of segment. 
      - Type: ``dict[str: np.array(shape=(num_subjects, num_dest_zones))]``
    - `mode_choice_log_sums_for_rubberbanding` -- Mode utility log sums for rubberbanding, for each mode choice table of segment. 
      - Type: ``dict[str: np.array(shape=(num_subjects, num_dest_zones))]``
    """
    used_mode_choice_tables = set(act_ex_parameters.segment['ModeChoiceTable'])
    mode_choice_log_sums = dict()
    mode_choice_log_sums_for_rubberbanding = dict()
    mode_id_to_interchangeable = dict(config.Visum.Net.Modes.GetMultipleAttributes(["ID", "Interchangeable"]))

    for mode_choice_table_name in used_mode_choice_tables:
        if len(mode_choice_table_name) == 0:
            continue

        mode_choice_segments = config.load_mode_choice_table(mode_choice_table_name)
        mode_matrix_expr = np.array(mode_choice_segments['MatrixExpr'], dtype=object)
        destination_impedance = np.array(mode_choice_segments['DestinationImpedance'], dtype=object)
        mode_attr_expr = np.array(mode_choice_segments['AttrExpr'], dtype=object)
        mode_betas = mode_choice_segments['Beta']  #shape = (num_terms, num_modes)

        usable_mode_inds = None
        if act_ex_parameters.mode_id is not None:
            # remove all terms which are irrelevant for mode_id (and interchangable modes, if allowed)
            usable_mode_inds = compute_usable_mode_inds(act_ex_parameters.mode_id,
                                                        mode_id_to_interchangeable,
                                                        mode_choice_segments['Choices'],
                                                        act_ex_parameters.allow_interchangable_modes)
            relevant_terms = get_relevant_terms(mode_betas, usable_mode_inds)

            mode_matrix_expr = mode_matrix_expr[relevant_terms]
            destination_impedance = destination_impedance[relevant_terms]
            mode_attr_expr = mode_attr_expr[relevant_terms]
            mode_betas = mode_betas[relevant_terms] # remove irrelevant terms
            mode_betas = mode_betas[:, usable_mode_inds] # remove columns of not usable modes

        obj_values_mode = abm_utilities.eval_attrexpr(act_ex_parameters.act_ex_set, mode_attr_expr)

        matrix_expr_per_term_and_list, matrix_expr_per_term_and_list_for_rubberbanding = abm_utilities.evaluate_matrix_expr_with_context_time_for_location_choice(
            act_ex_parameters, config, time_interval_start_times, time_interval_end_times, [mode_matrix_expr, destination_impedance])

        # assert not np.any(np.isnan(matrix_expr_per_term_and_list))
        # assert not np.any(np.isnan(matrix_expr_per_term_and_list_for_rubberbanding))
        # assert not np.any(np.isposinf(matrix_expr_per_term_and_list))
        # assert not np.any(np.isposinf(matrix_expr_per_term_and_list_for_rubberbanding))
        # assert not np.any(np.isneginf(matrix_expr_per_term_and_list)) # no mode is usable - currently not treated as error
        # assert not np.any(np.isneginf(matrix_expr_per_term_and_list_for_rubberbanding)) # no mode is usable - currently not treated as error

        assert len(matrix_expr_per_term_and_list) == len(matrix_expr_per_term_and_list_for_rubberbanding) == 2
        mode_matrix_expr_eval, dest_imp_eval = matrix_expr_per_term_and_list
        mode_matrix_expr_rubberbanding_eval, dest_imp_rubberbanding_eval = matrix_expr_per_term_and_list_for_rubberbanding

        mode_choice_log_sums[mode_choice_table_name] = mode_choice_engine.calc_mode_utility_logsum_per_dest_zone(
            act_ex_parameters.origin_zones, obj_values_mode, mode_matrix_expr_eval,
            dest_imp_eval, mode_betas)

        mode_choice_log_sums_for_rubberbanding[mode_choice_table_name] = mode_choice_engine.calc_mode_utility_logsum_per_dest_zone(
            act_ex_parameters.target_zones, obj_values_mode, mode_matrix_expr_rubberbanding_eval,
            dest_imp_rubberbanding_eval, mode_betas)

        # assert not np.any(np.isnan(mode_choice_log_sums[mode_choice_table_name]))
        # assert not np.any(np.isnan(mode_choice_log_sums_for_rubberbanding[mode_choice_table_name]))
        # assert not np.any(np.isposinf(mode_choice_log_sums[mode_choice_table_name]))
        # assert not np.any(np.isneginf(mode_choice_log_sums[mode_choice_table_name]))  # no mode is usable - currently not treated as error

    return mode_choice_log_sums, mode_choice_log_sums_for_rubberbanding


def get_relevant_terms(mode_betas : np.ndarray, usable_mode_inds : list[int]) -> np.ndarray:
    """
    Get only relevant terms, where beta is != 0 for any usable mode.

    Returns `relevant_terms` as 1-dimensional np.array(dtype=bool, size=mode_betas.shape(0))
    """
    relevant_terms = np.any(mode_betas[:, usable_mode_inds], axis=1)
    assert np.any(relevant_terms) # some terms must be != 0
    return relevant_terms


def compute_usable_mode_inds(mode_id, mode_id_to_interchangeable, chooseable_mode_ids, allow_interchangable_modes : bool) -> list[int]:
    """
    Computes usable mode indices.

    * If `allow_interchangable_modes` is True, all interchangeable modes may be used, plus the given mode.
    * If `allow_interchangable_modes` is False, only the given mode may be used.

    Returns `list[int]` of usable mode indices.
    """
    modeId_to_modeInd = dict(zip(chooseable_mode_ids, range(len(chooseable_mode_ids))))
    if allow_interchangable_modes:
        usable_mode_inds = []
        for chooseable_mode_id in chooseable_mode_ids:
            if mode_id_to_interchangeable[chooseable_mode_id]:
                usable_mode_inds.append(modeId_to_modeInd[chooseable_mode_id])
        if mode_id_to_interchangeable[mode_id] == 0:
            usable_mode_inds.append(modeId_to_modeInd[mode_id])
    else:
        usable_mode_inds = [modeId_to_modeInd[mode_id]]
    return usable_mode_inds


def choose_dest_locations(location_probs_inside_zone, chosen_zone_indices, zoneIndex_to_locationIndices, num_zones):
    logging.info('choose_dest_locations')

    unique_zone_indices, person_count_per_zone = np.unique(chosen_zone_indices, return_counts=True)
    person_count_per_zone_index = zip(unique_zone_indices, person_count_per_zone)

    chosen_location_indices_per_zone = dict()
    for zone_index, person_count in person_count_per_zone_index:
        # location indices associated with zone
        location_indices_for_zone = np.array(zoneIndex_to_locationIndices[zone_index], dtype=int)
        probs_for_one_choice = location_probs_inside_zone[zone_index]
        # long term choice is independent of person attributes
        prob = np.full(shape=(person_count, len(probs_for_one_choice)), fill_value=probs_for_one_choice)
        choices_parallel = choice_engine.choose2D_parallel(prob, chunk_size=abm_settings.chunk_size_locations)
        # convert dask array to np array if applicable
        chosen_indices = np.array(choices_parallel, dtype = int)
        # convert chosen indices back to location indices
        chosen_location_indices_per_zone[zone_index] = location_indices_for_zone[chosen_indices]

    choice_index_per_zone = np.zeros((num_zones,), dtype=int)
    chosen_location_indices = []
    for chosen_zone_index in chosen_zone_indices:
        chosen_location_indices.append(chosen_location_indices_per_zone[chosen_zone_index][choice_index_per_zone[chosen_zone_index]])
        choice_index_per_zone[chosen_zone_index] += 1
    logging.info('choose_dest_locations completed')

    return np.array(chosen_location_indices, dtype=int)

@dataclass
class Location_Choice_Parameter_Global:
    config: src.config.Config
    zones: np.array                 # array with all zone numbers
    locations: np.array             # array with all location numbers
    zoneNo_to_zoneInd: dict
    locationNo_to_locationInd: dict
    time_interval_start_times: np.array
    time_interval_end_times: np.array


@dataclass 
class Location_Choice_Parameter_Act_Ex_Data:
    act_ex_set: Any # com object Filtered Set

    # origin -> ... -> act_ex -> ... -> target
    # orgin -> act_ex and act_ex -> target are used for impedance
    # used ti for Context[FromTime] and CONTEXT[ToTime] are time_interval_indices_origin_path and time_interval_indices_target_path

    # for each element in act_ex_set one value 
    origin_zones: np.array              
    time_interval_indices_origin_path: list
    target_zones: np.array
    time_interval_indices_target_path: list

    segment: dict
    act_code: str                   # all act ex in act_ex_set should have this acitivty
    attraction_per_zone: np.array   # for each element in act_ex_set one value 

    mode_id: int = None
    allow_interchangable_modes: bool = False


def choose_and_set_locations(global_parameters,
                             act_ex_parameters,
                             chosen_zones_result_attr=None):
    """
    Choose locations in two steps: 
    1. Choose zones based on DestChoice parameters as well as mode utility log sums and location attractions.
    2. Choose locations within zones based on location attractions.

    Chosen locations are directly saved into ``LocationNo`` of activity executions.

    If `chosen_zones_result_attr` is provided, chosen zone nos are saved into that attribute.
    """
                    
    probs = calc_dest_zone_probs_with_logsums(act_ex_parameters,
                                              global_parameters.config,
                                              global_parameters.time_interval_start_times,
                                              global_parameters.time_interval_end_times)

    chosen_zoneInds = choose_dest_zones(probs, None)

    location_probs_inside_zone, zoneIndex_to_locationIndices = abm_utilities.build_location_probs_inside_zones(
    global_parameters.config.Visum, act_ex_parameters.act_code, global_parameters.zoneNo_to_zoneInd, global_parameters.locationNo_to_locationInd)
    chosen_locationInds = choose_dest_locations(
        location_probs_inside_zone, chosen_zoneInds, zoneIndex_to_locationIndices, global_parameters.zones.shape[0])

    chosen_zoneNos = global_parameters.zones[chosen_zoneInds]
    if chosen_zones_result_attr is not None and len(chosen_zones_result_attr) > 0:
        # sets major activity zone no at tour via indirect attribute
        # Tour\MajorActivityZoneNo is used for destination choice in subtours for rubberbanding
        visum_utilities.SetMulti(act_ex_parameters.act_ex_set, chosen_zones_result_attr, chosen_zoneNos, chunk_size=abm_settings.chunk_size_trips) 
    
    chosen_locationNos = global_parameters.locations[chosen_locationInds]
    visum_utilities.SetMulti(act_ex_parameters.act_ex_set, "LocationNo", chosen_locationNos, chunk_size=abm_settings.chunk_size_trips)

    logging.info('location set for %d activity executions', act_ex_parameters.act_ex_set.Count)


def choose_dest_zones(probs, originZonePerObj = None):
    logging.info('choose_dest_zones')

    if originZonePerObj is None:
        adapted_probs = probs
    else:
        originZonePerObj = da.from_array(originZonePerObj, chunks=abm_settings.chunk_size_zones)
        adapted_probs = probs[originZonePerObj]
        
    num_choices = adapted_probs.shape[0]
    choice2d = choice_engine.Choice2DParallel(num_choices, chunk_size=abm_settings.chunk_size_zones)
    choice2d.add_prob(adapted_probs)
    
    chosen_zone_indices = choice2d.choose()
    logging.info('choose_dest_zones completed')

    return chosen_zone_indices


    
