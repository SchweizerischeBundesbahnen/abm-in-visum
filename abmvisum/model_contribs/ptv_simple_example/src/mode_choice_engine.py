import collections
import logging

import numpy as np
import dask.array as da

import VisumPy.helpers as VPH

from src import abm_utilities, visum_utilities, choice_engine
from settings import abm_settings

def compute_node_stoparea_distances(Visum, tsys, max_walk_time_in_minutes):
    """
    Computes for every node the stop areas within distance max_walk_time_in_minutes 
    using links / turns open for the specified tsys.
    We do this by doing shortest path searches starting at the access node of stop areas.
    
    Note: we do only forward shortest path searches from the stop areas,
    i.e. we find the distance from the stop area to the node.
    We do not do backward shortest path searches, but we assume that distances are symmetric.
    """

    # set search_criterion:
    #   t0 = 0, tCur = 1, impedance = 2, distance = 3, addVal1 = 4, addVal2 = 5, addVal3 = 6,
    search_criterion = 0

    # build map nodeNo -> reachable stop areas (stopAreaNo, distance)
    node_to_stop_area_distances = collections.defaultdict(list)
    node_to_stop_area_indices = collections.defaultdict(list)

    max_walk_time_in_seconds = max_walk_time_in_minutes * 60 # internal unit for t0 is seconds

    # define filter for reachable nodes
    # NOTE: FilteredBy is re-evaluated on each usage (lazy evaluation)
    reachable_nodes_filter = Visum.Net.Nodes.FilteredBy(f'[IsocTimePrT] <= {max_walk_time_in_seconds}')

    stop_area_access_node_nos = VPH.GetMulti(Visum.Net.StopAreas, 'NodeNo')
    for stop_area_index, stop_area_access_node_no in enumerate(stop_area_access_node_nos):
        if stop_area_access_node_no == 0:
            continue

        # run isochrones calculation from stop area access node
        isoc_source_nodes = Visum.CreateNetElements()
        isoc_source_nodes.Add(Visum.Net.Nodes.ItemByKey(stop_area_access_node_no))
        Visum.Analysis.Isochrones.ExecutePrT(isoc_source_nodes, tsys, search_criterion, max_walk_time_in_seconds)

        # get reachable node values
        # NOTE: reachable_nodes_filter is re-evaluated each time
        reachable_nodes = reachable_nodes_filter.GetMultipleAttributes(['No', 'IsocTimePrT'])
        
        # put reachable nodes into map node -> reachable stop areas
        for isoc_dest_node, isoc_val in reachable_nodes:
            isoc_dest_node_int = int(isoc_dest_node)
            node_to_stop_area_distances[isoc_dest_node_int].append(int(isoc_val))
            node_to_stop_area_indices[isoc_dest_node_int].append(stop_area_index)

    return node_to_stop_area_indices, node_to_stop_area_distances


def get_stop_area_matrices(Visum, dseg_code, matrix_code, time_interval_start_times, time_interval_end_times):
    """
    Get stop area skim matrices for specified DSeg, matrix code and analysis time intervals
    """
    stop_area_matrices = {}
    for time_interval_index, (start_time, end_time)  in enumerate(zip(time_interval_start_times, time_interval_end_times)):
        matrix_ref = {
            "MATRIXTYPE": 4, # skim matrix
            "OBJECTTYPEREF": 4, # stop area
            "DSEGCODE": dseg_code,
            "CODE": matrix_code,
            "FROMTIME": start_time,
            "TOTIME": end_time }
        stop_area_matrix = VPH.GetMatrixRaw(Visum, matrix_ref)
        stop_area_matrices[time_interval_start_times[time_interval_index]] = stop_area_matrix
    return stop_area_matrices


def compute_put_impedance_major_mode_choice_for_one_direction(Visum,
                                                              node_to_stop_area_indices,
                                                              node_to_stop_area_distances,
                                                              time_interval_start_times,
                                                              time_interval_end_times,
                                                              dseg_code,
                                                              walking_time_impedance_factor,
                                                              major_act_ex_segment_filtered,
                                                              is_outbound_direction):

    """
    Compute put impedance for major activity executions.
    
    We do this by determining stop areas close to the from-node and to-node
    of the trip. The put impedance is then simply the minimum of [the (time-dependent)
    impedance between the from- and to-stop-area plus the impedance
    from-node -> from-stop-area plus the impedance to-stop-area -> to-node].
    
    NOTE: The unit of the stop area impedance matrix is minutes. For the
    walk to the stop area we assume a speed of 4 km/h. the walking time
    contributes to the impedance with a factor of 1.
    """

    # obtain trip times from activity execution start/end (outbound/inbound)
    trip_times = visum_utilities.GetMulti(
        major_act_ex_segment_filtered, 'StartTime' if is_outbound_direction else 'EndTime', chunk_size=abm_settings.chunk_size_trips)
    trip_time_interval_start_times = [abm_utilities.get_time_interval_start_time(
        time_point, time_interval_start_times, time_interval_end_times) for time_point in trip_times]

    # get from- and to-nodes of activity executions
    # - from node: nearest nodes of person's home
    # - to node: nearest node of location where activity is executed
    # (the other way around for inbound direction)
    from_node_nos, to_node_nos = get_from_and_to_node_nos_of_major_activity_executions(major_act_ex_segment_filtered)
    if not is_outbound_direction:
        from_node_nos, to_node_nos = to_node_nos, from_node_nos

    put_impedance = compute_put_impedance(Visum, node_to_stop_area_indices, node_to_stop_area_distances, time_interval_start_times, time_interval_end_times,
                                          dseg_code, walking_time_impedance_factor, trip_time_interval_start_times, from_node_nos, to_node_nos)

    return put_impedance


def compute_put_impedance(Visum, node_to_stop_area_indices,
                          node_to_stop_area_distances, time_interval_start_times, time_interval_end_times, dseg_code,
                          walking_time_impedance_factor, trip_time_interval_start_times,
                          from_node_nos, to_node_nos, compute_travel_times = False):
    """
    Compute PuT impedance (and optionally travel times) of shortest path consisting of:
    *  origin node -> origin stop area (via node_to_stop_area_distances)
    *  origin stop area -> destination stop area (via stop area matrices (matrix code: "IPD"))
    *  destination stop area -> destination node (via node_to_stop_area_distances)
    """
    # get time-dependent stop area matrices
    stop_area_impedance_matrices = get_stop_area_matrices(Visum, dseg_code, "IPD", time_interval_start_times, time_interval_end_times)
    stop_area_ride_time_matrices = get_stop_area_matrices(Visum, dseg_code, "RIT", time_interval_start_times, time_interval_end_times) if compute_travel_times else None
    stop_area_adaption_time_matrices = get_stop_area_matrices(Visum, dseg_code, "ADT", time_interval_start_times, time_interval_end_times) if compute_travel_times else None
    
    put_impedance = []
    travel_times = []
    # loop over trips
    for trip_time_interval_start_time, from_node_no, to_node_no in zip(trip_time_interval_start_times, from_node_nos, to_node_nos):
        # check whether there are stop areas close to the from- and to-node
        if (not from_node_no in node_to_stop_area_distances) or (not to_node_no in node_to_stop_area_distances):
            put_impedance.append(999999)
            if compute_travel_times:
                travel_times.append(999999)
            continue
        
        # get indices and distances of stop areas close to from- and to-node
        from_node_to_stop_area_dists   = np.array(node_to_stop_area_distances[from_node_no], dtype=np.float64)
        to_node_to_stop_area_dists     = np.array(node_to_stop_area_distances[to_node_no], dtype=np.float64)
        from_node_to_stop_area_indices = node_to_stop_area_indices[from_node_no]
        to_node_to_stop_area_indices   = node_to_stop_area_indices[to_node_no]

        # get stop area matrix corresponding to time of trip
        stop_area_matrix = stop_area_impedance_matrices[trip_time_interval_start_time]

        # get part of stop area matrix belonging to stop areas close to from- and to-node
        reachable_stop_area_matrix = stop_area_matrix[from_node_to_stop_area_indices,:][:,to_node_to_stop_area_indices]

        # put impedance matrix is sum of cost from-node -> from-stop-area, from-stop-area -> to-stop-area, to-stop-area -> to-node
        # note: stop area matrix is in minutes,
        #       from_node_distance_matrix and to_node_distance_matrix are in seconds
        put_impedance_matrix = reachable_stop_area_matrix + \
            (from_node_to_stop_area_dists[:,np.newaxis] + to_node_to_stop_area_dists) * ((1/60.0) * walking_time_impedance_factor)

        # the actual impedance is the minimum of the matrix
        index_of_minimum_impedance = np.unravel_index(np.argmin(put_impedance_matrix, axis=None), put_impedance_matrix.shape)

        min_from_stop_area_index = index_of_minimum_impedance[0]
        min_to_stop_area_index = index_of_minimum_impedance[1]

        cur_put_impedance = put_impedance_matrix[index_of_minimum_impedance]
        put_impedance.append(cur_put_impedance)

        if compute_travel_times:
            assert stop_area_ride_time_matrices is not None
            ride_time_matrix = stop_area_ride_time_matrices[trip_time_interval_start_time]
            adaption_time_matrix = stop_area_adaption_time_matrices[trip_time_interval_start_time]
            stop_area_ride_and_adaption_time = ride_time_matrix[from_node_to_stop_area_indices[min_from_stop_area_index],
                                              to_node_to_stop_area_indices[min_to_stop_area_index]] + \
                adaption_time_matrix[from_node_to_stop_area_indices[min_from_stop_area_index],
                                     to_node_to_stop_area_indices[min_to_stop_area_index]]

            # ride_time_matrix is in minutes
            cur_travel_time = 60 * stop_area_ride_and_adaption_time + \
                from_node_to_stop_area_dists[min_from_stop_area_index] + \
                to_node_to_stop_area_dists[min_to_stop_area_index]
            travel_times.append(cur_travel_time)
        
    if compute_travel_times:
        return (put_impedance, travel_times)
    else:
        return put_impedance

def compute_impedance_for_both_directions(compute_impedance_for_one_direction_fun):
    impedance_inbound = compute_impedance_for_one_direction_fun(False)
    impedance_outbound = compute_impedance_for_one_direction_fun(True)
    return [impedance_inbound, impedance_outbound]

def compute_put_impedance_for_major_mode_choice(Visum,
                                                node_to_stop_area_indices,
                                                node_to_stop_area_distances,
                                                time_interval_start_times,
                                                time_interval_end_times,
                                                dseg_code,
                                                walking_time_impedance_factor,
                                                major_act_ex_segment_filtered):
    compute_impedance_for_one_direction = lambda is_outbound_direction : compute_put_impedance_major_mode_choice_for_one_direction(Visum,
                                                                                                                                   node_to_stop_area_indices,
                                                                                                                                   node_to_stop_area_distances,
                                                                                                                                   time_interval_start_times,
                                                                                                                                   time_interval_end_times,
                                                                                                                                   dseg_code,
                                                                                                                                   walking_time_impedance_factor,
                                                                                                                                   major_act_ex_segment_filtered,
                                                                                                                                   is_outbound_direction)
    impedance = compute_impedance_for_both_directions(compute_impedance_for_one_direction)
    return impedance


def get_from_and_to_node_nos_of_major_activity_executions(major_act_ex_segment_filtered):
    from_node_nos = visum_utilities.GetMulti(major_act_ex_segment_filtered,
                                           r"Schedule\Person\Household\Residence\Location\NearestActiveNode\No", chunk_size=abm_settings.chunk_size_trips, reindex = True)
    to_node_nos = visum_utilities.GetMulti(major_act_ex_segment_filtered,
                                         r"Location\NearestActiveNode\No", chunk_size=abm_settings.chunk_size_trips)
    return from_node_nos, to_node_nos

def get_from_and_to_nodes_of_main_activity_executions(major_act_ex_segment_filtered, nodeno_to_node):
    from_node_nos, to_node_nos = get_from_and_to_node_nos_of_major_activity_executions(major_act_ex_segment_filtered)
    from_nodes = [nodeno_to_node[node_no] for node_no in from_node_nos]
    to_nodes   = [nodeno_to_node[node_no] for node_no in to_node_nos]
    return from_nodes, to_nodes

def compute_prt_impedance_minor_mode_choice(prt_shortest_path_searcher,
                                            from_nodes, 
                                            to_nodes, 
                                            trip_ti_codes,
                                            prt_impedance_attribute,
                                            prtsys,
                                            compute_travel_times = False):
    # the impedance definition is a link/turn/main turn/connector/restricted traffic area attribute.
    # if the attribute ends with (), then it is time-dependent and we expect that it has analysis time intervals as subattribute.
    prt_impedance_attribute_is_time_dependent = prt_impedance_attribute.endswith('()')
    if prt_impedance_attribute_is_time_dependent:
        prt_impedance_attribute = prt_impedance_attribute[:-2].strip() # remove () at end
    
    # build search requests
    tsyss = [prtsys] * len(from_nodes)
    if prt_impedance_attribute_is_time_dependent:
        prt_impedance_attributes = [f'{prt_impedance_attribute}({ti_code})' for ti_code in trip_ti_codes]
    else:
        prt_impedance_attributes = [prt_impedance_attribute] * len(from_nodes)

    # pass search requests to searcher
    search_request_list = list(zip(from_nodes, to_nodes, tsyss, prt_impedance_attributes))
    search_requests = prt_shortest_path_searcher.CreatePrTShortestPathRequests()
    search_requests.AddRequests(search_request_list)

    # search shortest paths    
    if compute_travel_times:
        # additional output attribute values:
        # t0 = 0, tCur = 1, impedance = 2, distance = 3, addVal1 = 4, addVal2 = 5, addVal3 = 6,
        additional_output_attributes = [1]
        search_results = prt_shortest_path_searcher.Search(search_requests, additional_output_attributes)
        trip_costs = np.array(list(search_results.Costs))
        trip_times = search_results.AdditionalAttributeValues
        trip_times = np.array(trip_times[0])
        trip_times[trip_costs >= 999999] = 999999
        trip_times[trip_times > 999999] = 999999
        return (trip_costs, trip_times)
    else:
        search_results = prt_shortest_path_searcher.Search(search_requests, [])
        return list(search_results.Costs)


def compute_prt_impedance_major_mode_choice_for_one_direction(prt_shortest_path_searcher,
                                                              major_act_ex_segment_filtered,
                                                              prt_impedance_attribute,
                                                              prtsys,
                                                              time_interval_start_times,
                                                              time_interval_end_times,
                                                              time_interval_codes,
                                                              nodeno_to_node,
                                                              is_outbound_direction):
    # the impedance definition is a link/turn/main turn/connector/restricted traffic area attribute.
    # if the attribute ends with (), then it is time-dependent and we expect that it has analysis time intervals as subattribute.
    prt_impedance_attribute_is_time_dependent = prt_impedance_attribute.endswith('()')
    if prt_impedance_attribute_is_time_dependent:
        prt_impedance_attribute = prt_impedance_attribute[:-2].strip() # remove () at end

    # determine from-node and to-node for each activity execution
    from_nodes, to_nodes = get_from_and_to_nodes_of_main_activity_executions(major_act_ex_segment_filtered, nodeno_to_node)
    if not is_outbound_direction:
        from_nodes, to_nodes = to_nodes, from_nodes
    
    # build search requests
    tsyss = [prtsys] * len(from_nodes)
    if prt_impedance_attribute_is_time_dependent:
        # get time intervals in which the activity executions start or end
        # we simply use the start time and end time of the activity execution
        trip_start_times = visum_utilities.GetMulti(
            major_act_ex_segment_filtered, 'StartTime' if is_outbound_direction else 'EndTime', chunk_size=abm_settings.chunk_size_trips, reindex = True)
        trip_start_time_intervals = [abm_utilities.get_time_interval_code(time_point, time_interval_start_times, time_interval_end_times, time_interval_codes) for time_point in trip_start_times]
        prt_impedance_attributes = [prt_impedance_attribute + "(" + str(ti_code) + ")" for ti_code in trip_start_time_intervals]
    else:
        prt_impedance_attributes = [prt_impedance_attribute] * len(from_nodes)

    # pass search requests to searcher
    search_request_list = list(zip(from_nodes, to_nodes, tsyss, prt_impedance_attributes))
    search_requests = prt_shortest_path_searcher.CreatePrTShortestPathRequests()
    search_requests.AddRequests(search_request_list)

    # search shortest paths
    search_results = prt_shortest_path_searcher.Search(search_requests, [])
    return list(search_results.Costs)


def compute_prt_impedance_for_major_mode_choice(prt_shortest_path_searcher,
                                                major_act_ex_segment_filtered,
                                                prt_impedance_attribute,
                                                prtsys,
                                                time_interval_start_times,
                                                time_interval_end_times,
                                                time_interval_codes,
                                                nodeno_to_node):
    compute_impedance_for_one_direction = lambda is_outbound_direction : compute_prt_impedance_major_mode_choice_for_one_direction(prt_shortest_path_searcher,
                                                                                                                                   major_act_ex_segment_filtered,
                                                                                                                                   prt_impedance_attribute,
                                                                                                                                   prtsys,
                                                                                                                                   time_interval_start_times,
                                                                                                                                   time_interval_end_times,
                                                                                                                                   time_interval_codes,
                                                                                                                                   nodeno_to_node,
                                                                                                                                   is_outbound_direction)
    impedance = compute_impedance_for_both_directions(compute_impedance_for_one_direction)
    return impedance


def compute_direct_distances_for_major_mode_choice(activity_executions, nodeno_to_longitude_latitude):
    # determine from-node and to-node for each activity execution
    from_node_nos = visum_utilities.GetMulti(
        activity_executions, r"Schedule\Person\Household\Residence\Location\NearestActiveNode\No", chunk_size=abm_settings.chunk_size_trips, reindex = True)
    to_node_nos = visum_utilities.GetMulti(activity_executions, r"Location\NearestActiveNode\No", chunk_size=abm_settings.chunk_size_trips)

    return compute_direct_distances_for_minor_mode_choice(from_node_nos, to_node_nos, nodeno_to_longitude_latitude)


def compute_direct_distances_for_minor_mode_choice(from_node_nos, to_node_nos, nodeno_to_longitude_latitude):

    # determine WGS84 coordinates of from-nodes and to-nodes
    from_node_long_lat = [nodeno_to_longitude_latitude[node_no] for node_no in from_node_nos]
    to_node_long_lat   = [nodeno_to_longitude_latitude[node_no] for node_no in to_node_nos]

    # compute direct distance    
    direct_dist = [abm_utilities.compute_direct_distance(p1[1], p1[0], p2[1], p2[0]) for p1, p2 in zip(from_node_long_lat, to_node_long_lat)]
    return direct_dist


def calc_mode_utility_logsum_per_dest_zone(origin_zones, obj_values, # values per act ex
                                            matrices_mode, destination_impedance_per_term, betas):
    """
    Calculate mode utility log sum for each destination zone.

    Parameters:
    - `obj_values` -- Evaluated ``mode_attr_expr`` per activity execution. Shape=(num_subjects, num_terms)
    - `matrices_mode` -- Evaluated ``mode_matrix_expr`` per activity execution. Shape=(num_terms, num_subjects, num_dest_zones)
    - `destination_impedance_per_term` -- Evaluated ``destination_impedance`` per activity execution. Shape=(num_terms, num_subjects, num_dest_zones)
    - `betas` -- Factors for each mode. Shape=(num_terms, num_modes)

    Returns ``np.array(shape=(num_subjects, num_dest_zones))``, where ``-inf`` is replaced by ``-999999``.
    """
    num_dest_zones = matrices_mode.shape[2]
    num_subjects = len(origin_zones)
    num_terms = obj_values.shape[1]
    num_modes = betas.shape[1]

    assert obj_values.shape                     == (num_subjects, num_terms)
    assert betas.shape                          == (num_terms, num_modes)
    assert matrices_mode.shape                  == (num_terms, num_subjects, num_dest_zones)
    assert destination_impedance_per_term.shape == (num_terms, num_subjects, num_dest_zones)

    obj_values = da.from_array(obj_values.T[:, :, np.newaxis], chunks=('auto', abm_settings.chunk_size_zones, 'auto'))
    matrices_mode = obj_values * matrices_mode

    # re-shape and inflate matrices for matching dimensions
    matrices_mode = da.transpose(matrices_mode, (1, 2, 0))
    destination_impedance_per_term = da.transpose(destination_impedance_per_term, (1, 2, 0))

    assert matrices_mode.shape                  == (num_subjects, num_dest_zones, num_terms)
    assert destination_impedance_per_term.shape == (num_subjects, num_dest_zones, num_terms)

    utility_per_dest_zone_and_term = matrices_mode * destination_impedance_per_term
    assert utility_per_dest_zone_and_term.shape == (num_subjects, num_dest_zones, num_terms)


    utility_per_mode = da.dot(utility_per_dest_zone_and_term, betas)
    assert utility_per_mode.shape == (num_subjects, num_dest_zones, num_modes)

    # NOTE: `log(0)` may lead to "divide by zero" warnings.
    #       These warnings cannot be turned off when using dask. (see: https://github.com/dask/dask/issues/3245)
    #       But when using numpy directly, they may be turned off by the following statement
    # with np.errstate(divide='ignore'):
    utility_logsum = da.log(da.sum(da.exp( utility_per_mode),2))
    # log (exp (-999999)) = -inf, but -inf leads to problems when multiplied with 0, so we set -inf to -999999
    utility_logsum[utility_logsum < -999999] = -999999

    assert utility_logsum.shape == (num_subjects, num_dest_zones)
    # assert not np.any(np.isnan(utility_logsum))
    # assert not np.any(np.isposinf(utility_logsum))
    # assert not np.any(np.isneginf(utility_logsum)) # no mode is usable - currently not treated as error

    return utility_logsum


def calc_mode_utility(utility_per_term : np.ndarray,
                      betas : np.ndarray,
                      mode_is_useable : np.ndarray = None,
                      mode_is_interchangeable : np.ndarray = None,
                      mode_ranks : np.ndarray = None) -> np.ndarray:
    """
    Calculate utility per mode.

    Optional arguments:
    - `mode_is_usable` -- ``np.array(shape=(num_modes), dtype=bool)`` -- 
                       if provided, set utility of unusable modes to ``MINUSINF``
    - `mode_is_interchangeable` -- ``np.array(shape=(num_modes), dtype=bool)`` -- 
                                if provided, compute nest utility and (via logsum) and assign to 
                                interchangeable mode of highest rank (other interchangeable modes get ``MINUSINF``)
    - `mode_ranks` -- ``np.array(shape=(num_modes), dtype=float)`` -- 
                   rank of each mode, must be provided if `mode_is_interchangeable` is provided
    
    Returns `np.array(shape=(num_subjects, num_modes))`.
    """
    num_subjects = utility_per_term.shape[0]
    num_terms = utility_per_term.shape[1]
    num_modes = betas.shape[1]

    # mark mode as unchoosaeable by setting utility to -9999 (-inf would lead to numerical issues)
    MINUSINF_UTILITY = -9999

    assert num_terms == betas.shape[0]

    assert utility_per_term.shape == (num_subjects, num_terms)

    utility_per_mode = np.dot(utility_per_term, betas)
    assert utility_per_mode.shape == (num_subjects, num_modes)

    if mode_is_useable is not None:
        assert len(mode_is_useable) == num_modes
        utility_per_mode[:, ~mode_is_useable] = MINUSINF_UTILITY

    if mode_is_interchangeable is not None:
        assert mode_ranks is not None
        assert len(mode_is_interchangeable) == num_modes == len(mode_ranks)
        mode_ranks = np.array(mode_ranks, dtype=float)
        mode_is_interchangeable = np.array(mode_is_interchangeable, dtype=bool)

        # calc nest utility of interchangeable modes (logsum)
        nest_utility = np.log(np.sum(np.exp(utility_per_mode[:, mode_is_interchangeable]), axis=1))

        # mode ranks for interchangeable modes must be unique
        assert mode_ranks[mode_is_interchangeable].shape == np.unique(mode_ranks[mode_is_interchangeable]).shape
        highest_interchangeable_mode_rank = max(mode_ranks[mode_is_interchangeable])

        # the interchangeable mode with highest rank represents the nest of interchangeable modes and is assigned the whole nest's utility
        # the other interchangeable modes with lower rank are assigned utility of -infinity (-9999)
        utility_per_mode[:, mode_is_interchangeable & (mode_ranks == highest_interchangeable_mode_rank)] = nest_utility[:, np.newaxis]
        utility_per_mode[:, mode_is_interchangeable & (mode_ranks  < highest_interchangeable_mode_rank)] = MINUSINF_UTILITY

    return utility_per_mode


def calc_major_mode_choice(subjects,
                           Visum,
                           segment,
                           inbound_mode_impedances_per_term,
                           outbound_mode_impedances_per_term,
                           origin_zones,
                           dest_zones,
                           time_interval_start_times,
                           time_interval_end_times,
                           inbound_trip_start_time_interval_indices,
                           outbound_trip_start_time_interval_indices,
                           mode_ids,
                           mode_is_interchangeable,
                           mode_ranks) -> np.ndarray:
    """
    Calculate major mode choice based on inbound and outbound mode impedance.

    Returns `np.array(shape=(num_subjects,))` containing chosen ``mode id`` for each subject.
    """

    # attrExpr, matrixExpr and mode_impedances_per_term are multiplied
    matrixExprs = segment['MatrixExpr']
    attrExprs = segment['AttrExpr']
    betas = segment['Beta']

    # attrExpr (for all terms)
    obj_values = abm_utilities.eval_attrexpr(subjects, attrExprs)
    assert obj_values.shape == (len(subjects), len(attrExprs))

    # mode_impedances_per_term (for all terms)
    inbound_mode_impedances_per_term  = np.column_stack(inbound_mode_impedances_per_term)
    outbound_mode_impedances_per_term = np.column_stack(outbound_mode_impedances_per_term)
    assert inbound_mode_impedances_per_term.shape == obj_values.shape
    assert outbound_mode_impedances_per_term.shape == obj_values.shape

    # in PuT 999999 usually means no path available, so we limit impedance from shortest path searches to 999999
    inbound_mode_impedances_per_term = np.minimum(inbound_mode_impedances_per_term, 999999)
    outbound_mode_impedances_per_term = np.minimum(outbound_mode_impedances_per_term, 999999)

    # input for matrix expr evaluation [inbound trips, oubtbound trips]
    origin_zones_for_matrix_expr = [origin_zones, dest_zones]
    dest_zones_for_matrix_expr = [dest_zones, origin_zones]
    time_interval_indices_for_matrix_expr = [inbound_trip_start_time_interval_indices, outbound_trip_start_time_interval_indices]

    matrix_utilities_per_term = abm_utilities.evaluate_matrix_expr_with_context_time_for_mode_choice(
        Visum,
        origin_zones_for_matrix_expr, dest_zones_for_matrix_expr, time_interval_indices_for_matrix_expr,
        time_interval_start_times, time_interval_end_times, matrixExprs)

    inbound_matrix_utilities_per_term = matrix_utilities_per_term[0]
    outbound_matrix_utilities_per_term =  matrix_utilities_per_term[1]

    assert inbound_matrix_utilities_per_term.shape == obj_values.shape
    assert outbound_matrix_utilities_per_term.shape == obj_values.shape

    # multiply attrExpr, matrixExpr and modeImpedance
    inbound_utility_per_term = inbound_mode_impedances_per_term * inbound_matrix_utilities_per_term
    outbound_utility_per_term = outbound_mode_impedances_per_term * outbound_matrix_utilities_per_term
    utility_per_term = obj_values * (inbound_utility_per_term + outbound_utility_per_term)

    utility_per_mode = calc_mode_utility(utility_per_term,
                                         betas,
                                         mode_is_useable=None,
                                         mode_is_interchangeable=mode_is_interchangeable,
                                         mode_ranks=mode_ranks)
    
    prob_per_mode = calc_prob_per_mode(utility_per_mode, mode_is_useable=None)
    chosen_indices = choice_engine.choose2D(prob_per_mode)

    mode_index_to_mode_id = np.array(mode_ids)

    chosen_mode_per_subject = mode_index_to_mode_id[chosen_indices]
    
    num_subjects = len(origin_zones)
    logging.info('executed %d mode choices', num_subjects)

    return chosen_mode_per_subject


def calc_minor_mode_choice(subjects,
                           Visum,
                           betas,
                           matrixExprs,
                           attrExprs,
                           mode_impedances_per_term,
                           origin_zones,
                           dest_zones,
                           time_interval_start_times,
                           time_interval_end_times,
                           trip_time_point_time_interval_indices,
                           mode_is_useable):

    # attrExpr (for all terms)
    obj_values = abm_utilities.eval_attrexpr(subjects, attrExprs)
    assert obj_values.shape == (len(subjects), len(attrExprs))
 
    # mode_impedances_per_term (for all terms)
    mode_impedances_per_term = np.column_stack(mode_impedances_per_term)
    assert mode_impedances_per_term.shape == obj_values.shape

    # in PuT 999999 usuabbly means no path available, so we limit impedance from shortest path searches to 999999
    mode_impedances_per_term = np.minimum(mode_impedances_per_term, 999999)

    matrix_utilities_per_term = abm_utilities.evaluate_matrix_expr_with_context_time_for_mode_choice(
        Visum,
        [origin_zones], [dest_zones], [trip_time_point_time_interval_indices],
        time_interval_start_times, time_interval_end_times, matrixExprs)[0]

    assert matrix_utilities_per_term.shape == obj_values.shape

    # multiply attrExpr, matrixExpr and modeImpedance
    utility_per_term = obj_values * mode_impedances_per_term * matrix_utilities_per_term

    utility_per_mode = calc_mode_utility(utility_per_term,
                                         betas,
                                         mode_is_useable,
                                         mode_is_interchangeable=None,
                                         mode_ranks=None)
    
    prob_per_mode = calc_prob_per_mode(utility_per_mode, mode_is_useable)
    chosen_indices = choice_engine.choose2D(prob_per_mode)

    num_subjects = len(origin_zones)
    logging.info('executed %d mode choices', num_subjects)

    return chosen_indices


def calc_prob_per_mode(utility_per_mode : np.ndarray, mode_is_useable : np.ndarray = None) -> np.ndarray:
    """
    Calculate probabilities per mode as ``exp_utility_per_mode=exp(utility)``, which is normalized to ``1`` for each subject.
    Use uniform distribution if all utilities are `0` for one subject, respecting unusable modes if provided.

    Parameters:
    - `utility_per_mode` -- ``np.array(shape=(num_subjects, num_modes))`` -- unuseable modes must have been set to ``MINUSINF``
    - `mode_is_usable` -- ``np.array(shape=(num_modes,))`` -- if provided, set probs of unusable modes to ``0``

    Returns `np.array(shape=(num_subjects, num_modes))` containing probabilities per mode.
    """
    num_modes = utility_per_mode.shape[1]
    num_used_modes = np.count_nonzero(mode_is_useable) if mode_is_useable is not None else num_modes

    exp_utility_per_mode = np.exp(utility_per_mode)
    exp_utility_mode_sum = exp_utility_per_mode.sum(1)[:, np.newaxis]

    if mode_is_useable is not None:
        uniform_dist = [1 / num_used_modes if mode_is_useable[mode_index]
                        else 0 for mode_index in range(num_modes)]
    else:
        uniform_dist = np.full((num_modes), 1/num_modes)

    # to finde persons with uniform distribution choices, uncomment this and set a breakpoint
    # if np.count_nonzero(exp_utility_mode_sum) != len(exp_utility_mode_sum):
    #     index_0 = np.where(exp_utility_mode_sum == 0)

    # the following statement often gives the warning "RuntimeWarning: invalid value encountered in true_divide"
    # exp_utility_per_mode / exp_utility_mode_sum is evaluated on all elements and causes this warning, it can be ignored
    prob_per_mode = np.where(exp_utility_mode_sum != 0,
                             exp_utility_per_mode / exp_utility_mode_sum, uniform_dist)

    prob_per_mode[prob_per_mode < 0] = 0

    assert np.allclose(np.sum(prob_per_mode, axis=1), 1.0)

    return prob_per_mode
