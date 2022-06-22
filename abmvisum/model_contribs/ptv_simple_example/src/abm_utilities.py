import logging
import math
import re

import numpy as np
import dask.array as da

import VisumPy.helpers as VPH
from src import visum_utilities
from settings import abm_settings

# ----------------------------------------------------------

def start_logging():
    logger = logging.getLogger()
    for handler in logger.handlers:
        logger.removeHandler(handler)
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

# ----------------------------------------------------------

def get_global_attribute(Visum, attribute_name, column):
    global_attributes = dict(Visum.Net.TableDefinitions.ItemByKey('Global attributes').TableEntries.GetMultipleAttributes(["Attribute", column]))
    return global_attributes[attribute_name]

# ----------------------------------------------------------

# evaluate attribute expressions
def eval_attrexpr(container, attrexpr, np_dtype=np.float):
    '''
    container = a Visum net object container
    attrexpr = a list of expressions, containing attributes of container as variables. Attribute IDs must be enclosed in []

    returns a np.array of shape (# of objects in container, # of expressions in attrexpr) filled with the results of the expressions

    Example:
    attrexpr = ['[NO]', '[NO]+[empl_pct]']
    container = Visum.Net.POICategories.ItemByKey(6).POIs
    res = eval_attrexpr(container, attrexpr)

    res[:10] =
    array([[5.56068163e+00, 2.60000000e+02],
           [5.56452041e+00, 2.61000000e+02],
           [5.92958914e+00, 3.76000000e+02],
           [7.59186171e+00, 1.98200000e+03],
           [7.59337419e+00, 1.98500000e+03],
           [7.60489448e+00, 2.00800000e+03],
           [7.70796153e+00, 2.22600000e+03],
           [7.72797554e+00, 2.27100000e+03],
           [7.73761628e+00, 2.29300000e+03],
           [7.78821156e+00, 2.41200000e+03]])
    '''
    num_exprs = len(attrexpr)
    num_objects = container.Count

    results = np.empty((num_objects, num_exprs), dtype=np_dtype)

    for i, expr in enumerate(attrexpr):
        if expr.strip() == "1" or not expr.strip():
            results[:, i] = 1
        else:
            results[:, i] = VPH.GetMultiByFormula(container, expr)
    return results


# evaluate matrix expressions
def eval_matexprs(Visum, matrix_expressions):
    count_expressions = len(matrix_expressions)
    count_zones = Visum.Net.Zones.Count
    result = np.ones((count_expressions, count_zones, count_zones), dtype=np.float) # matrices are later on multiplied => use 1 as default (not 0!)

    helper_matrix_no = 99999 # temporary result matrix for matrix formulas

    helper_matrix_exists = helper_matrix_no in VPH.GetMulti(Visum.Net.Matrices, "NO")
    if helper_matrix_exists:
        helper_matrix = Visum.Net.Matrices.ItemByKey(helper_matrix_no)
    else:
        helper_matrix = Visum.Net.AddMatrix(helper_matrix_no)

    for i in range(count_expressions):
        if matrix_expressions[i].strip() == "1" or not matrix_expressions[i].strip():
            # "1" or empty matrix expression => leave default (all ones)
            continue
        success = helper_matrix.SetValuesToResultOfFormula(matrix_expressions[i])
        assert success, "matrix expression invalid: " + matrix_expressions[i]
        result[i] = VPH.GetMatrixRaw(Visum, helper_matrix_no)

    return result


def evaluate_matrix_expr_with_context_time_for_location_choice(act_ex_parameters, 
                                                               config, 
                                                               time_interval_start_times : list, 
                                                               time_interval_end_times : list, 
                                                               matrix_exprs_list : list[list[str]]):
    """
    Evaluate matrix expressions including references to time intervals.

    ``CONTEXT[FromTime]`` and ``CONTEXT[ToTime]`` are replaced by start and end times of time intervals respectively.

    For performance reasons, all matrix values are obtained for all relevant time intervals beforehand 
    and then evaluated for the actual combinations of origin zones and time intervals.

    # Parameters:
    - `matrix_exprs_list` -- each entry is a list of matrix expressions corresponding to one column (e.g. ``MatrixExpr``, ``DestinationImpedance``)

    Returns 4-dimensional array of shape ``(len(matrix_exprs_list), num_terms, num_objects, num_zones)``
    containing evaulated matrix_expr for all dest zones.                                 
    """

    # find all from-times, find all relevant destinationImpedance definitions
    used_ti_indices = sorted(set(act_ex_parameters.time_interval_indices_origin_path).union(act_ex_parameters.time_interval_indices_target_path))

    matrix_expr_per_list_and_term = []
    matrix_expr_per_list_and_term_for_rubberbanding = []
    ti_index_to_used_ti_index = dict(zip(used_ti_indices, range(len(used_ti_indices))))

    origin_path_ti_indices = np.array([ti_index_to_used_ti_index[ti_index]
                                        for ti_index in act_ex_parameters.time_interval_indices_origin_path], dtype=int)
    target_path_ti_indices = np.array([ti_index_to_used_ti_index[ti_index]
                                        for ti_index in act_ex_parameters.time_interval_indices_target_path], dtype=int)

    from_time_context = 'CONTEXT[FromTime]'
    to_time_context = 'CONTEXT[ToTime]'

    for matrix_exprs in matrix_exprs_list:
        matrix_expr_per_term = []
        matrix_expr_per_term_for_rubberbanding = []
        for matrix_expr in matrix_exprs:
            if (from_time_context in matrix_expr) or (to_time_context in matrix_expr):
                matrix_expr_per_ti = [matrix_expr.replace(
                    from_time_context, str(time_interval_start_times[ti_index])).replace(
                        to_time_context, str(time_interval_end_times[ti_index])) 
                            for ti_index in used_ti_indices]

                matrix_expr_per_ti = eval_matexprs(config.Visum, matrix_expr_per_ti)
                # matrix_expr_per_ti.shape = (num_time_intervals, num_zones, num_zones)
                # matrix_result.shape = (num_objects, num_zones)
                matrix_result = da.from_array(
                    matrix_expr_per_ti[origin_path_ti_indices, act_ex_parameters.origin_zones], chunks=('auto', abm_settings.chunk_size_zones))
                matrix_result_for_rubberbanding = da.from_array(matrix_expr_per_ti.transpose((0, 2, 1))[
                                                                target_path_ti_indices, act_ex_parameters.target_zones], chunks=('auto', abm_settings.chunk_size_zones))
            else:    
                dest_imp_independent_of_ti = np.array(eval_matexprs(config.Visum, [matrix_expr])[0])
                matrix_result = da.from_array(
                    dest_imp_independent_of_ti[act_ex_parameters.origin_zones], chunks=('auto', abm_settings.chunk_size_zones))
                matrix_result_for_rubberbanding = da.from_array(dest_imp_independent_of_ti.transpose(
                    1, 0)[act_ex_parameters.target_zones], chunks=('auto', abm_settings.chunk_size_zones))
                
            # matrix_expr_term_result.shape = (num_objects, num_zones)                 
            matrix_expr_per_term.append(matrix_result)
            matrix_expr_per_term_for_rubberbanding.append(matrix_result_for_rubberbanding)

        matrix_expr_per_term = da.stack(matrix_expr_per_term)
        matrix_expr_per_term_for_rubberbanding = da.stack(matrix_expr_per_term_for_rubberbanding)

        # matrix_expr_per_term.shape = (num_terms, num_objects, num_zones)
        matrix_expr_per_list_and_term.append(matrix_expr_per_term)
        matrix_expr_per_list_and_term_for_rubberbanding.append(matrix_expr_per_term_for_rubberbanding)

    # matrix_expr_per_list_and_term.shape = (len(matrix_exprs_list), num_terms, num_objects, num_zones)
    matrix_expr_per_list_and_term = da.stack(matrix_expr_per_list_and_term)
    matrix_expr_per_list_and_term_for_rubberbanding = da.stack(matrix_expr_per_list_and_term_for_rubberbanding)
    return matrix_expr_per_list_and_term, matrix_expr_per_list_and_term_for_rubberbanding


def evaluate_matrix_expr_with_context_time_for_mode_choice(Visum, 
                                                           origin_zones_for_matrix_expr, dest_zones_for_matrix_expr, time_interval_indices_for_matrix_expr,
                                                           time_interval_start_times, time_interval_end_times, matrixExprs):
    num_evaluations_per_matrix = len(origin_zones_for_matrix_expr)
    assert len(dest_zones_for_matrix_expr) == num_evaluations_per_matrix
    assert len(time_interval_indices_for_matrix_expr) == num_evaluations_per_matrix

    # matrixExpr (one term at a time)
    matrix_utilities_per_term = [ [] for _ in range(num_evaluations_per_matrix) ]
    utility_for_current_term = [ [] for _ in range(num_evaluations_per_matrix) ]
    for matrixExpr in matrixExprs:
        from_time_context = 'CONTEXT[FromTime]'
        to_time_context = 'CONTEXT[ToTime]'
        if (from_time_context in matrixExpr) or (to_time_context in matrixExpr):
            matrixExpr_per_ti = [matrixExpr.replace(
                from_time_context, str(time_interval_start_times[ti_index])).replace(
                to_time_context, str(time_interval_end_times[ti_index]))
                for ti_index in range(len(time_interval_start_times))]
            utility_matrices_per_ti = eval_matexprs(Visum, matrixExpr_per_ti)
            for i in range(num_evaluations_per_matrix):
                utility_for_current_term[i] = utility_matrices_per_ti[time_interval_indices_for_matrix_expr[i],
                                                                      origin_zones_for_matrix_expr[i], dest_zones_for_matrix_expr[i]]
        else:
            utility_matrix = eval_matexprs(Visum, [matrixExpr])[0]
            for i in range(num_evaluations_per_matrix):
                utility_for_current_term[i] = utility_matrix[origin_zones_for_matrix_expr[i],
                                                             dest_zones_for_matrix_expr[i]]
        for i in range(num_evaluations_per_matrix):
            matrix_utilities_per_term[i].append(utility_for_current_term[i])

    for i in range(num_evaluations_per_matrix):
        matrix_utilities_per_term[i] = np.column_stack(matrix_utilities_per_term[i])

    return matrix_utilities_per_term


# ----------------------------------------------------------


def get_filtered_subjects(subjects, filter_expr):
    return subjects.GetFilteredSet(filter_expr) if filter_expr != "" else subjects


def build_activity_dict(Visum):
    return dict(Visum.Net.Activities.GetMultipleAttributes(["ID", "Code"]))


def build_activityLocation_dict(Visum):
    activityLocation_dict = dict()
    for location_no, act_code, activityLocation_key in Visum.Net.ActivityLocations.GetMultipleAttributes(["LocationNo", "ActivityCode", "Key"]):
        activityLocation_dict[(location_no, act_code)] = activityLocation_key
    return activityLocation_dict
    

def build_zone_index_to_attraction_array(Visum, activity_code):
    zone_attraction_attr = fr'SUM:ASSOCIATEDLOCATIONS\SUM:ACTIVITYLOCATIONS([ACTIVITYCODE]="{activity_code}")\AttractionPotential'
    zone_attraction = np.array(VPH.GetMulti(Visum.Net.Zones, zone_attraction_attr), dtype=float)
    # relation zone -> associated location may be empty
    # => replace NaN by 0
    np.nan_to_num(zone_attraction, nan=0.0, copy=False)
    return zone_attraction

def build_location_probs_inside_zones(Visum, activity_code, zoneNo_to_zoneInd, locationNo_to_locationInd):
    # location attractions
    location_attraction_attr_for_activity = fr'SUM:ACTIVITYLOCATIONS([ACTIVITYCODE]="{activity_code}")\AttractionPotential'
    location_attraction = np.array(VPH.GetMulti(Visum.Net.Locations, location_attraction_attr_for_activity), dtype=float)
    # relation may be empty
    #  => replace NaN by 0
    np.nan_to_num(location_attraction, nan=0.0, copy = False)
    
    # collect only relevant locations within zones (attraction > 0)
    zoneIndex_to_locationIndices = [ [] for _ in range(len(zoneNo_to_zoneInd)) ]
    location_zoneNos = Visum.Net.Locations.GetMultipleAttributes(["No","Zone\\No"])
    for (locationNo, zoneNo) in location_zoneNos:
        locationInd = locationNo_to_locationInd[locationNo]
        if location_attraction[locationInd] > 0:
            zoneIndex_to_locationIndices[zoneNo_to_zoneInd[zoneNo]].append(locationInd)

    # compute probs for location choice
    location_probs_inside_zone = []
    for location_indices in zoneIndex_to_locationIndices:
        probs_inside_current_zone = []
        if location_indices:
            attraction_sum = sum(location_attraction[location_indices])
            assert attraction_sum > 0.
            for location_index in location_indices:
                probs_inside_current_zone.append(location_attraction[location_index] / attraction_sum)
        location_probs_inside_zone.append(probs_inside_current_zone)
    
    return location_probs_inside_zone, zoneIndex_to_locationIndices

def nos_to_indices(nos, no_to_ind):
    assert np.all(nos > 0) # nos must be > 0
    # convert no -> index
    indices = np.fromiter(iter=(no_to_ind[no] for no in nos), dtype=int, count=nos.shape[0])
    return indices

def get_indices(subjects, no_attr_id, no_to_ind, chunks=None, reindex=False):
    nos = np.array(visum_utilities.GetMulti(subjects, no_attr_id, chunk_size=chunks, reindex=reindex), dtype=int)
    return nos_to_indices(nos, no_to_ind)


# returns list containing the index of each value in discrete_values
# i.e. returns value_indices s.t. discrete_values_list[value_indices[i]] == values[i]
def get_value_indices(discrete_values_list, values):
    discrete_value_to_index = dict()
    for i, value in enumerate(discrete_values_list):
        discrete_value_to_index[value] = i
        
    value_indices = [discrete_value_to_index[val] for val in values]
    return value_indices


def get_time_interval_index(time, sorted_interval_start_times, sorted_time_interval_end_times):
    assert len(sorted_interval_start_times) == len(sorted_time_interval_end_times)
    
    if time < sorted_interval_start_times[0]:
        logging.error(f'time point {time} lies before the first analysis time interval')
        return sorted_interval_start_times[0]

    for interval_index, interval_start_time in enumerate(sorted_interval_start_times):
        if time < interval_start_time:
            if time > sorted_time_interval_end_times[interval_index - 1]:
                logging.error(f'time point {time} lies outside the analysis time intervals')
            return interval_index - 1
    return len(sorted_interval_start_times) - 1


def get_time_interval_start_time(time, sorted_interval_start_times, sorted_time_interval_end_times):
    return sorted_interval_start_times[get_time_interval_index(time, sorted_interval_start_times, sorted_time_interval_end_times)]


def get_time_interval_code(time, sorted_interval_start_times, sorted_time_interval_end_times, time_interval_codes):
    assert len(sorted_interval_start_times) == len(time_interval_codes)
    return time_interval_codes[get_time_interval_index(time, sorted_interval_start_times, sorted_time_interval_end_times)]


# compute direct distance in meters between two points given in WGS84 coords
def compute_direct_distance(lat1, lng1, lat2, lng2):
    # taken from https://github.com/geopy/geopy/blob/master/geopy/distance.py#L445-L463
    radius = 6371e3 # earth radius in meters

    lat1, lng1 = math.radians(lat1), math.radians(lng1)
    lat2, lng2 = math.radians(lat2), math.radians(lng2)

    sin_lat1, cos_lat1 = math.sin(lat1), math.cos(lat1)
    sin_lat2, cos_lat2 = math.sin(lat2), math.cos(lat2)

    delta_lng = lng2 - lng1
    cos_delta_lng, sin_delta_lng = math.cos(delta_lng), math.sin(delta_lng)

    d = math.atan2(math.sqrt((cos_lat2 * sin_delta_lng) ** 2 +
                             (cos_lat1 * sin_lat2 - sin_lat1 * cos_lat2 * cos_delta_lng) ** 2),
                   sin_lat1 * sin_lat2 + cos_lat1 * cos_lat2 * cos_delta_lng)

    return radius * d


def get_coord_list_from_wkt_point_list(wkt_point_list : list[str]) -> list[tuple[float]]:
    """Convert list of WKT point strings (``POINT(longitude latitude)``) into list of coordinates (pairs of type float)."""
    parse_coords_regex = re.compile(r"POINT\((\S+)\s+(\S+)\)")
    return [tuple(map(float, parse_coords_regex.match(wkt_str).groups())) for wkt_str in wkt_point_list]


def get_nodeno_to_longitude_latitude(Visum) -> dict[int, tuple[float]]:
    """Get dict providing coordinates ``(longitude, latitude)`` for each node no."""
    node_wktpointstr = VPH.GetMulti(Visum.Net.Nodes, 'WKTLocWGS84')
    node_nos = VPH.GetMulti(Visum.Net.Nodes, 'No')
    coord_list = get_coord_list_from_wkt_point_list(node_wktpointstr)
    return dict(zip(node_nos, coord_list))
