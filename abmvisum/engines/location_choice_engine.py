import dask.array as da
import numpy as np

import abmvisum.engines.choice_engine as choice_engine
import abmvisum.tools.utilities as utilities

daskfactory = choice_engine.DaskFactory(10000)


def calc_dest_zone_utility_matrix(origin_zones, betas, utility_matrices,
                                  target_zones_for_rubberbanding,
                                  logging):
    assert (target_zones_for_rubberbanding is None) or len(target_zones_for_rubberbanding) == len(origin_zones)

    betas_adapted = np.array(betas)[:, np.newaxis, np.newaxis]

    # if all subjects' attrExprs are 1, utility depends only on origin and dest zone
    # so memory can be saved by only calculating a matrix of dimension (#zones x #zones) instead of (#subjects x #zones) in this case
    # could be further optimized: first calculate all terms with AttrExpr == 1 based on zones and then only calculate
    # the remaining terms with AttrExpr != 1 based on subjects
    calculate_zone_based = (target_zones_for_rubberbanding is None)

    if calculate_zone_based:
        utility_matrix = daskfactory.fromarray((utility_matrices * betas_adapted).sum(axis=0))
    else:
        #  we need to use dask here to reduce memory usage
        # as we are dealing with matrices of size (num_subjects x num_zones x num_terms)
        utility_matrices = da.from_array(utility_matrices, chunks=(-1, 10000, -1))
        origin_zones = daskfactory.fromarray(origin_zones)
        if target_zones_for_rubberbanding is None:
            utility_per_destZone = utility_matrices[:, origin_zones]
        else:
            target_zones_for_rubberbanding = daskfactory.fromarray(target_zones_for_rubberbanding)
            utility_per_destZone = utility_matrices[:, origin_zones] + \
                                   utility_matrices.transpose((0, 2, 1))[:, target_zones_for_rubberbanding]

        # assert utility_per_destZone.shape[0] == obj_values.shape[0] # number of subjects
        # assert utility_per_destZone.shape[1] == utility_matrices.shape[0] == utility_matrices.shape[1] # number of zones
        # assert utility_per_destZone.shape[2] == len(betas) # number of terms

        # obj_values = da.from_array(obj_values.T[:, :, np.newaxis], chunks=(-1, 10000, -1))
        utility_matrix = (utility_per_destZone * betas_adapted).sum(axis=0)
        # logging.info(utility_matrix.shape)

    return utility_matrix, calculate_zone_based


def calc_dest_zone_probs_from_utility_matrix(utility_matrix, attraction_per_dest_zone, logging):
    # build probability matrix for the members of the segment

    exp_util = da.multiply(da.exp(utility_matrix), attraction_per_dest_zone)
    exp_util_sum = exp_util.sum(1)
    exp_util_sum_T = da.asarray(exp_util_sum[:, np.newaxis])
    exp_util_sum_T = da.where(exp_util_sum_T <= 0, 1.0, exp_util_sum_T)
    probs = exp_util / exp_util_sum_T
    probs = da.maximum(probs, 0)
    # logging.info(probs.shape)

    # assert np.allclose(np.sum(probs, axis=1), 1.0)
    return probs


# uses rubberbanding if target_zones_for_rubberbanding are provided
def calc_dest_zone_probs_simple(segment, skims, act_id, Visum, logging):
    matrix_expr = segment['MatrixExpr']
    attr_expr = segment['AttrExpr']
    od_kalif_expr = segment['ODKaliFExpr']
    impedance_params = segment['ImpedanceParams']

    if impedance_params:
        nZones = Visum.Net.Zones.Count
        utility_matrices = np.empty((1, nZones, nZones), dtype=np.float)
        # logging.info('loading logsum matrix from impedance params')

        # for imp_expr in impedance_params:
        #    utility_matrices[0, :, :] += np.exp(utilities.eval_matexprs(Visum, [imp_expr])[0, :, :])

        utility_matrices[0, :, :] = utilities.eval_matexprs_cached_skims(Visum, impedance_params[act_id], skims)
        utility_matrices[0, :, :] = np.where(np.isneginf(np.log(utility_matrices[0, :, :])), -9999,
                                             np.log(utility_matrices[0, :, :]))
    else:
        logging.info("taking matrix expression")
        utility_matrices = utilities.eval_matexprs(Visum, matrix_expr)

    od_kalif_expr_act_id = [0]
    for i, attr in enumerate(attr_expr):
        if float(attr) == float(act_id):
            od_kalif_expr_act_id = [od_kalif_expr[i]]

    od_asc_matrices = utilities.eval_matexprs(Visum, od_kalif_expr_act_id)
    utility_matrices[0, :, :] += od_asc_matrices[0, :, :]
    utility_matrix = utility_matrices[0, :, :]
    return utility_matrix


# uses rubberbanding if target_zones_for_rubberbanding are provided
def calc_dest_zone_probs(filtered_subjects, skims, origin_zones, segment, attraction_per_dest_zone, Visum, logging,
                         target_zones_for_rubberbanding=None):
    matrix_expr = segment['MatrixExpr']
    attr_expr = segment['AttrExpr']
    betas = segment['Beta']
    od_kalif_expr = segment['ODKaliFExpr']
    impedance_params = segment['ImpedanceParams']

    if impedance_params is not None:
        nZones = Visum.Net.Zones.Count
        utility_matrices = np.empty((1, nZones, nZones), dtype=np.float)
        # logging.info('loading logsum matrix from impedance params')

        # for imp_expr in impedance_params:
        #    utility_matrices[0, :, :] += np.exp(utilities.eval_matexprs(Visum, [imp_expr])[0, :, :])

        utility_matrices[0, :, :] = utilities.eval_matexprs_cached_skims(Visum, impedance_params, skims)
        utility_matrices[0, :, :] = np.where(np.isneginf(np.log(utility_matrices[0, :, :])), -9999,
                                             np.log(utility_matrices[0, :, :]))
    else:
        utility_matrices = utilities.eval_matexprs(Visum, matrix_expr)

    # VPH.SetMatrixRaw(Visum, 99999, utility_matrices[0, :, :])
    # obj_values = utilities.eval_attrexpr(filtered_subjects, attr_expr)
    od_asc_matrices = utilities.eval_matexprs(Visum, od_kalif_expr)
    utility_matrices[0, :, :] += od_asc_matrices[0, :, :]
    attraction_per_dest_zone = daskfactory.fromarray(attraction_per_dest_zone)

    utility_matrix, is_utility_matrix_still_zone_based = calc_dest_zone_utility_matrix(origin_zones, betas,
                                                                                       utility_matrices,
                                                                                       target_zones_for_rubberbanding,
                                                                                       logging)
    probs = calc_dest_zone_probs_from_utility_matrix(utility_matrix, attraction_per_dest_zone, logging)
    return probs, utility_matrix, is_utility_matrix_still_zone_based


def choose_and_set_minor_destinations(matrix_cache, skims, act_id, Visum,
                                      segment,
                                      attraction_per_zone,
                                      origin_zones,
                                      logging,
                                      target_zones_for_rubberbanding=None):
    matrix_key = (act_id, segment['Specification'])
    if matrix_cache.has_matrix(matrix_key):
        utils = matrix_cache.get_matrix(matrix_key)
    else:
        utils = calc_dest_zone_probs_simple(segment, skims, act_id, Visum, logging)
        matrix_cache.add_matrix_to_cache(matrix_key, utils)

    origin_zones = daskfactory.fromarray(origin_zones)
    if (target_zones_for_rubberbanding is None) or (float(act_id) == 1.0) or (float(act_id) == 7.0):
        exp_utils = np.exp(utils)
        exp_utils_attr = np.multiply(exp_utils, attraction_per_zone)
        util_exp_attr_matrix = daskfactory.fromarray(exp_utils_attr)
        utility_tot = util_exp_attr_matrix[origin_zones]
    else:
        # utils at origin
        origin_weight = 0.5  # todo (PM) own assumption: verify
        exp_utils_attr = np.multiply(np.exp(origin_weight * utils), attraction_per_zone)
        util_exp_attr_matrix = daskfactory.fromarray(exp_utils_attr)
        util_mat_ik = util_exp_attr_matrix[origin_zones]

        # utils at target
        target_weight = 0.5  # todo (PM) own assumption: verify
        utils_T = np.transpose(np.exp(target_weight * utils))
        utility_matrix_T = daskfactory.fromarray(utils_T)
        target_zones_for_rubberbanding = daskfactory.fromarray(target_zones_for_rubberbanding)
        util_mat_kj = utility_matrix_T[target_zones_for_rubberbanding]

        utility_tot = da.multiply(util_mat_ik, util_mat_kj)

    exp_util_sum = da.asarray(utility_tot.sum(1)[:, np.newaxis])
    exp_util_sum = da.where(exp_util_sum <= 0, 1.0, exp_util_sum)
    probs = utility_tot / exp_util_sum
    probs = da.maximum(probs, 0)
    return choose_dest_zones(probs)


def choose_dest_zones(probs, originZonePerObj=None):
    if originZonePerObj is None:
        adapted_probs = probs
    else:
        originZonePerObj = da.from_array(originZonePerObj, chunks=10000)
        adapted_probs = probs[originZonePerObj]

    num_choices = adapted_probs.shape[0]
    choice2d = choice_engine.Choice2DParallel(num_choices, 10000)
    choice2d.add_prob(adapted_probs)

    chosen_zone_indices = choice2d.choose()
    return chosen_zone_indices
