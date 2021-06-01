import collections

import VisumPy.helpers as VPH
import dask.array as da
import numpy as np

import abmvisum.engines.choice_engine as choice_engine
import abmvisum.engines.location_choice_engine as location_choice_engine
import abmvisum.tools.utilities as utilities
from .facility_choice import prepare_choice_probs_for_act_id, choose_facilities

daskfactory = choice_engine.DaskFactory(10000)


def run_long_term_primary_location_choice(Visum, skims, segments, zones, zoneNo_to_zoneInd,
                                          a_multi, dist_mat, logging, use_IPF=True, cache=None):
    num_zones = Visum.Net.Zones.Count
    activityID_to_primLocZoneNoAttr = dict(
        Visum.Net.Activities.GetMultipleAttributes(["Id", "lt_loc_zone_number_attr"]))
    activityID_to_primLocDistAttr = dict(Visum.Net.Activities.GetMultipleAttributes(["Id", "lt_loc_distance_attr"]))
    activityID_to_primLocAccesibAttr = dict(
        Visum.Net.Activities.GetMultipleAttributes(["Id", "lt_loc_accessibility_attr"]))
    activityID_to_facilityOptions = get_facility_probs_for_act_types(Visum, segments)

    segments_by_attraction = get_segments_by_attraction(segments, use_IPF, logging)

    init_location_and_dist_attrs(Visum, segments, activityID_to_primLocZoneNoAttr, activityID_to_primLocDistAttr)

    for attraction in sorted(segments_by_attraction):
        segment_indices = segments_by_attraction[attraction]
        logging.info('primary destination choice for segments with attraction "%s"' % attraction)
        run_coupled_location_choice(Visum, skims, cache, segments, zoneNo_to_zoneInd,
                                    activityID_to_primLocZoneNoAttr, activityID_to_primLocDistAttr,
                                    activityID_to_primLocAccesibAttr, a_multi, zones, dist_mat,
                                    segment_indices, num_zones, use_IPF, activityID_to_facilityOptions,
                                    logging)

    if cache is not None:
        for i, k in enumerate(cache.prob_matrices.keys()):
            mat_no = i + 5000
            helper_matrix_exists = mat_no in VPH.GetMulti(Visum.Net.Matrices, "NO")
            if not helper_matrix_exists:
                mat_visum = Visum.Net.AddMatrix(mat_no, 2, 4)
                mat_visum.SetAttValue("Name", k)
                mat_visum.SetAttValue("Code", k)
            VPH.SetMatrixRaw(Visum, k, cache.get_prob_mat(k))


def get_facility_probs_for_act_types(Visum, segments):
    activity_id_cache = []
    activityID_to_facilityOptions = collections.defaultdict(list)
    for segment in segments:
        activity_id = segment['AddData']
        if activity_id in activity_id_cache:
            continue
        else:
            activity_id_cache.append(activity_id)
        activityID_to_facilityOptions[activity_id] = prepare_choice_probs_for_act_id(Visum, activity_id,
                                                                                     location_attribute='Key')

    return activityID_to_facilityOptions


def get_segments_by_activity(segments):
    segments_by_activity = collections.defaultdict(list)
    for i in range(len(segments)):
        curSegment = segments[i]
        activity_id = curSegment['AddData']
        segments_by_activity[activity_id].append(i)
    return segments_by_activity


def get_segments_by_attraction(segments, use_ipf, logging):
    segments_by_attraction = collections.defaultdict(list)
    for i in range(len(segments)):
        curSegment = segments[i]
        attraction = curSegment['Attraction']
        segments_by_attraction[attraction].append(i)
    return segments_by_attraction


def init_location_and_dist_attrs(Visum, segments, activityID_to_primLocZoneNoAttr, activityID_to_primLocDistAttr):
    activity_id_cache = []
    for segment in segments:
        activity_id = segment['AddData']
        if activity_id in activity_id_cache:
            continue
        else:
            activity_id_cache.append(activity_id)
        primary_location_zoneNo_attrID = activityID_to_primLocZoneNoAttr[activity_id]
        dist_attrID = activityID_to_primLocDistAttr[activity_id]

        # init locations
        Visum.Net.Persons.SetAllAttValues(primary_location_zoneNo_attrID, 0)
        Visum.Net.Persons.SetAllAttValues(dist_attrID, 0.0)


def run_coupled_location_choice(Visum, skims, cache, segments, zoneNo_to_zoneInd,
                                activityID_to_primLocZoneNoAttr, activityID_to_primLocDistAttr,
                                activityID_to_primLocAccesibAttr, a_multi, zones, dist_mat,
                                segment_indices, num_zones, use_IPF, activityID_to_facilityOptions, logging):
    attractionAttr = segments[segment_indices[0]]["Attraction"]
    attraction = np.array(VPH.GetMulti(Visum.Net.Zones, attractionAttr))
    act_id = segments[segment_indices[0]]["AddData"]
    facility_options, facility_probs = activityID_to_facilityOptions[act_id]

    # set attraction to zero if there is no facility in that zone
    counter = 0
    for i, opt in enumerate(facility_options):
        if opt[0] == 'None':
            if attraction[i] != 0:
                attraction[i] = 0
                counter += 1
    if counter > 0:
        logging.info("set attraction value of " + str(counter) + " zones to zero.")

    segment_data = init_segment_data(Visum, segments, zoneNo_to_zoneInd, segment_indices)

    num_persons_in_segment = sum(seg_data["num_filtered_persons"] for seg_data in segment_data.values())

    # logging.info('run primary destination choice for %d persons' % num_persons_in_segment)

    probs = dict()
    fric_mat = dict()
    is_utilMat_still_zone_based = dict()

    for segment_index in segment_indices:
        probs[segment_index], fric_mat[segment_index], is_utilMat_still_zone_based[
            segment_index] = run_choice_for_segment(Visum, skims,
                                                    attraction,
                                                    segment_data[
                                                        segment_index],
                                                    logging)
        # row_sums = np.sum(probs[segment_index], axis=1)
        # assert len(row_sums[row_sums == 0]) == 0

    # run IPF
    if use_IPF:
        ipf_iterations = segments[segment_indices[0]]["IPFIterations"]
        if ipf_iterations > 0:
            probs = run_IPF(segment_data, segment_indices, num_zones, attraction, is_utilMat_still_zone_based,
                            fric_mat, logging, ipf_iterations)

    for i in range(1):
        # run choice
        for segment_index in segment_indices:
            cur_segment_data = segment_data[segment_index]
            if cache is not None:
                segment = cur_segment_data["segment"]
                logging.info("put data into cache for segment " + segment)
                # cache.add_prob_mat_to_cache(segment["Specification"], probs[segment_index].compute())
            originZonePerObj = cur_segment_data["originZones"] if is_utilMat_still_zone_based[segment_index] else None
            cur_segment_data["chosen_zoneIndices"] = location_choice_engine.choose_dest_zones(probs[segment_index],
                                                                                              originZonePerObj)

        # write results
        for segment_index in segment_indices:
            write_results_for_segment(activityID_to_primLocZoneNoAttr, activityID_to_primLocDistAttr,
                                      activityID_to_primLocAccesibAttr, a_multi, zones, dist_mat,
                                      segment_data[segment_index])

        # utilities.od_to_demand_mat(Visum, i + 2000, "commuters_it_" + str(i + 1), Visum.Net.Persons,
        #                            "Household\\Residence\\Location\\ZoneNo", "work_zone_no")

    # write results
    for segment_index in segment_indices:
        cur_segment_data = segment_data[segment_index]
        chosen_facilities = choose_facilities(cur_segment_data["chosen_zoneIndices"], facility_probs,
                                              facility_options, logging)

        personNos = VPH.GetMulti(cur_segment_data["filtered_persons"], "No")
        Visum.Net.AddMultiLongTermChoices(list(zip(personNos, chosen_facilities)))

    logging.info('processed %d persons' % num_persons_in_segment)


def init_segment_data(Visum, segments, zoneNo_to_zoneInd, segment_indices):
    segment_data = collections.defaultdict(dict)

    for segment_index in segment_indices:
        segment = segments[segment_index]

        # apply filter
        filtered_persons = utilities.get_filtered_subjects(Visum.Net.Persons, segment['Filter'])

        # fetch homezone indices
        originZones = utilities.get_zone_indices(filtered_persons, r'Household\Residence\Location\Zone\NO',
                                                 zoneNo_to_zoneInd)

        segment_data[segment_index]["segment"] = segment
        segment_data[segment_index]["filtered_persons"] = filtered_persons
        segment_data[segment_index]["num_filtered_persons"] = filtered_persons.Count
        segment_data[segment_index]["originZones"] = originZones
        segment_data[segment_index]["chosen_zoneIndices"] = []

    return segment_data


def run_choice_for_segment(Visum, skims, attraction, cur_segment_data, logging):
    if cur_segment_data["num_filtered_persons"] == 0:
        return

    segment = cur_segment_data["segment"]
    filtered_persons = cur_segment_data["filtered_persons"]

    logging.info('primary destination choice for segment  %s: %s ' % (segment['Specification'], segment['Comment']))

    return location_choice_engine.calc_dest_zone_probs(filtered_persons, skims, cur_segment_data["originZones"],
                                                       segment,
                                                       attraction, Visum, logging)


def run_IPF(segment_data, segment_indices, num_zones, attraction, is_utilMat_still_zone_based, fric_mat, logging,
            iterations):
    logging.info('run IPF with ' + str(iterations) + ' iterations')

    all_fric_matrix, all_row_targets = prepare_matrix_and_targets_for_IPF(segment_data, segment_indices, num_zones,
                                                                          is_utilMat_still_zone_based, fric_mat)
    all_probs_matrix = calculate_IPF(all_fric_matrix, all_row_targets, attraction, logging, max_iterations=iterations)

    if len(segment_indices) > 1:
        segment_index_map = list(fric_mat.keys())

        shape0s = [0] + list(np.cumsum([probs_i.shape[0] for probs_i in fric_mat.values()]))
        split_matrices = [all_probs_matrix[shape0s[i]:shape0s[i + 1]] for i in range(len(shape0s) - 1)]

        # assert (len(split_matrices) == len(probs.values()))
        for i in range(len(split_matrices)):
            fric_mat[segment_index_map[i]] = split_matrices[i]
    else:
        fric_mat[segment_indices[0]] = all_probs_matrix

    return fric_mat


def prepare_matrix_and_targets_for_IPF(segment_data, segment_indices, num_zones, is_utility_matrix_still_zone_based,
                                       probs):
    row_targets = dict()
    for segment_index in segment_indices:
        cur_segment_data = segment_data[segment_index]
        if is_utility_matrix_still_zone_based[segment_index]:
            # one line per orig zone
            row_targets[segment_index] = np.bincount(cur_segment_data["originZones"], minlength=num_zones)
            # multiply probs for origZone X with number of persons starting at zone X
            # probs[segment_index] = probs[segment_index] * (row_targets[segment_index])[:, np.newaxis]
        else:
            # one line per person
            row_targets[segment_index] = np.ones(len(cur_segment_data["originZones"]))
        row_targets[segment_index] = daskfactory.fromarray(row_targets[segment_index])

    if len(segment_indices) > 1:
        all_probs_matrix = da.concatenate([probs_i for probs_i in probs.values()], axis=0).rechunk((10000, -1))
        all_row_targets = da.concatenate([row_targets_i for row_targets_i in row_targets.values()]).rechunk((10000, -1))
    else:
        all_probs_matrix = probs[segment_indices[0]]
        all_row_targets = row_targets[segment_indices[0]]
    # assert (all_row_targets.shape[0] == all_probs_matrix.shape[0])

    return all_probs_matrix, all_row_targets


def calculate_IPF(matrix, row_targets, col_targets, logging, max_iterations=10):
    col_targets = col_targets * (row_targets.sum() / col_targets.sum())
    col_targets_t = col_targets.copy()
    row_targets_t = row_targets.copy()

    with np.errstate(invalid='ignore', divide='ignore'):
        for iter_no in range(int(max_iterations)):
            # update columns
            exp_util = da.multiply(da.exp(matrix), col_targets)
            exp_util_sum = exp_util.sum(1)
            exp_util_sum_T = da.asarray(exp_util_sum[:, np.newaxis])
            exp_util_sum_T = da.where(exp_util_sum_T <= 0, 1.0, exp_util_sum_T)
            trips = da.asarray(row_targets[:, np.newaxis]) * exp_util / exp_util_sum_T

            computed_attr = trips.sum(axis=0)
            computed_attr[computed_attr == 0] = 1
            col_targets = col_targets * (col_targets_t / computed_attr)

            computed_prod = trips.sum(axis=1)
            computed_prod[computed_prod == 0] = 1
            row_targets = row_targets * (row_targets_t / computed_prod)

    exp_util = da.multiply(da.exp(matrix), col_targets)
    exp_util_sum = exp_util.sum(1)
    exp_util_sum_T = da.asarray(exp_util_sum[:, np.newaxis])
    exp_util_sum_T = da.where(exp_util_sum_T <= 0, 1.0, exp_util_sum_T)
    probs = exp_util / exp_util_sum_T
    return probs


def write_results_for_segment(activityID_to_primLocZoneNoAttr, activityID_to_primLocDistAttr,
                              activityID_to_primLocAccesibAttr, a_multi, zones, dist_mat,
                              cur_segment_data):
    activity_id = cur_segment_data["segment"]['AddData']
    zoneNo_attr_ID = activityID_to_primLocZoneNoAttr[activity_id]
    distance_attr_ID = activityID_to_primLocDistAttr[activity_id]
    accessib_attr_ID = activityID_to_primLocAccesibAttr[activity_id]

    chosen_zoneNos = zones[cur_segment_data["chosen_zoneIndices"]]
    utilities.SetMulti(cur_segment_data["filtered_persons"], zoneNo_attr_ID, chosen_zoneNos)
    chosen_accessib_values = a_multi[cur_segment_data["chosen_zoneIndices"]]
    utilities.SetMulti(cur_segment_data["filtered_persons"], accessib_attr_ID, chosen_accessib_values)

    distances = dist_mat[cur_segment_data["originZones"], cur_segment_data["chosen_zoneIndices"]]
    utilities.SetMulti(cur_segment_data["filtered_persons"], distance_attr_ID,
                       distances)  # set distance from home to primary locations
