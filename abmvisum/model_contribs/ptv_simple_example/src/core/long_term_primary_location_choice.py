import logging
import collections

import numpy as np
import dask.array as da

import VisumPy.helpers as VPH

from src import abm_utilities, visum_utilities, choice_engine, location_choice_engine
from settings import abm_settings

def run(Visum, segments, locations, zoneNo_to_zoneInd, locationNo_to_locationInd, globalPersonFilter, use_IPF=True):
    
    # global person filter can only be used if generation is done in this script
    if Visum.Net.Tours.Count > 0:
        globalPersonFilter = ''
    
    activityLocation_dict = abm_utilities.build_activityLocation_dict(Visum)
    segments_by_activity = get_segments_by_activity(segments)

    num_zones = Visum.Net.Zones.Count

    activityID_to_activityCode = abm_utilities.build_activity_dict(Visum)

    for activity_id in sorted(segments_by_activity):
        segment_indices = segments_by_activity[activity_id]
        activity_code = activityID_to_activityCode[activity_id]
        logging.info(f'primary destination choice for segments with activity "{activity_code}"')

        num_coupled_segments = len(segment_indices)
        if num_coupled_segments > 1:
            logging.info(f'  couple jointly on destination side for {num_coupled_segments} segments')

        run_coupled_location_choice(Visum, segments, zoneNo_to_zoneInd, locationNo_to_locationInd, activityLocation_dict,
                                    locations, activity_code, segment_indices, num_zones, globalPersonFilter, use_IPF)
    
    _set_is_primary_for_tours(Visum)


def get_segments_by_activity(segments):
    segments_by_activity = collections.defaultdict(list)
    for segment_index, segment in enumerate(segments):
        activity_id = segment['AddData']
        segments_by_activity[activity_id].append(segment_index)
    return segments_by_activity


def run_coupled_location_choice(Visum, segments, zoneNo_to_zoneInd, locationNo_to_locationInd, activityLocation_dict,
                                locations, activity_code, segment_indices, num_zones, globalPersonFilter, use_IPF):

    zone_attraction = abm_utilities.build_zone_index_to_attraction_array(Visum, activity_code)

    if sum(zone_attraction) == 0:
        logging.warning('attraction for activity %s is 0 for every location associated to a zone', activity_code)

    location_probs_inside_zone, zoneIndex_to_locationIndices = abm_utilities.build_location_probs_inside_zones(
        Visum, activity_code, zoneNo_to_zoneInd, locationNo_to_locationInd)

    segment_data = init_segment_data(Visum, segments, zoneNo_to_zoneInd, segment_indices, globalPersonFilter)

    num_persons_in_coupled_segments = sum(seg_data["num_filtered_persons"] for seg_data in segment_data.values())

    logging.info(f'run primary destination choice for {num_persons_in_coupled_segments} persons')
    if num_persons_in_coupled_segments == 0:
        return

    probs = dict()
    is_utilMat_still_zone_based = dict()

    for segment_index in segment_indices:
        cur_segment_data = segment_data[segment_index]
        if cur_segment_data["num_filtered_persons"] == 0:
            continue

        probs[segment_index], is_utilMat_still_zone_based[segment_index] = run_choice_for_segment(Visum, zone_attraction,
                                                                                                  cur_segment_data)
        # row_sums = np.sum(probs[segment_index], axis=1)
        # sum_total = np.sum(row_sums)
        # row_sums_comupted = row_sums
        # zero_row_sums = row_sums[row_sums == 0.0]
        # assert len(zero_row_sums) == 0

    # run IPF
    if use_IPF:
        run_IPF(segment_data, segment_indices, num_zones, zone_attraction, is_utilMat_still_zone_based, probs)

    # run choice
    for segment_index in segment_indices:
        cur_segment_data = segment_data[segment_index]
        if cur_segment_data["num_filtered_persons"] == 0:
            continue

        originZonePerObj = cur_segment_data["originZones"] if is_utilMat_still_zone_based[segment_index] else None
        cur_segment_data["chosen_zoneIndices"] = location_choice_engine.choose_dest_zones(probs[segment_index], originZonePerObj)
        cur_segment_data["chosen_locationIndices"] = location_choice_engine.choose_dest_locations(
            location_probs_inside_zone, cur_segment_data["chosen_zoneIndices"], zoneIndex_to_locationIndices, num_zones)

    # write results
    for segment_index in segment_indices:
        write_results_for_segment(Visum, activityLocation_dict,
                                  locations, activity_code, segment_data[segment_index])

    logging.info(f'processed {num_persons_in_coupled_segments} persons')


def init_segment_data(Visum, segments, zoneNo_to_zoneInd, segment_indices, globalPersonFilter):
    segment_data = collections.defaultdict(dict)

    for segment_index in segment_indices:
        segment = segments[segment_index]

        # apply filter
        if globalPersonFilter == '':
            segment_filter = segment['Filter']
        elif segment['Filter'] == '':
            segment_filter = globalPersonFilter
        else:
            segment_filter = segment['Filter'] + ' & ' + globalPersonFilter
        filtered_persons = abm_utilities.get_filtered_subjects(Visum.Net.Persons, segment_filter)

        # fetch homezone indices
        originZones = abm_utilities.get_indices(filtered_persons, r'Household\Residence\Location\Zone\NO', zoneNo_to_zoneInd)

        segment_data[segment_index]["segment"] = segment
        segment_data[segment_index]["filtered_persons"] = filtered_persons
        segment_data[segment_index]["num_filtered_persons"] = filtered_persons.Count
        segment_data[segment_index]["originZones"] = originZones
        segment_data[segment_index]["chosen_zoneIndices"] = []
        segment_data[segment_index]["chosen_locationIndices"] = []

    return segment_data


def run_choice_for_segment(Visum, attraction, cur_segment_data):
    segment = cur_segment_data["segment"]
    filtered_persons = cur_segment_data["filtered_persons"]

    logging.info('primary destination choice for segment %s: %s ', segment['Specification'], segment['Comment'])

    return location_choice_engine.calc_dest_zone_probs_for_long_term_choice(filtered_persons, cur_segment_data["originZones"], segment, attraction, Visum)


def run_IPF(segment_data, segment_indices, num_zones, attraction, is_utilMat_still_zone_based, probs):
    logging.info('run IPF')

    all_probs_matrix, all_row_targets = prepare_matrix_and_targets_for_IPF(segment_data, segment_indices, num_zones,
                                                                           is_utilMat_still_zone_based, probs)
    all_probs_matrix = calculate_IPF(all_probs_matrix, all_row_targets, attraction)

    if len(segment_indices) > 1:
        segment_index_map = list(probs.keys())

        shape0s = [0] + list(np.cumsum([probs_i.shape[0] for probs_i in probs.values()]))
        split_matrices = [all_probs_matrix[shape0s[i]:shape0s[i+1]] for i in range(len(shape0s)-1)]

        #assert (len(split_matrices) == len(probs.values()))
        for i, matrix in enumerate(split_matrices):
            probs[segment_index_map[i]] = matrix
    else:
        probs[segment_indices[0]] = all_probs_matrix

    for segment_index in segment_indices:
        row_sums = probs[segment_index].sum(axis=1)
        row_sums = row_sums.reshape(-1, 1).rechunk((abm_settings.chunk_size_zones, -1))
        row_sums = da.where(row_sums <= 0, 1., row_sums)  # reset to 1 to avoid dividing by zero
        probs[segment_index] = probs[segment_index] / row_sums


dask_factory = choice_engine.DaskFactory(abm_settings.chunk_size_zones)

def prepare_matrix_and_targets_for_IPF(segment_data, segment_indices, num_zones, is_utility_matrix_still_zone_based, probs):
    row_targets = dict()
    for segment_index in segment_indices:
        cur_segment_data = segment_data[segment_index]
        if is_utility_matrix_still_zone_based[segment_index]:
            # one line per orig zone
            row_targets[segment_index] = np.bincount(cur_segment_data["originZones"], minlength=num_zones)
            # multiply probs for origZone X with number of persons starting at zone X
            probs[segment_index] = probs[segment_index] * (row_targets[segment_index])[:, np.newaxis]
        else:
            # one line per person
            row_targets[segment_index] = np.ones(len(cur_segment_data["originZones"]))
        row_targets[segment_index] = dask_factory.from_array(row_targets[segment_index])

    if len(segment_indices) > 1:
        all_probs_matrix = da.concatenate([probs_i for probs_i in probs.values()], axis=0).rechunk((abm_settings.chunk_size_zones,'auto'))
        all_row_targets = da.concatenate([row_targets_i for row_targets_i in row_targets.values()]).rechunk((abm_settings.chunk_size_zones,'auto'))
    else:
        all_probs_matrix = probs[segment_indices[0]]
        all_row_targets = row_targets[segment_indices[0]]
    #assert (all_row_targets.shape[0] == all_probs_matrix.shape[0])

    return all_probs_matrix, all_row_targets


def calculate_IPF(matrix, row_targets, col_targets, max_iterations=5):
    #row_sums = matrix.sum(axis=1)
    #assert len(row_sums[row_sums == 0]) == len(row_targets[row_targets == 0]) # all row sums for targets > 0 must be > 0
    #debug_helper.warn_NaN_in_dask_matrix(matrix)

    with np.errstate(invalid='ignore', divide='ignore'):
        for _ in range(max_iterations):
            # update columns
            col_factors = col_targets / matrix.sum(axis=0)
            col_factors[~ da.isinf(col_factors)] = 1
            matrix *= col_factors

            # update rows
            row_factors = row_targets / matrix.sum(axis=1)
            row_factors[~ da.isinf(row_factors)] = 1
            row_factors = row_factors.reshape(-1, 1).rechunk((abm_settings.chunk_size_zones, -1))
            matrix *= row_factors

    #row_sums = matrix.sum(axis=1)
    #assert len(row_sums[row_sums == 0]) == len(row_targets[row_targets == 0]) # all row sums for targets > 0 must be > 0
    return matrix

def write_results_for_segment(Visum, activityLocation_dict, locations, activity_code, cur_segment_data):
    chosen_locationNos = locations[cur_segment_data["chosen_locationIndices"]]
    chosen_activityLocations = [activityLocation_dict[(location, activity_code)] for location in chosen_locationNos]
    assert chosen_activityLocations.count('') == 0

    personNos = VPH.GetMulti(cur_segment_data["filtered_persons"], "No")
    Visum.Net.AddMultiLongTermChoices(list(zip(personNos, chosen_activityLocations)))


def _set_is_primary_for_tours(Visum):
    if Visum.Net.Tours.Count == 0:
        return

    tour_major_activity_code = VPH.GetMulti(Visum.Net.Tours, "MajorActCode")
    tour_long_term_act_codes = VPH.GetMulti(Visum.Net.Tours, r"SCHEDULE\PERSON\CONCATENATE:LONGTERMCHOICES\ACTIVITYCODE")

    tour_is_primary = [(act_code != "" and act_code in long_term_act_code) for (
        act_code, long_term_act_code) in zip(tour_major_activity_code, tour_long_term_act_codes)]
    VPH.SetMulti(Visum.Net.Tours, 'Is_Primary', tour_is_primary)

    subtour_act_ex = Visum.Net.ActivityExecutions.GetFilteredSet("[IsPartOfSubtour] = 1")
    from_trip_tour_is_primary = visum_utilities.GetMulti(
        subtour_act_ex, r'FromTrip\Tour\Is_Primary', chunk_size=abm_settings.chunk_size_trips, reindex = True)
    to_trip_tour_is_primary = visum_utilities.GetMulti(
        subtour_act_ex, r'ToTrip\Tour\Is_Primary', chunk_size=abm_settings.chunk_size_trips)
    is_part_of_subtour = list(map( lambda x, y: x == 1 or y == 1, from_trip_tour_is_primary , to_trip_tour_is_primary))
    num_changed_values = is_part_of_subtour.count(False)
    if num_changed_values > 0:
        logging.warning(
            rf"IsPartOfSubtour was changed to False for {num_changed_values} activity executions. Their person had no long term choice for the tour's major activity.")
    visum_utilities.SetMulti(
        subtour_act_ex, r'IsPartOfSubtour', is_part_of_subtour, chunk_size=abm_settings.chunk_size_trips)
