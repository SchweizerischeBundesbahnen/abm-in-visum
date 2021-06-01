import collections
import operator

import numpy as np

import abmvisum.tools.utilities as utilities
from abmvisum.engines.choice_engine import Choice2D


def run_start_time_choice(segments, visum_com, logging, rand):
    # only one segment for time of day scheduling
    assert len(segments) == 1
    segment = segments[0]

    _a, _b, _c, _d, _e, _f = prepare_time_of_day_choice(visum_com, segment, logging)
    # _a = tour_order_nos
    # _b = time_series_dict
    # _c = act_time_series_no
    # _d = dur_before_major_act
    # _e = dur_after_major_act
    # _f = all_act_time_series_no

    final_valid_tours, final_trip_times, final_actex_times = run_time_of_day_choice(visum_com, _a, _d, _e, _c, _f,
                                                                                    _b, segment, rand, logging)

    # write results into Visum
    utilities.SetMulti(visum_com.Net.Tours, "valid_start_times", final_valid_tours, chunks=20000000)
    utilities.SetMulti(visum_com.Net.Trips, "SchedDepTime", final_trip_times, chunks=20000000)
    utilities.SetMulti(visum_com.Net.ActivityExecutions.GetFilteredSet(r'[FromTrip\Index] >= 1'),
                       r'StartTime', final_actex_times, chunks=20000000)

    write_results_and_sort_tours(visum_com, logging)


def prepare_time_of_day_choice(v, segment, logging):
    # 1. first initialise the chosen start times
    init(v, logging)

    num_tours = v.Net.Tours.Count
    num_activity_executions = v.Net.ActivityExecutions.Count
    num_trips = v.Net.Trips.Count
    logging.info('start time choice model: %d tours with %d trips and %d activity executions' % (num_tours, num_trips,
                                                                                                 num_activity_executions))

    # 2. load all the relevant information
    actex_wo_first_home = v.Net.ActivityExecutions.GetFilteredSet(r'[FromTrip\Index] >= 1')

    filter_condition = r'([is_main_tour_activity] = 1) & ([FromTrip\FromActivityExecution\is_part_of_subtour] != 1)'
    relevant_act_arr = np.array(utilities.GetMultiByFormula(actex_wo_first_home, 'IF(%s,1,0)' % filter_condition),
                                dtype=int)
    empty_arr = np.zeros_like(relevant_act_arr)

    all_act_time_series_no = assign_time_series_to_act_ex(segment, actex_wo_first_home, empty_arr, logging)

    act_time_series_no = all_act_time_series_no[relevant_act_arr == 1]
    assert act_time_series_no.shape[0] == num_tours

    time_series_dict = load_time_series_dict(v, np.unique(all_act_time_series_no), logging)

    # 3. sort tour according to SBBs' logic
    relevant_acts = v.Net.ActivityExecutions.GetFilteredSet(filter_condition)
    dur_before_major_act, dur_after_major_act = get_duration_before_after_major_act(relevant_acts)
    assert dur_before_major_act.shape[0] == dur_after_major_act.shape[0] == num_tours

    tour_order_nos = set_tour_orders(v, dur_before_major_act + dur_after_major_act, logging)
    assert tour_order_nos.shape[0] == num_tours

    # 4. return infos to perform time of day scheduling
    return tour_order_nos, time_series_dict, act_time_series_no, dur_before_major_act, dur_after_major_act, all_act_time_series_no


def init(v, logging):
    logging.info('initialising all relevant variables')
    v.Net.ActivityExecutions.SetAllAttValues("StartTime", None)
    v.Net.Trips.SetAllAttValues("SchedDepTime", None)
    v.Net.Tours.SetAllAttValues("start_times_order_number", 0)
    v.Net.Tours.SetAllAttValues("valid_start_times", False)


def assign_time_series_to_act_ex(segment, acts, empty_arr, logging):
    logging.info('assigning time series number for all activities')

    act_time_series_no = np.copy(empty_arr)
    for ind, filt_expr in enumerate(segment["AttrExpr"], start=0):
        filt_expr = filt_expr.replace("_A_", "ActivityCode")
        filt_expr = filt_expr.replace("_D_", "FromTrip\\distance")
        filt_expr = filt_expr.replace("_P_", "Schedule\\Person")
        # logging.info('assigning time series number for all activities in %s' % (segment["Subgroup_Comment"][ind]))
        time_series_no = int(float(segment["TimeSeriesNo"][ind]))
        _arr = np.array(utilities.GetMultiByFormula(acts, 'IF(%s,%d,0)' % (filt_expr, time_series_no)))
        act_time_series_no[_arr > 0] = _arr[_arr > 0]

    assert len(act_time_series_no[act_time_series_no == 0]) == 0
    return act_time_series_no


def load_time_series_dict(v, unique_time_series, logging):
    if int(v.VersionNumber[0:2]) >= 21:
        share_factor = 100
    else:
        share_factor = 1

    logging.info('loading probabilities for all time series')
    time_series_dict = collections.defaultdict(list)
    for time_series_no in sorted(unique_time_series):
        time_series_items = v.Net.TimeSeriesCont.ItemByKey(time_series_no).TimeSeriesItems.GetAll
        for time_series_item in time_series_items:
            time_series_dict[time_series_no].append(
                [time_series_item.AttValue(attr)
                 for attr in ["StartTime", "EndTime"]] + [time_series_item.AttValue("Share") * share_factor])

    time_series_dict_arr = collections.defaultdict(np.array)
    for k, val in time_series_dict.items():
        time_series_dict_arr[k] = np.array(val)

    return time_series_dict_arr


def get_duration_before_after_major_act(relevant_acts):
    tour_data_1 = np.array(utilities.GetMulti(relevant_acts,
                                              r'FromTrip\Tour\Sum:Trips([ToActivityExecution\Index] < ([Tour\main_activity_execution_index] + 1))\Duration'))
    tour_data_2 = np.array(utilities.GetMulti(relevant_acts,
                                              r'FromTrip\Tour\Sum:Trips([ToActivityExecution\Index] < ([Tour\main_activity_execution_index] + 1))\FromActivityExecution\Duration'))
    tour_data_3 = np.array(utilities.GetMulti(relevant_acts,
                                              r'FromTrip\Tour\Sum:Trips([FromActivityExecution\Index] >= [Tour\main_activity_execution_index])\Duration'))
    tour_data_4 = np.array(utilities.GetMulti(relevant_acts,
                                              r'FromTrip\Tour\Sum:Trips([FromActivityExecution\Index] >= [Tour\main_activity_execution_index])\FromActivityExecution\Duration'))
    dur_before_major_act = tour_data_1 + tour_data_2
    dur_after_major_act = tour_data_3 + tour_data_4

    return dur_before_major_act, dur_after_major_act  # time in seconds


def set_tour_orders(v, tot_tour_durations, logging):
    all_tours = v.Net.Tours

    person_nos = np.array(utilities.GetMulti(all_tours, r'Schedule\PersonNo'))
    tour_no = np.array(utilities.GetMulti(all_tours, r'No'))
    major_act_id = np.array(utilities.GetMulti(all_tours, r'main_activity_id'))
    major_act_id_to_ranking = {2: 10, 1: 9, 3: 8, 0: 7}  # todo: very SBB specific...
    major_act_id_order_no = np.array([major_act_id_to_ranking[x] for x in major_act_id])

    logging.info('setting tour ranking according to main activity and total tour durations')

    assert person_nos.shape[0] == tour_no.shape[0] == major_act_id_order_no.shape[0] == tot_tour_durations.shape[0]

    person_tour_dict = collections.defaultdict(list)
    for i, no in enumerate(person_nos):
        person_tour_dict[no].append([major_act_id_order_no[i], tot_tour_durations[i], tour_no[i]])

    def get_elements(elem):
        return elem[0], elem[1]

    tour_order_nos = []
    for tour_info in person_tour_dict.values():
        tour_info.sort(key=get_elements, reverse=True)
        tour_order_nos.extend([a[2] for a in tour_info])

    tour_order_nos = np.array(tour_order_nos, dtype=int)
    # all tours must have a valid order no
    assert tour_order_nos[tour_order_nos == 0].shape[0] == 0
    return tour_order_nos


def run_time_of_day_choice(v, tour_order_nos, dur_before_major_act, dur_after_major_act,
                           rel_act_time_series_no, all_act_time_series, time_series_dict, segment, rand, logging):
    time_series_to_expbeta = dict(zip(segment['TimeSeriesNo'].astype(float),
                                      segment['ExpBeta'].astype(float)))
    min_time_intervals = next(iter(time_series_dict.values()))[:, 0]

    # get all the necessary information
    trip_info = np.concatenate((np.array(utilities.GetMulti(v.Net.Trips, r'Index'))[:, np.newaxis],
                                np.array(utilities.GetMulti(v.Net.Trips, r'Duration'))[:, np.newaxis],
                                np.array(utilities.GetMulti(v.Net.Trips, r'PersonNo'))[:, np.newaxis]),
                               axis=1).astype(int)
    person_nos, trips_to_persons = np.unique(trip_info[:, 2], return_inverse=True)

    actex_wo_first_home = v.Net.ActivityExecutions.GetFilteredSet(r'[FromTrip\Index] >= 1')
    actex_info = np.concatenate((np.array(utilities.GetMulti(actex_wo_first_home, r'FromTrip\Index'))[:, np.newaxis],
                                 np.array(utilities.GetMulti(actex_wo_first_home, r'Duration'))[:, np.newaxis],
                                 np.array(utilities.GetMulti(actex_wo_first_home, r'PersonNo'))[:, np.newaxis]),
                                axis=1).astype(int)
    person_nos, actex_to_persons = np.unique(actex_info[:, 2], return_inverse=True)

    max_trip_index_per_tour = np.concatenate(
        (np.array(utilities.GetMulti(v.Net.Tours, r'Max:Trips\Index'))[:, np.newaxis],
         np.array(utilities.GetMulti(v.Net.Tours, r'PersonNo'))[:, np.newaxis],
         np.array(utilities.GetMulti(v.Net.Tours, r'No'))[:, np.newaxis]),
        axis=1).astype(int)
    person_nos, tours_to_persons = np.unique(max_trip_index_per_tour[:, 1], return_inverse=True)
    nb_tours_per_person = np.bincount(tours_to_persons)

    # this is a filter for unfinished plans
    person_complete = np.zeros(person_nos.shape[0], dtype=int)
    person_best_score = np.zeros(person_nos.shape[0], dtype=float)
    person_best_valid_tours = np.zeros(person_nos.shape[0], dtype=int)

    final_trip_times = np.zeros(trip_info.shape[0], dtype=int)
    final_actex_times = np.zeros(actex_info.shape[0], dtype=int)
    final_valid_tours = np.zeros(max_trip_index_per_tour.shape[0], dtype=int)

    # start with the iterations here and process unfinished plans only
    for iteration in range(1, 101):
        uncompl_persons = (person_complete == 0)
        unf_persons = person_complete[uncompl_persons]
        if unf_persons.shape[0] == 0:
            break
        logging.info(
            ' - iteration %d of start time choice with %d unfinished persons' % (iteration, unf_persons.shape[0]))

        # trip information
        unf_trip_mask = (person_complete[trips_to_persons] == 0)
        unf_trips = trip_info[unf_trip_mask]

        unf_trips_to_persons = trips_to_persons[unf_trip_mask]
        _, unf_trips_to_persons_reind = np.unique(unf_trips_to_persons, return_inverse=True)

        # activity execution information
        unf_act_ex_mask = (person_complete[actex_to_persons] == 0)
        unf_act_ex = actex_info[unf_act_ex_mask]

        unf_act_ex_to_persons = actex_to_persons[unf_act_ex_mask]
        _, unf_act_ex_to_persons_reind = np.unique(unf_act_ex_to_persons, return_inverse=True)

        # tour information
        unf_tour_mask = (person_complete[tours_to_persons] == 0)
        unf_tour_max_trip_ind = max_trip_index_per_tour[unf_tour_mask][:, 0]

        unf_act_time_series = rel_act_time_series_no[unf_tour_mask]
        unf_tour_order_nos = tour_order_nos[unf_tour_mask]
        unf_dur_before_major_act = dur_before_major_act[unf_tour_mask]
        unf_dur_after_major_act = dur_after_major_act[unf_tour_mask]

        unf_tours_to_persons = tours_to_persons[person_complete[tours_to_persons] == 0]
        _, unf_tours_to_persons_reind = np.unique(unf_tours_to_persons, return_inverse=True)

        # choose start time for every tour
        temp_tour_start_time_choices, temp_tour_valid_choices = run_tour_start_time_choices(unf_tours_to_persons,
                                                                                            unf_act_time_series,
                                                                                            unf_dur_before_major_act,
                                                                                            unf_dur_after_major_act,
                                                                                            unf_tour_order_nos,
                                                                                            time_series_dict, rand)

        # we now have a start time for each tour -> assign start times to trips and activity executions
        temp_trip_times = np.zeros(unf_trips.shape[0], dtype=int)
        temp_actex_times = np.zeros(unf_act_ex.shape[0], dtype=int)
        for i in range(1, np.max(unf_tour_max_trip_ind) + 1):
            cur_times = temp_tour_start_time_choices[i <= unf_tour_max_trip_ind]
            cur_times[cur_times <= 0] = 0

            cur_trips = unf_trips[unf_trips[:, 0] == i]
            cur_actex = unf_act_ex[unf_act_ex[:, 0] == i]

            temp_trip_times[unf_trips[:, 0] == i] = cur_times
            cur_times[cur_times > 0] = cur_times[cur_times > 0] + cur_trips[cur_times > 0][:, 1]
            temp_actex_times[unf_act_ex[:, 0] == i] = cur_times
            cur_times[cur_times > 0] = cur_times[cur_times > 0] + cur_actex[cur_times > 0][:, 1]

            temp_tour_start_time_choices[i <= unf_tour_max_trip_ind] = cur_times

        # if the plan contains more valid tours than the best plan -> take this plan
        person_sum_valid_tours = np.bincount(unf_tours_to_persons_reind, weights=temp_tour_valid_choices).astype(int)
        update_choice = np.zeros(unf_persons.shape[0], dtype=int)

        # for each start time, get the index of the corresponding time interval
        start_time_indices = np.searchsorted(min_time_intervals, temp_actex_times, side='right') - 1
        # get probs and betas for each choice
        probs_post_choice = np.array([[3600 / (time_series_dict[series][start_time_indices[ind], 1] - time_series_dict[series][start_time_indices[ind], 0]) *
                                       time_series_dict[series][start_time_indices[ind], 2] / 100,
                                       time_series_to_expbeta[series]]
                                      for ind, series
                                      in enumerate(all_act_time_series[unf_act_ex_mask])])
        # score choice
        scores_per_act_ex = np.exp(probs_post_choice[:, 0] * probs_post_choice[:, 1])
        scores_per_unf_person = np.bincount(unf_act_ex_to_persons_reind, weights=scores_per_act_ex)

        # do binary choice
        binary_choice_probs = 1 / (1 + np.exp(-5 * (scores_per_unf_person - person_best_score[uncompl_persons])))
        random_numbers = rand.rand(binary_choice_probs.shape[0])
        update_choice[(person_sum_valid_tours > person_best_valid_tours[uncompl_persons]) |
                      ((binary_choice_probs > random_numbers) &
                       (person_sum_valid_tours == person_best_valid_tours[uncompl_persons]))] = 1

        __ = np.copy(person_best_score[uncompl_persons])
        __[update_choice == 1] = scores_per_unf_person[update_choice == 1]
        person_best_score[uncompl_persons] = __
        logging.info(
            ' -- avg. updated score: %f; avg. best score: %f' % (np.average(scores_per_unf_person[update_choice == 1]),
                                                                 np.average(person_best_score)))

        __ = np.copy(person_best_valid_tours[uncompl_persons])
        __[update_choice == 1] = person_sum_valid_tours[update_choice == 1]
        person_best_valid_tours[uncompl_persons] = __
        logging.info(' --- updating choice of ' + str(update_choice[update_choice == 1].shape[0]) + ' persons')

        # add to final trip choice or not
        upd_trip_mask = (update_choice[unf_trips_to_persons_reind] == 1)
        __ = np.copy(final_trip_times[unf_trip_mask])
        __[upd_trip_mask] = temp_trip_times[upd_trip_mask]
        final_trip_times[unf_trip_mask] = __

        upd_act_ex_mask = (update_choice[unf_act_ex_to_persons_reind] == 1)
        __ = np.copy(final_actex_times[unf_act_ex_mask])
        __[upd_act_ex_mask] = temp_actex_times[upd_act_ex_mask]
        final_actex_times[unf_act_ex_mask] = __

        upd_tour_mask = (update_choice[unf_tours_to_persons_reind] == 1)
        __ = np.copy(final_valid_tours[unf_tour_mask])
        __[upd_tour_mask] = temp_tour_valid_choices[upd_tour_mask]
        final_valid_tours[unf_tour_mask] = __

        min_iterations = 5
        if iteration >= min_iterations:
            __ = np.copy(unf_persons)
            __[nb_tours_per_person[uncompl_persons] == person_sum_valid_tours] = 1
            person_complete[uncompl_persons] = __

    return final_valid_tours, final_trip_times, final_actex_times


def run_tour_start_time_choices(unf_tours_to_persons, unf_act_time_series, unf_dur_before_major_act,
                                unf_dur_after_major_act, unf_tour_order_nos, time_series_dict, rand):
    temp_tour_start_time_choices = np.zeros(unf_act_time_series.shape[0], dtype=int)
    temp_tour_valid_choices = np.zeros(unf_act_time_series.shape[0], dtype=int)

    # Latest start time (last home activity) must be within 25h
    free_times = collections.defaultdict(lambda: [(0, 25 * 60 * 60)])

    for tour_ind in range(1, max(unf_tour_order_nos) + 1):
        cur_tours = unf_act_time_series[unf_tour_order_nos == tour_ind]

        # print(' --- tour order number %d: choosing start times for %d tours' % (tour_ind, cur_tours.shape[0]))

        cur_tours_before_dur = unf_dur_before_major_act[unf_tour_order_nos == tour_ind].astype(int)
        cur_tours_after_dur = unf_dur_after_major_act[unf_tour_order_nos == tour_ind].astype(int)
        cur_tours_person_ind = unf_tours_to_persons[unf_tour_order_nos == tour_ind]

        # get weight_matrix
        weight_matrix, intersection_intervals = get_weight_mat(free_times, cur_tours, cur_tours_person_ind,
                                                               cur_tours_before_dur, cur_tours_after_dur,
                                                               time_series_dict)

        start_times_vector, is_tour_valid_vector = make_choices(weight_matrix, intersection_intervals,
                                                                cur_tours_before_dur, rand)

        free_times = update_free_times(free_times, start_times_vector, is_tour_valid_vector,
                                       cur_tours_before_dur + cur_tours_after_dur, cur_tours_person_ind)

        temp_tour_start_time_choices[unf_tour_order_nos == tour_ind] = start_times_vector
        temp_tour_valid_choices[unf_tour_order_nos == tour_ind] = is_tour_valid_vector

    return temp_tour_start_time_choices, temp_tour_valid_choices


def make_choices(weight_matrix, intersection_intervals, cur_tours_before_dur, rand):
    weight_row_sums = weight_matrix.sum(axis=1)
    # set zero rows to ones to avoid warnings => reset choice to -1 afterwards
    weight_matrix[weight_row_sums == 0] = np.ones(weight_matrix.shape[1])
    prob_matrix = weight_matrix / weight_matrix.sum(axis=1)[:, np.newaxis]

    _choice = Choice2D(prob_matrix.shape[0])
    _choice.add_prob(prob_matrix)
    chosen_ind = _choice.choose(rand)

    # reset choice
    chosen_ind[weight_row_sums == 0] = -1

    start_times_vector = np.zeros(chosen_ind.shape[0], dtype=int)
    is_tour_valid_vector = np.ones(chosen_ind.shape[0], dtype=int)
    interval_nb = rand.rand(chosen_ind.shape[0])

    for j, chosen_index in enumerate(chosen_ind):
        # handle invalid choices
        if chosen_index == -1:
            start_times_vector[j] = 0
            is_tour_valid_vector[j] = 0
            continue

        possible_intervals = intersection_intervals[j][chosen_index]
        if len(possible_intervals) > 1:
            chosen_interval = rand.choice(possible_intervals)
        else:
            chosen_interval = possible_intervals[0]

        interval_start = chosen_interval[0]
        interval_end = chosen_interval[1]
        precise_time = int(interval_start + (interval_end - interval_start) * interval_nb[j])
        start_times_vector[j] = precise_time - cur_tours_before_dur[j]

    return start_times_vector, is_tour_valid_vector


def get_weight_mat(free_times, cur_tours, cur_tours_person_ind, cur_tours_before_dur, cur_tours_after_dur,
                   time_series_dict):
    weight_mat = np.zeros((cur_tours.shape[0], next(iter(time_series_dict.values())).shape[0]))
    intersecting_intervals = collections.defaultdict(lambda: collections.defaultdict(list))

    for tour_index, cur_tour in enumerate(cur_tours):
        person_ind = cur_tours_person_ind[tour_index]
        current_time_series = time_series_dict[cur_tours[tour_index]]
        pre_duration = cur_tours_before_dur[tour_index]
        after_duration = cur_tours_after_dur[tour_index]

        for free_times_interval_begin, free_times_interval_end in free_times[person_ind]:
            constrained_begin = free_times_interval_begin + pre_duration + 15 * 60  # buffer for home act
            constrained_end = free_times_interval_end - after_duration - 15 * 60  # buffer for home act
            if constrained_end - constrained_begin > 0:
                for time_series_item_index, (item_begin, item_end, item_share) in enumerate(current_time_series):
                    intersecting_interval_begin = max(constrained_begin, item_begin)
                    intersecting_interval_end = min(constrained_end, item_end)
                    intersecting_interval_length = intersecting_interval_end - intersecting_interval_begin
                    if intersecting_interval_length > 0:
                        weight_mat[tour_index, time_series_item_index] = item_share
                        intersecting_intervals[tour_index][time_series_item_index].append(
                            (intersecting_interval_begin, intersecting_interval_end))

    return weight_mat, intersecting_intervals


def update_free_times(free_times, start_times, is_valid,
                      cur_tours_durations, cur_tours_person_ind):
    for j, start_time in enumerate(start_times):
        if is_valid[j] == 0:
            continue

        blocked_interval_begin = start_time
        blocked_interval_end = start_time + cur_tours_durations[j]
        personNo = cur_tours_person_ind[j]

        for interval_index, (interval_begin, interval_end) in enumerate(free_times[personNo]):
            # blocked interval is outside of interval
            if blocked_interval_begin >= interval_end or blocked_interval_end <= interval_begin:
                continue
            # blocked interval is inside of interval
            elif blocked_interval_begin > interval_begin and blocked_interval_end < interval_end:
                # split interval
                free_times[personNo][interval_index] = (interval_begin, blocked_interval_begin)
                free_times[personNo].append((blocked_interval_end, interval_end))
            # blocked interval surrounds interval
            elif blocked_interval_begin <= interval_begin and blocked_interval_end >= interval_end:
                del free_times[personNo][interval_index]
            # blocked interval reduces interval from the right
            elif blocked_interval_begin > interval_begin:
                assert blocked_interval_end >= interval_end
                free_times[personNo][interval_index] = (interval_begin, blocked_interval_begin)
            # blocked interval reduces interval from the left
            elif blocked_interval_end < interval_end:
                assert blocked_interval_begin <= interval_begin
                free_times[personNo][interval_index] = (blocked_interval_end, interval_end)

        # sort time intervals
        free_times[personNo].sort(key=operator.itemgetter(0))
    return free_times


def write_results_and_sort_tours(v, logging):
    # handle invalid tours
    invalid_trips = v.Net.Trips.GetFilteredSet(r'[Tour\valid_start_times]=0')
    num_invalid_trips = invalid_trips.Count
    invalid_trips.SetMultipleAttributes([r'SchedDepTime'],
                                        np.array([None] * num_invalid_trips)[:, np.newaxis])
    invalid_trips.SetMultipleAttributes([r'ToActivityExecution\StartTime'],
                                        np.array([None] * num_invalid_trips)[:, np.newaxis])

    # reset all home activities
    home_actExs = v.Net.ActivityExecutions.GetFilteredSet(
        r'[ActivityCode]="H" & [ToTrip\TourNo] > 0')  # get only activity executions with existing ToTrip
    num_home_actExs = home_actExs.Count
    if num_home_actExs > 0:
        utilities.SetMulti(home_actExs, r"ToTrip\FromActivityExecutionIndex", np.array([None] * num_home_actExs))

    # order tours according to start times and set the order number
    first_trips_in_valid_tours = v.Net.Trips.GetFilteredSet(r'[Index]=[Tour\First:Trips\Index] & [SchedDepTime]>0')
    num_trip_start_times = first_trips_in_valid_tours.Count
    order_nos = np.zeros(num_trip_start_times, dtype=int)

    trip_start_times = np.concatenate((np.array(utilities.GetMulti(first_trips_in_valid_tours, r'PersonNo'))[:, np.newaxis],
                                       np.array(utilities.GetMulti(first_trips_in_valid_tours, r'TourNo'))[:, np.newaxis],
                                       np.array(utilities.GetMulti(first_trips_in_valid_tours, r'SchedDepTime'))[:, np.newaxis]),
                                      axis=1)
    start_times_by_person = collections.defaultdict(list)
    for index_in_container, (personNo, tourNo, start_time) in enumerate(trip_start_times):
        start_times_by_person[personNo].append((index_in_container, tourNo, start_time))
    for person_data in start_times_by_person.values():
        person_data.sort(key=operator.itemgetter(2))
    for personNo, tours in start_times_by_person.items():
        for order_no, (index_in_container, tourNo, start_time) in enumerate(tours, start=1):
            order_nos[index_in_container] = order_no

    utilities.SetMulti(first_trips_in_valid_tours, r"Tour\start_times_order_number", order_nos)

    # concatenate tours according to order number
    cur_home_actExs = collections.defaultdict(lambda: 1)  # all tours start with activity execution index = 1
    max_order_no = np.max(order_nos)

    for order_no in range(1, max_order_no + 1):
        first_trips_cur_order_no = v.Net.Trips.GetFilteredSet(
            r'[Tour\start_times_order_number]=%d & [Tour\valid_start_times]=1 & [Index]=[Tour\First:Trips\Index]' % order_no)
        num_first_trips_cur_order_no = first_trips_cur_order_no.Count

        logging.info('Concatenate valid tours: order_no=%d (%d objects)' % (order_no, num_first_trips_cur_order_no))
        if num_first_trips_cur_order_no > 0:
            trip_data = np.concatenate((np.array(utilities.GetMulti(first_trips_cur_order_no, r'PersonNo'))[:, np.newaxis],
                                        np.array(utilities.GetMulti(first_trips_cur_order_no, r'Tour\Last:Trips\ToActivityExecutionIndex'))[:, np.newaxis],
                                        np.array(utilities.GetMulti(first_trips_cur_order_no, r'SchedDepTime'))[:, np.newaxis]),
                                       axis=1).astype(int)
            from_actExs = [cur_home_actExs[personNo] for personNo, to_actEx_index, schedDepTime in trip_data]
            sched_dep_times = [sched_dep_time for personNo, to_actEx_index, sched_dep_time in trip_data]

            # concatenate first trip with current home activity execution
            utilities.SetMulti(first_trips_cur_order_no, r"FromActivityExecutionIndex", np.array(from_actExs))

            if order_no == 1:
                # set startTime of home activity execution (first home activity execution has duration=0)
                utilities.SetMulti(first_trips_cur_order_no, r"FromActivityExecution\StartTime",
                                   np.array(sched_dep_times))
            else:
                # set EndTime of home activity execution (has already valid StartTime! EndTime sets Duration implicitly)
                utilities.SetMulti(first_trips_cur_order_no, r"FromActivityExecution\EndTime",
                                   np.array(sched_dep_times))

            for personNo, to_actEx_index, schedDepTime in trip_data:
                cur_home_actExs[personNo] = to_actEx_index

    # last step of the scheduling: resorting the schedules
    v.Net.Schedules.SortEachSchedule()
