import logging
import collections
import operator

import numpy as np

from src import abm_utilities, visum_utilities, choice_engine
from settings import abm_settings

def run(segments, Visum):
    # start time choice of major activtiy executions (not for subtours)
    init(Visum)

    # only one segment in this example
    assert len(segments) == 1
    segment = segments[0]

    num_tours = Visum.Net.Tours.Count

    # for tours containing a subtour, the major activity is split up
    # => choose start time for the first appearance of the major activity, which is the one not preceeded by a subtour activity
    filter_condition = r'([IsMajorActivity] = 1) & ([FromTrip\FromActivityExecution\IsPartOfSubtour] != 1)'
    major_actExs_of_tours = Visum.Net.ActivityExecutions.GetFilteredSet(filter_condition)
    num_major_activity_executions = major_actExs_of_tours.Count

    logging.info('start time choice model: %d tours with %d major activity executions', num_tours, num_major_activity_executions)

    # actExs_for_person contains list of: [index_in_filtered_subjects,
    #                                      subsegment_index,
    #                                      ActExIndex,
    #                                      'FromTrip\Tour\is_primary',
    #                                      'Duration',
    #                                      'FromTrip\Tour\No']
    time_series_dict, actExs_data_per_person = load_subsegment_data(Visum, major_actExs_of_tours, segment)

    logging.info('start time choice model: order tours by longest major activity execution, primary tours first (%d persons)', len(actExs_data_per_person))
    set_tour_orders_by_longest_major_actEx(Visum, actExs_data_per_person, num_tours)

    subsegment_indices, time_series_shares_for_actEx = get_all_time_series_shares_for_actExs(actExs_data_per_person, time_series_dict)

    choose_start_times(Visum, major_actExs_of_tours, subsegment_indices, time_series_shares_for_actEx, time_series_dict)

    _sort_tours_by_major_act_ex_start_time(Visum)


def _sort_tours_by_major_act_ex_start_time(Visum):
    """
    Sort tours according to start time of major activity execution.

    NOTE: `MajorActExIndex` is being updated according to the applied shift.
    """
    # store tour-relative index of second activity execution in tour
    second_act_ex_index_in_tour_before_sort = np.array(visum_utilities.GetMulti(Visum.Net.Tours, r'FIRST:TRIPS\TOACTIVITYEXECUTION\INDEX'), dtype=int)

    # sort schedules by major activity start time
    # (which at this point is the only activity execution with a defined start time in each tour)
    Visum.Net.Schedules.SortEachSchedule()

    # obtain difference of indices => major activity of tour was shifted by the same value
    second_act_ex_index_in_tour_after_sort = np.array(visum_utilities.GetMulti(Visum.Net.Tours, r'FIRST:TRIPS\TOACTIVITYEXECUTION\INDEX'), dtype=int)
    act_ex_index_diff_in_tour = second_act_ex_index_in_tour_after_sort - second_act_ex_index_in_tour_before_sort

    major_act_ex_index = np.array(visum_utilities.GetMulti(Visum.Net.Tours, r'MAJORACTEXINDEX'), dtype=int) + act_ex_index_diff_in_tour
    visum_utilities.SetMulti(Visum.Net.Tours, r'MAJORACTEXINDEX', major_act_ex_index)

    # re-connect home ActivityExecutions to their new ToTrips
    disconnected_trips_from_home = Visum.Net.Trips.GetFilteredSet(r'[FromActivityExecutionIndex]=0')
    if disconnected_trips_from_home.Count > 0:
        to_act_ex_index = np.array(visum_utilities.GetMulti(disconnected_trips_from_home, r"ToActivityExecutionIndex", chunk_size=10000000, reindex = True), dtype=int)
        # activity executions where renumbered according to tour order
        from_act_ex_index = to_act_ex_index - 1
        visum_utilities.SetMulti(disconnected_trips_from_home, r"FromActivityExecutionIndex", from_act_ex_index, chunk_size=10000000)


def init(Visum):
    Visum.Net.ActivityExecutions.SetAllAttValues("StartTime", None)
    Visum.Net.Trips.SetAllAttValues("SchedDepTime", None)
    visum_utilities.insert_UDA_if_missing(Visum.Net.Tours, "StartTimeChoice_order_no")
    Visum.Net.Tours.SetAllAttValues("StartTimeChoice_order_no", None)
    visum_utilities.insert_UDA_if_missing(Visum.Net.Tours, "Order_no")
    Visum.Net.Tours.SetAllAttValues("Order_no", None)

    # temporarily disconnect home ActivityExecutions from their ToTrips to allow re-ordering after start time choice
    home_actExs = Visum.Net.ActivityExecutions.GetFilteredSet(r'[ActivityCode]="H" & [ToTrip\TourNo] > 0') # get only activity executions with existing ToTrip
    num_home_actExs = home_actExs.Count
    if num_home_actExs > 0:
        visum_utilities.SetMulti(home_actExs, r"ToTrip\FromActivityExecutionIndex", [None] * num_home_actExs, chunk_size=10000000)


def load_subsegment_data(Visum, major_actExs_of_tours, segment):
    act_ex_person_nos = visum_utilities.GetMulti(major_actExs_of_tours, 'PersonNo', chunk_size=abm_settings.chunk_size_trips, reindex = True)
    act_ex_indices = visum_utilities.GetMulti(major_actExs_of_tours, 'Index', chunk_size=abm_settings.chunk_size_trips)
    container_data = np.column_stack([act_ex_person_nos, act_ex_indices])
    indices_in_container = collections.defaultdict(lambda: collections.defaultdict(list))

    for index_in_container, (personNo, actExIndex) in enumerate(container_data, start=0):
        indices_in_container[personNo][actExIndex] = index_in_container

    time_series_nos_by_subsegment = dict()
    actExs_data_per_person = collections.defaultdict(list)

    for subsegment_index, sub_segment_filter_expr in enumerate(segment["AttrExpr"]):
        major_actExs_for_sub_segment = abm_utilities.get_filtered_subjects(major_actExs_of_tours, sub_segment_filter_expr)

        num_major_actExs_for_sub_segment = major_actExs_for_sub_segment.Count

        logging.info('  load data for subgroup "%s" (%d objects)', segment['Subgroup_Comment'][subsegment_index], num_major_actExs_for_sub_segment)

        if num_major_actExs_for_sub_segment > 0:
            sub_seg_act_ex_person_nos = visum_utilities.GetMulti(
                major_actExs_for_sub_segment, 'PersonNo', chunk_size=abm_settings.chunk_size_trips, reindex = True)
            sub_seg_act_ex_indices = visum_utilities.GetMulti(
                major_actExs_for_sub_segment, 'Index', chunk_size=abm_settings.chunk_size_trips)
            sub_seg_act_ex_tour_is_primary = visum_utilities.GetMulti(
                major_actExs_for_sub_segment, r'FromTrip\Tour\is_primary', chunk_size=abm_settings.chunk_size_trips)
            sub_seg_act_ex_duraction = visum_utilities.GetMulti(
                major_actExs_for_sub_segment, 'Duration', chunk_size=abm_settings.chunk_size_trips)
            sub_seg_act_ex_tour_no = visum_utilities.GetMulti(
                major_actExs_for_sub_segment, r'FromTrip\TourNo', chunk_size=abm_settings.chunk_size_trips)

            actExs_data = np.column_stack([sub_seg_act_ex_person_nos, sub_seg_act_ex_indices,
                                          sub_seg_act_ex_tour_is_primary, sub_seg_act_ex_duraction, sub_seg_act_ex_tour_no])

            for actEx_data in actExs_data:
                personNo = int(actEx_data[0])
                actEx_index = int(actEx_data[1])
                index_in_container = indices_in_container[personNo][actEx_index]
                actExs_data_per_person[personNo].append([index_in_container, subsegment_index] + list(actEx_data[1:])) # all but personNo
            time_series_nos_by_subsegment[subsegment_index] = segment["TimeSeriesNo"][subsegment_index]

    time_series_dict = load_time_series_dict(Visum, time_series_nos_by_subsegment)

    return time_series_dict, actExs_data_per_person


def load_time_series_dict(Visum, time_series_nos_by_subsegment):
    time_series_dict = collections.defaultdict(list)

    # data structures for checking input data
    time_intervals = Visum.Net.CalendarPeriod.AnalysisTimeIntervalSet.TimeIntervals
    time_interval_start_times = np.array(visum_utilities.GetMulti(time_intervals, 'StartTime'), dtype=int)

    for subsegment_index, time_series_no in sorted(time_series_nos_by_subsegment.items()):
        time_series_items = Visum.Net.TimeSeriesCont.ItemByKey(time_series_no).TimeSeriesItems.GetAll
        time_series_items_of_subsegment = time_series_dict[subsegment_index]
        for time_series_item in time_series_items:
            time_series_items_of_subsegment.append([time_series_item.AttValue(attr)
                                                    for attr in ["StartTime", "EndTime", "Share"]])

        # check for model error: that time series items are contained in AP
        if any (intervalData[2] > 0 and intervalData[0] < time_interval_start_times[0] for intervalData in time_series_items_of_subsegment):
            logging.error("time series '%s' used for start time choice lies outside of the analysis time intervals", time_series_no)

    return time_series_dict


def set_tour_orders_by_longest_major_actEx(Visum, actExs_data_per_person, num_tours):
    all_tours = Visum.Net.Tours
    # get no and personNo from all tours
    tour_nos = visum_utilities.GetMulti(all_tours, 'No', chunk_size=abm_settings.chunk_size_trips, reindex = True)
    person_nos = visum_utilities.GetMulti(all_tours, r'Schedule\PersonNo', chunk_size=abm_settings.chunk_size_trips)

    # build dict
    tour_indices_dict = collections.defaultdict(dict)
    for tour_index, (tourNo, personNo) in enumerate(zip(tour_nos, person_nos), start=0):
        tour_indices_dict[personNo][tourNo] = tour_index

    # order tours by (1) isPrimary and (2) longest major activity execution
    STC_order_nos = np.zeros(num_tours, dtype=int)
    for personNo in sorted(actExs_data_per_person):
        actExs_data_of_person = actExs_data_per_person[personNo]
        actExs_data_of_person.sort(key=operator.itemgetter(3, 4), reverse=True)  # index of isPrimaryTour = 3, index of duration = 4
        # actExs_data_of_person: [index_in_container,
        #                         subsegment_index,
        #                         ActExIndex,
        #                         r'FromTrip\Tour\is_primary',
        #                         'Duration',
        #                         r'FromTrip\Tour\No']
        tours_ordered = [actEx_data_of_person[5] for actEx_data_of_person in actExs_data_of_person]  # index of tourNo = 5

        for order_no, tourNo in enumerate(tours_ordered, start=1):
            tour_index = tour_indices_dict[personNo][tourNo]
            STC_order_nos[tour_index] = order_no

    # all tours must have a valid order no
    assert len(STC_order_nos[STC_order_nos == 0]) == 0
    visum_utilities.SetMulti(all_tours, "StartTimeChoice_order_no", STC_order_nos, chunk_size=10000000)


def get_all_time_series_shares_for_actExs(actExs_data_per_person, time_series_dict):
    subsegment_indices = collections.defaultdict(lambda: collections.defaultdict(int))
    time_series_shares_for_actEx = collections.defaultdict(lambda: collections.defaultdict(list))

    for personNo, data_per_person in sorted(actExs_data_per_person.items()):
        for data_per_tour in data_per_person:
            subsegment_index = data_per_tour[1]
            actEx_index = data_per_tour[2]
            subsegment_indices[personNo][actEx_index] = subsegment_index
            time_series_shares_for_actEx[personNo][actEx_index] = [time_series[2] for time_series in time_series_dict[subsegment_index]]

    return subsegment_indices, time_series_shares_for_actEx


def choose_start_times(Visum, major_actExs_of_tours, subsegment_indices, time_series_shares_for_actEx, time_series_dict):
    rand = np.random.RandomState(42)

    max_STC_order_no = int(Visum.Net.AttValue(r'MAX:TOURS\StartTimeChoice_order_no'))
    for STC_order_no in range(1, max_STC_order_no + 1):
        major_actExs_for_order_no = major_actExs_of_tours.GetFilteredSet(fr'[FromTrip\Tour\StartTimeChoice_order_no]={STC_order_no}')
        assert major_actExs_for_order_no.Count > 0
        choose_start_times_for_order_no(rand,
                                        STC_order_no,
                                        major_actExs_for_order_no,
                                        subsegment_indices,
                                        time_series_shares_for_actEx,
                                        time_series_dict)


def choose_start_times_for_order_no(rand,
                                    STC_order_no,
                                    major_actExs_for_order_no,
                                    subsegment_indices,
                                    time_series_shares_for_actEx,
                                    time_series_dict):
    num_tours = major_actExs_for_order_no.Count
    logging.info('start time choice for tours with StartTimeChoice_order_no=%d (%d objects)', STC_order_no, num_tours)

    logging.info('  load time series data')

    # NOTE: first trip in tour has no FromActivityExecution at this point! => we need to use [ToActivityExecution\Index] < ([Tour\MajorActExIndex] + 1)
    major_act_ex_person_nos = visum_utilities.GetMulti(major_actExs_for_order_no, 'PersonNo', chunk_size=abm_settings.chunk_size_trips, reindex = True)
    major_act_ex_indices = visum_utilities.GetMulti(major_actExs_for_order_no, 'Index', chunk_size=abm_settings.chunk_size_trips)

    logging.info('  time interval choice for %d tours', num_tours)
    chosen_interval_indices = choose_time_intervals(major_act_ex_person_nos, major_act_ex_indices, time_series_shares_for_actEx)

    logging.info('  -> draw random start times in chosen intervals')
    start_times = choose_random_start_times_in_intervals(time_series_dict,
                                                         subsegment_indices,
                                                         chosen_interval_indices,
                                                         np.column_stack([major_act_ex_person_nos, major_act_ex_indices]),
                                                         num_tours,
                                                         rand)

    logging.info('  set start times of major activity executions (%d objects)', num_tours)
    visum_utilities.SetMulti(major_actExs_for_order_no, "StartTime", start_times, chunk_size=10000000)

def choose_time_intervals(act_ex_person_nos, act_ex_indices, time_series_shares_for_actEx):
    time_series_shares = np.array([time_series_shares_for_actEx[int(personNo)][int(index)]
                                   for personNo, index in zip(act_ex_person_nos, act_ex_indices)]) / 100
    scaled_time_series_shares_row_sums = time_series_shares.sum(1)
    scaled_time_series_shares_zero_rows = scaled_time_series_shares_row_sums == 0
    # set zero rows to ones to avoid warnings => reset choice to -1 afterwards
    time_series_shares[scaled_time_series_shares_zero_rows] = np.ones(time_series_shares.shape[1])
    time_series_shares = time_series_shares / time_series_shares.sum(1)[:, np.newaxis]
    chosen_indices = choice_engine.choose2D(time_series_shares)
    # reset choices for (invalid) zero rows
    chosen_indices[scaled_time_series_shares_zero_rows] = -1
    return chosen_indices



def choose_random_start_times_in_intervals(time_series_dict,
                                           subsegment_indices,
                                           chosen_indices,
                                           major_actExs_data,
                                           num_tours,
                                           rand):
    start_times = np.zeros(num_tours, dtype=int)

    for index_in_container, chosen_index in enumerate(chosen_indices):
        # chosen_index == -1 results from all-zero time series
        assert chosen_index != -1

        personNo, index = major_actExs_data[index_in_container]
        time_series = time_series_dict[subsegment_indices[personNo][index]]

        chosen_time_series_item = time_series[chosen_index]
        time_series_item_duration = chosen_time_series_item[1] - chosen_time_series_item[0]

        start_times[index_in_container] = np.floor(rand.rand() * time_series_item_duration + chosen_time_series_item[0])

    return start_times
