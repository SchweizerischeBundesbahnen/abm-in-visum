import logging

import numpy as np

import VisumPy.helpers as VPH

from src import abm_utilities, location_choice_engine, mode_choice_engine, visum_utilities
from src import config
from settings import abm_settings

def run(Visum, segmentsLocationChoice, segmentsModeChoice, config_object, zones, locations, zoneNo_to_zoneInd, locationNo_to_locationInd,
        time_interval_start_times, time_interval_end_times):
    activityID_to_activityCode = abm_utilities.build_activity_dict(Visum)
    nodeno_to_node = dict(zip(VPH.GetMulti(Visum.Net.Nodes, 'No'), Visum.Net.Nodes.GetAll))
    nodeno_to_longitude_latitude = abm_utilities.get_nodeno_to_longitude_latitude(Visum)
    
    logging.info('destination choice for minor activity executions (outbound stops, subtour stops, homebound stops)')
       
    # location is chosen for activity executions with [MinorDestAndModeChoiceSeqIndex] >= 1
    # example: activity executions of one tour:
    # Tour: [MajorActExIndex] = 14
    # (previous Tour)
    # 10 H                        ([MinorDestAndModeChoiceSeqIndex] = 0, [IsPartOfSubtour] = 0)
    # 12 ..                       ([MinorDestAndModeChoiceSeqIndex] = 2, [IsPartOfSubtour] = 0)
    # 13 ..                       ([MinorDestAndModeChoiceSeqIndex] = 1, [IsPartOfSubtour] = 0)
    # 14 W  Major Act (1st part)  ([MinorDestAndModeChoiceSeqIndex] = 0, [IsPartOfSubtour] = 1, [IsMajorActEx] = 1)
    # 15 .. Major Act of Subtour  ([MinorDestAndModeChoiceSeqIndex] = 1, [IsPartOfSubtour] = 1, [IsMajorActEx] = 0, [IsMajorActOfSubtour] = 1)
    # 16 ..                       ([MinorDestAndModeChoiceSeqIndex] = 2, [IsPartOfSubtour] = 1, [IsMajorActEx] = 0, [IsMajorActOfSubtour] = 0)
    # 17 W  Major Act (2nd part)  ([MinorDestAndModeChoiceSeqIndex] = 3, [IsPartOfSubtour] = 1, [IsMajorActEx] = 1)
    # 18 ..                       ([MinorDestAndModeChoiceSeqIndex] = 4, [IsPartOfSubtour] = 0)
    # 19 H                        ([MinorDestAndModeChoiceSeqIndex] = 0, [IsPartOfSubtour] = 0, [ActivityCode] = "H")

    relevant_act_exs = Visum.Net.ActivityExecutions.GetFilteredSet(r'[MinorDestAndModeChoiceSeqIndex] > 0')

    # attraction depends on the activity of the activity execution and the chosen location depends on the activity
    unique_activity_IDs = list(activityID_to_activityCode.keys())

    # impedance matrices may be chosen per analysis time interval
    time_interval_codes = VPH.GetMulti(Visum.Net.CalendarPeriod.AnalysisTimeIntervalSet.TimeIntervals, 'Code')

    mode_ids = np.array(VPH.GetMulti(Visum.Net.Modes, r'ID'), dtype=int)
    mode_id_to_interchangeable_raw = Visum.Net.Modes.GetMultipleAttributes(["ID", "Interchangeable"])
    mode_id_to_interchangeable = {int(mode_id): bool(is_interchangeable) for mode_id, is_interchangeable in mode_id_to_interchangeable_raw}
    mode_id_to_firstDSegCode_raw = Visum.Net.Modes.GetMultipleAttributes(["ID", r'First:DemandSegments\Code'])
    mode_id_to_firstDSegCode = {int(mode_id): dseg_code for mode_id, dseg_code in mode_id_to_firstDSegCode_raw}

    global_parameter = location_choice_engine.Location_Choice_Parameter_Global(
        config_object, zones, locations, zoneNo_to_zoneInd, locationNo_to_locationInd, time_interval_start_times, time_interval_end_times)

    prt_shortest_path_searcher = Visum.Net.CreatePrTShortestPathSearcher()

    num_chosen_locs = 0
    
    # dest zones are chosen consecutively (starting with [MinorDestAndModeChoiceSeqIndex] = 1) using rubberbanding    
    # NOTE: the different groups (outbound, subtour, inbound) can be calculated simultaneously, since their origin and 
    #       target location is already fixed: it is either the major act ex or home
    cur_local_act_ex_index = 1
    while True:
        act_ex_with_cur_local_index = relevant_act_exs.GetFilteredSet(r'[MinorDestAndModeChoiceSeqIndex]=' + str(cur_local_act_ex_index))
        if act_ex_with_cur_local_index.Count == 0:
            break # there cannot be any more activity executions with higher local index

        for (segmentLocationChoice, segmentModeChoice) in zip(segmentsLocationChoice, segmentsModeChoice):
            assert segmentLocationChoice['Filter'] == segmentModeChoice['Filter']
            cur_segment_act_ex = abm_utilities.get_filtered_subjects(act_ex_with_cur_local_index, segmentLocationChoice['Filter'])
            if cur_segment_act_ex.Count == 0:
                continue      
            
            # location choice depends on destination activity - for zone attraction
            for act_id in unique_activity_IDs:
                act_code = activityID_to_activityCode[act_id]
                curSegment_curAct_act_ex = cur_segment_act_ex.GetFilteredSet('[ActivityCode]="' + act_code + '"')
                minor_location_choice_for_activity(Visum, act_code,
                                                   mode_ids, mode_id_to_interchangeable, global_parameter, cur_local_act_ex_index,
                                                   segmentLocationChoice, curSegment_curAct_act_ex)
                num_chosen_locs += curSegment_curAct_act_ex.Count

            # mode choice does not depend on activity
            minor_mode_choice(global_parameter, prt_shortest_path_searcher, cur_segment_act_ex, nodeno_to_node, nodeno_to_longitude_latitude, time_interval_codes,
                              cur_local_act_ex_index, segmentModeChoice, mode_id_to_interchangeable, mode_id_to_firstDSegCode)

        cur_local_act_ex_index += 1

    assert num_chosen_locs == relevant_act_exs.Count

    # mode choice for trip to and from home location
    home_act_exs = Visum.Net.ActivityExecutions.GetFilteredSet(r'[ActivityCode] = "H"')
    for segmentModeChoice in segmentsModeChoice:
        cur_segment_home_act_exs = abm_utilities.get_filtered_subjects(home_act_exs, segmentModeChoice['Filter'])
        minor_mode_choice_for_home_trips(global_parameter, prt_shortest_path_searcher, cur_segment_home_act_exs, nodeno_to_node, nodeno_to_longitude_latitude, time_interval_codes,
                                         segmentModeChoice, mode_id_to_interchangeable, mode_id_to_firstDSegCode)

    update_main_mode_of_tours_with_interchangeable_main_mode(Visum)



def minor_location_choice_for_activity(Visum, act_code,
                                       mode_ids, mode_id_to_interchangeable, global_parameter, 
                                       cur_local_act_ex_index, segmentLocationChoice, curSegment_curAct_act_exs):

    if curSegment_curAct_act_exs.Count == 0:
        return
            
    logging.info('destination choice for minor activity executions (local index: %d, segment %s, activity %s): %s ',
        cur_local_act_ex_index, segmentLocationChoice['Specification'], act_code, segmentLocationChoice['Comment'])
            
    # toopt: attraction_per_zone can be precalculated for each act_id
    attraction_per_zone = abm_utilities.build_zone_index_to_attraction_array(Visum, act_code)
    num_chosen_locs = 0

    # consider main mode of tour for mode choice table term
    for main_mode_id in mode_ids:
        logging.info('destination choice for minor activity executions (local index: %d, segment: %s, activity: %s, main mode: %s): %s ',
            cur_local_act_ex_index, segmentLocationChoice['Specification'], act_code, main_mode_id, segmentLocationChoice['Comment'])
        if mode_id_to_interchangeable[main_mode_id]:
            # case 1: main mode is interchangeable
            # => all interchangeable modes are possible for every trip of the tour
            # NOTE: No need to differentiate between subtours and non-subtours, 
            #       since subtours do not have any additional mode option
            curSegment_curAct_curMode_act_exs = curSegment_curAct_act_exs.GetFilteredSet(
                r'[FromTrip\Tour\MainDSeg\Mode\ID] = ' + str(main_mode_id))

            choose_and_set_locations(global_parameter,
                                     segmentLocationChoice, act_code, attraction_per_zone,
                                     main_mode_id, curSegment_curAct_curMode_act_exs, allow_interchangable_modes=True)
            num_chosen_locs += curSegment_curAct_curMode_act_exs.Count
                                   
        else:
            # case 2: main mode is non-interchangeable
            # * non-subtour: only main mode can be used
            # * subtour: main mode and all interchangeable modes can be used

            # non-subtour destinations: use main mode only
            curSegment_curAct_curMode_act_exs = curSegment_curAct_act_exs.GetFilteredSet(
                r'[FromTrip\Tour\MainDSeg\Mode\ID] = ' + str(main_mode_id) + ' & [IsPartOfSubtour] = 0')
            choose_and_set_locations(global_parameter,
                                     segmentLocationChoice, act_code, attraction_per_zone,
                                     main_mode_id, curSegment_curAct_curMode_act_exs, allow_interchangable_modes=False)
            num_chosen_locs += curSegment_curAct_curMode_act_exs.Count

            # subtour destinations: allow main mode and all interchangeable modes
            # NOTE: first activity of subtour is considered the major activity of subtour
            #       => if the (non-interchangeable!) main mode is chosen as subtour major mode, 
            #          all subsequent subtour trips must use that mode
            if cur_local_act_ex_index == 1:
                # first activity of subtour (= major activity of subtour): 
                # => both main mode and all interchangable modes are available
                curSegment_curAct_curMode_act_exs = curSegment_curAct_act_exs.GetFilteredSet(
                    r'[FromTrip\Tour\MainDSeg\Mode\ID] = ' + str(main_mode_id) + ' & [IsPartOfSubtour] = 1')

                choose_and_set_locations(global_parameter,
                                         segmentLocationChoice, act_code, attraction_per_zone,
                                         main_mode_id, curSegment_curAct_curMode_act_exs, allow_interchangable_modes=True)
                num_chosen_locs += curSegment_curAct_curMode_act_exs.Count

            else:
                # major mode of subtour has been chosen:
                # * if it's interchangeable, all interchangeable modes are available again
                # * if it's the (non-interchangeable!) main mode of tour, that mode must be used

                # subtour major mode == main mode of tour: use main mode for whole subtour
                curSegment_curAct_curMode_act_exs = curSegment_curAct_act_exs.GetFilteredSet(
                                    r'[FromTrip\Tour\MainDSeg\Mode\ID] = ' + str(main_mode_id) + 
                                    ' & [IsPartOfSubtour] = 1 & ' +
                                    r'[FromTrip\Tour\MajorModeIDOfSubtour] = ' + str(main_mode_id))
                            
                choose_and_set_locations(global_parameter,
                                         segmentLocationChoice, act_code, attraction_per_zone,
                                         main_mode_id, curSegment_curAct_curMode_act_exs, allow_interchangable_modes=False)
                num_chosen_locs += curSegment_curAct_curMode_act_exs.Count

                # subtour major mode is interchangeable: all interchangeable modes are available
                curSegment_curAct_curMode_act_exs = curSegment_curAct_act_exs.GetFilteredSet(
                                    r'[FromTrip\Tour\MainDSeg\Mode\ID] = ' + str(main_mode_id) + 
                                    ' & [IsPartOfSubtour] = 1 & ' +
                                    r'[FromTrip\Tour\MajorModeIDOfSubtour] != ' + str(main_mode_id))
                            
                choose_and_set_locations(global_parameter,
                                         segmentLocationChoice, act_code, attraction_per_zone,
                                         main_mode_id, curSegment_curAct_curMode_act_exs, allow_interchangable_modes=True)
                num_chosen_locs += curSegment_curAct_curMode_act_exs.Count

    assert num_chosen_locs == curSegment_curAct_act_exs.Count


def choose_and_set_locations(global_parameter,
                             segmentDestChoice, act_code, attraction_per_zone,
                             mode_id, curSegment_curAct_curMode_act_ex, allow_interchangable_modes : bool):
    if curSegment_curAct_curMode_act_ex.Count == 0:
        return

    origin_zones, act_ex_origin_ti_for_imp, target_zones_for_rubberbanding, act_ex_target_ti_for_imp = compute_orig_dest_zones_and_time_intervals_for_dest_choice(
        curSegment_curAct_curMode_act_ex, global_parameter.zoneNo_to_zoneInd, global_parameter.time_interval_start_times, global_parameter.time_interval_end_times)

    act_ex_parameter = location_choice_engine.Location_Choice_Parameter_Act_Ex_Data(
        curSegment_curAct_curMode_act_ex,
        origin_zones, act_ex_origin_ti_for_imp,
        target_zones_for_rubberbanding, act_ex_target_ti_for_imp,
        segmentDestChoice, act_code, attraction_per_zone, mode_id, allow_interchangable_modes)

    location_choice_engine.choose_and_set_locations(global_parameter, act_ex_parameter, chosen_zones_result_attr=None)


def compute_orig_dest_zones_and_time_intervals_for_dest_choice(filterd_act_ex, zoneNo_to_zoneInd, time_interval_start_times, time_interval_end_times):
    """
    Obtain origin and destination zones for destination choice with rubberbanding.

    We have different types of activity executions:
    1. Outbound:              home                 -> [?] -> major activity
    2. Inbound (subtour):     1st major activity   -> [?] -> 2nd major activity
    3. Inbound (non-subtour): (2nd) major activity -> [?] -> home

    Since time points are obtained from (1st) major activity execution outwards, we need to choose orig/dest zones accordingly:
    1. Outbound:              home act ex       -> [?] -> succeeding act ex
    2. Inbound (subtour):     preceeding act ex -> [?] -> 2nd major act ex
    3. Inbound (non-subtour): preceeding act ex -> [?] -> home act ex

    Time points are obtained accordingly, respecting duration of the current activity execution.
    """    
    is_outbound = np.array(VPH.GetMultiByFormula(filterd_act_ex, formula=r'([Index] < [FromTrip\Tour\MajorActExIndex]) & ([IsPartOfSubTour]=0)'), 
                           dtype=bool)
    is_inbound_subtour = np.array(VPH.GetMultiByFormula(filterd_act_ex, formula=r'[IsPartOfSubTour]=1'), dtype=bool)
    is_inbound_nonsubtour = ~is_outbound & ~is_inbound_subtour
    # subtours are considered inbound
    assert not np.any(is_outbound & is_inbound_subtour) 

    num_act_exs = len(is_outbound)

    prec_act_ex_zone_no = np.array(visum_utilities.GetMulti(filterd_act_ex, r'FromTrip\FromActivityExecution\Location\Zone\No', chunk_size=abm_settings.chunk_size_trips, reindex = True))
    residence_zone_no   = np.array(visum_utilities.GetMulti(filterd_act_ex, r'Schedule\Person\Household\Residence\Location\Zone\No', chunk_size=abm_settings.chunk_size_trips))
    major_act_zone_no   = np.array(visum_utilities.GetMulti(filterd_act_ex, r'FromTrip\Tour\MajorActivityZoneNo', chunk_size=abm_settings.chunk_size_trips))
    succ_act_ex_zone_no = np.array(visum_utilities.GetMulti(filterd_act_ex, r'ToTrip\ToActivityExecution\Location\Zone\No', chunk_size=abm_settings.chunk_size_trips))

    # origin zones
    origin_zone_nos = np.empty(shape=(num_act_exs,), dtype=int)
    origin_zone_nos[~is_outbound] = prec_act_ex_zone_no[~is_outbound] # inbound (including subtour): preceeding act ex
    origin_zone_nos[ is_outbound] = residence_zone_no[is_outbound]    # outbound: home location
    origin_zones = abm_utilities.nos_to_indices(origin_zone_nos, zoneNo_to_zoneInd)

    # destination zones
    dest_zone_nos = np.empty(shape=(num_act_exs,), dtype=int)
    dest_zone_nos[is_outbound] = succ_act_ex_zone_no[is_outbound]                   # outbound: succeeding act ex
    dest_zone_nos[is_inbound_subtour] = major_act_zone_no[is_inbound_subtour]       # inbound (subtour): major act ex
    dest_zone_nos[is_inbound_nonsubtour] = residence_zone_no[is_inbound_nonsubtour] # inbound (non-subtour): home location
    dest_zones_for_rubberbanding = abm_utilities.nos_to_indices(dest_zone_nos, zoneNo_to_zoneInd)

    # obtain time intervals used in impedance calculation
    act_ex_duration        = np.array(visum_utilities.GetMulti(filterd_act_ex, r'Duration', chunk_size=abm_settings.chunk_size_trips))
    prec_act_ex_end_time   = np.array(visum_utilities.GetMulti(filterd_act_ex, r'FromTrip\FromActivityExecution\EndTime', chunk_size=abm_settings.chunk_size_trips))
    succ_act_ex_start_time = np.array(visum_utilities.GetMulti(filterd_act_ex, r'ToTrip\ToActivityExecution\StartTime', chunk_size=abm_settings.chunk_size_trips))

    act_ex_origin_time_points_for_imp = np.empty(shape=(num_act_exs,), dtype=int)
    act_ex_dest_time_points_for_imp = np.empty(shape=(num_act_exs,), dtype=int)

    # outbound
    act_ex_dest_time_points_for_imp[is_outbound]   = succ_act_ex_start_time[is_outbound]
    act_ex_origin_time_points_for_imp[is_outbound] = succ_act_ex_start_time[is_outbound] - act_ex_duration[is_outbound]

    # inbound (including subtours)
    act_ex_origin_time_points_for_imp[~is_outbound] = prec_act_ex_end_time[~is_outbound]
    act_ex_dest_time_points_for_imp[~is_outbound]   = prec_act_ex_end_time[~is_outbound] + act_ex_duration[~is_outbound]

    DAY_SECONDS = 60 * 60 * 24
    act_ex_origin_ti_for_imp = [abm_utilities.get_time_interval_index(
        time_point % DAY_SECONDS, time_interval_start_times, time_interval_end_times) for time_point in act_ex_origin_time_points_for_imp]
    act_ex_dest_ti_for_imp = [abm_utilities.get_time_interval_index(
        time_point % DAY_SECONDS, time_interval_start_times, time_interval_end_times) for time_point in act_ex_dest_time_points_for_imp]

    return origin_zones, act_ex_origin_ti_for_imp, dest_zones_for_rubberbanding, act_ex_dest_ti_for_imp


def minor_mode_choice(global_parameter, prt_shortest_path_searcher, curSegment_act_ex, nodeno_to_node, nodeno_to_longitude_latitude, time_interval_codes,
                      cur_local_act_ex_index, segmentModeChoice, mode_id_to_interchangeable, mode_id_to_firstDSegCode):

    if curSegment_act_ex.Count == 0:
        return

    logging.info('mode choice for minor activity executions (local index: %d, segment %s): %s ',
        cur_local_act_ex_index, segmentModeChoice['Specification'], segmentModeChoice['Comment'])
        
    chooseable_mode_ids = segmentModeChoice['Choices']
    main_mode_attr = r'[FromTrip\Tour\MainDSeg\Mode\ID]'
    major_mode_id_of_subtour_attr = r'[FromTrip\Tour\MajorModeIDOfSubtour]'
    major_act_ex_index_attr = r'[FromTrip\Tour\MajorActExIndex]'

    num_chosen_modes = minor_mode_choice_internal(global_parameter, prt_shortest_path_searcher, 
                                                  curSegment_act_ex,
                                                  nodeno_to_node, nodeno_to_longitude_latitude,
                                                  time_interval_codes, cur_local_act_ex_index,
                                                  segmentModeChoice, mode_id_to_interchangeable, 
                                                  mode_id_to_firstDSegCode, chooseable_mode_ids,
                                                  main_mode_attr, major_mode_id_of_subtour_attr, major_act_ex_index_attr)

    assert num_chosen_modes == curSegment_act_ex.Count


def minor_mode_choice_for_home_trips(global_parameter, prt_shortest_path_searcher, cur_segment_home_act_exs, nodeno_to_node, nodeno_to_longitude_latitude, time_interval_codes,
                                     segmentModeChoice, mode_id_to_interchangeable, mode_id_to_firstDSegCode):
    """
    Minor mode choice for home trips.

    Modes are only chosen, when main mode is interchangeable, but travel times must always be computed.

    NOTE: For intermediate home stops, start time is chosen from preceeding tour. This may not be consistent with succeeding tour.
    """

    cur_local_act_ex_index_for_home_trips = -1 # mark as home trips
    # for all home act ex having a from trip, 
    # choose mode and set trip time for the inbound trip X -> H and the corresponding time for H
    home_act_ex_with_from_trip = cur_segment_home_act_exs.GetFilteredSet(r'[FromTrip\Index] != 0')

    logging.info('mode choice for minor activity executions (inbound home trips X -> H, segment %s): %s (%s choices)' ,
        segmentModeChoice['Specification'], segmentModeChoice['Comment'], home_act_ex_with_from_trip.Count)

    chooseable_mode_ids = segmentModeChoice['Choices']
    prec_main_mode_attr = r'[FromTrip\Tour\MainDSeg\Mode\ID]'
    prec_major_mode_id_of_subtour_attr = '' # not needed
    prec_major_act_ex_index_attr = r'[FromTrip\Tour\MajorActExIndex]'
    minor_mode_choice_internal(global_parameter, prt_shortest_path_searcher,
                               home_act_ex_with_from_trip,
                               nodeno_to_node, nodeno_to_longitude_latitude,
                               time_interval_codes, cur_local_act_ex_index_for_home_trips,
                               segmentModeChoice, mode_id_to_interchangeable,
                               mode_id_to_firstDSegCode, chooseable_mode_ids,
                               prec_main_mode_attr, prec_major_mode_id_of_subtour_attr,
                               prec_major_act_ex_index_attr,
                               set_act_ex_start_time=True)

    # for all home act ex without from trip, 
    # choose mode and set trip for the outbound trip H -> X and the corresponding start time for H
    home_act_ex_without_from_trip = cur_segment_home_act_exs.GetFilteredSet(r'[FromTrip\Index] = 0')

    succ_main_mode_attr = r'[ToTrip\Tour\MainDSeg\Mode\ID]'
    succ_major_mode_id_of_subtour_attr = '' # not needed
    succ_major_act_ex_index_attr = r'[ToTrip\Tour\MajorActExIndex]'

    logging.info('mode choice for minor activity executions (outbound home trips H -> X (first tour), segment %s): %s (%s choices)',
                 segmentModeChoice['Specification'], segmentModeChoice['Comment'], home_act_ex_without_from_trip.Count)

    minor_mode_choice_internal(global_parameter, prt_shortest_path_searcher,
                               home_act_ex_without_from_trip,
                               nodeno_to_node, nodeno_to_longitude_latitude,
                               time_interval_codes, cur_local_act_ex_index_for_home_trips,
                               segmentModeChoice, mode_id_to_interchangeable,
                               mode_id_to_firstDSegCode, chooseable_mode_ids,
                               succ_main_mode_attr, succ_major_mode_id_of_subtour_attr,
                               succ_major_act_ex_index_attr,
                               set_act_ex_start_time=True)


    # for all home act ex having a from trip, 
    # choose mode and set trip time for the outbound trip H -> X without changing the start time of H

    logging.info('mode choice for minor activity executions (outbound home trips H -> X (later tours), segment %s): %s (%s choices)' ,
        segmentModeChoice['Specification'], segmentModeChoice['Comment'], home_act_ex_with_from_trip.Count)

    minor_mode_choice_internal(global_parameter, prt_shortest_path_searcher,
                               home_act_ex_with_from_trip,
                               nodeno_to_node, nodeno_to_longitude_latitude,
                               time_interval_codes, cur_local_act_ex_index_for_home_trips,
                               segmentModeChoice, mode_id_to_interchangeable,
                               mode_id_to_firstDSegCode, chooseable_mode_ids,
                               succ_main_mode_attr, succ_major_mode_id_of_subtour_attr,
                               succ_major_act_ex_index_attr,
                               set_act_ex_start_time=False)

    # update the duration of H in X -> H -> Y, if the times are valid:
    # duration(H) = max(0, startTime(Y) - duration(H -> Y) - startTime(H))

    intermediate_home_act_ex  = home_act_ex_with_from_trip.GetFilteredSet(r'[ToTrip\Index] != 0')
    act_ex_start_time = np.array(visum_utilities.GetMulti(
        intermediate_home_act_ex, r'StartTime', chunk_size=abm_settings.chunk_size_trips, reindex = True), dtype=int)
    succ_act_ex_start_time = np.array(visum_utilities.GetMulti(
        intermediate_home_act_ex, r'ToTrip\ToActivityExecution\StartTime', chunk_size=abm_settings.chunk_size_trips), dtype=int)
    to_trip_duration = np.array(visum_utilities.GetMulti(
        intermediate_home_act_ex, r'ToTrip\Duration', chunk_size=abm_settings.chunk_size_trips), dtype=int)
    intermediate_home_act_ex_duration = succ_act_ex_start_time - to_trip_duration - act_ex_start_time
    # set duration to 0 for overlapping tours
    intermediate_home_act_ex_duration[intermediate_home_act_ex_duration < 0] = 0
    visum_utilities.SetMulti(intermediate_home_act_ex, r'Duration', intermediate_home_act_ex_duration, chunk_size=abm_settings.chunk_size_trips)


def minor_mode_choice_internal(global_parameter, prt_shortest_path_searcher, curSegment_act_ex, 
                               nodeno_to_node, nodeno_to_longitude_latitude, time_interval_codes,
                               cur_local_act_ex_index, segmentModeChoice, mode_id_to_interchangeable, mode_id_to_firstDSegCode, chooseable_mode_ids,
                               main_mode_attr, major_mode_id_of_subtour_attr, major_act_ex_index_attr, set_act_ex_start_time = True) -> int:
    """
    Compute minor mode choice.

    # Relevant parameters
    * `cur_local_act_ex_index` : `int` (`-1` for home trips) -- index for correct processing sequence of activity executions (from major activity execution outwards)
    
    Returns `num_chosen_modes : int`.
    """
    if curSegment_act_ex.Count == 0:
        return 0

    num_chosen_modes = 0

    # consider main mode of tour for mode choice table term
    for main_mode_id in chooseable_mode_ids:
        local_index_string = f'local index: {cur_local_act_ex_index}' if cur_local_act_ex_index >= 0 else 'home trips'
        logging.info('mode choice for minor activity executions (%s, segment %s, main mode %s): %s ',
            local_index_string, segmentModeChoice['Specification'], main_mode_id, segmentModeChoice['Comment'])
        if mode_id_to_interchangeable[main_mode_id]:
            # all interchangeable mode are possible for every trip of the tour
            curSegment_curMode_act_ex = curSegment_act_ex.GetFilteredSet(f'{main_mode_attr} = {main_mode_id}')

            num_chosen_modes += choose_and_set_modes(global_parameter, prt_shortest_path_searcher, nodeno_to_node,
                                                     nodeno_to_longitude_latitude, time_interval_codes,
                                                     segmentModeChoice,
                                                     main_mode_id,
                                                     curSegment_curMode_act_ex,
                                                     mode_id_to_interchangeable, mode_id_to_firstDSegCode, 
                                                     major_act_ex_index_attr, set_act_ex_start_time, 
                                                     allow_interchangable_modes=True)
                                  
        else:
            # mode is fix for all non-subtour destinations
            curSegment_curMode_act_ex = curSegment_act_ex.GetFilteredSet(f'{main_mode_attr} = {main_mode_id} & [IsPartOfSubtour] = 0')
            
            # step is needed for the also calculated trip time
            # set fixed mode and calculate time for this mode
            num_chosen_modes += choose_and_set_modes(global_parameter, prt_shortest_path_searcher, nodeno_to_node,
                                                     nodeno_to_longitude_latitude, time_interval_codes,
                                                     segmentModeChoice,
                                                     main_mode_id, curSegment_curMode_act_ex,
                                                     mode_id_to_interchangeable, mode_id_to_firstDSegCode, 
                                                     major_act_ex_index_attr, set_act_ex_start_time, 
                                                     allow_interchangable_modes=False)

            # subtours may use main mode and interchangable mode
            if cur_local_act_ex_index == 1:
                # first activity of subtour (= major activity of subtour): 
                # choose either main mode of subtour or interchangable mode
                curSegment_curMode_act_ex = curSegment_act_ex.GetFilteredSet(f'{main_mode_attr} = {main_mode_id} & [IsPartOfSubtour] = 1')

                # choose and set major mode id of subtour
                num_chosen_modes += choose_and_set_modes(global_parameter, prt_shortest_path_searcher, nodeno_to_node, 
                                                         nodeno_to_longitude_latitude, time_interval_codes,
                                                         segmentModeChoice,
                                                         main_mode_id, curSegment_curMode_act_ex,
                                                         mode_id_to_interchangeable, mode_id_to_firstDSegCode, 
                                                         major_act_ex_index_attr, set_act_ex_start_time, 
                                                         allow_interchangable_modes=True,
                                                         is_major_mode_of_subtour_choice=True)

            elif cur_local_act_ex_index > 1:
                # if main mode of subtour is main mode of tour, use this (its not interchangeable)
                curSegment_curMode_act_ex = curSegment_act_ex.GetFilteredSet(
                    f'{main_mode_attr} = {main_mode_id} & [IsPartOfSubtour] = 1 & {major_mode_id_of_subtour_attr} = {main_mode_id}')

                num_chosen_modes += choose_and_set_modes(global_parameter, prt_shortest_path_searcher, nodeno_to_node, 
                                                         nodeno_to_longitude_latitude, time_interval_codes,
                                                         segmentModeChoice,
                                                         main_mode_id, curSegment_curMode_act_ex,
                                                         mode_id_to_interchangeable, mode_id_to_firstDSegCode,
                                                         major_act_ex_index_attr, set_act_ex_start_time, 
                                                         allow_interchangable_modes=False)

                # if interchangable mode is used as main mode of subtour, use all interchangeable modes
                curSegment_curMode_act_ex = curSegment_act_ex.GetFilteredSet(
                    f'{main_mode_attr} = {main_mode_id} & [IsPartOfSubtour] = 1 & {major_mode_id_of_subtour_attr} != {main_mode_id}')

                num_chosen_modes += choose_and_set_modes(global_parameter, prt_shortest_path_searcher, nodeno_to_node, 
                                                         nodeno_to_longitude_latitude, time_interval_codes,
                                                         segmentModeChoice,
                                                         main_mode_id, curSegment_curMode_act_ex,
                                                         mode_id_to_interchangeable, mode_id_to_firstDSegCode, 
                                                         major_act_ex_index_attr, set_act_ex_start_time, 
                                                         allow_interchangable_modes=True)
    return num_chosen_modes


def choose_and_set_modes(global_parameter,
                         prt_shortest_path_searcher,
                         nodeno_to_node,
                         nodeno_to_longitude_latitude,
                         time_interval_codes,
                         segmentModeChoice,
                         main_mode_id,
                         curSegment_curMode_act_ex,
                         mode_id_to_interchangeable, mode_id_to_firstDSegCode,
                         major_act_ex_index_attr : str,
                         set_act_ex_start_time : bool,
                         allow_interchangable_modes : bool,
                         is_major_mode_of_subtour_choice : bool = False) -> int:

    if curSegment_curMode_act_ex.Count == 0:
        return 0

    Visum = global_parameter.config.Visum

    origin_zones, target_zones, trip_time_points, origin_node_nos, dest_node_nos, is_outbound_act_ex = _compute_input_for_mode_choice(
        curSegment_curMode_act_ex, global_parameter.zoneNo_to_zoneInd, major_act_ex_index_attr)

    trip_ti_start_times = [abm_utilities.get_time_interval_start_time(
        time_point, global_parameter.time_interval_start_times, global_parameter.time_interval_end_times) for time_point in trip_time_points]
    trip_ti_codes = [abm_utilities.get_time_interval_code(
        time_point, global_parameter.time_interval_start_times, global_parameter.time_interval_end_times, time_interval_codes) for time_point in trip_time_points]
    
    chooseable_mode_ids = np.array(segmentModeChoice['Choices'], dtype=int)
    mode_betas = segmentModeChoice['Beta']  # shape = (num_terms, num_modes)

    # some terms are not relevant because all currently allowed modes have beta = 0 in those terms
    usable_mode_inds = location_choice_engine.compute_usable_mode_inds(
        main_mode_id, mode_id_to_interchangeable, chooseable_mode_ids, allow_interchangable_modes)
    relevant_terms = location_choice_engine.get_relevant_terms(mode_betas, usable_mode_inds)

    mode_impedance_types       = np.array(segmentModeChoice["ModeImpedanceTypes"], dtype=object)[relevant_terms]
    mode_impedance_definitions = np.array(segmentModeChoice["ModeImpedanceDefinitions"], dtype=object)[relevant_terms]
    mode_impedance_PrTSys      = np.array(segmentModeChoice["ModeImpedancePrTSys"], dtype=object)[relevant_terms]

    mode_betas = mode_betas[relevant_terms]
    matrixExpr = np.array(segmentModeChoice['MatrixExpr'], dtype=object)[relevant_terms]
    attrExpr =   np.array(segmentModeChoice['AttrExpr'], dtype=object)[relevant_terms]

    # compute mode impedance for each term
    # mode impedance for one term is a list of length len(major_act_ex_segment_filtered)
    mode_impedances_per_term = []
    mode_tripTimes_per_subject = np.full(shape=(len(chooseable_mode_ids), curSegment_curMode_act_ex.Count), fill_value=-1, dtype=float)
    for [mode_impedance_type, mode_impedance_definition, prtsys, mode_beta] in zip(mode_impedance_types, 
                                                                                  mode_impedance_definitions, 
                                                                                  mode_impedance_PrTSys, 
                                                                                  mode_betas):
        if mode_impedance_type == config.ModeImpedanceType.PrTShortestPathImpedance:
            # use prt impedance
            origin_nodes = [nodeno_to_node[node_no] for node_no in origin_node_nos]
            dest_nodes = [nodeno_to_node[node_no] for node_no in dest_node_nos]
            shortestPath_search_result = mode_choice_engine.compute_prt_impedance_minor_mode_choice(prt_shortest_path_searcher,
                                                                                                    origin_nodes, dest_nodes,
                                                                                                    trip_ti_codes,
                                                                                                    mode_impedance_definition,
                                                                                                    prtsys,
                                                                                                    compute_travel_times=True)
            mode_impedances = shortestPath_search_result[0]
            mode_trip_times = np.array(shortestPath_search_result[1])
        elif mode_impedance_type == config.ModeImpedanceType.PuTShortestPathImpedance:
            walk_tsys = abm_utilities.get_global_attribute(
                Visum, 'WalkPrTSys', 'Name')
            max_walk_time_in_minutes = abm_utilities.get_global_attribute(
                Visum, 'MaxWalkTimeInMinutesForPutShortestPathSearch', 'Value')
            node_to_stop_area_indices, node_to_stop_area_distances = mode_choice_engine.compute_node_stoparea_distances(
                Visum, walk_tsys, max_walk_time_in_minutes)

            dseg_code = mode_impedance_definition
            walking_time_impedance_factor = abm_utilities.get_global_attribute(
                Visum, 'WalkTimeImpedanceFactor', 'Value')

            mode_impedances, mode_trip_times = mode_choice_engine.compute_put_impedance(Visum, node_to_stop_area_indices, node_to_stop_area_distances,
                                                                                        global_parameter.time_interval_start_times, global_parameter.time_interval_end_times,
                                                                                        dseg_code, walking_time_impedance_factor,
                                                                                        trip_ti_start_times, origin_node_nos, dest_node_nos,
                                                                                        compute_travel_times=True)

        elif mode_impedance_type == config.ModeImpedanceType.DirectDistance:
            direct_distance_speed_in_km_h = abm_utilities.get_global_attribute(Visum, 'DirectDistanceSpeed', 'Value') # speed in km/h
            direct_distance_sec_per_m = 3.6 / direct_distance_speed_in_km_h # 1 m/sec = 3.6 km/h 
            mode_impedances = np.array(mode_choice_engine.compute_direct_distances_for_minor_mode_choice(
                origin_node_nos, dest_node_nos, nodeno_to_longitude_latitude), dtype=float)
            mode_trip_times = mode_impedances * direct_distance_sec_per_m
        else:
            mode_impedances = [1.0] * len(curSegment_curMode_act_ex)
            mode_trip_times = None

        if mode_trip_times is not None:
            assert np.count_nonzero(mode_beta) == 1, "only one mode beta should be != 0 at once"
            assert np.all(mode_tripTimes_per_subject[mode_beta != 0] == -1), "trip times should be set only once"
            mode_tripTimes_per_subject[mode_beta != 0] = mode_trip_times

        mode_impedances_per_term.append(mode_impedances)

    outbound_act_ex_count = np.count_nonzero(is_outbound_act_ex)
    inbound_and_subtour_act_ex_count = len(is_outbound_act_ex) - outbound_act_ex_count

    curSegment_curMode_outbound_act_ex = curSegment_curMode_act_ex.GetFilteredSet(
        f'([Index] < {major_act_ex_index_attr}) & ([IsPartOfSubTour]=0)')
    curSegment_curMode_inbound_and_subtour_act_ex = curSegment_curMode_act_ex.GetFilteredSet(
        f'([Index] >= {major_act_ex_index_attr}) | ([IsPartOfSubTour]=1)')

    if not mode_id_to_interchangeable[main_mode_id] and not allow_interchangable_modes:
        # origin_zones is only used for the shape (num objects)
        chosen_mode_ids = np.full_like(origin_zones, main_mode_id)
        assert len(usable_mode_inds) == 1
        chosen_mode_indices = np.full_like(origin_zones, usable_mode_inds[0])
    else:
        if allow_interchangable_modes:
            mode_is_useable = np.fromiter(((mode_id_to_interchangeable[cur_mode_id] or cur_mode_id == main_mode_id)
                                                  for cur_mode_id in chooseable_mode_ids), 
                                          dtype=bool)
        else: 
            mode_is_useable = chooseable_mode_ids == main_mode_id

        trip_time_point_time_interval_indices = [abm_utilities.get_time_interval_index(
            time_point, global_parameter.time_interval_start_times, global_parameter.time_interval_end_times) for time_point in trip_time_points]

        chosen_mode_indices = mode_choice_engine.calc_minor_mode_choice(curSegment_curMode_act_ex,
                                                                        Visum,
                                                                        mode_betas,
                                                                        matrixExpr,
                                                                        attrExpr,
                                                                        mode_impedances_per_term,
                                                                        origin_zones,
                                                                        target_zones,
                                                                        global_parameter.time_interval_start_times,
                                                                        global_parameter.time_interval_end_times,
                                                                        trip_time_point_time_interval_indices,
                                                                        mode_is_useable)
        chosen_mode_ids = chooseable_mode_ids[chosen_mode_indices]
                  
    if outbound_act_ex_count > 0:
        set_mode_and_dSeg(curSegment_curMode_outbound_act_ex,
                          chosen_mode_ids[is_outbound_act_ex],
                          mode_id_to_firstDSegCode,
                          is_major_mode_of_subtour_choice,
                          r'ToTrip\DSegCode',
                          r'ToTrip\Tour\MajorModeIDOfSubtour')

    if inbound_and_subtour_act_ex_count > 0:
        set_mode_and_dSeg(curSegment_curMode_inbound_and_subtour_act_ex,
                          chosen_mode_ids[~is_outbound_act_ex],
                          mode_id_to_firstDSegCode,
                          is_major_mode_of_subtour_choice,
                          r'FromTrip\DSegCode',
                          r'FromTrip\Tour\MajorModeIDOfSubtour')

    logging.info('minor mode set for %d trips', len(chosen_mode_ids))

    travel_time_per_trip = mode_tripTimes_per_subject[chosen_mode_indices, range(len(chosen_mode_indices))]
    assert np.all(travel_time_per_trip >= 0)

    travel_time_per_trip = np.rint(travel_time_per_trip)
    travel_time_per_trip = np.minimum(travel_time_per_trip, 999999)

    if outbound_act_ex_count > 0:
        outbound_trip_travel_times = travel_time_per_trip[is_outbound_act_ex]
        
        # end time of trip is start time of succeeding activity execution
        start_time_succeeding = visum_utilities.GetMulti(
            curSegment_curMode_outbound_act_ex, r'ToTrip\ToActivityExecution\StartTime', chunk_size=abm_settings.chunk_size_trips, reindex = True)

        # start time of trip is start time of succeeding act ex - duration of trip
        outboud_trips_start_times = start_time_succeeding - outbound_trip_travel_times
        shift_tours_if_necessary(Visum, outbound_act_ex_set=curSegment_curMode_outbound_act_ex, start_times=outboud_trips_start_times)
        visum_utilities.SetMulti(container=curSegment_curMode_outbound_act_ex, attribute=r'ToTrip\Duration', values=outbound_trip_travel_times,
                                 chunk_size=abm_settings.chunk_size_trips)
        visum_utilities.SetMulti(container=curSegment_curMode_outbound_act_ex, attribute=r'ToTrip\SchedDepTime', values=outboud_trips_start_times,
                                 chunk_size=abm_settings.chunk_size_trips)

        if set_act_ex_start_time:
            # end time of act ex is start time of trip
            duration = visum_utilities.GetMulti(
                curSegment_curMode_outbound_act_ex, r'Duration', chunk_size=abm_settings.chunk_size_trips)
            start_time_act_ex = outboud_trips_start_times - duration
            shift_tours_if_necessary(Visum, outbound_act_ex_set=curSegment_curMode_outbound_act_ex, start_times=start_time_act_ex)
            assert np.all(start_time_act_ex >= 0)
            visum_utilities.SetMulti(curSegment_curMode_outbound_act_ex, "StartTime",
                                    start_time_act_ex, chunk_size=abm_settings.chunk_size_trips)

    if inbound_and_subtour_act_ex_count > 0:
        inbound_and_subtour_trip_travel_times = travel_time_per_trip[~is_outbound_act_ex]
        end_times_preceeding = visum_utilities.GetMulti(
            curSegment_curMode_inbound_and_subtour_act_ex, r'FromTrip\FromActivityExecution\EndTime', chunk_size=abm_settings.chunk_size_trips, reindex = True)
        visum_utilities.SetMulti(container=curSegment_curMode_inbound_and_subtour_act_ex, attribute=r'FromTrip\Duration', values=inbound_and_subtour_trip_travel_times,
                                 chunk_size=abm_settings.chunk_size_trips)
        visum_utilities.SetMulti(container=curSegment_curMode_inbound_and_subtour_act_ex, attribute=r'FromTrip\SchedDepTime', values=end_times_preceeding,
                                 chunk_size=abm_settings.chunk_size_trips)

        if set_act_ex_start_time:
            # start time of act ex is end time of preceeding act ex + duration of trip
            start_time_act_ex = end_times_preceeding + inbound_and_subtour_trip_travel_times
            assert np.all(start_time_act_ex >= 0)
            visum_utilities.SetMulti(curSegment_curMode_inbound_and_subtour_act_ex, "StartTime",
                                    start_time_act_ex, chunk_size=abm_settings.chunk_size_trips)

    logging.info('duration set for %d trips', len(travel_time_per_trip))

    return len(chosen_mode_indices)

def shift_tours_if_necessary(Visum, outbound_act_ex_set, start_times):
    """
    Shift tours with negative start times by enough days to start within `[0, 23:59:59]`.

    * `outbound_act_ex_set` must contain only *one* outbound activity execution per tour.

    NOTE:
    Tours may be shifted multiple times through consecutive calls to this function, 
    since outbound trips are processed backwards from major act ex towards home act ex.
    """
    if any(start_time < 0 for start_time in start_times):
        DAY_SEC = 60 * 60 * 24

        # act ex key is (PersonNo, ScheduleNo, Index)
        person_nos = visum_utilities.GetMulti(
            outbound_act_ex_set, r'PersonNo', chunk_size=abm_settings.chunk_size_trips, reindex = True)
        schedule_nos = visum_utilities.GetMulti(
            outbound_act_ex_set, r'ScheduleNo', chunk_size=abm_settings.chunk_size_trips)
        tour_nos = visum_utilities.GetMulti(
            outbound_act_ex_set, r'ToTrip\Tour\No', chunk_size=abm_settings.chunk_size_trips)
        act_ex_tour_keys = zip(person_nos, schedule_nos, tour_nos)
        tour_shift_data = [-(start_time // DAY_SEC) * DAY_SEC if start_time < 0 else 0 for start_time in start_times]
        tour_shift_dict = dict(zip(act_ex_tour_keys, tour_shift_data))

        # adapt data from current activity executions    
        start_times += tour_shift_data

        # adapt data from following activity executions 
        act_ex_with_from_trip = Visum.Net.ActivityExecutions.GetFilteredSet(r'[FromTrip\Index] > 0')

        person_no = visum_utilities.GetMulti(
            act_ex_with_from_trip, r'PersonNo', chunk_size=abm_settings.chunk_size_trips, reindex = True)
        schedule_no = visum_utilities.GetMulti(
            act_ex_with_from_trip, r'ScheduleNo', chunk_size=abm_settings.chunk_size_trips)
        tour_no = visum_utilities.GetMulti(
            act_ex_with_from_trip, r'FromTrip\Tour\No', chunk_size=abm_settings.chunk_size_trips)
        act_ex_start_time = visum_utilities.GetMulti(
            act_ex_with_from_trip, r'StartTime', chunk_size=abm_settings.chunk_size_trips)
        from_trip_duration = visum_utilities.GetMulti(
            act_ex_with_from_trip, r'FromTrip\Duration', chunk_size=abm_settings.chunk_size_trips)
        
        all_act_ex_time_shift = [tour_shift_dict.get(tuple(key), 0) for key in zip(person_no, schedule_no, tour_no)]
        all_act_ex_new_times = np.array([(start_time + time_shift, start_time + time_shift - duration if duration is not None else None) 
                                if start_time is not None else (None, None)
                                for start_time, duration, time_shift in zip(act_ex_start_time, from_trip_duration, all_act_ex_time_shift)])

        visum_utilities.SetMulti(act_ex_with_from_trip, r'StartTime', all_act_ex_new_times[:,0],chunk_size=abm_settings.chunk_size_trips)
        visum_utilities.SetMulti(act_ex_with_from_trip, r'FromTrip\SchedDepTime', all_act_ex_new_times[:,1],chunk_size=abm_settings.chunk_size_trips)


def set_mode_and_dSeg(act_ex,
                      chosen_mode_ids,
                      mode_id_to_firstDSegCode,
                      is_major_mode_of_subtour_choice,
                      trip_dSeg_code_string,
                      major_mode_id_of_subtour_string):

    chosen_dseg_codes = [mode_id_to_firstDSegCode[mode_id] for mode_id in chosen_mode_ids]

    visum_utilities.SetMulti(act_ex, trip_dSeg_code_string,
                             chosen_dseg_codes, chunk_size=abm_settings.chunk_size_trips)

    if is_major_mode_of_subtour_choice:
        visum_utilities.SetMulti(act_ex, major_mode_id_of_subtour_string, chosen_mode_ids, chunk_size=abm_settings.chunk_size_trips)


def _compute_input_for_mode_choice(filterd_act_ex, zoneNo_to_zoneInd, major_act_ex_index_attr : str):
    # mode choice is done for
    #   the from trip   for activity execution on the way to their home location (inbound) 
    #                   and for activity executions in subtours
    #   the to trip     for activity execution on the way to the major activity (outbound)

    is_outbound = np.array(VPH.GetMultiByFormula(filterd_act_ex, formula=f'([Index] < {major_act_ex_index_attr}) & ([IsPartOfSubTour]=0)'), 
                           dtype=int)

    prev_act_ex_zone_no = visum_utilities.GetMulti(
            filterd_act_ex, r'FromTrip\FromActivityExecution\Location\Zone\No', chunk_size=abm_settings.chunk_size_trips, reindex = True)
    act_ex_zone_no = visum_utilities.GetMulti(
            filterd_act_ex, r'Location\Zone\No', chunk_size=abm_settings.chunk_size_trips)
    succ_act_ex_zone_no = visum_utilities.GetMulti(
            filterd_act_ex, r'ToTrip\ToActivityExecution\Location\Zone\No', chunk_size=abm_settings.chunk_size_trips)

    prev_act_ex_node_no = visum_utilities.GetMulti(
            filterd_act_ex, r'FromTrip\FromActivityExecution\Location\NearestActiveNode\No', chunk_size=abm_settings.chunk_size_trips)
    act_ex_node_no = visum_utilities.GetMulti(
            filterd_act_ex, r'Location\NearestActiveNode\No', chunk_size=abm_settings.chunk_size_trips)
    succ_act_ex_node_no = visum_utilities.GetMulti(
            filterd_act_ex, r'ToTrip\ToActivityExecution\Location\NearestActiveNode\No', chunk_size=abm_settings.chunk_size_trips)

    origin_zones = abm_utilities.nos_to_indices(np.column_stack([prev_act_ex_zone_no, act_ex_zone_no])[np.arange(len(is_outbound)), is_outbound], 
                                                zoneNo_to_zoneInd)
    target_zones = abm_utilities.nos_to_indices(np.column_stack([act_ex_zone_no, succ_act_ex_zone_no])[np.arange(len(is_outbound)), is_outbound], 
                                                zoneNo_to_zoneInd)

    from_node_nos = np.column_stack([prev_act_ex_node_no, act_ex_node_no])[np.arange(len(is_outbound)), is_outbound]
    to_node_nos = np.column_stack([act_ex_node_no, succ_act_ex_node_no])[np.arange(len(is_outbound)), is_outbound]

    # the time intervall used in the impedance calculation is 
    #   for inbound stops and subtours:
    #       end of the preceeding activity execution
    #   for outbount stops:    
    #       the start of the succeeding activity execution 
    prev_act_ex_end_time = visum_utilities.GetMulti(
            filterd_act_ex, r'FromTrip\FromActivityExecution\EndTime', chunk_size=abm_settings.chunk_size_trips)
    succ_act_ex_start_time = visum_utilities.GetMulti(
        filterd_act_ex, r'ToTrip\ToActivityExecution\StartTime', chunk_size=abm_settings.chunk_size_trips)
    trip_time_points = np.column_stack([prev_act_ex_end_time, succ_act_ex_start_time])[np.arange(len(is_outbound)), is_outbound]

    is_outbound = np.array(is_outbound, dtype=bool)
    return origin_zones, target_zones, trip_time_points, from_node_nos, to_node_nos, is_outbound


def update_main_mode_of_tours_with_interchangeable_main_mode(Visum):
    logging.info('Update main mode of tours with interchangeable main mode to the mode with highest rank of all the actually chosen minor modes')

    interchangeable_mode_rank_and_firstDSegCode = Visum.Net.Modes.GetFilteredSet(r'[Interchangeable]=1').GetMultipleAttributes(["Rank", r'First:DemandSegments\Code'])
    interchangeable_mode_rank_to_firstDSegCode = dict(interchangeable_mode_rank_and_firstDSegCode)
    assert len(interchangeable_mode_rank_to_firstDSegCode) == len(interchangeable_mode_rank_and_firstDSegCode), "every interchangeable mode must have a unique rank"

    tours_with_interchangeable_main_mode = Visum.Net.Tours.GetFilteredSet(r'([MainDSeg\Mode\Interchangeable] = 1)')

    new_tour_main_mode_ranks = visum_utilities.GetMulti(
        tours_with_interchangeable_main_mode, r'Max:Trips([DSeg\Mode\Interchangeable]=1)\DSeg\Mode\Rank', chunk_size=10000000, reindex=True)
    new_tour_main_dseg_codes = [interchangeable_mode_rank_to_firstDSegCode[mode_rank] for mode_rank in new_tour_main_mode_ranks]
    visum_utilities.SetMulti(tours_with_interchangeable_main_mode, r'MainDSegCode', new_tour_main_dseg_codes, chunk_size=abm_settings.chunk_size_trips)
    logging.info('MainDSegCode updated for all tours with interchangeable main mode (%d tours)', len(new_tour_main_dseg_codes))

    logging.info('Update main mode of subtours with interchangeable main mode to the mode with highest rank of all the actually chosen minor modes of subtour')

    interchangeable_mode_rank_and_modeid = Visum.Net.Modes.GetFilteredSet(r'[Interchangeable]=1').GetMultipleAttributes(["Rank", "ID"])
    interchangeable_mode_rank_to_modeid = dict(interchangeable_mode_rank_and_modeid)
    assert len(interchangeable_mode_rank_to_firstDSegCode) == len(interchangeable_mode_rank_and_firstDSegCode), "every interchangeable mode must have a unique rank"

    tours_with_subtour = Visum.Net.Tours.GetFilteredSet(r'[subtour_stops] > 0')

    if tours_with_subtour.Count > 0 :
        new_tour_main_subtour_mode_rank = visum_utilities.GetMulti(
            tours_with_subtour,
            r'Max:Trips([FromActivityExecution\IsPartOfSubtour]=1 & [DSeg\Mode\Interchangeable]=1)\DSeg\Mode\Rank', 
            chunk_size=abm_settings.chunk_size_trips, reindex=True)
        tour_main_mode_ids = visum_utilities.GetMulti(
            tours_with_subtour, r'MainDSeg\Mode\ID', chunk_size=abm_settings.chunk_size_trips)
        new_tour_main_subtour_mode_ids = [interchangeable_mode_rank_to_modeid[mode_rank] if mode_rank is not None else main_mode_id
                                        for (mode_rank, main_mode_id) in zip(new_tour_main_subtour_mode_rank, tour_main_mode_ids)]

        visum_utilities.SetMulti(tours_with_subtour, r'MajorModeIDOfSubtour', new_tour_main_subtour_mode_ids, chunk_size=abm_settings.chunk_size_trips)
        logging.info('MajorModeIDOfSubtour updated for all tours with interchangeable subtour mode (%d tours)', sum(np.array(new_tour_main_subtour_mode_rank) != None))
