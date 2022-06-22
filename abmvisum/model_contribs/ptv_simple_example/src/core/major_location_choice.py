import logging

import numpy as np

import VisumPy.helpers as VPH

from src import abm_utilities, location_choice_engine, visum_utilities
from settings import abm_settings

def major_location_choice_non_primary_tours(global_location_choice_parameter, segments):
     # in contrast to primary location choice, [IsMajorActivity] = 1 is sufficient here: 
     # secondary tours don't have any subtours that would split the major activity into two parts
    major_act_ex_filter_expr = r'([FromTrip\Tour\Is_Primary] = 0) & ([IsMajorActivity] = 1)'
    origin_zone_no_attr_id = r'Schedule\Person\Household\Residence\Location\Zone\No'
    result_act_ex_attr = r'FromTrip\Tour\MajorActivityZoneNo'

    Visum = global_location_choice_parameter.config.Visum
    activityID_to_activityCode = abm_utilities.build_activity_dict(Visum)
    num_chosen_major_locs = 0

    major_act_ex = Visum.Net.ActivityExecutions.GetFilteredSet(major_act_ex_filter_expr)

    for segment in segments:
        cur_segment_major_act_ex = abm_utilities.get_filtered_subjects(major_act_ex, segment['Filter'])
        
        # determine set of occurring activities, as the attraction_per_zone depends on the activity of the activity execution and the chosen location depends on the activity
        unique_activity_IDs = np.unique(visum_utilities.GetMulti(
            cur_segment_major_act_ex, r'Activity\ID', chunk_size=abm_settings.chunk_size_trips, reindex = True))
        for act_id in unique_activity_IDs:
            act_code = activityID_to_activityCode[act_id]
            filtered_act_ex = cur_segment_major_act_ex.GetFilteredSet(f'[ActivityCode]="{act_code}"')
            num_act_ex = filtered_act_ex.Count
            if num_act_ex == 0:
                continue

            logging.info(f"location choice for major activity executions of secondary tours " \
                         f"(segment {segment['Specification']} and activity {act_code}): {segment['Comment']}")

            attraction_per_zone = abm_utilities.build_zone_index_to_attraction_array(Visum, act_code)
                        
            origin_zone_indices = abm_utilities.get_indices(filtered_act_ex, origin_zone_no_attr_id, global_location_choice_parameter.zoneNo_to_zoneInd, chunks=10000000, reindex = True)
            act_ex_start_times = np.array(visum_utilities.GetMulti(container=filtered_act_ex, attribute='StartTime', chunk_size=10000000), dtype=int)
            act_ex_end_times = np.array(visum_utilities.GetMulti(container=filtered_act_ex, attribute='EndTime', chunk_size=10000000), dtype=int)
            # start and end times must have been set for major activity executions
            assert not np.any(np.isnan(act_ex_start_times))
            assert not np.any(np.isnan(act_ex_end_times))

            obj_start_ti_indices = [abm_utilities.get_time_interval_index(
                time_point, global_location_choice_parameter.time_interval_start_times, global_location_choice_parameter.time_interval_end_times) for time_point in act_ex_start_times]
            obj_end_ti_indices = [abm_utilities.get_time_interval_index(
                time_point, global_location_choice_parameter.time_interval_start_times, global_location_choice_parameter.time_interval_end_times) for time_point in act_ex_end_times]

            act_ex_parameter = location_choice_engine.Location_Choice_Parameter_Act_Ex_Data(
                        act_ex_set=filtered_act_ex, 
                        origin_zones=origin_zone_indices,
                        time_interval_indices_origin_path=obj_start_ti_indices,
                        target_zones=origin_zone_indices, # major location choice: always return back to origin zone
                        time_interval_indices_target_path=obj_end_ti_indices,
                        segment=segment, act_code=act_code, attraction_per_zone=attraction_per_zone)

            location_choice_engine.choose_and_set_locations(global_location_choice_parameter, act_ex_parameter, result_act_ex_attr)

            num_chosen_major_locs += num_act_ex

            logging.info(f'location set for {num_act_ex} activity executions')

    return num_chosen_major_locs


def run(Visum, segments, config,
        zones, locations, zoneNo_to_zoneInd, locationNo_to_locationInd,
        time_interval_start_times, time_interval_end_times):

    global_location_choice_parameter = location_choice_engine.Location_Choice_Parameter_Global(
        config, zones, locations, zoneNo_to_zoneInd, locationNo_to_locationInd,
        time_interval_start_times, time_interval_end_times)

    set_long_term_choices_in_tours(Visum)

    # set location for major activity executions of secondary tours
    num_chosen_locs = major_location_choice_non_primary_tours(global_location_choice_parameter, segments)
    # for every secondary tour the location of one activity location has been set
    assert num_chosen_locs == Visum.Net.Tours.GetFilteredSet(r"[Is_Primary] = 0").Count


def set_long_term_choices_in_tours(Visum):

    activity_codes = VPH.GetMulti(Visum.Net.Activities, "Code")

    major_act_ex_filter_expr = r'([FromTrip\Tour\Is_Primary] = 1) & ([IsMajorActivity] = 1)'
    major_act_ex = Visum.Net.ActivityExecutions.GetFilteredSet(major_act_ex_filter_expr)

    primary_tours = Visum.Net.Tours.GetFilteredSet(r'[Is_Primary] = 1')

    for activity_code in activity_codes:
        major_act_ex_for_activity = major_act_ex.GetFilteredSet(fr'[ActivityCode] = "{activity_code}"')
 
        if major_act_ex_for_activity.Count == 0:
            continue
        location_no_attr_for_activity = fr'SCHEDULE\PERSON\Min:LONGTERMCHOICES([ACTIVITYCODE] = "{activity_code}")\LOCATIONNO'
        major_act_ex_location_no = visum_utilities.GetMulti(
            major_act_ex_for_activity, location_no_attr_for_activity, chunk_size=abm_settings.chunk_size_trips, reindex = True)
        visum_utilities.SetMulti(major_act_ex_for_activity, r'LocationNo',
                                 major_act_ex_location_no, chunk_size=abm_settings.chunk_size_trips)

        primary_tours_with_cur_major_activity = primary_tours.GetFilteredSet(fr'[MajorActCode] = "{activity_code}"')
        zone_no_attr_for_activity = fr'SCHEDULE\PERSON\Min:LONGTERMCHOICES([ACTIVITYCODE] = "{activity_code}")\LOCATION\ZONE\NO'
        major_activity_zone_no = visum_utilities.GetMulti(
            primary_tours_with_cur_major_activity, zone_no_attr_for_activity, chunk_size=abm_settings.chunk_size_trips, reindex = True)
        visum_utilities.SetMulti(primary_tours_with_cur_major_activity, r'MajorActivityZoneNo',
                                 major_activity_zone_no, chunk_size=abm_settings.chunk_size_trips)
