import logging

import VisumPy.helpers as VPH

from src import abm_utilities, visum_utilities, mode_choice_engine, config
from settings import abm_settings

def run(Visum, segments, zoneNo_to_zoneInd):
    major_mode_choice(Visum, segments, zoneNo_to_zoneInd) # major mode for tour

def major_mode_choice(Visum, segments, zoneNo_to_zoneInd):
    num_chosen_modes = 0
    
    # example: activity executions of one tour:
    # Tour: [MajorActExIndex] = 14
    # (previous Tour)
    # 11 H                        ([MinorDestAndModeChoiceSeqIndex] = 0, [IsPartOfSubtour] = 0)
    # 12 ..                       ([MinorDestAndModeChoiceSeqIndex] = 2, [IsPartOfSubtour] = 0)
    # 13 ..                       ([MinorDestAndModeChoiceSeqIndex] = 1, [IsPartOfSubtour] = 0)
    # 14 W  Major Act (1st part)  ([MinorDestAndModeChoiceSeqIndex] = 0, [IsPartOfSubtour] = 1, [IsMajorActEx] = 1)
    # 15 .. Major Act of Subtour  ([MinorDestAndModeChoiceSeqIndex] = 1, [IsPartOfSubtour] = 1, [IsMajorActEx] = 0, [IsMajorActOfSubtour] = 1)
    # 16 ..                       ([MinorDestAndModeChoiceSeqIndex] = 2, [IsPartOfSubtour] = 1, [IsMajorActEx] = 0, [IsMajorActOfSubtour] = 0)
    # 17 W  Major Act (2nd part)  ([MinorDestAndModeChoiceSeqIndex] = 3, [IsPartOfSubtour] = 1, [IsMajorActEx] = 1)
    # 18 ..                       ([MinorDestAndModeChoiceSeqIndex] = 4, [IsPartOfSubtour] = 0)
    # 19 H                        ([MinorDestAndModeChoiceSeqIndex] = 0, [IsPartOfSubtour] = 0, [ActivityCode] = "H")

    # collect all major activity executions first (the first part if split by subtour)
    major_act_exs = Visum.Net.ActivityExecutions.GetFilteredSet(r'([IsMajorActivity] = 1) & ([FromTrip\FromActivityExecution\IsPartOfSubtour] != 1)')
        
    mode_id_to_interchangeable = dict(Visum.Net.Modes.GetMultipleAttributes(["ID", "Interchangeable"]))
    mode_id_to_rank = dict(Visum.Net.Modes.GetMultipleAttributes(["ID", "Rank"]))
    mode_id_to_firstDSegCode = dict(Visum.Net.Modes.GetMultipleAttributes(["ID", r'First:DemandSegments\Code']))

    time_interval_start_times = VPH.GetMulti(Visum.Net.CalendarPeriod.AnalysisTimeIntervalSet.TimeIntervals, 'StartTime')
    time_interval_end_times   = VPH.GetMulti(Visum.Net.CalendarPeriod.AnalysisTimeIntervalSet.TimeIntervals, 'EndTime')
    time_interval_codes       = VPH.GetMulti(Visum.Net.CalendarPeriod.AnalysisTimeIntervalSet.TimeIntervals, 'Code')
    nodeno_to_node = dict(zip(VPH.GetMulti(Visum.Net.Nodes, 'No'), Visum.Net.Nodes.GetAll))
    nodeno_to_longitude_latitude = abm_utilities.get_nodeno_to_longitude_latitude(Visum)

    prt_shortest_path_searcher = Visum.Net.CreatePrTShortestPathSearcher()

    for segment in segments:
        major_act_ex_segment_filtered = abm_utilities.get_filtered_subjects(major_act_exs, segment['Filter'])
        num_major_act_ex_segment_filtered = major_act_ex_segment_filtered.Count
        if num_major_act_ex_segment_filtered == 0:
            continue
        
        logging.info(f"major mode choice for segment {segment['Specification']}: {segment['Comment']}")

        chooseable_mode_ids = segment['Choices']
        
        # compute mode impedance for each term
        # mode impedance for one term is a list of length len(major_act_ex_segment_filtered)
        inbound_mode_impedances_per_term = []
        outbound_mode_impedances_per_term = []
        for [mode_impedance_type, mode_impedance_definition, prtsys] in zip(segment["ModeImpedanceTypes"], segment["ModeImpedanceDefinitions"], segment["ModeImpedancePrTSys"]):
            if mode_impedance_type == config.ModeImpedanceType.PrTShortestPathImpedance:
                # prt impedance
                mode_impedances = mode_choice_engine.compute_prt_impedance_for_major_mode_choice(prt_shortest_path_searcher,
                                                                                                 major_act_ex_segment_filtered,
                                                                                                 mode_impedance_definition,
                                                                                                 prtsys,
                                                                                                 time_interval_start_times,
                                                                                                 time_interval_end_times,
                                                                                                 time_interval_codes,
                                                                                                 nodeno_to_node)
            elif mode_impedance_type == config.ModeImpedanceType.PuTShortestPathImpedance:
                # put impedance
                walk_tsys     = abm_utilities.get_global_attribute(Visum, 'WalkPrTSys', 'Name')
                max_walk_time_in_minutes = abm_utilities.get_global_attribute(Visum, 'MaxWalkTimeInMinutesForPutShortestPathSearch', 'Value')
                node_to_stop_area_indices, node_to_stop_area_distances = mode_choice_engine.compute_node_stoparea_distances(Visum, walk_tsys, max_walk_time_in_minutes)

                dseg_code = mode_impedance_definition
                walking_time_impedance_factor = abm_utilities.get_global_attribute(Visum, 'WalkTimeImpedanceFactor', 'Value')
                mode_impedances = mode_choice_engine.compute_put_impedance_for_major_mode_choice(Visum,
                                                                                                 node_to_stop_area_indices,
                                                                                                 node_to_stop_area_distances,
                                                                                                 time_interval_start_times,
                                                                                                 time_interval_end_times,
                                                                                                 dseg_code,
                                                                                                 walking_time_impedance_factor,
                                                                                                 major_act_ex_segment_filtered)
            elif mode_impedance_type == config.ModeImpedanceType.DirectDistance:
                # direct distance
                mode_impedances = mode_choice_engine.compute_direct_distances_for_major_mode_choice(major_act_ex_segment_filtered, nodeno_to_longitude_latitude)
                mode_impedances = [mode_impedances, mode_impedances] # multiply by 2 in order to account for way back
            else:
                # no impedance specified => use 1.0, since this impedance is multiplied by attrExpr and matrixExp
                mode_impedances = [[1.0] * len(major_act_ex_segment_filtered)] * 2
                
            inbound_mode_impedances_per_term.append(mode_impedances[0])
            outbound_mode_impedances_per_term.append(mode_impedances[1])

        mode_is_interchangeable = [mode_id_to_interchangeable[mode_id] for mode_id in chooseable_mode_ids]
        mode_ranks = [mode_id_to_rank[mode_id] for mode_id in chooseable_mode_ids]        
    
        origin_zones = abm_utilities.get_indices(major_act_ex_segment_filtered, r'Schedule\Person\Household\Residence\Location\Zone\No', zoneNo_to_zoneInd, reindex = True)
        dest_zones   = abm_utilities.get_indices(major_act_ex_segment_filtered, r'Location\Zone\No', zoneNo_to_zoneInd)

        inbound_trip_start_times = visum_utilities.GetMulti(
            major_act_ex_segment_filtered, 'StartTime', chunk_size=abm_settings.chunk_size_trips)
        outbound_trip_start_times = visum_utilities.GetMulti(
            major_act_ex_segment_filtered, 'EndTime', chunk_size=abm_settings.chunk_size_trips)
        inbound_trip_start_time_interval_indices = [abm_utilities.get_time_interval_index(
            time_point, time_interval_start_times, time_interval_end_times) for time_point in inbound_trip_start_times]
        outbound_trip_start_time_interval_indices = [abm_utilities.get_time_interval_index(
            time_point, time_interval_start_times, time_interval_end_times) for time_point in outbound_trip_start_times]

        chosen_mode_ids = mode_choice_engine.calc_major_mode_choice(major_act_ex_segment_filtered,
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
                                                                    chooseable_mode_ids,
                                                                    mode_is_interchangeable,
                                                                    mode_ranks)
        
        chosen_main_dseg_codes = [mode_id_to_firstDSegCode[chosen_mode_id] for chosen_mode_id in chosen_mode_ids]
        visum_utilities.SetMulti(major_act_ex_segment_filtered, r'FromTrip\Tour\MainDSegCode', chosen_main_dseg_codes, chunk_size=abm_settings.chunk_size_trips)
        
        logging.info(f'major mode set for {num_major_act_ex_segment_filtered} tours')

        num_chosen_modes += num_major_act_ex_segment_filtered
    
    assert num_chosen_modes == Visum.Net.Tours.Count

    
