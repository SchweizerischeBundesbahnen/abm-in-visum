import VisumPy.helpers as VPH
import numpy as np

from .activity_duration_choice import choose_activity_durations
from .facility_choice import prepare_choice_probs_for_act_id, choose_facilities
import abmvisum.engines.location_choice_engine as location_choice_engine
import abmvisum.tools.utilities as utilities


def run_combined_destination_and_duration_choice(matrix_cache, Visum, filtered_act_ex, filtered_trips, skims,
                                                 segments_dest, segments_dur, zones,
                                                 zoneNo_to_zoneInd, rand, logging, performing_budget=16.0,
                                                 travel_time_budget=5.0,
                                                 out_of_home_time_budget=18.0):
    secActID_to_activityCode = utilities.build_secondary_act_dict(Visum)
    activityID_to_activityCode = utilities.build_activity_dict(Visum)
    activityID_to_attractionAttr = utilities.build_activityID_to_attractionAttr_dict(Visum)
    activityID_to_primLocZoneNoAttr = dict(Visum.Net.Activities.GetMultipleAttributes(["Id", "lt_loc_zone_number_attr"]))
    prim_actID_to_primLocZoneNoAttr = {}
    attraction_per_act = {}
    location_probs_per_act = {}

    for key, value in activityID_to_primLocZoneNoAttr.items():
        if value != '':
            prim_actID_to_primLocZoneNoAttr[key] = value

    # location is chosen for activity executions with [MinorDestChoiceSeqIndex] >= 1
    # example: activity executions of one tour:
    # Tour: [MajorActExIndex] = 14
    # (previous Tour)
    # 10 H                        ([MinorDestChoiceSeqIndex] = 0, [IsPartOfSubtour] = 0)
    # 12 ..                       ([MinorDestChoiceSeqIndex] = 1, [IsPartOfSubtour] = 0)
    # 13 ..                       ([MinorDestChoiceSeqIndex] = 2, [IsPartOfSubtour] = 0)
    # 14 W  Major Act (1st part)  ([MinorDestChoiceSeqIndex] = 0, [IsPartOfSubtour] = 1, [IsMajorActEx] = 1)
    # 15 .. Major Act of Subtour  ([MinorDestChoiceSeqIndex] = 1, [IsPartOfSubtour] = 1, [IsMajorActEx] = 0, [IsMajorActOfSubtour] = 1)
    # 16 ..                       ([MinorDestChoiceSeqIndex] = 2, [IsPartOfSubtour] = 1, [IsMajorActEx] = 0, [IsMajorActOfSubtour] = 0)
    # 17 W  Major Act (2nd part)  ([MinorDestChoiceSeqIndex] = 0, [IsPartOfSubtour] = 1, [IsMajorActEx] = 1)
    # 18 ..                       ([MinorDestChoiceSeqIndex] = 1, [IsPartOfSubtour] = 0)
    # 19 H                        ([MinorDestChoiceSeqIndex] = 0, [IsPartOfSubtour] = 0, [ActivityCode] = "H")

    init_act_ex_and_set_prim_loc(filtered_act_ex, segments_dest, prim_actID_to_primLocZoneNoAttr,
                                 activityID_to_activityCode)
    travel_times = get_tt_matrix(skims)

    # dest zones are chosen consecutively (starting with [MinorDestChoiceSeqIndex] = 1) using rubberbanding the
    # different groups (outbound, subtour, inbound) can be calculated together (as their origin and target location
    # is already fixed: it is either the major act ex or home)

    for segment in segments_dest:

        filter_expression = segment['Filter'].replace('_P_', 'Tour\\Schedule\\Person')
        trips_dest_choice_segment = utilities.get_filtered_subjects(filtered_trips, filter_expression)
        if trips_dest_choice_segment.Count == 0:
            continue

        # persons who optimise their plans
        persons_unique, person_indices = np.unique(np.array(utilities.GetMulti(trips_dest_choice_segment, "PersonNo"),
                                                            dtype=np.int), return_inverse=True)
        nb_of_persons = persons_unique.shape[0]

        # build duration choice set
        number_of_plans_dur = 10
        act_performing_choice_set, tot_performing_budget_choice_set = build_performing_choice_set(segments_dur,
                                                                                                  nb_of_persons,
                                                                                                  person_indices,
                                                                                                  trips_dest_choice_segment,
                                                                                                  rand, logging,
                                                                                                  number_of_plans_dur)

        # build destination choice set
        number_of_plans_dest = 3
        dest_zones_choice_set, tot_travel_time_budget_choice_set = build_destination_choice_set(Visum, segment,
                                                                                                nb_of_persons,
                                                                                                person_indices,
                                                                                                trips_dest_choice_segment,
                                                                                                zoneNo_to_zoneInd,
                                                                                                secActID_to_activityCode,
                                                                                                activityID_to_attractionAttr,
                                                                                                attraction_per_act,
                                                                                                matrix_cache, skims,
                                                                                                travel_times, zones,
                                                                                                location_probs_per_act,
                                                                                                logging,
                                                                                                number_of_plans_dest)

        # handling of persons with minimal performing time higher than y hours
        chosen_dur_plan_ind = np.zeros(nb_of_persons, dtype=int)
        max_performing_time_cond = (np.min(tot_performing_budget_choice_set, axis=1) >= performing_budget)
        chosen_dur_plan_ind[max_performing_time_cond] = np.argmin(tot_performing_budget_choice_set[
                                                                      max_performing_time_cond], axis=1)
        nb_durations_to_opt = chosen_dur_plan_ind[~max_performing_time_cond].shape[0]
        tot_performing_budget_choice_set_red = tot_performing_budget_choice_set[~max_performing_time_cond]

        # handling of persons with minimal travel time higher than x hours
        chosen_dest_plan_ind = np.zeros(nb_of_persons, dtype=int)
        max_travel_time_cond = (np.min(tot_travel_time_budget_choice_set, axis=1) >= travel_time_budget)
        chosen_dest_plan_ind[max_travel_time_cond] = np.argmin(tot_travel_time_budget_choice_set[max_travel_time_cond],
                                                               axis=1)
        nb_destinations_to_opt = chosen_dest_plan_ind[~max_travel_time_cond].shape[0]
        tot_travel_time_budget_choice_set_red = tot_travel_time_budget_choice_set[~max_travel_time_cond]

        logging.info("number of persons: " + str(nb_of_persons))
        logging.info("number of persons with minimal performing time higher than " + str(performing_budget) +
                     " hours: " + str(chosen_dur_plan_ind[max_performing_time_cond].shape[0]))
        logging.info("number of persons with minimal travel time higher than " + str(travel_time_budget)
                     + " hours: " + str(chosen_dest_plan_ind[max_travel_time_cond].shape[0]))

        # here starts the plan choice procedure
        best_chosen_dur_plan_ind = np.zeros(nb_of_persons, dtype=int)
        best_chosen_dest_plan_ind = np.zeros(nb_of_persons, dtype=int)
        best_total_out_of_home_time = np.zeros(nb_of_persons, dtype=float)
        for __ in range(200):
            chosen_dur_plan_ind_temp = rand.randint(number_of_plans_dur, size=nb_durations_to_opt)
            chosen_dest_plan_ind_temp = rand.randint(number_of_plans_dest, size=nb_destinations_to_opt)
            chosen_performing_time_budgets_temp = tot_performing_budget_choice_set_red[np.arange(nb_durations_to_opt),
                                                                                       chosen_dur_plan_ind_temp]
            chosen_travel_time_budgets_temp = tot_travel_time_budget_choice_set_red[np.arange(nb_destinations_to_opt),
                                                                                    chosen_dest_plan_ind_temp]

            # make sure that no plan has total performing times higher than 16 hours
            for i in range(number_of_plans_dur):
                mask = (chosen_performing_time_budgets_temp >= performing_budget)
                if chosen_performing_time_budgets_temp[mask].shape[0] == 0:
                    break
                else:
                    chosen_dur_plan_ind_temp[mask] = chosen_dur_plan_ind_temp[mask] - 1
                    chosen_dur_plan_ind_temp[chosen_dur_plan_ind_temp == -1] = number_of_plans_dur - 1
                    chosen_performing_time_budgets_temp = tot_performing_budget_choice_set_red[
                        np.arange(nb_durations_to_opt), chosen_dur_plan_ind_temp]

            # make sure that no plan has total travel times higher than 5 hours
            for i in range(number_of_plans_dest):
                mask = (chosen_travel_time_budgets_temp >= travel_time_budget)
                if chosen_travel_time_budgets_temp[mask].shape[0] == 0:
                    break
                else:
                    chosen_dest_plan_ind_temp[mask] = chosen_dest_plan_ind_temp[mask] - 1
                    chosen_dest_plan_ind_temp[chosen_dest_plan_ind_temp == -1] = number_of_plans_dest - 1
                    chosen_travel_time_budgets_temp = tot_travel_time_budget_choice_set_red[
                        np.arange(nb_destinations_to_opt), chosen_dest_plan_ind_temp]

            # choose all durations
            chosen_dur_plan_ind[~max_performing_time_cond] = chosen_dur_plan_ind_temp
            chosen_performing_time_budgets = tot_performing_budget_choice_set[np.arange(nb_of_persons),
                                                                              chosen_dur_plan_ind]
            # choose all destinations
            chosen_dest_plan_ind[~max_travel_time_cond] = chosen_dest_plan_ind_temp
            chosen_travel_time_budgets = tot_travel_time_budget_choice_set[np.arange(nb_of_persons),
                                                                           chosen_dest_plan_ind]
            # total out of home time
            total_out_of_home_time = chosen_performing_time_budgets + chosen_travel_time_budgets

            if __ == 0:
                best_chosen_dur_plan_ind = np.copy(chosen_dur_plan_ind)
                best_chosen_dest_plan_ind = np.copy(chosen_dest_plan_ind)
                best_total_out_of_home_time = np.copy(total_out_of_home_time)
            else:
                mask = ((best_total_out_of_home_time > out_of_home_time_budget) &
                        (total_out_of_home_time < best_total_out_of_home_time))
                best_chosen_dur_plan_ind[mask] = chosen_dur_plan_ind[mask]
                best_chosen_dest_plan_ind[mask] = chosen_dest_plan_ind[mask]
                best_total_out_of_home_time[mask] = total_out_of_home_time[mask]

        logging.info("number of persons with total out-of-home-time higher than " + str(out_of_home_time_budget)
                     + " hours: " +
                     str(best_total_out_of_home_time[best_total_out_of_home_time >= out_of_home_time_budget].shape[0]))

        # write results back to Visum
        chosen_durations = act_performing_choice_set[np.arange(person_indices.shape[0]),
                                                     best_chosen_dur_plan_ind[person_indices]]
        utilities.SetMulti(trips_dest_choice_segment, r"ToActivityExecution\Duration", chosen_durations)

        chosen_zones = dest_zones_choice_set[np.arange(person_indices.shape[0]),
                                             best_chosen_dest_plan_ind[person_indices]]
        utilities.SetMulti(trips_dest_choice_segment, r"ToActivityExecution\zone_id", chosen_zones)

        origin_zones = utilities.get_zone_indices(trips_dest_choice_segment, r"FromActivityExecution\zone_id",
                                                  zoneNo_to_zoneInd)
        dest_zones = utilities.get_zone_indices(trips_dest_choice_segment, r"ToActivityExecution\zone_id",
                                                zoneNo_to_zoneInd)
        trip_tt = travel_times[origin_zones, dest_zones]
        utilities.SetMulti(trips_dest_choice_segment, "Duration", trip_tt)
        trip_d = skims.get_skim("car_net_distance_sym")[origin_zones, dest_zones]
        utilities.SetMulti(trips_dest_choice_segment, "distance", trip_d)

    # choose specific locations in zones
    relevant_act_ex = filtered_act_ex.GetFilteredSet(r'[secondary_activity_index] > 0')
    for act_id, act_code in activityID_to_activityCode.items():
        curAct_act_ex = relevant_act_ex.GetFilteredSet('[ActivityCode]="' + act_code + '"')
        if curAct_act_ex.Count == 0:
            continue

        if act_id in attraction_per_act.keys():
            # todo cache that
            facility_options, facility_probs = location_probs_per_act[act_id]
        else:
            facility_options, facility_probs = prepare_choice_probs_for_act_id(Visum, act_id)

        chosen_zone_ind = utilities.get_zone_indices(curAct_act_ex, r'zone_id', zoneNo_to_zoneInd)
        chosen_facilities = choose_facilities(chosen_zone_ind, facility_probs, facility_options, logging)
        utilities.SetMulti(curAct_act_ex, "LocationNo", chosen_facilities)


def build_performing_choice_set(segments_dur, nb_of_persons, person_indices, trips_dest_choice_segment, rand, logging,
                                number_of_plans_dur=15):
    act_performing_choice_set = np.zeros((person_indices.shape[0], number_of_plans_dur), dtype=float)
    tot_performing_budget_choice_set = np.zeros((nb_of_persons, number_of_plans_dur), dtype=float)
    for act_dur_segment in segments_dur:
        trips_act_dur_segment = np.array(utilities.GetMultiByFormula(trips_dest_choice_segment,
                                                                     "if(" + act_dur_segment['Filter'] + ";1;0)"))
        nb_relevant_activities = act_performing_choice_set[trips_act_dur_segment == 1].shape[0]
        logging.info(
            'choosing activity durations (number of activities: %d, segment %s, )' % (nb_relevant_activities,
                                                                                      act_dur_segment[
                                                                                          'Specification']))

        for subsegment_index, sub_segment_filter_expr in enumerate(act_dur_segment["AttrExpr"]):
            sub_segment_filter_expr = sub_segment_filter_expr.replace('_A_', 'ToActivityExecution\\ActivityCode')
            sub_segment_filter_expr = sub_segment_filter_expr.replace('_P_', 'Tour\\Schedule\\Person')

            trips_act_dur_sub_segment = np.array(utilities.GetMultiByFormula(trips_dest_choice_segment,
                                                                             "if(" + sub_segment_filter_expr + ";1;0)"))
            cond = ((trips_act_dur_segment == 1) & (trips_act_dur_sub_segment == 1))
            nb_relevant_sub_activities = act_performing_choice_set[cond].shape[0]

            if nb_relevant_sub_activities > 0:
                distribution_data = act_dur_segment["distribution_data"][subsegment_index]
                for i in range(number_of_plans_dur):
                    result = choose_activity_durations(nb_relevant_sub_activities,  distribution_data, rand)
                    act_performing_choice_set[cond, i] = result
                    assert (result.shape[0] == nb_relevant_sub_activities)

    for i in range(number_of_plans_dur):
        tot_performing_budget_choice_set[:, i] = np.bincount(person_indices,
                                                             weights=act_performing_choice_set[:, i]) / 3600
    logging.info("built duration choice set consisting of " + str(number_of_plans_dur) + " plans.")
    return act_performing_choice_set, tot_performing_budget_choice_set


def build_destination_choice_set(Visum, segment, nb_of_persons, person_indices, trips_dest_choice_segment,
                                 zoneNo_to_zoneInd, secActID_to_activityCode, activityID_to_attractionAttr,
                                 attraction_per_act, matrix_cache, skims, travel_times, zones,
                                 location_probs_per_act, logging, number_of_plans_dest=5):
    trips_dest_ind = np.array(utilities.GetMulti(trips_dest_choice_segment,
                                                 "ToActivityExecution\\secondary_activity_index"), dtype=np.int)
    trips_to_act_id = np.array(utilities.GetMulti(trips_dest_choice_segment, "ToActivityExecution\\Activity\\Id"),
                               dtype=np.int)
    trips_target_zone_ind, trips_origin_zone_ind, trips_to_act_zone_ind = collect_data_for_trip_segment(
        trips_dest_choice_segment, zoneNo_to_zoneInd)

    nb_relevant_trips = trips_to_act_zone_ind[trips_dest_ind > 0].shape[0]
    logging.info('choosing trip destinations  (number of trips: %d, segment %s, )' % (nb_relevant_trips,
                                                                                      segment['Specification']))

    travel_time_choice_set = np.zeros((nb_of_persons, number_of_plans_dest), dtype=float)
    dest_zones_choice_set = np.zeros((trips_target_zone_ind.shape[0], number_of_plans_dest), dtype=int)
    for i in range(number_of_plans_dest):
        logging.info('building destination choice set: plan ' + str(i + 1))
        cur_dest_ind = 1
        while True:
            if trips_origin_zone_ind[trips_dest_ind == cur_dest_ind].shape[0] == 0:
                break

            for act_id in secActID_to_activityCode.keys():
                attraction_per_zone = update_attr_and_loc_probs(attraction_per_act, location_probs_per_act, act_id,
                                                                activityID_to_attractionAttr, Visum, logging)

                # full tour trips
                trips_origin_zone_ind, trips_to_act_zone_ind = choose_destinations(cur_dest_ind, trips_dest_ind,
                                                                                   act_id, trips_to_act_id,
                                                                                   trips_target_zone_ind,
                                                                                   trips_origin_zone_ind,
                                                                                   trips_to_act_zone_ind, False,
                                                                                   matrix_cache,
                                                                                   skims,
                                                                                   Visum, segment,
                                                                                   attraction_per_zone, logging)
                # rubberbanding trips
                trips_origin_zone_ind, trips_to_act_zone_ind = choose_destinations(cur_dest_ind, trips_dest_ind,
                                                                                   act_id, trips_to_act_id,
                                                                                   trips_target_zone_ind,
                                                                                   trips_origin_zone_ind,
                                                                                   trips_to_act_zone_ind, True,
                                                                                   matrix_cache,
                                                                                   skims,
                                                                                   Visum, segment,
                                                                                   attraction_per_zone, logging)
            cur_dest_ind += 1

        trip_travel_times = travel_times[trips_origin_zone_ind, trips_to_act_zone_ind]
        travel_time_choice_set[:, i] = np.bincount(person_indices, weights=trip_travel_times) / 3600

        assert trips_origin_zone_ind[trips_origin_zone_ind == -1].shape[0] == 0
        assert trips_to_act_zone_ind[trips_to_act_zone_ind == -1].shape[0] == 0
        dest_zones_choice_set[:, i] = zones[trips_to_act_zone_ind]

    logging.info("built destination choice set consisting of " + str(number_of_plans_dest) + " plans.")
    return dest_zones_choice_set, travel_time_choice_set


def choose_destinations(cur_dest_ind, trips_dest_ind, act_id, trips_to_act_id, trips_target_zone_ind,
                        trips_origin_zone_ind, trips_to_act_zone_ind, rubberbanding, matrix_cache, skims,
                        Visum, segment, attraction_per_zone, logging):
    if rubberbanding:
        mask = ((trips_dest_ind == cur_dest_ind) &
                (trips_to_act_id == act_id) &
                (trips_target_zone_ind >= 0))
    else:
        mask = ((trips_dest_ind == cur_dest_ind) &
                (trips_to_act_id == act_id) &
                (trips_target_zone_ind == -1))
    origins = trips_origin_zone_ind[mask]
    if origins.shape[0] > 0:
        assert origins[origins == -1].shape[0] == 0
        targets = None
        if rubberbanding:
            targets = trips_target_zone_ind[mask]
            assert targets[targets == -1].shape[0] == 0

        chosen_zone_ind = location_choice_engine.choose_and_set_minor_destinations(matrix_cache, skims, act_id,
                                                                                   Visum, segment, attraction_per_zone,
                                                                                   origins, logging, targets)
        o_temp = np.roll(trips_origin_zone_ind, -1)
        o_temp[mask] = chosen_zone_ind
        trips_origin_zone_ind = np.roll(o_temp, 1)
        trips_to_act_zone_ind[mask] = chosen_zone_ind
    return trips_origin_zone_ind, trips_to_act_zone_ind


def update_attr_and_loc_probs(attraction_per_act, location_probs_per_act, act_id,
                              activityID_to_attractionAttr, Visum, logging):
    if act_id in attraction_per_act.keys():
        attraction_per_zone = attraction_per_act[act_id]
    else:
        attraction_attr = activityID_to_attractionAttr[act_id]
        assert len(attraction_attr) > 0
        attraction_per_zone = np.array(VPH.GetMulti(Visum.Net.Zones, attraction_attr))

        facility_options, facility_probs = prepare_choice_probs_for_act_id(Visum, act_id)
        # set attraction to zero if there is no facility in that zone
        counter = 0
        for i, opt in enumerate(facility_options):
            if opt[0] == 'None':
                if attraction_per_zone[i] != 0:
                    attraction_per_zone[i] = 0
                    counter += 1
        if counter > 0:
            logging.info("set attraction " + attraction_attr + " value of " + str(counter) + " zones to zero.")

        # todo cache that
        location_probs_per_act[act_id] = facility_options, facility_probs
        attraction_per_act[act_id] = attraction_per_zone
    return attraction_per_zone


def collect_data_for_trip_segment(trips_segment, zoneNo_to_zoneInd):
    trips_to_act_zone_no = np.array(utilities.GetMulti(trips_segment, "ToActivityExecution\\zone_id"),
                                    dtype=np.int)
    trips_origin_zone_no = np.array(utilities.GetMulti(trips_segment, "FromActivityExecution\\zone_id"),
                                    dtype=np.int)
    isHomeboundActEx = utilities.eval_attrexpr(trips_segment, [
        r'([ToActivityExecution\Index] > [Tour\main_activity_execution_index]) & ([ToActivityExecution\is_part_of_subtour]=0)'],
                                               np.int)[:, 0]
    majorActZoneNosAndHomeLocs = np.array(trips_segment.GetMultipleAttributes(
        [r'Tour\main_activity_zone_id', r'Tour\Schedule\Person\Household\Residence\Location\Zone\No']), dtype=int)
    trips_target_zone_no = majorActZoneNosAndHomeLocs[np.arange(len(isHomeboundActEx)), isHomeboundActEx]

    trips_target_zone_ind = np.negative(np.ones(trips_target_zone_no.shape[0], dtype=int))
    trips_origin_zone_ind = np.negative(np.ones(trips_origin_zone_no.shape[0], dtype=int))
    trips_to_act_zone_ind = np.negative(np.zeros(trips_to_act_zone_no.shape[0], dtype=int))

    for i in range(trips_target_zone_ind.shape[0]):
        if (trips_target_zone_no[i] > 0) and (majorActZoneNosAndHomeLocs[i, 0] > 0):
            trips_target_zone_ind[i] = zoneNo_to_zoneInd[trips_target_zone_no[i]]
        if trips_origin_zone_no[i] > 0:
            trips_origin_zone_ind[i] = zoneNo_to_zoneInd[trips_origin_zone_no[i]]
        if trips_to_act_zone_no[i] > 0:
            trips_to_act_zone_ind[i] = zoneNo_to_zoneInd[trips_to_act_zone_no[i]]

    return trips_target_zone_ind, trips_origin_zone_ind, trips_to_act_zone_ind


def get_tt_matrix(skims):
    travel_times_car = (skims.get_skim("car_travel_times_sym") * 60.0 +
                        skims.get_skim("at_car") + skims.get_skim("at_car").T +
                        60 * skims.get_skim("pc_car") * 2 * 0.047 / 0.125)
    travel_times_pt = 60 * (skims.get_skim("pt_travel_times_train_sym") + skims.get_skim("pt_travel_times_bus_sym") +
                            skims.get_skim("pt_access_times_sym") + skims.get_skim("pt_egress_times_sym") +
                            skims.get_skim("pt_transfers_sym") * 5)  # own assumption: 5 minutes per transfer
    travel_times = np.where(travel_times_car > 2 * travel_times_pt, travel_times_pt,
                            np.where(travel_times_pt > 2 * travel_times_car, travel_times_car,
                                     (travel_times_car + travel_times_pt) / 2))
    travel_times = np.where(skims.get_skim("car_net_distance_sym") < 1.5,
                            60 * skims.get_skim("car_net_distance_sym") / 0.1, travel_times)
    return 1.2 * travel_times


def init_act_ex_and_set_prim_loc(filtered_act_ex, segments, prim_actID_to_primLocZoneNoAttr,
                                 activityID_to_activityCode):
    relevant_act_ex = filtered_act_ex.GetFilteredSet(r'[secondary_activity_index] > 0')
    relevant_act_ex.SetAllAttValues("zone_id", 0)
    relevant_act_ex.SetAllAttValues("LocationNo", None)

    # set primary locations
    for act_id, prim_zone_attr in prim_actID_to_primLocZoneNoAttr.items():
        act_code = activityID_to_activityCode[act_id]
        curSegment_curAct_act_ex = relevant_act_ex.GetFilteredSet('[ActivityCode]="' + act_code + '"')
        num_act_ex = curSegment_curAct_act_ex.Count
        if num_act_ex == 0:
            continue
        prim_zones = utilities.GetMulti(curSegment_curAct_act_ex, 'Schedule\\Person\\' + prim_zone_attr)
        utilities.SetMulti(curSegment_curAct_act_ex, r'zone_id', prim_zones)
