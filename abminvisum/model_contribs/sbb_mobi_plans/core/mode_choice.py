import numpy as np

import abminvisum.tools.utilities as utilities
from abminvisum.engines.choice_engine import Choice2D


def run_mode_choice(v, config, zoneNo_to_zoneInd, logging, rand):
    imped_para_dict = config.impedance_expr
    modes_interchangeable = dict(v.Net.Modes.GetMultipleAttributes(["Name", "Interchangeable"]))

    # find home activity id
    visum_activities = v.Net.Activities.GetMultipleAttributes(["Id", "IsHomeActivity"])
    home_act_id = -1
    for act_id, is_home_act in visum_activities:
        if is_home_act == 1:
            home_act_id = int(act_id)
    assert home_act_id >= 0

    # mode index to mode name
    modeInd_to_modeName = {}
    for imp_para_values in imped_para_dict.values():
        for mode_ind, (mode, util_formulation) in enumerate(imp_para_values[1]):
            modeInd_to_modeName[mode_ind] = mode

    # 1. handle non-subtour trips
    # rule-based approach: in each tour, choice choice based on the longest-distance trip is made
    non_subt_trip_expr = r'(([FromActivityExecution\is_part_of_subtour] = 0) | ([ToActivityExecution\is_part_of_subtour] = 0))'
    non_subt_trips = v.Net.Trips.GetFilteredSet(non_subt_trip_expr)
    longest_non_subt_trips_expr = r'([Tour\Min:Trips(%s & ([Tour\Max:Trips%s\distance] = [distance]))\Index] = [Index])' % (non_subt_trip_expr, non_subt_trip_expr)
    # there might be two trips with the same distance
    longest_non_subt_trips = non_subt_trips.GetFilteredSet(longest_non_subt_trips_expr)
    assert longest_non_subt_trips.Count == v.Net.Tours.Count

    # get all the necessary infos (demand segment, O-D, etc.)
    non_subt_trips_to_tours, to_act_codes, from_act_zone_id, to_act_zone_id = get_non_subtour_trip_info(
        non_subt_trips,
        longest_non_subt_trips,
        home_act_id, zoneNo_to_zoneInd,
        logging)

    # 2. handle subtour trips
    # rule-based approach: in each subtour, choice based on the longest-distance trip is made
    subt_trip_expr = r'(([FromActivityExecution\is_part_of_subtour] = 1) & ([ToActivityExecution\is_part_of_subtour] = 1))'
    subt_trips = v.Net.Trips.GetFilteredSet(subt_trip_expr)
    longest_subt_trip_expr = r'([Tour\Min:Trips(%s & ([Tour\Max:Trips%s\distance] = [distance]))\Index] = [Index])' % (subt_trip_expr, subt_trip_expr)
    longest_subt_trips = subt_trips.GetFilteredSet(longest_subt_trip_expr)
    assert longest_subt_trips.Count < subt_trips.Count

    # get all the necessary infos (demand segment, O-D, etc.)
    subt_trips_to_tours, sub_to_act_codes, subt_from_act_zone_id, subt_to_act_zone_id = get_subtour_trip_info(
        subt_trips,
        longest_subt_trips,
        zoneNo_to_zoneInd,
        logging)

    # 3. calc utility matrix based on trip infos and skims
    non_subt_mode_utils, subt_mode_utils = calc_util_matrix(longest_non_subt_trips, to_act_codes, from_act_zone_id,
                                                            to_act_zone_id, longest_subt_trips, sub_to_act_codes,
                                                            subt_from_act_zone_id, subt_to_act_zone_id,
                                                            modeInd_to_modeName, imped_para_dict,
                                                            config, logging)

    # 4. make choices for all non-subtour trips
    chosen_modes_per_tour = make_mode_choice(non_subt_mode_utils, modeInd_to_modeName, rand, logging)
    chosen_modes_per_non_subt_trip = chosen_modes_per_tour[non_subt_trips_to_tours]
    assert non_subt_trips.Count == chosen_modes_per_non_subt_trip.shape[0]

    # 4.1 write results back into Visum
    utilities.SetMulti(non_subt_trips, 'mode', chosen_modes_per_non_subt_trip, chunks=20000000)

    # 5. make choices for all subtour trips - constraint by non subtour mode
    # if mode is not interchangeable (car) and tour mode is not car -> remove car from choice set
    subt_main_modes = np.array(utilities.GetMulti(longest_subt_trips, r'Tour\First:Trips\mode'))
    for mode_ind, mode_name in modeInd_to_modeName.items():
        if modes_interchangeable[mode_name] == 0:
            __ = subt_mode_utils[subt_main_modes != mode_name]
            __[:, mode_ind] = 0
            subt_mode_utils[subt_main_modes != mode_name] = __

    chosen_modes_per_tour = make_mode_choice(subt_mode_utils, modeInd_to_modeName, rand, logging)
    chosen_modes_per_subt_trip = chosen_modes_per_tour[subt_trips_to_tours]
    assert subt_trips.Count == chosen_modes_per_subt_trip.shape[0]

    # 5.1 write results back into Visum
    utilities.SetMulti(subt_trips, 'mode', chosen_modes_per_subt_trip)

    # 6. reset car modes in the case the person has no car avail
    no_car_avail_expr = r'([Tour\Schedule\Person\car_available] = 0)'
    car_trips_no_caravail = v.Net.Trips.GetFilteredSet(r'([mode] = "car") & %s' % no_car_avail_expr)
    utilities.SetMulti(car_trips_no_caravail, 'mode', (['ride'] * car_trips_no_caravail.Count))

    # 7. update travel time according to chosen mode
    all_trip_modes = np.array(utilities.GetMulti(v.Net.Trips, r'mode', chunks=20000000, reindex=True))
    from_act_zone_id = utilities.get_zone_indices(v.Net.Trips, r'FromActivityExecution\zone_id',
                                                  zoneNo_to_zoneInd, chunks=20000000)
    to_act_zone_id = utilities.get_zone_indices(v.Net.Trips, r'ToActivityExecution\zone_id',
                                                zoneNo_to_zoneInd, chunks=20000000)
    all_trip_travel_times = np.zeros(all_trip_modes.shape[0], dtype=float)
    tt_matrix_dict = get_tt_matrix_dict(config.skim_matrices)
    for mode in modeInd_to_modeName.values():
        cur_from_act_zone_id = from_act_zone_id[all_trip_modes == mode]
        cur_to_act_zone_id = to_act_zone_id[all_trip_modes == mode]
        cur_tt_mat = tt_matrix_dict[mode]
        all_trip_travel_times[all_trip_modes == mode] = cur_tt_mat[cur_from_act_zone_id, cur_to_act_zone_id]
    utilities.SetMulti(v.Net.Trips, 'Duration', all_trip_travel_times, chunks=20000000)


def get_non_subtour_trip_info(non_subtour_trips, longest_non_subtour_trips, home_act_id, zoneNo_to_zoneInd,
                              logging):
    logging.info('Collecting data for non-subtour trips...')

    trip_info = np.concatenate((np.array(utilities.GetMulti(non_subtour_trips, r'PersonNo'))[:, np.newaxis],
                                np.array(utilities.GetMulti(non_subtour_trips, r'TourNo'))[:, np.newaxis]),
                               axis=1).astype(int)
    __, non_subt_trips_to_tours = np.unique(trip_info, axis=0, return_inverse=True)

    from_act_codes = np.array(utilities.GetMulti(longest_non_subtour_trips, r'FromActivityExecution\Activity\Id'),
                              dtype=int)
    to_act_codes = np.array(utilities.GetMulti(longest_non_subtour_trips, r'ToActivityExecution\Activity\Id'),
                            dtype=int)

    to_act_codes[to_act_codes == home_act_id] = from_act_codes[to_act_codes == home_act_id]

    from_act_zone_id = utilities.get_zone_indices(longest_non_subtour_trips, r'FromActivityExecution\zone_id',
                                                  zoneNo_to_zoneInd)
    to_act_zone_id = utilities.get_zone_indices(longest_non_subtour_trips, r'ToActivityExecution\zone_id',
                                                zoneNo_to_zoneInd)
    assert np.max(to_act_codes) < 9
    assert to_act_codes[to_act_codes == 0].shape[0] == 0
    assert to_act_codes.shape[0] == longest_non_subtour_trips.Count

    return non_subt_trips_to_tours, to_act_codes, from_act_zone_id, to_act_zone_id


def get_subtour_trip_info(subtour_trips, longest_subtour_trips, zoneNo_to_zoneInd, logging):
    logging.info('Collecting data for subtour trips...')

    trip_info = np.concatenate((np.array(utilities.GetMulti(subtour_trips, r'PersonNo'))[:, np.newaxis],
                                np.array(utilities.GetMulti(subtour_trips, r'TourNo'))[:, np.newaxis]),
                               axis=1).astype(int)
    __, subt_trips_to_tours = np.unique(trip_info, axis=0, return_inverse=True)

    from_act_codes = np.array(utilities.GetMulti(longest_subtour_trips, r'FromActivityExecution\Activity\Id'),
                              dtype=int)
    to_act_codes = np.array(utilities.GetMulti(longest_subtour_trips, r'ToActivityExecution\Activity\Id'), dtype=int)
    major_act_code = np.array(utilities.GetMulti(longest_subtour_trips, r'Tour\main_activity_id'), dtype=int)
    to_act_codes[to_act_codes == major_act_code] = from_act_codes[to_act_codes == major_act_code]

    from_act_zone_id = utilities.get_zone_indices(longest_subtour_trips, r'FromActivityExecution\zone_id',
                                                  zoneNo_to_zoneInd)
    to_act_zone_id = utilities.get_zone_indices(longest_subtour_trips, r'ToActivityExecution\zone_id',
                                                zoneNo_to_zoneInd)
    assert np.max(to_act_codes) < 9
    assert to_act_codes[to_act_codes == 0].shape[0] == 0
    assert to_act_codes.shape[0] == longest_subtour_trips.Count

    return subt_trips_to_tours, to_act_codes, from_act_zone_id, to_act_zone_id


def calc_util_matrix(non_subt_trips, to_act_codes, from_act_zone_id, to_act_zone_id,
                     subt_trips, sub_to_act_codes, subt_from_act_zone_id, subt_to_act_zone_id,
                     modeInd_to_modeName, imped_para_dict, config, logging):
    non_subt_mode_utils = np.zeros((non_subt_trips.Count, len(modeInd_to_modeName)), dtype=float)
    subt_mode_utils = np.zeros((subt_trips.Count, len(modeInd_to_modeName)), dtype=float)

    logging.info('Need to make %d choices for non-subtour trips' % non_subt_trips.Count)
    logging.info('Need to make %d choices for subtour trips' % subt_trips.Count)
    logging.info('Calculating utility matrices for all demand segments')

    nb_tot_non_subt_trips = 0
    nb_tot_subt_trips = 0

    for imp_para_key, imp_para_values in imped_para_dict.items():
        cur_act_id = int(imp_para_key[0])
        filt_expr = imp_para_values[0]
        logging.info(' - current segment: %s' % imp_para_key[1])
        if filt_expr is None:
            continue

        filt_expr = filt_expr.replace('_P_', 'Tour\\Schedule\\Person')
        d_seg_trips = np.array(utilities.GetMultiByFormula(non_subt_trips, 'IF(%s,1,0)' % filt_expr))
        assert d_seg_trips.shape[0] == non_subt_trips.Count
        __mask = ((d_seg_trips == 1) & (to_act_codes == cur_act_id))
        subt_d_seg_trips = np.array(utilities.GetMultiByFormula(subt_trips, 'IF(%s,1,0)' % filt_expr))
        assert subt_d_seg_trips.shape[0] == subt_trips.Count
        __subt_mask = ((subt_d_seg_trips == 1) & (sub_to_act_codes == cur_act_id))

        # get utility matrices per mode
        nb_non_subt_trips = d_seg_trips[__mask].shape[0]
        nb_tot_non_subt_trips += nb_non_subt_trips
        temp_mode_utils = np.zeros((nb_non_subt_trips, len(modeInd_to_modeName)), dtype=float)

        nb_subt_trips = subt_d_seg_trips[__subt_mask].shape[0]
        nb_tot_subt_trips += nb_subt_trips
        subt_temp_mode_utils = np.zeros((nb_subt_trips, len(modeInd_to_modeName)), dtype=float)
        for mode_ind, (mode, util_formulation) in enumerate(imp_para_values[1]):
            if util_formulation != '-9999':
                mode_util_mat = utilities.calc_utils_for_matr_expr(config.skim_matrices, util_formulation)
                # non subtour trips
                cur_from_zones = from_act_zone_id[__mask]
                cur_to_zones = to_act_zone_id[__mask]
                util_vector = mode_util_mat[cur_from_zones, cur_to_zones]
                temp_mode_utils[:, mode_ind] = util_vector
                # subtour trips
                subt_cur_from_zones = subt_from_act_zone_id[__subt_mask]
                subt_cur_to_zones = subt_to_act_zone_id[__subt_mask]
                subt_util_vector = mode_util_mat[subt_cur_from_zones, subt_cur_to_zones]
                subt_temp_mode_utils[:, mode_ind] = subt_util_vector

        non_subt_mode_utils[__mask] = temp_mode_utils
        subt_mode_utils[__subt_mask] = subt_temp_mode_utils

    assert nb_tot_non_subt_trips == non_subt_trips.Count
    assert nb_tot_subt_trips == subt_trips.Count
    return non_subt_mode_utils, subt_mode_utils


def make_mode_choice(mode_utils, modeInd_to_modeName, rand, logging):
    logging.info('making choices for %d trips' % mode_utils.shape[0])
    prob_row_sums = mode_utils.sum(axis=1)
    assert prob_row_sums[prob_row_sums == 0].shape[0] == 0
    prob_matrix = mode_utils / prob_row_sums[:, np.newaxis]

    _choice = Choice2D(prob_matrix.shape[0])
    _choice.add_prob(prob_matrix)
    chosen_mode_ind = _choice.choose(rand)
    chosen_mode_names = np.array([modeInd_to_modeName[__] for __ in chosen_mode_ind])

    return chosen_mode_names


def get_tt_matrix_dict(skims):
    tt_matrix_dict = {}

    tt_mat_walk = 60 * skims.get_skim("car_net_distance_sym") / 0.078333333336
    tt_mat_pt = 60 * (skims.get_skim("pt_travel_times_train_sym") + skims.get_skim("pt_travel_times_bus_sym") +
                      skims.get_skim("pt_access_times_sym") + skims.get_skim("pt_egress_times_sym") +
                      skims.get_skim("pt_transfers_sym") * 5)
    tt_matrix_dict['walk'] = tt_mat_walk
    tt_matrix_dict['bike'] = 60 * skims.get_skim("car_net_distance_sym") / 0.216666666666
    tt_matrix_dict['car'] = (60 * skims.get_skim("car_travel_times_sym") + skims.get_skim("at_car") +
                             skims.get_skim("at_car").T + 60 * skims.get_skim("pc_car") * 2 * 0.047 / 0.125)
    tt_matrix_dict['ride'] = (60 * skims.get_skim("car_travel_times_sym") + skims.get_skim("at_ride") +
                              skims.get_skim("at_ride").T + 60 * skims.get_skim("pc_ride") * 2 * 0.047 / 0.125)
    tt_matrix_dict['pt'] = np.where(tt_mat_pt < tt_mat_walk, tt_mat_pt, tt_mat_walk)

    return tt_matrix_dict
