import VisumPy.helpers as VPH
import numpy as np


def create_trips(Visum, filtered_persons, filtered_tours, rand, logging):
    logging.info('preparing data for creating trips...')

    Net = Visum.Net
    tours = filtered_tours
    persons = filtered_persons

    activityID_to_activityCode = dict(Net.Activities.GetMultipleAttributes(["Id", "Code"]))
    activityID_to_primLocZoneNoAttr = dict(Net.Activities.GetMultipleAttributes(["Id", "lt_loc_zone_number_attr"]))

    person_data = np.array(
        persons.GetFilteredSet("[active]=1").
            GetMultipleAttributes(['No', 'Sum:Schedules\\Count:Tours', 'Household\\Residence\\LocationNo',
                                   'Household\\Residence\\Location\\ZoneNo']), dtype=np.int)

    # tour_data = np.array(Net.Tours.GetMultipleAttributes(['inb_Stops', 'outb_stops', 'subtour_stops', 'no', 'PrimTourMajorActivityID', 'is_primary']), dtype=np.int)
    tour_data0 = np.array(VPH.GetMulti(tours, 'inb_Stops'), dtype=np.int)
    tour_data1 = np.array(VPH.GetMulti(tours, 'outb_stops'), dtype=np.int)
    tour_data2 = np.array(VPH.GetMulti(tours, 'subtour_stops'), dtype=np.int)
    tour_data3 = np.array(VPH.GetMulti(tours, 'No'), dtype=np.int)
    tour_data4 = np.array(VPH.GetMulti(tours, 'main_activity_id'), dtype=np.int)
    tour_data5 = np.array(VPH.GetMulti(tours, 'is_primary'), dtype=np.int)
    # tour_data6 = np.array(VPH.GetMulti(tours, 'MajorActivityZoneNo'), dtype=np.int)

    activity_ids_of_primary_tours = np.unique(
        np.array(VPH.GetMulti(tours.GetFilteredSet("[is_primary]=1"), 'main_activity_id'), dtype=np.int))

    activityID_to_primary_loc_zoneNos = dict()
    for act_id in activity_ids_of_primary_tours:
        primLocZoneNoAttr = activityID_to_primLocZoneNoAttr[act_id]
        activityID_to_primary_loc_zoneNos[act_id] = np.array(VPH.GetMulti(persons.GetFilteredSet("[active]=1"),
                                                                          primLocZoneNoAttr), dtype=np.int)

    long_term_choices = np.array(persons.GetFilteredSet("[active]=1").
            GetMultipleAttributes(['Concatenate:LongTermChoices\\ActivityCode',
                                   'Concatenate:LongTermChoices\\LocationNo']))
    major_act_locations = {}
    for prim_act in activity_ids_of_primary_tours:
        major_act_locations[activityID_to_activityCode[prim_act]] = np.zeros(long_term_choices.shape[0])

    for i, choice in enumerate(long_term_choices):
        if choice[0] == '':
            continue
        for j, act_code in enumerate(choice[0].split(',')):
            locations = choice[1].split(',')
            if act_code in major_act_locations.keys():
                major_act_locations[act_code][i] = locations[j]

    global_tour_index = 0
    global_actEx_keys = []
    global_actEx_activityCodes = []
    global_actEx_isMajorActivity = []
    global_actEx_isPartOfSubtour = []
    global_actEx_isMajorActivityOfSubtour = []
    global_actEx_MinorDestChoiceSeqIndex = []
    global_actEx_locationNo = []
    global_actEx_zoneNo = []
    global_trip_keys = []
    global_trip_fromTo_actEx_indices = []
    global_tour_majorActExIndex = []
    global_tour_majorAct_zoneNos = []

    for person_ind in range(len(person_data)):
        person_no = person_data[person_ind][0]
        num_tours = person_data[person_ind][1]
        home_loc_no = person_data[person_ind][2]
        home_zone_no = person_data[person_ind][3]

        if num_tours == 0:
            continue

        # every person starts with an ActEx at Home
        actEx_activityCodes = []
        actEx_activityCodes.append('H')
        global_actEx_isMajorActivity.append(False)
        global_actEx_isPartOfSubtour.append(False)
        global_actEx_isMajorActivityOfSubtour.append(False)
        global_actEx_MinorDestChoiceSeqIndex.append(0)
        global_actEx_locationNo.append(home_loc_no)
        global_actEx_zoneNo.append(home_zone_no)

        for t in range(num_tours):
            num_outboundStops = tour_data1[global_tour_index]
            num_subtour_stops = tour_data2[global_tour_index]
            num_inboundStops = tour_data0[global_tour_index]
            major_activity_id = tour_data4[global_tour_index]
            is_primary_tour = tour_data5[global_tour_index]

            major_act_code = ''
            major_act_zoneNo = 0
            major_act_locationNo = None  # None represents empty value (don't set this to 0, it will slow down the script)
            if is_primary_tour:
                major_act_code = activityID_to_activityCode[major_activity_id]
                major_act_zoneNo = activityID_to_primary_loc_zoneNos[major_activity_id][person_ind]
                if major_act_zoneNo == 0:
                    logging.info(person_no)
                assert major_act_zoneNo > 0
                major_act_locationNo = major_act_locations[major_act_code][person_ind]
                # major_act_locationNo = zoneNoActCode_to_locationNo[(major_act_zoneNo, major_act_code)]
                # assert major_act_locationNo > 0
            else:
                if major_activity_id == 1:
                    major_act_code = activityID_to_activityCode[major_activity_id]
                    if num_inboundStops > 0:
                        num_outboundStops = int(rand.randint(num_inboundStops + 1))
                    else:
                        num_outboundStops = 0
                    num_inboundStops -= num_outboundStops

            tour_no = tour_data3[global_tour_index]

            starting_actEx_index = len(actEx_activityCodes)

            # outbound stops have empty activity
            actEx_activityCodes += ([''] * num_outboundStops)
            global_actEx_isMajorActivity += ([False] * num_outboundStops)
            global_actEx_isPartOfSubtour += ([False] * num_outboundStops)
            global_actEx_isMajorActivityOfSubtour += ([False] * num_outboundStops)
            if is_primary_tour:
                global_actEx_MinorDestChoiceSeqIndex += list(range(1, num_outboundStops + 1))
            global_actEx_locationNo += ([
                                            None] * num_outboundStops)  # None represents empty value (don't set this to 0, it will slow down the script)
            global_actEx_zoneNo += ([0] * num_outboundStops)
            # primary stop activity from tour activity
            actEx_activityCodes.append(major_act_code)
            global_actEx_isMajorActivity.append(True)
            global_actEx_isPartOfSubtour.append(True if num_subtour_stops > 0 else False)
            global_actEx_isMajorActivityOfSubtour.append(False)
            if is_primary_tour:
                global_actEx_MinorDestChoiceSeqIndex.append(0)
            global_actEx_locationNo.append(
                major_act_locationNo)  # set for primary tours only (set later in choice model 'DestMajor' for secondary tours)
            global_actEx_zoneNo.append(
                major_act_zoneNo)
            global_tour_majorActExIndex.append(len(actEx_activityCodes))
            # subtour stops have empty activity
            actEx_activityCodes += ([''] * num_subtour_stops)
            global_actEx_isMajorActivity += ([False] * num_subtour_stops)
            global_actEx_isPartOfSubtour += ([True] * num_subtour_stops)
            # first activity of subtour is the major activity of the subtour
            global_actEx_isMajorActivityOfSubtour += ([False] * num_subtour_stops) if num_subtour_stops > 0 else []
            global_actEx_MinorDestChoiceSeqIndex += list(range(1, num_subtour_stops + 1))
            global_actEx_locationNo += ([
                                            None] * num_subtour_stops)  # None represents empty value (don't set this to 0, it will slow down the script)
            global_actEx_zoneNo += ([0] * num_subtour_stops)
            if num_subtour_stops > 0:
                # if subtour, then one extra ActEx for second part of primary ActEx, also adds one trip
                actEx_activityCodes.append(major_act_code)
                global_actEx_isMajorActivity.append(True)
                global_actEx_isPartOfSubtour.append(True)
                global_actEx_isMajorActivityOfSubtour.append(False)
                global_actEx_MinorDestChoiceSeqIndex.append(0)
                global_actEx_locationNo.append(major_act_locationNo)  # zone will be chosen later
                global_actEx_zoneNo.append(major_act_zoneNo)
            # inbound stops have empty activity
            actEx_activityCodes += ([''] * num_inboundStops)
            global_actEx_isMajorActivity += ([False] * num_inboundStops)
            global_actEx_isPartOfSubtour += ([False] * num_inboundStops)
            global_actEx_isMajorActivityOfSubtour += ([False] * num_inboundStops)
            if is_primary_tour:
                global_actEx_MinorDestChoiceSeqIndex += list(range(1, num_inboundStops + 1))
            else:
                global_actEx_MinorDestChoiceSeqIndex += list(range(1, num_inboundStops + num_outboundStops + 2))
            global_actEx_locationNo += ([
                                            None] * num_inboundStops)  # None represents empty value (don't set this to 0, it will slow down the script)
            global_actEx_zoneNo += ([0] * num_inboundStops)
            # every tour ends with one ActEx at Home again
            actEx_activityCodes.append('H')
            global_actEx_isMajorActivity.append(False)
            global_actEx_isPartOfSubtour.append(False)
            global_actEx_isMajorActivityOfSubtour.append(False)
            global_actEx_MinorDestChoiceSeqIndex.append(0)
            global_actEx_locationNo.append(home_loc_no)
            global_actEx_zoneNo.append(home_zone_no)

            num_trips = len(actEx_activityCodes) - starting_actEx_index

            global_trip_keys += ([(0, person_no, 1, tour_no)] * num_trips)
            global_trip_fromTo_actEx_indices += [(starting_actEx_index + i, starting_actEx_index + i + 1) for i in
                                                 range(num_trips)]
            # starting_actEx_index += num_trips
            global_tour_majorAct_zoneNos.append(major_act_zoneNo)
            global_tour_index += 1

        global_actEx_keys += [(0, person_no, 1)] * len(actEx_activityCodes)
        global_actEx_activityCodes += actEx_activityCodes

    logging.info('create %d activity executions...' % len(global_actEx_keys))
    logging.info('set activity execution attributes...')

    actEx_attrvalues = list(zip(global_actEx_activityCodes, global_actEx_isMajorActivity, global_actEx_isPartOfSubtour,
                                global_actEx_zoneNo, global_actEx_MinorDestChoiceSeqIndex,
                                global_actEx_locationNo))

    assert len(global_actEx_keys) == len(global_actEx_activityCodes) == len(global_actEx_isMajorActivity) == \
           len(global_actEx_isPartOfSubtour) == len(global_actEx_isMajorActivityOfSubtour) == \
           len(global_actEx_MinorDestChoiceSeqIndex) == len(global_actEx_locationNo) == len(global_actEx_zoneNo)
    for i in range(int(np.ceil(len(global_actEx_keys) / 1000000))):
        actEx = Net.AddMultiActivityExecutions(global_actEx_keys[i * 1000000:(i + 1) * 1000000])
        actEx.SetMultipleAttributes(['ActivityCode', 'is_main_tour_activity', 'is_part_of_subtour', 'zone_id',
                                     'secondary_activity_index', 'LocationNo'],
                                    actEx_attrvalues[i * 1000000:(i + 1) * 1000000])

    logging.info('create %d trips...' % len(global_trip_keys))
    logging.info('set trip attributes...')

    for i in range(int(np.ceil(len(global_trip_keys) / 1000000))):
        trips = Net.AddMultiTrips(global_trip_keys[i * 1000000:(i + 1) * 1000000])
        trips.SetMultipleAttributes(['FromActivityExecutionIndex', 'ToActivityExecutionIndex'],
                                    global_trip_fromTo_actEx_indices[i * 1000000:(i + 1) * 1000000])

    tours.SetMultipleAttributes(['main_activity_zone_id', 'main_activity_execution_index'],
                                list(zip(global_tour_majorAct_zoneNos, global_tour_majorActExIndex)))
