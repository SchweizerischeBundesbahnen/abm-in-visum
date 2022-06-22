import logging

import numpy as np

import VisumPy.helpers as VPH

from settings import abm_settings
from src import visum_utilities

def run(Visum):
    logging.info('preparing data for creating trips...')

    Net = Visum.Net
    tours = Net.Tours

    visum_utilities.insert_UDA_if_missing(Visum.Net.Tours, 'MajorActExIndex')
    visum_utilities.insert_UDA_if_missing(Visum.Net.Tours, 'MajorActCode', value_type=5)

    activityID_to_activityCode = dict(Net.Activities.GetMultipleAttributes(["ID", "Code"]))
    person_data = np.array(Net.Persons.GetMultipleAttributes(['No', 'Sum:Schedules\\Count:Tours', 'Household\\Residence\\LocationNo']), dtype=np.int)

    #tour_data = np.array(Net.Tours.GetMultipleAttributes(['inb_stops', 'outb_stops', 'subtour_stops', 'no', 'PrimTourMajorActivityID', 'is_primary']), dtype=np.int)
    tour_data0 = np.array(VPH.GetMulti(tours, 'inb_stops'), dtype=np.int)
    tour_data1 = np.array(VPH.GetMulti(tours, 'outb_stops'), dtype=np.int)
    tour_data2 = np.array(VPH.GetMulti(tours, 'subtour_stops'), dtype=np.int)
    tour_data3 = np.array(VPH.GetMulti(tours, 'no'), dtype=np.int)
    tour_data4 = np.array(VPH.GetMulti(tours, 'PrimTourMajorActivityID'), dtype=np.int)
    tour_data5 = np.array(VPH.GetMulti(tours, 'is_primary'), dtype=np.int)

    global_tour_index = 0
    global_actEx_keys = []
    global_actEx_activityCodes = []
    global_actEx_isMajorActivity = []
    global_actEx_isPartOfSubtour = []
    global_actEx_MinorDestAndModeChoiceSeqIndex = []
    global_actEx_locationNo = []
    global_trip_keys = []
    global_trip_fromTo_actEx_indices = []
    global_tour_majorActExIndex = []
    global_tour_majorAct_codes = []

    for current_person_data in person_data:
        person_no = current_person_data[0]
        num_tours = current_person_data[1]
        home_loc_no = current_person_data[2]

        if num_tours == 0:
            continue

        # every person starts with an ActEx at Home
        actEx_activityCodes = []
        actEx_activityCodes.append('H')
        global_actEx_isMajorActivity.append(False)
        global_actEx_isPartOfSubtour.append(False)
        global_actEx_MinorDestAndModeChoiceSeqIndex.append(0)
        global_actEx_locationNo.append(home_loc_no)

        for _ in range(num_tours):
            num_outbound_stops = tour_data1[global_tour_index]
            num_subtour_stops = tour_data2[global_tour_index]
            num_inbound_stops = tour_data0[global_tour_index]
            major_activity_id = tour_data4[global_tour_index]
            is_primary_tour = tour_data5[global_tour_index]
            assert (is_primary_tour == 1) == (major_activity_id > 0)

            major_act_code = activityID_to_activityCode[major_activity_id] if is_primary_tour else ''
  
            tour_no = tour_data3[global_tour_index]

            starting_actEx_index = len(actEx_activityCodes)

            # outbound stops have empty activity
            actEx_activityCodes += ([''] * num_outbound_stops)
            global_actEx_isMajorActivity += ([False] * num_outbound_stops)
            global_actEx_isPartOfSubtour += ([False] * num_outbound_stops)
            global_actEx_MinorDestAndModeChoiceSeqIndex += list(reversed(range(1, num_outbound_stops + 1)))
            global_actEx_locationNo += ([None] * num_outbound_stops)  # None represents empty value (don't set this to 0, it will slow down the script)
            # primary stop activity from tour activity
            actEx_activityCodes.append(major_act_code)
            global_actEx_isMajorActivity.append(True)
            global_actEx_isPartOfSubtour.append(True if num_subtour_stops > 0 else False)
            global_actEx_MinorDestAndModeChoiceSeqIndex.append(0)
            global_actEx_locationNo.append(None)  # primloc is set later in choice model 'DestMajor'
            global_tour_majorActExIndex.append(len(actEx_activityCodes))
            # subtour stops have empty activity
            actEx_activityCodes += ([''] * num_subtour_stops)
            global_actEx_isMajorActivity += ([False] * num_subtour_stops)
            global_actEx_isPartOfSubtour += ([True] * num_subtour_stops)
            # first activity of subtour is the major activity of the subtour
            global_actEx_MinorDestAndModeChoiceSeqIndex += list(range(1, num_subtour_stops + 1))
            global_actEx_locationNo += ([None] * num_subtour_stops)  # None represents empty value (don't set this to 0, it will slow down the script)
            if num_subtour_stops > 0:
                # if subtour, then one extra ActEx for second part of primary ActEx, also adds one trip
                actEx_activityCodes.append(major_act_code)
                global_actEx_isMajorActivity.append(True)
                global_actEx_isPartOfSubtour.append(True)
                global_actEx_MinorDestAndModeChoiceSeqIndex.append(num_subtour_stops + 1)
                global_actEx_locationNo.append(None)  # zone will be chosen later
            # inbound stops have empty activity
            actEx_activityCodes += ([''] * num_inbound_stops)
            global_actEx_isMajorActivity += ([False] * num_inbound_stops)
            global_actEx_isPartOfSubtour += ([False] * num_inbound_stops)
            nextMinorDestandModeChoiceIndex = num_subtour_stops + 2 if num_subtour_stops > 0 else 1
            global_actEx_MinorDestAndModeChoiceSeqIndex += list(range(nextMinorDestandModeChoiceIndex, nextMinorDestandModeChoiceIndex + num_inbound_stops))
            global_actEx_locationNo += ([None] * num_inbound_stops)  # None represents empty value (don't set this to 0, it will slow down the script)
            # every tour ends with one ActEx at Home again
            actEx_activityCodes.append('H')
            global_actEx_isMajorActivity.append(False)
            global_actEx_isPartOfSubtour.append(False)
            global_actEx_MinorDestAndModeChoiceSeqIndex.append(0)
            global_actEx_locationNo.append(home_loc_no)

            num_trips = len(actEx_activityCodes) - starting_actEx_index

            global_trip_keys += ([(0, person_no, 1, tour_no)] * num_trips)
            global_trip_fromTo_actEx_indices += [(starting_actEx_index + i, starting_actEx_index + i + 1) for i in
                                                 range(num_trips)]
            starting_actEx_index += num_trips
            global_tour_majorAct_codes.append(major_act_code)
            global_tour_index += 1

        global_actEx_keys += [(0, person_no, 1)] * len(actEx_activityCodes)
        global_actEx_activityCodes += actEx_activityCodes

    logging.info('create %d activity executions...', len(global_actEx_keys))
    logging.info('set activity execution attributes...')

    actEx_attrvalues = list(zip(global_actEx_activityCodes, global_actEx_isMajorActivity, global_actEx_isPartOfSubtour,
                             global_actEx_MinorDestAndModeChoiceSeqIndex, global_actEx_locationNo))

    assert len(global_actEx_keys) == len(global_actEx_activityCodes) == len(global_actEx_isMajorActivity) == \
           len(global_actEx_isPartOfSubtour) == \
           len(global_actEx_MinorDestAndModeChoiceSeqIndex) == len(global_actEx_locationNo)
    
    for i in range(int(np.ceil(len(global_actEx_keys) / abm_settings.chunk_size_trips))):
        actEx = Net.AddMultiActivityExecutions(global_actEx_keys[i * abm_settings.chunk_size_trips:(i + 1) * abm_settings.chunk_size_trips])
        actEx.SetMultipleAttributes(['ActivityCode', 'IsMajorActivity', 'IsPartOfSubtour', 'MinorDestAndModeChoiceSeqIndex', 'LocationNo'],
                                    actEx_attrvalues[i * abm_settings.chunk_size_trips:(i + 1) * abm_settings.chunk_size_trips])

    logging.info(f'create {len(global_trip_keys)} trips...')
    logging.info('set trip attributes...')
    for i in range(int(np.ceil(len(global_trip_keys) / abm_settings.chunk_size_trips))):
        trips = Net.AddMultiTrips(global_trip_keys[i*abm_settings.chunk_size_trips:(i+1)*abm_settings.chunk_size_trips])
        trips.SetMultipleAttributes(['FromActivityExecutionIndex', 'ToActivityExecutionIndex'],
                                    global_trip_fromTo_actEx_indices[i*abm_settings.chunk_size_trips:(i+1)*abm_settings.chunk_size_trips])

    Net.Tours.SetMultipleAttributes(['MajorActCode'               , 'MajorActExIndex'],
                                    list(zip(global_tour_majorAct_codes, global_tour_majorActExIndex)))
