import logging
import os
import csv
from operator import itemgetter

import numpy as np

from src import visum_utilities
from settings import abm_settings


# Visum value types
VISUM_VALUETYPE_INT = 1
VISUM_VALUETYPE_FLOAT = 2
VISUM_VALUETYPE_STRING = 5


def run(Visum, model_dir):

    logging.info('delete all persons and households')
    clear_population(Visum)
    
    max_existing_location_no = np.amax(visum_utilities.GetMulti(Visum.Net.Locations, "No"))

    demand_file_name = os.path.join(model_dir, r"HHI.dmd") 
    syn_pop_HH_file_name = os.path.join(model_dir, r"SynPopHH.csv") 
    syn_pop_person_file_name = os.path.join(model_dir, r"SynPopPerson.csv") 
    
    logging.info('read surey data: %s', demand_file_name)

    # load demand file
    # demand file need to contain example persons (propably from a household survey)
    # persons have schedules with
    # tours with activity executions with set
    #   Duration
    #   IsMajorActivity 
    #   IsPartOfSubtour
    #   StartTime (only for major activity execution)
    visum_utilities.insert_UDA_if_missing(Visum.Net.Households, 'ORIGINALHHID', value_type=VISUM_VALUETYPE_STRING)
    try:
        AddNetReadController = Visum.IO.CreateAddNetReadController()
        AddNetReadController.SetUseNumericOffset("Location", True)
        AddNetReadController.SetNumericOffset("Location", max_existing_location_no + 100)
        Visum.IO.LoadDemandFile(demand_file_name, True, AddNetReadController)
    except Exception as error:
        logging.error(f"Demand file could not be read: {error}")
        raise

    # check survey data

    # all trips must have FromActivityExecution and ToActivityExecution
    # <=> number of trips needs to be equal to number of trips with FromActivityExecution 
    #     as well as to number of trips with ToActivityExecution
    survey_tours = Visum.Net.Tours

    if survey_tours.Count > 0:
        tour_check_data = np.array(survey_tours.GetMultipleAttributes(
            [r'Count:Trips', r'Sum:Trips\Count:FromActivityExecution', r'Sum:Trips\Count:ToActivityExecution']))

        if any(tour_check_data[:, 0]-tour_check_data[:, 1]) or any(tour_check_data[:, 0]-tour_check_data[:, 2]):
            logging.error("All trips need to have a FromActivityExecution and a ToActivityExecution.")
            raise Exception("Invalid survey data: All trips need to have a FromActivityExecution and a ToActivityExecution.")
    
        Visum.Net.Schedules.SortEachSchedule()

    # we now consider all persons as example persons, 
    # their households with their locations are removed after inserting the synthetic population
    logging.info('collect information from survey') 
    survery_persons = Visum.Net.Persons
    survey_persons_nos = survery_persons.GetMultipleAttributes(["No"])

    # person data
    person_UDA_IDs = visum_utilities.get_UDA_IDs(survery_persons)
    survey_person_UDA_data = survery_persons.GetMultipleAttributes(person_UDA_IDs)
    survey_persons_schedules_data = survery_persons.GetMultipleAttributes(
        [r'Count:Schedules', r'Concatenate:Schedules\Count:Tours', r'Concatenate:Schedules\Concatenate:Tours\Count:Trips', 
         r'Concatenate:Schedules\Concatenate:Tours\Count:Trips([FromActivityExecution\IsPartOfSubtour])', 
         r'Concatenate:Schedules\Concatenate:Tours\First:Trips\FromActivityExecution\Index'])

    # household data
    survery_households = Visum.Net.Households
    hh_UDA_IDs = visum_utilities.get_UDA_IDs(survery_households)
    survey_hh_UDA_data = survery_households.GetMultipleAttributes(hh_UDA_IDs)

    # activity execution data
    survey_activity_executions = Visum.Net.ActivityExecutions
    act_ex_editable_attributes = [attr.ID for attr in survey_activity_executions.Attributes.GetAll if attr.Editable]

    act_ex_editable_attributes.remove('ACTIVITYLOCATIONKEY')
    act_ex_editable_attributes.remove('ENDTIME')
    act_ex_editable_attributes.remove('LOCATIONNO')
    act_ex_attributes = ["PersonNo"] + act_ex_editable_attributes
    survey_activity_execution_data = survey_activity_executions.GetMultipleAttributes(act_ex_attributes)

    survey_person_no_to_act_ex_data_range = create_person_range_dict(survey_activity_execution_data)

    # collect MinorDestAndModeChoiceSeqIndex, MajorActExIndex, MajorActCode, inbound_stops, outbound_stops, subtour_stops
    survey_tour_key_data, survey_tour_data, tour_attributes, survey_minor_dest_and_mode_choice_seq_index = compute_survey_tour_data(
        survey_persons_nos, survey_persons_schedules_data, act_ex_attributes, survey_activity_execution_data)

    survey_person_no_to_tour_data_range = create_person_range_dict(survey_tour_key_data, person_no_index=1)

    # trip data
    survey_trips = Visum.Net.Trips
    trip_editable_attributes = ['FROMACTIVITYEXECUTIONINDEX', 'TOACTIVITYEXECUTIONINDEX']
    trip_attribute = ["PersonNo"] + trip_editable_attributes
    survey_trip_data = survey_trips.GetMultipleAttributes(trip_attribute)
    survey_person_no_to_trip_data_range = create_person_range_dict(survey_trip_data)

    # remove survey data from Visum
    clear_population(Visum, min_survey_location_no=max_existing_location_no + 100)
    
    # insert necessary UDAs  
    visum_utilities.insert_UDA_if_missing(Visum.Net.ActivityExecutions, 'MinorDestAndModeChoiceSeqIndex')
    visum_utilities.insert_UDA_if_missing(Visum.Net.Tours, 'MajorActExIndex')
    visum_utilities.insert_UDA_if_missing(Visum.Net.Tours, 'MajorActCode', value_type=5)
    visum_utilities.insert_UDA_if_missing(Visum.Net.Tours, 'inb_stops')
    visum_utilities.insert_UDA_if_missing(Visum.Net.Tours, 'outb_stops')
    visum_utilities.insert_UDA_if_missing(Visum.Net.Tours, 'subtour_stops')

    # create synthetic households
    logging.info('read synthetic population: households file %s', syn_pop_HH_file_name)
    location_no_to_home_act_loc = create_location_dict(Visum)
    synthetic_pop_HH_data, synthetic_pop_HH_key_data = read_synthetic_households(syn_pop_HH_file_name, location_no_to_home_act_loc)

    logging.info('insert synthetic population: households')
    syn_pop_households = Visum.Net.AddMultiHouseholds(synthetic_pop_HH_key_data)
    set_uda_values_from_survey_objects(
        synthetic_pop_HH_data, hh_UDA_IDs, survey_hh_UDA_data, syn_pop_households, orig_id_attr="ORIGINALHHID")
    
    # create synthetic persons
    logging.info('read synthetic population: persons file %s', syn_pop_person_file_name)
    synthetic_pop_person_data, synthetic_pop_person_key_data = read_synthetic_persons(syn_pop_person_file_name)

    logging.info('insert synthetic population: persons')
    syn_pop_persons = Visum.Net.AddMultiPersons(synthetic_pop_person_key_data)
    orig_person_id_to_survey_data_index = set_uda_values_from_survey_objects(
        synthetic_pop_person_data, person_UDA_IDs, survey_person_UDA_data, syn_pop_persons, orig_id_attr="ORIGINALPERSONID")

    # create schedules and tours
    logging.info('insert synthetic population: schedules, tours, trips and activity executions')
    syn_pop_tours_key_data, syn_pop_act_ex_key_data, syn_pop_trip_key_data = add_schedules(
        Visum, synthetic_pop_person_data, survey_persons_schedules_data, orig_person_id_to_survey_data_index)
    
    # set attribute values to tours
    logging.info('insert synthetic population: tour data')
    if len(survey_tour_data) == 0:
        return

    set_syn_pop_data(survey_tour_data, 0, survey_person_no_to_tour_data_range, survey_persons_nos,
                     syn_pop_tours_key_data, tour_attributes, synthetic_pop_person_data, orig_person_id_to_survey_data_index,
                     Visum.Net.AddMultiTours)

    # set attribute values to activity executions
    logging.info('insert synthetic population: activtiy executions data')
    if len(survey_activity_execution_data) == 0:
        return

    # set MinorDestAndModeChoiceSeqIndex in survey data
    try:
        # MinorDestAndModeChoiceSeqIndex attribute exists in survey data => overwrite data
        minor_choice_seq_index_UDA_index = act_ex_attributes.index('MINORDESTANDMODECHOICESEQINDEX')
        survey_activity_execution_data = [list(data)[:minor_choice_seq_index_UDA_index] + [survey_minor_dest_and_mode_choice_seq_index[i]] + list(data)[minor_choice_seq_index_UDA_index + 1:]
                                          for i, data in enumerate(survey_activity_execution_data)]
    except ValueError:
        # MinorDestAndModeChoiceSeqIndex does not exist in survey data => add UDA and data
        visum_utilities.insert_UDA_if_missing(Visum.Net.ActivityExecutions, 'MinorDestAndModeChoiceSeqIndex')
        act_ex_editable_attributes += ['MINORDESTANDMODECHOICESEQINDEX']
        survey_activity_execution_data = [ data + (seq_ind,) for data, seq_ind in zip(survey_activity_execution_data, survey_minor_dest_and_mode_choice_seq_index) ]

    set_syn_pop_data(survey_activity_execution_data, 1, survey_person_no_to_act_ex_data_range, survey_persons_nos,
                     syn_pop_act_ex_key_data, act_ex_editable_attributes, synthetic_pop_person_data, orig_person_id_to_survey_data_index,
                     Visum.Net.AddMultiActivityExecutions)

    home_act_ex = Visum.Net.ActivityExecutions.GetFilteredSet(r'[ActivityCode] = "H"')
    home_location_no_attr = r'SCHEDULE\PERSON\HOUSEHOLD\RESIDENCE\LOCATIONNO'
    home_location_nos = visum_utilities.GetMulti(home_act_ex, home_location_no_attr, chunk_size=abm_settings.chunk_size_trips)
    visum_utilities.SetMulti(home_act_ex, r'LocationNo', home_location_nos, chunk_size=abm_settings.chunk_size_trips)

    # set attribute values to trips
    logging.info('insert synthetic population: trip data')
    set_syn_pop_data(survey_trip_data, 1, survey_person_no_to_trip_data_range, survey_persons_nos,
                     syn_pop_trip_key_data, trip_editable_attributes, synthetic_pop_person_data, orig_person_id_to_survey_data_index,
                     Visum.Net.AddMultiTrips)

def compute_survey_tour_data(survey_persons_nos, survey_persons_schedules_data, act_ex_attribute, survey_activity_execution_data):
    start_time_UDA_index = act_ex_attribute.index('STARTTIME')
    activity_code_UDA_index = act_ex_attribute.index('ACTIVITYCODE')
    is_major_act_ex_UDA_index = act_ex_attribute.index('ISMAJORACTIVITY')
    start_time_values = list(map(itemgetter(start_time_UDA_index), survey_activity_execution_data))
    activity_code_values = list(map(itemgetter(activity_code_UDA_index), survey_activity_execution_data))
    is_major_act_ex_values = list(map(itemgetter(is_major_act_ex_UDA_index), survey_activity_execution_data))

    survey_tour_key_data = []
    survey_tour_major_act_index = []
    survey_tour_major_act_code = []
    survey_tour_num_outbound_stops = []
    survey_tour_num_subtour_stops = []
    survey_tour_num_inbound_stops = []
    survey_minor_dest_and_mode_choice_seq_index = []

    act_ex_previous_schedules = 0
    for person_index, cur_survey_object_schedule_data in enumerate(survey_persons_schedules_data):
        person_no = int(survey_persons_nos[person_index][0])
        num_schedules = int(cur_survey_object_schedule_data[0])
        num_tours_per_schedule = list(map(int, list(cur_survey_object_schedule_data[1])))
        if sum(num_tours_per_schedule) == 0:
            continue 
        num_trips_per_tour = cur_survey_object_schedule_data[2].split(',')
        num_subtour_act_ex_per_tour = np.array(cur_survey_object_schedule_data[3].split(','), int)
        first_act_ex_index_per_tour = np.array(cur_survey_object_schedule_data[4].split(','), int)
        # Major Activity is set two times 'part of subtour' if a subtour exists
        num_subtour_stops_per_tour = np.maximum(0, num_subtour_act_ex_per_tour - 2)
        person_tour_index = 0
        for schedule_no in range(1, num_schedules+1):
            num_act_ex_cur_schedule = 0
            num_tours = num_tours_per_schedule[schedule_no-1]
            act_ex_in_schedule_prev_tours = 0
            survey_minor_dest_and_mode_choice_seq_index.append(0) # start with H

            first_act_ex_index_to_seq_indices = []
            for tour_no in range(1, num_tours+1):
                first_act_ex_index = first_act_ex_index_per_tour[person_tour_index]
                survey_tour_key_data.append([tour_no, person_no, schedule_no])

                num_trips = int(num_trips_per_tour[person_tour_index])
                num_act_ex_cur_schedule += num_trips + 1 if tour_no == 1 else num_trips
                num_subtour_stops = num_subtour_stops_per_tour[person_tour_index]
                next_home_act_ex_index = first_act_ex_index + num_trips
                next_home_act_ex_index_in_survey_data = act_ex_previous_schedules + next_home_act_ex_index-1
                person_no_in_schedule_data = int(
                    survey_activity_execution_data[next_home_act_ex_index_in_survey_data][0])
                if person_no_in_schedule_data != person_no:
                    error_msg = "Schedules are not allowed to have cycles."
                    raise_tour_error(person_no, schedule_no, tour_no, error_msg)

                major_act_ex_index_cur_tour = 0
                major_act_ex_code_cur_tour = None
                num_outbound_stops_cur_tour = 0
                if activity_code_values[
                        act_ex_previous_schedules + first_act_ex_index - 1] != 'H' or activity_code_values[
                        next_home_act_ex_index_in_survey_data] != 'H':
                    error_msg = "All tours need to start and end with activity 'H'."
                    raise_tour_error(person_no, schedule_no, tour_no, error_msg)

                for act_ex_index in range(first_act_ex_index, next_home_act_ex_index + 1) :
                    if is_major_act_ex_values[act_ex_previous_schedules + act_ex_index - 1]: 
                        if start_time_values[act_ex_previous_schedules + act_ex_index - 1] == 0:
                            error_msg = "All major activities need to have a start time."
                            raise_tour_error(person_no, schedule_no, tour_no, error_msg)

                        act_ex_index_in_tour = act_ex_index - first_act_ex_index + 1
                        assert act_ex_index_in_tour >= 1 # tour starts with H
                        num_outbound_stops_cur_tour = act_ex_index_in_tour - 2 # index 2 means H -> major activity
                        major_act_ex_index_cur_tour = act_ex_index
                        major_act_ex_code_cur_tour = activity_code_values[act_ex_previous_schedules + act_ex_index - 1]
                        break

                if major_act_ex_index_cur_tour == 0:
                    error_msg = "All tours need to have major activity."
                    raise_tour_error(person_no, schedule_no, tour_no, error_msg)

                num_stops_major_act_and_subtour_stops = np.maximum(1, num_subtour_act_ex_per_tour[person_tour_index])
                # numtrips - 1 = non home act ex 
                num_inbound_stops_cur_tour = num_trips - 1 - num_stops_major_act_and_subtour_stops - num_outbound_stops_cur_tour
                cur_tour_seq_indices = []
                cur_tour_seq_indices += list(reversed(range(1, num_outbound_stops_cur_tour + 1)))
                cur_tour_seq_indices.append(0) # major activity
                if num_subtour_stops > 0:
                    # subtour stops get `1, 2, ..., num_subtour_stops`
                    # major activity execution appears for a second time and therefore gets `num_subtour_stops + 1`
                    cur_tour_seq_indices += list(range(1, num_subtour_stops + 2))

                next_seq_index_cur_tour = cur_tour_seq_indices[-1] + 1
                cur_tour_seq_indices += list(range(next_seq_index_cur_tour, next_seq_index_cur_tour + num_inbound_stops_cur_tour))
                cur_tour_seq_indices.append(0) # tour ends with H
                first_act_ex_index_to_seq_indices.append((first_act_ex_index, cur_tour_seq_indices))

                survey_tour_major_act_index.append(major_act_ex_index_cur_tour)
                survey_tour_major_act_code.append(major_act_ex_code_cur_tour)
                survey_tour_num_outbound_stops.append(num_outbound_stops_cur_tour)
                survey_tour_num_subtour_stops.append(num_subtour_stops)
                survey_tour_num_inbound_stops.append(num_inbound_stops_cur_tour)
                act_ex_in_schedule_prev_tours += num_trips
                person_tour_index += 1
            first_act_ex_index_to_seq_indices.sort(key=lambda x: x[0])
            for first_act_ex_index, cur_tour_seq_indices in first_act_ex_index_to_seq_indices:
                survey_minor_dest_and_mode_choice_seq_index += cur_tour_seq_indices
            act_ex_previous_schedules += num_act_ex_cur_schedule

    survey_tour_data = list(zip(survey_tour_num_subtour_stops, survey_tour_num_inbound_stops,
                            survey_tour_num_outbound_stops, survey_tour_major_act_index, survey_tour_major_act_code))
    tour_attributes = ['subtour_stops', 'inb_stops','outb_stops', 'MajorActExIndex', 'MajorActCode']

    return survey_tour_key_data, survey_tour_data, tour_attributes, survey_minor_dest_and_mode_choice_seq_index

def raise_tour_error(person_no, schedule_no, tour_no, error_msg):
    tour_data = f"Error in tour ({person_no},{schedule_no},{tour_no})"
    logging.error(error_msg + " " + tour_data)
    # raise Exception(f'Invalid survey data: {error_msg} {tour_data}')
  

def set_syn_pop_data(survey_data, num_leading_key_cols_in_survey_data, survey_person_no_to_survey_data_range, survey_persons_nos,
                     syn_pop_keys, syn_pop_editable_attributes, synthetic_pop_person_data, orig_person_id_to_survey_data_index,
                     add_syn_pop_objects_to_visum_func):

    syn_pop_data = []
    syn_pop_key_index = 0 # used only for assertions
    for syn_pop_person_no, _, original_id  in synthetic_pop_person_data:
        survey_person_data_index = orig_person_id_to_survey_data_index[original_id]
        survey_person_no = int(survey_persons_nos[survey_person_data_index][0])
        if survey_person_no in survey_person_no_to_survey_data_range:
            index_range_start, index_range_end = survey_person_no_to_survey_data_range[survey_person_no]
            for survey_data_index in range(index_range_start, index_range_end):
                assert syn_pop_keys[syn_pop_key_index][1] == syn_pop_person_no
                syn_pop_data.append(survey_data[survey_data_index][num_leading_key_cols_in_survey_data:])
                syn_pop_key_index += 1

    chuck_size = int(abm_settings.chunk_size_trips / len(syn_pop_editable_attributes))

    num_syn_pop_objects = len(syn_pop_keys)
    assert num_syn_pop_objects == len(syn_pop_data)

    for i in range(int(np.ceil(num_syn_pop_objects / chuck_size))):
        syn_pop_objects = add_syn_pop_objects_to_visum_func(syn_pop_keys[i * chuck_size:(i + 1) * chuck_size])
        syn_pop_objects.SetMultipleAttributes(syn_pop_editable_attributes,
                                              syn_pop_data[i * chuck_size:(i + 1) * chuck_size])


def create_person_range_dict(survey_data, person_no_index = 0):
    survey_person_no_to_data_range = dict()

    range_start_index = None
    cur_person_no = None
    for data_row_index, data_row in enumerate(survey_data):
        if cur_person_no != data_row[person_no_index]:
            if range_start_index is not None:
                survey_person_no_to_data_range[cur_person_no] = (range_start_index, data_row_index)
            range_start_index = data_row_index
            cur_person_no = data_row[person_no_index]
    if range_start_index is not None:
        survey_person_no_to_data_range[cur_person_no] = (range_start_index, len(survey_data))
    return survey_person_no_to_data_range

def add_schedules(Visum, synthetic_pop_person_data, survey_persons_schedules_data, orig_person_id_to_survey_data_index):

    syn_pop_schedules_key_data = []
    syn_pop_tours_key_data = []
    syn_pop_act_ex_key_data = []
    syn_pop_trip_key_data = []
    for cur_syn_pop_object in synthetic_pop_person_data :
        original_id = cur_syn_pop_object[2]
        person_no = int(cur_syn_pop_object[0])
        cur_syn_pop_object_schedule_data = survey_persons_schedules_data[orig_person_id_to_survey_data_index[original_id]]
        num_schedules = int(cur_syn_pop_object_schedule_data[0])
        num_tours_per_schedule = list(cur_syn_pop_object_schedule_data[1])
        num_trips_per_tour = cur_syn_pop_object_schedule_data[2].split(',')
        global_tour_index = 0
        for schedule_no in range(1, num_schedules+1):
            syn_pop_schedules_key_data.append([schedule_no, person_no]) 
            num_tours = int(num_tours_per_schedule[schedule_no-1])
            for tour_no in range(1, num_tours+1):
                num_trips = int(num_trips_per_tour[global_tour_index])
                syn_pop_tours_key_data.append([tour_no, person_no, schedule_no])
                syn_pop_act_ex_key_data += [(0, person_no, schedule_no)] * (
                    (num_trips + 1) if tour_no == 1 else num_trips)
                syn_pop_trip_key_data += [(0, person_no, schedule_no, tour_no)] * num_trips
                global_tour_index += 1
    if len(syn_pop_schedules_key_data) > 0 :
        Visum.Net.AddMultiSchedules(syn_pop_schedules_key_data)

    return syn_pop_tours_key_data, syn_pop_act_ex_key_data, syn_pop_trip_key_data


def set_uda_values_from_survey_objects(synthetic_pop_data, UDA_IDs, survey_UDA_data, syn_pop_objects, orig_id_attr):

    orig_object_id_UDA_index = UDA_IDs.index(orig_id_attr)
    orig_ids_of_survey_objects = list(map(itemgetter(orig_object_id_UDA_index), survey_UDA_data))
    orig_id_attribute_type = visum_utilities.get_attribute_type(syn_pop_objects, orig_id_attr)
    if orig_id_attribute_type == VISUM_VALUETYPE_INT:
        orig_ids_of_survey_objects = list(map(int, orig_ids_of_survey_objects))

    orig_id_to_survey_data_index = dict(
        zip(orig_ids_of_survey_objects, range(len(orig_ids_of_survey_objects))))

    syn_pop_UDA_data = []
    for cur_syn_pop_object in synthetic_pop_data:
        # works especially if orig_id_attribute_type == VISUM_VALUETYPE_STRING
        original_id = cur_syn_pop_object[2]

        if orig_id_attribute_type == VISUM_VALUETYPE_INT:
            original_id = int(original_id)
        elif orig_id_attribute_type == VISUM_VALUETYPE_FLOAT:
            original_id = float(original_id)

        syn_pop_UDA_data.append(survey_UDA_data[orig_id_to_survey_data_index[original_id]])

    syn_pop_objects.SetMultipleAttributes(UDA_IDs, syn_pop_UDA_data)
    return orig_id_to_survey_data_index


def read_synthetic_persons(syn_pop_person_file_name):
    synthetic_pop_person_data = []
    synthetic_pop_person_key_data = []  # person key is [No (int), Household (int)]
    with open(syn_pop_person_file_name, newline='') as csvfile:
        file_reader = csv.reader(csvfile, delimiter=';', quotechar='*')
        # first read header
        header = next(file_reader)
        # NO;HHNO;ORIGINALPERSONID
        if header[0] != "NO" or header[1] != "HHNO" or header[2] != "ORIGINALPERSONID" :
            logging.warning("Synthetic population person file needs format NO;HHNO;ORIGINALPERSONID")
        for row in file_reader:
            if len(row) > 0:
                # NOTE: column 2 may be of type string
                row = [int(row[0]), int(row[1]), row[2]]
                synthetic_pop_person_data.append(row)
                synthetic_pop_person_key_data.append(row[0:2])

    sorted(synthetic_pop_person_data,key=lambda entry:entry[1])
    sorted(synthetic_pop_person_key_data,key=lambda entry:entry[1])

    return synthetic_pop_person_data, synthetic_pop_person_key_data


def read_synthetic_households(syn_pop_HH_file_name, location_no_to_home_act_loc):
    synthetic_pop_HH_data = []
    synthetic_pop_HH_key_data = []   # household key is ["No", IActivityLocation]
    with open(syn_pop_HH_file_name, newline='') as csvfile:
        file_reader = csv.reader(csvfile, delimiter=';', quotechar='*')

        # first read header
        header = next(file_reader)
        # NO;LOCATIONNO;ORIGINALHHID
        if header[0] != "NO" or header[1] != "LOCATIONNO" or header[2] != "ORIGINALHHID" :
            logging.warning("Synthetic population household file needs format NO;LOCATIONNO;ORIGINALHHID")
        for row in file_reader:
            if len(row) > 0:
                # NOTE: column 2 may be of type string
                row = [int(row[0]), int(row[1]), row[2]] 
                synthetic_pop_HH_data.append(row)
                home_act_loc_obj = location_no_to_home_act_loc[row[1]]
                synthetic_pop_HH_key_data.append([row[0], home_act_loc_obj])

    sorted(synthetic_pop_HH_data,key=lambda entry:entry[1])

    return synthetic_pop_HH_data, synthetic_pop_HH_key_data


def create_location_dict(Visum):
    home_act_locs = Visum.Net.ActivityLocations.GetFilteredSet(r'[ActivityCode]="H"')
    home_act_loc_nos = np.array(home_act_locs.GetMultiAttValues('LocationNo'), dtype=int)[:, 1]
    location_no_to_home_act_loc = dict(zip(home_act_loc_nos, home_act_locs))
    return location_no_to_home_act_loc


def clear_population(Visum, min_survey_location_no = None):
    Visum.Net.Tours.RemoveAll()
    Visum.Net.Schedules.RemoveAll()
    Visum.Net.Persons.RemoveAll()
    Visum.Net.Households.RemoveAll()

    if min_survey_location_no is not None:
        survey_activity_locations = Visum.Net.ActivityLocations.GetFilteredSet(rf'[Location\No] >= {min_survey_location_no}')
        survey_activity_locations.RemoveAll()
        survey_locations = Visum.Net.Locations.GetFilteredSet(rf'[No] >= {min_survey_location_no}')
        survey_locations.RemoveAll()
