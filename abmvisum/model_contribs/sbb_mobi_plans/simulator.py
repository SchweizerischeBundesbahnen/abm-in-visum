import logging
from logging import StreamHandler
import importlib

import numpy as np
import pandas as pd

import abmvisum.engines.choice_engine as choice_engine
import abmvisum.tools.utilities as utilities
from .config import Config
from .core.activity_choice import run_activity_choice
from .core.combined_destination_and_duration_choice import run_combined_destination_and_duration_choice
from .core.create_trips import create_trips
from .core.long_term_primary_location_choice import run_long_term_primary_location_choice
from .core.mode_choice import run_mode_choice
from .core.start_time_choice import run_start_time_choice
from .core.tour_frequency_choice import run_tour_frequency_choice
from .matrix_cache import MatrixCache

importlib.reload(choice_engine)


class MOBiPlansSimulator(StreamHandler):
    def __init__(self, Visum, seed_val=42, add_skims_to_cache=True):
        self.Visum = Visum
        self.rand = np.random.RandomState(seed_val)

        StreamHandler.__init__(self)
        logger = logging.getLogger()
        logger.addHandler(self)

        logging.info('--- initialize data and config ---')
        self.initialize_data_and_config(add_skims_to_cache)

    def emit(self, record):
        try:
            msg = self.format(record)
            self.Visum.Log(20480, msg)  # message priority: Note
        except:
            pass

    def initialize_data_and_config(self, add_skims_to_cache=True):
        # initialize all filters
        self.Visum.Filters.InitAll()

        # load zones
        self.zones = np.array(self.Visum.Net.Zones.GetMultiAttValues('No'), dtype=int)[:, 1]
        self.zoneNo_to_zoneInd = dict(zip(self.zones, range(len(self.zones))))

        # init config
        self.config = Config(self.Visum, logging, add_skims_to_cache)

    def long_term_location_choice(self, cache=None, use_IPF=True, accsib_multimodal_attribute='accsib_mul'):
        logging.info('--- location choice ---')
        segments = self.config.load_choice_para('PrimLoc')
        accessibility_multi = np.array(self.Visum.Net.Zones.GetMultiAttValues(accsib_multimodal_attribute), dtype=int)[
                              :, 1]

        logging.info('--- initialize long-term locations keys ---')
        logging.info('remove existing long term choices')
        self.Visum.Net.Persons.RemoveAllLongTermChoices()

        logging.info('remove existing schedules')
        self.Visum.Net.Schedules.RemoveAll()
        self.Visum.Net.Persons.SetAllAttValues("out_of_home_time", 0.0)

        dist_mat = self.config.skim_matrices.get_skim("car_net_distance_sym")
        run_long_term_primary_location_choice(self.Visum, self.config.skim_matrices, segments, self.zones,
                                              self.zoneNo_to_zoneInd, accessibility_multi,
                                              dist_mat, logging, use_IPF, cache)

    def plan_generation(self, time_budget_dict=None, out_of_home_budget=18.0):
        matrix_cache = MatrixCache(logging)

        nb_iters = len(time_budget_dict)

        for i in range(nb_iters):
            logging.info("--- plan adjustment iteration " + str(i + 1) + " ---")

            active_persons = self.Visum.Net.Persons.GetFilteredSet("[active]=1")
            if i == 0:
                active_persons.SetAllAttValues("out_of_home_time", 99.0)
                filtered_persons = active_persons

                logging.info('--- initialize schedules ---')
                logging.info('remove existing schedules')
                self.Visum.Net.Schedules.RemoveAll()
                # create one schedule per person
                logging.info('create %d schedules (one per active person)' % active_persons.Count)
                keys = np.array(active_persons.GetMultipleAttributes(['No', 'No']))
                keys[:, 0] = 1
                self.Visum.Net.AddMultiSchedules(keys)
            else:
                filtered_persons = active_persons.GetFilteredSet("[out_of_home_time]>=" + str(out_of_home_budget))
                utilities.init_schedules(self.Visum, filtered_persons, logging)

            logging.info('--- tour frequency choice ---')
            self.tour_frequency_choice(filtered_persons)

            filtered_tours = self.Visum.Net.Tours.GetFilteredSet(r"[Schedule\Person\out_of_home_time]>=" +
                                                                 str(out_of_home_budget))

            logging.info('--- stop frequency choice ---')
            self.stop_frequency_choice(filtered_tours)

            logging.info('--- subtour choice ---')
            self.subtour_choice(filtered_persons, filtered_tours)

            filtered_act_ex = self.Visum.Net.ActivityExecutions.GetFilteredSet(
                r"[Schedule\Person\out_of_home_time]>=" + str(out_of_home_budget))
            filtered_trips = self.Visum.Net.Trips.GetFilteredSet(
                r"[Tour\Schedule\Person\out_of_home_time]>=" + str(out_of_home_budget))

            logging.info('--- activity type choice ---')
            self.activity_choice(filtered_act_ex)

            logging.info('--- combined destination and activity duration choice ---')
            budget_for_iteration = time_budget_dict[i]
            self.combined_dest_and_dur_choice(matrix_cache, filtered_act_ex, filtered_trips,
                                              performing_budget=budget_for_iteration['performing_budget'],
                                              travel_time_budget=budget_for_iteration['travel_time_budget'],
                                              out_of_home_time_budget=budget_for_iteration['out_of_home_time_budget'])

            utilities.set_out_of_home_time(filtered_persons)

    def tour_frequency_choice(self, filtered_persons):
        segments_prim = self.config.load_choice_para('TourFreqPrim')
        for segment in segments_prim:
            result_attr = segment['ResAttr']
            filtered_persons.SetAllAttValues(result_attr, 0)

        segments_sec = self.config.load_choice_para('TourFreqSec')
        for segment in segments_sec:
            result_attr = segment['ResAttr']
            filtered_persons.SetAllAttValues(result_attr, 0)

        run_tour_frequency_choice(self.Visum, self.rand, filtered_persons, segments_prim, is_primary=True)
        run_tour_frequency_choice(self.Visum, self.rand, filtered_persons, segments_sec, is_primary=False)

    def stop_frequency_choice(self, filtered_tours):
        segments = self.config.load_choice_para('TourStopFreq')
        for segment in segments:
            result_attr = segment['ResAttr']
            filtered_tours.SetAllAttValues(result_attr, 0)

        for segment in segments:
            logging.info('stop frequency choice model %s: %s' % (segment['Specification'], segment['Comment']))
            choice_engine.run_simple_choice(filtered_tours, segment, self.rand)

    def subtour_choice(self, filtered_persons, filtered_tours):
        segments = self.config.load_choice_para('SubtourFreq')
        for segment in segments:
            result_attr = segment['ResAttr']
            filtered_tours.SetAllAttValues(result_attr, 0)

        for segment in segments:
            logging.info('subtour choice model %s: %s' % (segment['Specification'], segment['Comment']))
            choice_engine.run_simple_choice(filtered_tours, segment, self.rand)

        logging.info('--- create trips ---')
        create_trips(self.Visum, filtered_persons, filtered_tours, self.rand, logging)

    def activity_choice(self, filtered_act_ex):
        segments = self.config.load_choice_para('ActType')
        run_activity_choice(segments, self.Visum, filtered_act_ex, self.rand, logging)

    def combined_dest_and_dur_choice(self, matrix_cache, filtered_act_ex, filtered_trips, performing_budget=16.0,
                                     travel_time_budget=5.0, out_of_home_time_budget=18.0):
        segments_dest = self.config.load_choice_para('SecDest')
        segments_dur = self.config.load_act_dur_para('ActDur')
        run_combined_destination_and_duration_choice(matrix_cache, self.Visum, filtered_act_ex, filtered_trips,
                                                     self.config.skim_matrices, segments_dest,
                                                     segments_dur, self.zones, self.zoneNo_to_zoneInd,
                                                     self.rand, logging, performing_budget, travel_time_budget,
                                                     out_of_home_time_budget)

    def mode_choice(self):
        logging.info('--- mode choice ---')
        run_mode_choice(self.Visum, self.config, self.zoneNo_to_zoneInd, logging, self.rand)

    def start_time_choice(self):
        logging.info('--- start time choice ---')
        segments = self.config.load_start_times_para('StartTime')
        run_start_time_choice(segments, self.Visum, logging, self.rand)

    def railaccess_choiceset(self):
        logging.info('--- railaccess choice set ---')
        segments = self.config.load_choice_para('RailAccess')
        subjects = self.Visum.Net.Persons
        activities = sorted(list(set(self.Visum.Net.Activities.GetMultipleAttributes(["Name"]))))

        for segment in segments:
            logging.info(f'{segment["Specification"]}->{segment["ResAttr"]}')
            betas = [b for (a, b) in zip(segment['AttrExpr'], segment['Beta']) if a != '0']
            att_expr = [a for a in segment['AttrExpr'] if a != '0']
            betas_act = [b for (a, b) in zip(segment['AttrExpr'], segment['Beta']) if a == '0']
            acts = [b for (a, b) in zip(segment['AttrExpr'], segment['Comments']) if a == '0']
            filtered_subjects = utilities.get_filtered_subjects(subjects, segment['Filter'])
            if segment['ResAttr'][-4:]=="_act":
                choice_per_subject = [0]*len(filtered_subjects)
                for (activity, beta) in zip(acts, betas_act):
                    assert activity in activities, f"{segment['Specification']}, activity {activity} not defined"
                    bit = 2**activities.index(activity)
                    choice_per_subject = choice_per_subject + choice_engine.calc_binary_probabilistic_choice_per_subject(filtered_subjects,
                                                                                         ['1'],
                                                                                         [beta[0]],
                                                                                         segment['Choices'], self.rand)*bit
            else:
                choice_per_subject = choice_engine.calc_binary_probabilistic_choice_per_subject(filtered_subjects,
                                                                                         att_expr,
                                                                                         [b[0] for b in betas],
                                                                                         segment['Choices'], self.rand)
            result_attr = segment['ResAttr']
            filtered_subjects.SetAllAttValues(result_attr, 0)
            utilities.SetMulti(filtered_subjects, result_attr, choice_per_subject)

            logging.info('%s set for %d objects' % (result_attr, len(choice_per_subject)))

    def ownership_models(self):
        logging.info('--- mobility tool ownership models ---')
        segments = self.config.load_choice_para('MobilityTools')
        for segment in segments:
            shadow_util_dict = None
            subject_attribute = None
            choice_object = None

            if segment["Specification"] == "DriversLicense":
                table_name = "Driving_License_Shadow_Utils"
                table_entries = self.Visum.Net.TableDefinitions.ItemByKey(table_name).TableEntries
                shadow_utils = pd.DataFrame(
                    table_entries.GetMultipleAttributes(['aggregate_id', 'option_0', 'option_1']),
                    columns=['aggregate_id', 'option_0', 'option_1']).set_index('aggregate_id')
                shadow_util_dict = {index: (row[0], row[1]) for index, row in shadow_utils.iterrows()}
                subject_attribute = table_entries.Attributes.GetLongName('aggregate_id')
                choice_object = self.Visum.Net.Persons

            elif segment["Specification"] == "CarsInHH":
                table_name = "NB_Cars_HH_Shadow_Utils"
                table_entries = self.Visum.Net.TableDefinitions.ItemByKey(table_name).TableEntries
                shadow_utils = pd.DataFrame(
                    table_entries.GetMultipleAttributes(['aggregate_id', 'option_0', 'option_1',
                                                         'option_2', 'option_3']),
                    columns=['aggregate_id', 'option_0', 'option_1', 'option_2', 'option_3']).set_index('aggregate_id')
                shadow_util_dict = {index: (row[0], row[1], row[2], row[3]) for index, row in shadow_utils.iterrows()}
                subject_attribute = table_entries.Attributes.GetLongName('aggregate_id')
                choice_object = self.Visum.Net.Households

            elif segment["Specification"] == "PTSubscr_u18":
                choice_object = self.Visum.Net.Persons

            elif segment["Specification"] == "PTSubscr":
                table_name = "PT_Subscription_Shadow_Utils"
                table_entries = self.Visum.Net.TableDefinitions.ItemByKey(table_name).TableEntries
                shadow_utils = pd.DataFrame(
                    table_entries.GetMultipleAttributes(['aggregate_id', 'option_0', 'option_1', 'option_2',
                                                         'option_3', 'option_4']),
                    columns=['aggregate_id', 'option_0', 'option_1', 'option_2',
                             'option_3', 'option_4']).set_index('aggregate_id')
                shadow_util_dict = {index: (row[0], row[1], row[2], row[3],
                                            row[4]) for index, row in shadow_utils.iterrows()}
                subject_attribute = table_entries.Attributes.GetLongName('aggregate_id')
                choice_object = self.Visum.Net.Persons

            logging.info(segment)
            choice_engine.run_simple_choice(choice_object, segment, self.rand, shadow_util_dict, subject_attribute)
