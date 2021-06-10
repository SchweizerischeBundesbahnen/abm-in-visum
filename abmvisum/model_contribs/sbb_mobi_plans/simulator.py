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

importlib.reload(choice_engine)  # prevents Visum from caching imports


class MOBiPlansSimulator(StreamHandler):
    """
        MOBiPlansSimulator is the central simulator class for an activity-based model. With the initialization, it
        loads all necessary parameters which are stored within Visum. It provides all methods to generate
        activity-based travel demand for all persons of a synthetic population as stored in the Persons list
        in Visum.

        Parameters:
            Visum: Instance of the PTV Visum software.
            seed_val (int): Fixes the random seed to this number.
            add_skims_to_cache: Possibility to load all skim matrices as defined in the config into the Python cache.

        Attributes:
            Visum: Instance of the PTV Visum software.
            rand: Random state of numpy.
            config (Config): Config object containing all the necessary parameters.
            zones: List of all zone numbers.
            zoneNo_to_zoneInd: Mapping between zone number and position of the zone index in the matrices.
    """

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

        # load zones and create mapping to come from zone numbers to indices (for matrix intexing)
        self.zones = np.array(self.Visum.Net.Zones.GetMultiAttValues('No'), dtype=int)[:, 1]
        self.zoneNo_to_zoneInd = dict(zip(self.zones, range(len(self.zones))))

        # init config
        self.config = Config(self.Visum, logging, add_skims_to_cache)

    def long_term_location_choice(self, cache=None, use_IPF=True, accsib_multimodal_attribute='accsib_mul'):
        """
            Simulation of the long-term location choice (work or school place).
            It is a two-staged process: Firstly, a traffic zone is chosen. Secondly, the choice of one specific
            location within the chosen zone is performed.

            Parameters:
                cache (MatrixCache): MatrixCache object containing all cached matrices.
                use_IPF (bool): Possibility to use the iterative fitting approach or not.
                accsib_multimodal_attribute (str): the multimodal accessibility of the long-term location's zone.

            Returns:
                Updated Visum Persons tables with information about their long-term locations.
        """
        logging.info('--- location choice ---')
        segments = self.config.load_choice_para('PrimLoc')
        # the multimodal accessibility of the long-term location is a necessary input for the subtour choice models
        accessibility_multi = np.array(self.Visum.Net.Zones.GetMultiAttValues(accsib_multimodal_attribute),
                                       dtype=int)[:, 1]

        logging.info('--- initialize long-term locations keys ---')
        logging.info('remove existing long term choices')
        self.Visum.Net.Persons.RemoveAllLongTermChoices()

        logging.info('remove existing schedules')
        self.Visum.Net.Schedules.RemoveAll()
        self.Visum.Net.Persons.SetAllAttValues("out_of_home_time", 0.0)

        # we use car distances from residence to long-term location as an input for the tour frequency models.
        dist_mat = self.config.skim_matrices.get_skim("car_net_distance_sym")

        # simulating the long-term location choice step
        run_long_term_primary_location_choice(self.Visum, self.config.skim_matrices, segments, self.zones,
                                              self.zoneNo_to_zoneInd, accessibility_multi,
                                              dist_mat, logging, use_IPF, cache)

    def plan_generation(self, time_budget_dict=None, out_of_home_budget=18.0):
        """
            This method simulates the choice dimensions activity generation, duration and secondary
            destinations. It is built as an iterative process which considers time budgets.

            Parameters:
                time_budget_dict: Dictionary with information about time budgets in specific iterations
                out_of_home_budget (float): Total out-of-home budget.

            Returns:
                Create Visum Tours/ Trips/ ActivityExecutions tables and fills them with all necessary information.
        """
        matrix_cache = MatrixCache(logging)

        nb_iters = len(time_budget_dict)

        for i in range(nb_iters):
            logging.info("--- plan generation iteration " + str(i + 1) + " ---")

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
                # initializes schedules of all persons that break the out-of-home time budget.
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
            # reset tour frequency choices
            result_attr = segment['ResAttr']
            filtered_persons.SetAllAttValues(result_attr, 0)

        segments_sec = self.config.load_choice_para('TourFreqSec')
        for segment in segments_sec:
            # reset tour frequency choices
            result_attr = segment['ResAttr']
            filtered_persons.SetAllAttValues(result_attr, 0)

        # run primary tour frequency
        run_tour_frequency_choice(self.Visum, self.rand, filtered_persons, segments_prim, is_primary=True)
        # run secondary tour frequency
        run_tour_frequency_choice(self.Visum, self.rand, filtered_persons, segments_sec, is_primary=False)

    def stop_frequency_choice(self, filtered_tours):
        segments = self.config.load_choice_para('TourStopFreq')
        for segment in segments:
            # reset stop frequency
            result_attr = segment['ResAttr']
            filtered_tours.SetAllAttValues(result_attr, 0)

        for segment in segments:
            logging.info('stop frequency choice model %s: %s' % (segment['Specification'], segment['Comment']))
            choice_engine.run_simple_choice(filtered_tours, segment, self.rand)

    def subtour_choice(self, filtered_persons, filtered_tours):
        segments = self.config.load_choice_para('SubtourFreq')
        for segment in segments:
            # reset subtour frequency
            result_attr = segment['ResAttr']
            filtered_tours.SetAllAttValues(result_attr, 0)

        for segment in segments:
            logging.info('subtour choice model %s: %s' % (segment['Specification'], segment['Comment']))
            choice_engine.run_simple_choice(filtered_tours, segment, self.rand)

        logging.info('--- create trips ---')
        # create trips and activity excecutions
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
