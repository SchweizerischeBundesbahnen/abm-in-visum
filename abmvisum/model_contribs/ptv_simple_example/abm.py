import logging

import numpy as np

from src import abm_utilities

import src.config
import src.choice_engine

import src.core.activity_choice
import src.core.activity_duration_choice
import src.core.create_trips
import src.core.long_term_primary_location_choice
import src.core.major_location_choice
import src.core.major_mode_choice
import src.core.minor_location_and_mode_choice
import src.core.start_time_choice
import src.core.stop_frequency_choice
import src.core.tour_frequency_choice

import src.visum_utilities
import src.checks
import src.import_synthetic_population


class ABM(logging.StreamHandler):
    # construct from existing Visum (construct from version file by using from_version_file() class method)
    def __init__(self, Visum, model_dir, log_to_Visum=True):
        self.Visum = Visum
        self.model_dir = model_dir

        if log_to_Visum:
            logging.StreamHandler.__init__(self)
            logger = logging.getLogger()
            logger.addHandler(self)

    # construct from version file
    @classmethod
    def from_version_file(cls, version_filename, visum_version_no, model_dir):
        try:
            # pylint: disable=import-outside-toplevel
            import win32com.client
            Visum = win32com.client.Dispatch("Visum.Visum." + str(visum_version_no))
            Visum.IO.LoadVersion(version_filename)
            return cls(Visum, model_dir, log_to_Visum=False)

        except ImportError:
            logging.warning("import win32com.client failed.")
            raise

    # implementation of StreamHandler
    def emit(self, record):
         # pylint: disable=bare-except
        try:
            msg = self.format(record)
            level = record.levelname
            if level == 'INFO':
                self.Visum.Log(20480, msg)  # message priority: Note
            elif level == 'WARNING':
                self.Visum.Log(16384, msg)  # message priority: Warning
            elif level == 'ERROR':
                self.Visum.Log(12288, msg)  # message priority: Error
            else:
                self.Visum.Log(20480, msg)  # message priority: Note
        except:
            pass  # swallow all exceptions

    def initialize_data_and_config(self):
        # load zones
        self.zones = np.array(src.visum_utilities.GetMulti(self.Visum.Net.Zones, r'No'), dtype=int)
        self.zoneNo_to_zoneInd = dict(zip(self.zones, range(len(self.zones))))

        # load locations
        self.locations = np.array(src.visum_utilities.GetMulti(self.Visum.Net.Locations, r'No'), dtype=int)
        self.locationNo_to_locationInd = dict(zip(self.locations, range(len(self.locations))))

        time_intervals = self.Visum.Net.CalendarPeriod.AnalysisTimeIntervalSet.TimeIntervals
        self.time_interval_start_times = np.array(src.visum_utilities.GetMulti(time_intervals, 'StartTime'), dtype=int)
        self.time_interval_end_times = np.array(src.visum_utilities.GetMulti(time_intervals, 'EndTime'), dtype=int)

        self.config = src.config.Config(self.Visum)

    def check_specifications(self, check_generation_steps=True):
        src.checks.check_specifications(self.Visum, self.config, self.time_interval_start_times, self.time_interval_end_times, check_generation_steps=check_generation_steps)

    def run_full_abm(self, write_model_state=lambda step_name, only_demand: None):
        write_model_state('0_orig', False)

        logging.info('--- Run full ABM ---')
        self.Visum.Graphic.StopDrawing = True

        logging.info('--- initialize data and config ---')
        self.initialize_data_and_config()

        # import synthetic population
        logging.info('--- generate trips from synthetic population files ---')
        self.import_syn_pop(self.model_dir)

        logging.info('--- check specifications ---')
        self.check_specifications()

        logging.info('--- initialize schedules ---')
        self.init_schedules()
        write_model_state('1_schedules_initialized', True)

        logging.info('--- location choice ---')
        self.long_term_primary_location_choice(use_IPF=True)
        write_model_state('2_long_term_primary_location_choice', True)

        logging.info('--- tour frequency choice ---')
        self.tour_frequency_choice()
        write_model_state('3_tour_frequency_choice', True)

        logging.info('--- stop frequency choice ---')
        self.stop_frequency_choice()
        write_model_state('4_stop_frequency_choice', True)

        logging.info('--- subtour choice ---')
        self.subtour_choice()
        write_model_state('5_subtour_choice', True)

        logging.info('--- create trips ---')
        self.create_trips()
        write_model_state('6_create_trips', True)

        logging.info('--- activity type choice ---')
        self.activity_choice()
        write_model_state('7_activity_choice', True)

        logging.info('--- activity duration choice ---')
        self.activity_duration_choice()
        write_model_state('8_activity_duration_choice', True)

        logging.info('--- major activity start time choice ---')
        self.start_time_choice()
        write_model_state('9_start_time_choice', True)

        logging.info('--- major activity location choice ---')
        self.major_location_choice()
        write_model_state('10_major_location_choice', True)

        logging.info('--- major activity mode choice ---')
        self.major_mode_choice()
        write_model_state('11_major_mode_choice', True)

        logging.info('--- minor activity location and mode choice ---')
        self.minor_location_and_mode_choice()
        write_model_state('12_minor_location_and_mode_choice', True)

        self.Visum.Graphic.StopDrawing = False

        logging.info('--- ABM Run completed ---')


    def run_full_abm_with_syn_pop(self, write_model_state=lambda step_name, only_demand: None):
        write_model_state('0_orig', False)

        logging.info('--- Run full ABM with synthetic population ---')
        self.Visum.Graphic.StopDrawing = True

        logging.info('--- initialize data and config ---')
        self.initialize_data_and_config()

        logging.info('--- generate trips from synthetic population files ---')
        self.import_syn_pop(self.model_dir)

        logging.info('--- check specifications ---')
        self.check_specifications(check_generation_steps=False)

        logging.info('--- location choice ---')
        self.long_term_primary_location_choice(use_IPF=True)
        write_model_state('2_long_term_primary_location_choice', True)

        logging.info('--- major activity location choice ---')
        self.major_location_choice()
        write_model_state('3_major_location_choice', True)

        logging.info('--- major activity mode choice ---')
        self.major_mode_choice()
        write_model_state('4_major_mode_choice', True)

        logging.info('--- minor activity location and mode choice ---')
        self.minor_location_and_mode_choice()
        write_model_state('5_minor_location_and_mode_choice', True)

        self.Visum.Graphic.StopDrawing = False

        logging.info('--- ABM Run completed ---')


    def tripchain_choice(self):
        logging.info('--- Run trip chain choice ---')
        self.Visum.Graphic.StopDrawing = True

        logging.info('--- initialize schedules ---')
        self.init_schedules()

        logging.info('--- initialize data and config ---')
        self.initialize_data_and_config()

        logging.info('--- tour frequency choice ---')
        self.tour_frequency_choice()

        logging.info('--- stop frequency choice ---')
        self.stop_frequency_choice()

        logging.info('--- subtour choice ---')
        self.subtour_choice()

        logging.info('--- create trips ---')
        self.create_trips()

        logging.info('--- activity type choice ---')
        self.activity_choice()

        logging.info('--- activity duration choice ---')
        self.activity_duration_choice()

        logging.info('--- major activity start time choice ---')
        self.start_time_choice()

        self.Visum.Graphic.StopDrawing = False

        logging.info('--- trip chain choice completed ---')


    def load_synpop_with_tripchains(self):

        logging.info('--- Load synthetic population including trip chain skeletons ---')
        self.Visum.Graphic.StopDrawing = True

        logging.info('--- generate trips from synthetic population ---')
        self.import_syn_pop(self.model_dir)

        self.Visum.Graphic.StopDrawing = False

        logging.info('--- Load synthetic population including trip chain skeletons completed ---')


    def run_check_specifications(self):

        logging.info('--- Check specifications ---')

        logging.info('--- initialize data and config ---')
        self.initialize_data_and_config()

        self.check_specifications()

        logging.info('--- Check specifications completed ---')


    def longterm_location_choice(self):

        logging.info('--- Run long term location choice ---')
        self.Visum.Graphic.StopDrawing = True

        logging.info('--- initialize data and config ---')
        self.initialize_data_and_config()

        logging.info('--- location choice ---')

        # remove any old results
        logging.info('remove existing long term choices')
        self.Visum.Net.Persons.RemoveAllLongTermChoices()

        self.long_term_primary_location_choice(use_IPF=True)

        self.Visum.Graphic.StopDrawing = False

        logging.info('--- Long term location choice completed ---')


    def destination_and_mode_choice(self):
        logging.info('--- Run destination and mode choice ---')
        self.Visum.Graphic.StopDrawing = True

        logging.info('--- initialize data and config ---')
        self.initialize_data_and_config()

        logging.info('--- major activity location choice ---')
        self.major_location_choice()

        logging.info('--- major activity mode choice ---')
        self.major_mode_choice()

        logging.info('--- minor activity location and mode choice ---')
        self.minor_location_and_mode_choice()

        self.Visum.Graphic.StopDrawing = False

        logging.info('--- Destination and mode choice completed ---')


    def init_schedules(self):

        logging.info('remove existing schedules')
        self.Visum.Net.Tours.RemoveAll()
        self.Visum.Net.Schedules.RemoveAll()

        num_persons = self.Visum.Net.Persons.Count
        logging.info('create %d schedules (one per person)', num_persons)

        # create one schedule per person
        keys = np.array(self.Visum.Net.Persons.GetMultipleAttributes(['No', 'No']))
        keys[:, 0] = 1
        self.Visum.Net.AddMultiSchedules(keys)

    def long_term_primary_location_choice(self, use_IPF=True):
        segments = self.config.load_choice_para('PrimLoc')

        # remove all existing long term choices to enable feedback loop
        self.Visum.Net.Persons.RemoveAllLongTermChoices()

        try:
            globalPersonFilter = abm_utilities.get_global_attribute(self.Visum, attribute_name='GlobalPersonFilter', column='Name')
        except:
            globalPersonFilter = "" # definition is optional
        src.core.long_term_primary_location_choice.run(self.Visum, segments, self.locations, self.zoneNo_to_zoneInd, self.locationNo_to_locationInd,
                                                       globalPersonFilter, use_IPF)


    def tour_frequency_choice(self):
        try:
            globalPersonFilter = abm_utilities.get_global_attribute(self.Visum, 'GlobalPersonFilter', 'Name')
        except:
            globalPersonFilter = "" # definition is optional
            
        segments = self.config.load_choice_para('TourFreqPrim')
        src.core.tour_frequency_choice.run(self.Visum, segments, globalPersonFilter, is_primary=True)

        segments = self.config.load_choice_para('TourFreqSec')
        src.core.tour_frequency_choice.run(self.Visum, segments, globalPersonFilter, is_primary=False)

    def stop_frequency_choice(self):
        segments = self.config.load_choice_para('StopFreqInb')
        src.core.stop_frequency_choice.run(self.Visum, segments, is_inbound=True)

        segments = self.config.load_choice_para('StopFreqOutb')
        src.core.stop_frequency_choice.run(self.Visum, segments, is_inbound=False)

    def subtour_choice(self):
        segments = self.config.load_choice_para('SubtourFreq')

        resAttr = 'subtour_stops'
        src.visum_utilities.insert_UDA_if_missing(self.Visum.Net.Tours, resAttr)
        # reset result attribute
        self.Visum.Net.Tours.SetAllAttValues(resAttr, 0)

        for segment in segments:
            logging.info('subtour choice model %s: %s', segment['Specification'], segment['Comment'])
            src.choice_engine.run_simple_choice(self.Visum.Net.Tours, segment, resAttr)

    def create_trips(self):
        src.core.create_trips.run(self.Visum)

    def activity_choice(self):
        segments = self.config.load_choice_para('ActType')
        src.core.activity_choice.run(segments, self.Visum)

    def major_location_choice(self):
        segments = self.config.load_choice_para('DestMajor')
        src.core.major_location_choice.run(self.Visum, segments, self.config,
                                           self.zones, self.locations, self.zoneNo_to_zoneInd, self.locationNo_to_locationInd,
                                           self.time_interval_start_times, self.time_interval_end_times)

    def major_mode_choice(self):
        segments = self.config.load_choice_para('ModeMajor')
        src.core.major_mode_choice.run(self.Visum, segments, self.zoneNo_to_zoneInd)

    def minor_location_and_mode_choice(self):
        segmentsLocationChoice = self.config.load_choice_para(choice_model='DestAndModeMinor', specification_column='Specification')
        segmentsModeChoice     = self.config.load_choice_para(choice_model='DestAndModeMinor', specification_column='Specification2')

        src.core.minor_location_and_mode_choice.run(self.Visum, segmentsLocationChoice, segmentsModeChoice, self.config,
                                                    self.zones, self.locations, self.zoneNo_to_zoneInd, self.locationNo_to_locationInd,
                                                    self.time_interval_start_times, self.time_interval_end_times)

    def activity_duration_choice(self):
        # reset durations
        self.Visum.Net.ActivityExecutions.SetAllAttValues("Duration", 0)

        segments = self.config.load_act_dur_para('ActDur')
        src.core.activity_duration_choice.run(segments, self.Visum)

    def start_time_choice(self):
        segments = self.config.load_start_times_para('StartTime')
        src.core.start_time_choice.run(segments, self.Visum)

    def import_syn_pop(self, model_dir):
        src.import_synthetic_population.run(Visum=self.Visum, model_dir=model_dir)
