import logging

import numpy as np

import VisumPy.helpers as VPH

import abm
from src import visum_utilities, abm_utilities
from settings import abm_settings

def run(Visum):
    abm.ABM(Visum=Visum, model_dir=r'.\\') # used for logging messages to Visum

    time_intervals = Visum.Net.CalendarPeriod.AnalysisTimeIntervalSet.TimeIntervals
    time_interval_start_times = np.array(visum_utilities.GetMulti(time_intervals, 'StartTime'), dtype=int)
    time_interval_end_times = np.array(visum_utilities.GetMulti(time_intervals, 'EndTime'), dtype=int)
    num_time_intervals = time_intervals.Count

    zone_nos = np.array(Visum.Net.Zones.GetMultiAttValues('No'), dtype=int)[:, 1]
    zoneNo_to_zoneInd = dict(zip(zone_nos, range(len(zone_nos))))
    num_zones = zone_nos.shape[0]

    trips_with_dep_time = Visum.Net.Trips.GetFilteredSet('NUMTOSTR([SCHEDDEPTIME]) != ""')

    dseg_codes = visum_utilities.GetMulti(trips_with_dep_time, r'DSEGCODE', chunk_size=abm_settings.chunk_size_trips, reindex=True)
    unique_dseg_codes = set(dseg_codes)

    for dseg_code in unique_dseg_codes:
        trips_cur_dseg = trips_with_dep_time.GetFilteredSet(f'[DSegCode] = "{dseg_code}"')

        dep_times = visum_utilities.GetMulti(trips_cur_dseg, r'SCHEDDEPTIME', chunk_size=abm_settings.chunk_size_trips, reindex=True)
        
        trip_ti_indices = [abm_utilities.get_time_interval_index(
            time_point, time_interval_start_times, time_interval_end_times) for time_point in dep_times]
        from_zone_indices = abm_utilities.nos_to_indices(np.array(visum_utilities.GetMulti(
            trips_cur_dseg, r"FROMACTIVITYEXECUTION\LOCATION\ZONE\NO", chunk_size=abm_settings.chunk_size_trips), dtype=int), zoneNo_to_zoneInd)
        to_zone_indices = abm_utilities.nos_to_indices(np.array(visum_utilities.GetMulti(
            trips_cur_dseg, r"TOACTIVITYEXECUTION\LOCATION\ZONE\NO", chunk_size=abm_settings.chunk_size_trips), dtype=int), zoneNo_to_zoneInd)

        trip_count_matrices = np.full(shape=(num_time_intervals, num_zones, num_zones), fill_value=0, dtype=int)

        for trip_ti_index, from_zone_index, to_zone_index in zip(trip_ti_indices, from_zone_indices, to_zone_indices):
            trip_count_matrices[trip_ti_index, from_zone_index, to_zone_index] += 1
        
        # collect matrices for demand
        for time_interval_index, (start_time, end_time) in enumerate(zip(time_interval_start_times, time_interval_end_times)):
            matrix_ref = {
                "MATRIXTYPE": 3, # demand matrix
                "OBJECTTYPEREF": 2, # zone
                "DSEGCODE": dseg_code,
                "Name": "assignment",
                "FROMTIME": start_time,
                "TOTIME": end_time }

            try:
                VPH.SetMatrixRaw(Visum, matrix_ref, trip_count_matrices[time_interval_index])
            except Exception as error:
                logging.info(f"demand matrix could not be written: {error}")


if __name__ == "__main__":
    # start logging engine
    abm_utilities.start_logging()

    if 'Visum' in globals():
        run(Visum=globals()['Visum'])
    else:
        raise Exception('Please start from inside Visum or use run inside another script.')
