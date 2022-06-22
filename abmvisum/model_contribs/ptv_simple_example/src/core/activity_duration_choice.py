import logging
import numpy as np

from src import abm_utilities, visum_utilities


def run(segments, Visum):
    subjects = Visum.Net.ActivityExecutions
    for segment in segments:
        segment_spec = segment['Specification']
        filtered_subjects_for_segment = abm_utilities.get_filtered_subjects(subjects, segment['Filter'])
        num_filtered_subjects_for_segment = filtered_subjects_for_segment.Count
        logging.info('activity duration choice model %s: %d objects [%s]', segment_spec, num_filtered_subjects_for_segment, segment['Comment'])

        sum_of_filtered_subjects_for_sub_segment = 0
        for subsegment_index, sub_segment_filter_expr in enumerate(segment["AttrExpr"]):
            filtered_subjects_for_sub_segment = abm_utilities.get_filtered_subjects(filtered_subjects_for_segment, sub_segment_filter_expr)
            num_filtered_subjects = filtered_subjects_for_sub_segment.Count

            logging.info('activity duration choice model %s: subgroup "%s" (%d objects)', 
                segment_spec, segment['Subgroup_Comment'][subsegment_index], num_filtered_subjects)

            if num_filtered_subjects > 0:
                distribution_data = segment["distribution_data"][subsegment_index]

                result = choose_activity_durations(num_filtered_subjects, distribution_data)

                assert len(result) == num_filtered_subjects
                visum_utilities.SetMulti(filtered_subjects_for_sub_segment, 'Duration', result, chunk_size=10000000)
                sum_of_filtered_subjects_for_sub_segment += num_filtered_subjects

        assert num_filtered_subjects_for_segment == sum_of_filtered_subjects_for_sub_segment


def choose_activity_durations(num_filtered_subjects, distribution_data, max_iter=5):
    # returns durations (in seconds) by chosen distribution
    assert num_filtered_subjects > 0

    rand = np.random.RandomState(42)
    
    # init with nan
    duration = np.full(shape=num_filtered_subjects, fill_value=np.nan, dtype=float)

    # draw random numbers until all are valid (up to `max_iter` times)
    for _ in range(max_iter):
        # invalid_durations = durations that are uninitialized (= nan), too short (< min) or too long (> max)
        with np.errstate(invalid='ignore'):
            invalid_durations = np.isnan(duration) + (duration < distribution_data["MIN"]) + (duration > distribution_data["MAX"])
        num_invalid_durations = np.count_nonzero(invalid_durations)

        if num_invalid_durations == 0:
            break  # all numbers are valid

        if distribution_data["DISTRIBUTION"] == 'normal':
            duration[invalid_durations] = rand.normal(distribution_data["LOCATION"], distribution_data["SCALE"], num_invalid_durations)
        elif distribution_data["DISTRIBUTION"] == 'lognormal':
            duration[invalid_durations] = rand.lognormal(distribution_data["LOCATION"], distribution_data["SCALE"], num_invalid_durations)
        elif distribution_data["DISTRIBUTION"] == 'Weibull':
            duration[invalid_durations] = distribution_data["SCALE"] * rand.weibull(distribution_data["LOCATION"], num_invalid_durations)

    assert np.count_nonzero(np.isnan(duration)) == 0
    duration[duration < distribution_data["MIN"]] = distribution_data["MIN"]
    duration[duration > distribution_data["MAX"]] = distribution_data["MAX"]

    # convert result from hours -> seconds
    duration = 3600 * duration

    return duration
