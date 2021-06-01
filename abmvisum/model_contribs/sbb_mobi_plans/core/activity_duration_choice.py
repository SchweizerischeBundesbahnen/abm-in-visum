import numpy as np


def choose_activity_durations(num_filtered_subjects, distribution_data, rand, max_iter=5):
    # returns durations (in seconds) by chosen distribution
    assert(num_filtered_subjects > 0)
    
    # init with nan
    duration = np.full(num_filtered_subjects, np.nan, dtype=float)

    for _ in range(max_iter):
        # invalid_durations = durations that are uninitialized (= nan), too short (< min) or too long (> max)
        with np.errstate(invalid='ignore'):
            invalid_durations = (np.isnan(duration) + (duration < distribution_data["MIN"]) +
                                 (duration > distribution_data["MAX"]))
        num_invalid_durations = np.count_nonzero(invalid_durations)

        if num_invalid_durations == 0:
            break

        if distribution_data["DISTRIBUTION"] == 'normal':
            duration[invalid_durations] = rand.normal(distribution_data["AVGDURATION"], distribution_data["STDDEV"],
                                                      num_invalid_durations)
        elif distribution_data["DISTRIBUTION"] == 'lognormal':
            duration[invalid_durations] = rand.lognormal(distribution_data["AVGDURATION"], distribution_data["STDDEV"],
                                                         num_invalid_durations)
        elif distribution_data["DISTRIBUTION"] == 'Weibull':
            duration[invalid_durations] = distribution_data["STDDEV"] * rand.weibull(distribution_data["AVGDURATION"],
                                                                                     num_invalid_durations)

    assert(np.count_nonzero(np.isnan(duration)) == 0)
    duration[duration < distribution_data["MIN"]] = distribution_data["MIN"]
    duration[duration > distribution_data["MAX"]] = distribution_data["MAX"]

    # convert result from hours -> seconds
    duration = 3600 * duration
    return duration
