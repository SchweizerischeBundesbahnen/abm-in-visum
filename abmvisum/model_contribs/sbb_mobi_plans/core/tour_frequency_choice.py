import logging

import abmvisum.engines.choice_engine as choice_engine


def run_tour_frequency_choice(Visum, rand, filtered_persons, segments, is_primary):
    subjects = filtered_persons
    for segment in segments:
        logging.info('tour frequency choice model %s: %s' % (segment['Specification'], segment['Comment']))

        choice_engine.run_simple_choice(subjects, segment, rand)
        create_tours(Visum, subjects, segment["AddData"], is_primary, segment['ResAttr'])


def create_tours(Visum, subjects, activity_id, is_primary, num_tours_attr):
    person_data = subjects.GetMultipleAttributes(["No", num_tours_attr])

    tour_keys = [(0, person_no, 1) for person_no, num_tours in person_data
                 for __ in range(int(num_tours))]
    if len(tour_keys) == 0:
        return
    tours = Visum.Net.AddMultiTours(tour_keys)

    tour_attrs = [(is_primary, activity_id) for person_no, num_tours in person_data
                  for __ in range(int(num_tours))]
    tours.SetMultipleAttributes(['is_primary', 'main_activity_id'], tour_attrs)
