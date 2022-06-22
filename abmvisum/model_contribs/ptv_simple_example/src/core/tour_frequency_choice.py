import logging

from src import choice_engine, abm_utilities

def run(Visum, segments, globalPersonFilter, is_primary):
    subjects = Visum.Net.Persons
    for segment in segments:
        logging.info('tour frequency choice model %s: %s', segment['Specification'], segment['Comment'])
        
        if globalPersonFilter == '':
            segment_filter = segment['Filter']
        elif segment['Filter'] == '':
            segment_filter = globalPersonFilter
        else:
            segment_filter = segment['Filter'] + ' & ' + globalPersonFilter

        filtered_persons = abm_utilities.get_filtered_subjects(subjects, segment_filter)
        if len(filtered_persons) == 0:
            return

        num_tours_per_filtered_person = choice_engine.calc_simple_choice_per_subject(filtered_persons, segment['AttrExpr'], segment['Beta'], segment['Choices'])

        activity_id = segment["AddData"]
        create_tours(Visum, filtered_persons, activity_id, is_primary, num_tours_per_filtered_person)


def create_tours(Visum, persons, activity_id, is_primary, num_tours_per_person):
    person_nos = persons.GetMultiAttValues("No")
    person_data = zip(person_nos, num_tours_per_person)
    tour_no = 0 # use next free no
    schedule_no = 1 # we have one schedule per person
    tour_keys = [(tour_no, person_no[1], schedule_no) for person_no, num_tours in person_data
                                                      for i in range(int(num_tours))]
    tour_attrs = [(is_primary, activity_id) for num_tours in num_tours_per_person
                                            for i in range(int(num_tours))]

    if len(tour_keys) == 0:
        return
    tours = Visum.Net.AddMultiTours(tour_keys)
    tours.SetMultipleAttributes(['is_primary', 'PrimTourMajorActivityID'], tour_attrs)
