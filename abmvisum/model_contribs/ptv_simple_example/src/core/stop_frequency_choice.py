import logging
import src.choice_engine
import src.visum_utilities


def run(Visum, segments, is_inbound):
    # reset result attribute
    result_attribute = 'inb_stops' if is_inbound else 'outb_stops'
    src.visum_utilities.insert_UDA_if_missing(Visum.Net.Tours, result_attribute)
    Visum.Net.Tours.SetAllAttValues(result_attribute, 0)

    for segment in segments:
        logging.info('stop frequency choice model %s: %s', segment['Specification'], segment['Comment'])
        src.choice_engine.run_simple_choice(Visum.Net.Tours, segment, result_attribute)
