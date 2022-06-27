import logging
from src import visum_utilities, choice_engine

def run(segments, Visum):
    activityId_to_activityCode = dict(Visum.Net.Activities.GetMultipleAttributes(["ID", "Code"]))
    for segment in segments:
        logging.info(f"activity type choice model {segment['Specification']}: {segment['Comment']}")

        filtered_actExs = Visum.Net.ActivityExecutions.GetFilteredSet(segment['Filter'])
        chosen_act_ids = choice_engine.calc_simple_choice_per_subject(filtered_actExs, segment['AttrExpr'], segment['Beta'], segment['Choices'])

        # convert chosen activities from id to code
        act_codes = [activityId_to_activityCode[id] for id in chosen_act_ids]
        visum_utilities.SetMulti(filtered_actExs, "ActivityCode", act_codes, chunk_size=10000000)

        logging.info(f'ActivityCode set for {len(act_codes)} activity executions')
