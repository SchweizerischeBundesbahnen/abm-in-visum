import abminvisum.engines.choice_engine as choice_engine
import abminvisum.tools.utilities as utilities


def run_activity_choice(segments, Visum, active_act_ex, rand, logging):
    activityId_to_activityCode = dict(Visum.Net.Activities.GetMultipleAttributes(["Id", "Code"]))

    for segment in segments:
        logging.info('activity type choice model %s: %s' % (segment['Specification'], segment['Comment']))

        segment_filter = segment['Filter'].replace('_P_', 'Schedule\\Person')
        filtered_actExs = active_act_ex.GetFilteredSet(segment_filter)

        segment_attr_expression = [expr.replace('_P_', 'Schedule\\Person') for expr in segment['AttrExpr']]
        chosen_act_ids = choice_engine.calc_probabilistic_choice_per_subject(filtered_actExs, segment_attr_expression,
                                                                             segment['Beta'], segment['Choices'],
                                                                             rand)

        # convert chosen activities from id to code
        act_codes = [activityId_to_activityCode[id] for id in chosen_act_ids]
        utilities.SetMulti(filtered_actExs, "ActivityCode", act_codes, chunks=20000000)

        logging.info('ActivityCode set for %d activity executions' % len(act_codes))
