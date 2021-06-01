import numpy as np
import pandas as pd

import abminvisum.engines.choice_engine as choice_engine

daskfactory = choice_engine.DaskFactory(10000)


def prepare_choice_probs_for_act_id(visum, act_id, location_attribute='LocationNo'):
    filtered_act_loc_attr = np.array(visum.Net.ActivityLocations.FilteredBy(
        r'[Activity\ID] = ' + str(act_id) + ' & [Location\ZoneNo] > 0').GetMultipleAttributes(
        [r'Location\ZoneNo', location_attribute, 'AttractionPotential'], True))

    df = pd.DataFrame(data=filtered_act_loc_attr[:, :], columns=['ZoneNo', location_attribute, 'AttractionPotential'])
    df = df.astype({'AttractionPotential': 'float', 'ZoneNo': 'float'})
    df['AttractionTot'] = df.groupby('ZoneNo').AttractionPotential.transform('sum')
    df['Prob'] = df['AttractionPotential'] / df['AttractionTot']

    df_compl = df[['ZoneNo', location_attribute, 'Prob']].groupby('ZoneNo').agg(lambda x: list(x))

    zones = np.array(visum.Net.Zones.GetMultipleAttributes(['No'], True))
    df_zones = pd.DataFrame(data=zones, columns=['ZoneNo'])

    df_merged = df_zones.merge(df_compl, left_on="ZoneNo", right_index=True, how='left')
    max_length = 0
    for x in df_compl['Prob'].values:
        cur_length = len(x)
        if cur_length > max_length:
            max_length = cur_length

    choice_matrix = np.empty((zones.shape[0], max_length), dtype=object)
    prob_matrix = np.zeros((zones.shape[0], max_length))
    for i, x in enumerate(df_merged[[location_attribute, 'Prob']].values):
        if not isinstance(x[0], list):
            choice_matrix[i, 0] = "None"
            prob_matrix[i, 0] = 1
        else:
            choice_matrix[i, 0:len(x[0])] = x[0]
            prob_matrix[i, 0:len(x[1])] = x[1]

    return choice_matrix, prob_matrix


def choose_facilities(zone_indices, facility_probs, facility_options, logging):
    zone_ind = daskfactory.fromarray(zone_indices)
    probs = daskfactory.fromarray(facility_probs)
    all_prob_mat = probs[zone_ind]

    num_choices = all_prob_mat.shape[0]
    choice2d = choice_engine.Choice2DParallel(num_choices, 10000)
    choice2d.add_prob(all_prob_mat)
    choices_ind = choice2d.choose()

    return facility_options[zone_indices, choices_ind]
