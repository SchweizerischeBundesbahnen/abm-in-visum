import logging

import dask.array as da
import numpy as np

import abminvisum.tools.utilities as utilities


class DaskFactory:
    def __init__(self, chunksize):
        self.chunks = chunksize

    def fromarray(self, arraylike):
        if arraylike.ndim == 1:
            return da.from_array(arraylike, chunks=(self.chunks,))
        else:
            return da.from_array(arraylike, chunks=(self.chunks, -1))


class Choice2D:
    def __init__(self, num_objects):
        self.num_objects = num_objects
        self.prob_blocks = []

    def add_prob(self, prob):
        assert len(prob.shape) == 2
        assert prob.shape[0] == self.num_objects
        self.prob_blocks.append(prob)

    def choose(self, rand):
        prob = np.concatenate(self.prob_blocks)
        cum_prob = np.cumsum(prob, axis=1)
        draws = rand.rand(prob.shape[0])[:, np.newaxis]
        choices = (cum_prob > draws).argmax(axis=1)
        assert len(choices) == self.num_objects
        return choices


class Choice2DParallel:

    def __init__(self, N, chunks=None):
        self.N = N
        self.chunks = chunks if chunks is not None else int(N / 10)
        self.px = np.arange(N, dtype=int)
        self.px_list = []
        self.probblocks = []

    def add_prob(self, prob, filtered=None):
        self.probblocks.append(prob)
        self.filtered = (filtered is not None)
        if self.filtered:
            self.px_list.append(self.px[filtered])

    def choose(self):
        prob = da.concatenate(self.probblocks)
        cumprob = da.cumsum(prob, axis=1)
        draws = da.random.random(prob.shape[0], chunks=self.chunks)[:, np.newaxis]
        chosen = (cumprob > draws).argmax(axis=1).compute()
        if self.filtered:
            px_list = np.concatenate(self.px_list)
            px_inv = px_list.argsort()
            result = np.zeros(self.N, dtype=int) - 1
            result[px_list] = chosen[px_inv]
        else:
            result = chosen
        return result


def choose2D(prob, rand):
    assert len(prob.shape) == 2
    choice2d = Choice2D(num_objects=prob.shape[0])
    choice2d.add_prob(prob)
    return choice2d.choose(rand)


def calc_binary_probabilistic_choice_per_subject(filtered_subjects, attr_exprs, betas, choices, rand):
    x = utilities.eval_attrexpr(filtered_subjects, attr_exprs)
    utils = np.dot(x, betas)
    utils[utils < 0] = 0
    utils[utils > 1] = 1
    utils = np.array([1 - utils, utils])
    probs = utils.transpose()

    probs[probs < 0] = 0

    choice2d = Choice2D(len(x))
    choice2d.add_prob(probs)
    chosen_indices = choice2d.choose(rand)
    discrete_choice_values = np.array(choices)

    choice_per_subject = discrete_choice_values[chosen_indices]

    return choice_per_subject


'''
Generic  execution of simple probabilistic choice model
'''


def calc_probabilistic_choice_per_subject(filtered_subjects, attr_exprs, betas, choices, rand):
    x = utilities.eval_attrexpr(filtered_subjects, attr_exprs)
    utils = np.dot(x, betas)
    probs = utils / utils.sum(1)[:, np.newaxis]
    probs[probs < 0] = 0

    choice2d = Choice2D(len(x))
    choice2d.add_prob(probs)
    chosen_indices = choice2d.choose(rand)
    discrete_choice_values = np.array(choices)

    choice_per_subject = discrete_choice_values[chosen_indices]

    return choice_per_subject


'''
Generic  execution of simple logit choice model
'''


def calc_simple_choice_per_subject(filtered_subjects, attr_exprs, betas, choices, rand, shadow_util_dict=None,
                                   subject_attribute=None):
    x = utilities.eval_attrexpr(filtered_subjects, attr_exprs)
    utils = np.dot(x, betas)

    if shadow_util_dict is not None and subject_attribute is not None:
        residence_amr_id = np.array(
            filtered_subjects.GetMultipleAttributes([subject_attribute]))[:, 0]
        shadow_util_array_persons = np.array([shadow_util_dict[amr_id] for amr_id in residence_amr_id])
        exp_utils = np.exp(utils + shadow_util_array_persons)

    else:
        exp_utils = np.exp(utils)

    probs = exp_utils / exp_utils.sum(1)[:, np.newaxis]
    probs[probs < 0] = 0

    choice2d = Choice2D(len(x))
    choice2d.add_prob(probs)
    chosen_indices = choice2d.choose(rand)
    discrete_choice_values = np.array(choices)

    choice_per_subject = discrete_choice_values[chosen_indices]

    return choice_per_subject


def run_simple_choice(subjects, segment, rand, shadow_util_dict=None, subject_attribute=None):
    filtered_subjects = utilities.get_filtered_subjects(subjects, segment['Filter'])
    if len(filtered_subjects) == 0:
        return

    choice_per_subject = calc_simple_choice_per_subject(filtered_subjects, segment['AttrExpr'], segment['Beta'],
                                                        segment['Choices'], rand, shadow_util_dict, subject_attribute)

    result_attr = segment['ResAttr']
    filtered_subjects.SetAllAttValues(result_attr, 0)
    utilities.SetMulti(filtered_subjects, result_attr, choice_per_subject)

    if segment["MaxPlusOne"] > 0:
        # logging.info("Max plus one alternative enabled with a weight of " + str(segment["MaxPlusOne"]))
        max_choice = np.max(np.array(segment['Choices']))
        max_choice_subject = utilities.get_filtered_subjects(filtered_subjects,
                                                             "([" + result_attr + "]=" + str(max_choice) + ")")
        max_plus_one_choices = max_choice_subject.Count
        if max_plus_one_choices > 0:
            draws = rand.rand(max_plus_one_choices)
            choice_max_plus_one = np.where(draws <= segment["MaxPlusOne"], max_choice + 1, max_choice)
            utilities.SetMulti(max_choice_subject, result_attr, choice_max_plus_one)

    logging.info('%s set for %d objects' % (result_attr, len(choice_per_subject)))
