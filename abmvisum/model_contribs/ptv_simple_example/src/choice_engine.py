import logging

import numpy as np
import dask.array as da

from src import abm_utilities, visum_utilities

class DaskFactory:
    def __init__(self, chunk_size):
        self.chunk_size = chunk_size

    def from_array(self, array_like):
        if array_like.ndim == 1:
            return da.from_array(array_like, chunks=(self.chunk_size,))
        else:
            return da.from_array(array_like, chunks=(self.chunk_size, -1))


class Choice2D:
    def __init__(self, num_objects):
        self.num_objects = num_objects
        self.prob_blocks = []

    def add_prob(self, prob):
        assert len(prob.shape) == 2
        assert prob.shape[0] == self.num_objects
        self.prob_blocks.append(prob)

    def choose(self, rand=None):
        if rand is None:
            rand = np.random.RandomState(42)

        prob = np.concatenate(self.prob_blocks)
        cum_prob = np.cumsum(prob, axis=1)
        draws = rand.rand(prob.shape[0])[:,np.newaxis]
        choices = (cum_prob > draws).argmax(axis=1)
        assert len(choices) == self.num_objects
        return choices


class Choice2DParallel:

    def __init__(self, num_objects, chunk_size=None):
        self.num_objects = num_objects
        self.chunk_size = chunk_size if chunk_size is not None else int(num_objects / 10)
        self.prob_blocks = []

    def add_prob(self, prob):
        self.prob_blocks.append(prob)

    def choose(self, rand=None):
        if rand is None:
            rand = da.random.RandomState(42)

        prob = da.concatenate(self.prob_blocks)
        cum_prob = da.cumsum(prob, axis=1)
        draws = rand.random(prob.shape[0], chunks=self.chunk_size)[:, np.newaxis]
        chosen = da.argmax(da.greater(cum_prob, draws), axis=1).compute() # the results need to be computed
        return chosen


def choose2D(prob):
    assert len(prob.shape) == 2
    choice2d = Choice2D(num_objects=prob.shape[0])
    choice2d.add_prob(prob)
    return choice2d.choose()


def choose2D_parallel(prob, chunk_size=None):
    assert len(prob.shape) == 2
    choice2d = Choice2DParallel(num_objects=prob.shape[0], chunk_size=chunk_size)
    choice2d.add_prob(prob)
    return choice2d.choose()


def calc_simple_choice_per_subject(filtered_subjects, attr_exprs, betas, choices):
    """
    Generic  execution of simple logit choice model
    """
    x = abm_utilities.eval_attrexpr(filtered_subjects, attr_exprs)
    utils = np.exp(np.dot(x, betas))
    probs = utils / utils.sum(1)[:, np.newaxis]
    probs[probs < 0] = 0

    choice2d = Choice2D(len(x))
    choice2d.add_prob(probs)
    chosen_indices = choice2d.choose()
    discrete_choice_values = np.array(choices)

    choice_per_subject = discrete_choice_values[chosen_indices]

    logging.info(f'executed {str(len(x))} choices')

    return choice_per_subject


def run_simple_choice(subjects, segment, result_attr):
    filtered_subjects = abm_utilities.get_filtered_subjects(subjects, segment['Filter'])
    if len(filtered_subjects) == 0:
        return

    choice_per_subject = calc_simple_choice_per_subject(filtered_subjects, segment['AttrExpr'], segment['Beta'], segment['Choices'])

    visum_utilities.SetMulti(filtered_subjects, result_attr, choice_per_subject)

    logging.info('%s set for %d objects', result_attr, len(choice_per_subject))
