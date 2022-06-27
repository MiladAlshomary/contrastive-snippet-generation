import logging
from typing import List, Dict

from shared.DataHandler import DataHandler
from FeaturedArgument import FeaturedArgument
from MMDBase import MMDBase
from shared.myutils import argmax


class Inference(MMDBase):
    def __init__(self, param_gamma, param_lambda, snippet_length, thetaA, thetaB):
        """
        Initializes MMD-based summarizer

        :param param_gamma: parameter for Gaussian Kernel
        :param param_lambda: parameter for objective functions
        :param snippet_length: number of sentences in a snippet
        :param thetaA:
        :param thetaB:
        """
        super().__init__(param_gamma, param_lambda)
        self.log = logging.getLogger(Inference.__name__)
        self.snippet_length = snippet_length
        self.thetaA = thetaA
        self.thetaB = thetaB

    def fit(self, arguments: List[FeaturedArgument]):
        """
        Performs inference for the given arguments.
        :param arguments: arguments for which snippets should be generated
        :return: objective values returned by the inference
        """
        context_keys = DataHandler.get_query_context_keys(arguments)
        objectives = dict()
        for context_key in context_keys:
            context_arguments = list(filter(lambda a: a.query == context_key, arguments))
            objs = self.fit_context(context_arguments)
            objectives.update(objs)
        return objectives

    def fit_context(self, arguments: List[FeaturedArgument]) -> Dict:
        """
        Performs inference for the given arguments of the same context.
        :param arguments: arguments for which snippets should be generated
        :return: objective values returned by the inference
        """
        objective_values = dict()
        for argument in arguments:
            context = [a for a in arguments if not a == argument]
            context_arg = FeaturedArgument.context_argument_dummy(
                context)  # merge context in data structure
            snippet_idxs, obj_val = self.greedy_opt(argument, context_arg)
            argument.excerpt_indices = snippet_idxs
            argument.excerpt = [argument.sentences[idx] for idx in snippet_idxs]
            objective_values[argument.arg_id] = obj_val
        return objective_values

    def predict_from_triplets(self, triplets: List[MMDBase.triplet]):
        """
        Performs inference for the given triplets
        :param triplets: triplets as those for training
        :return: objective values returned by the inference
        """
        objective_values = dict()
        for t in triplets:
            argument = t.V_A
            context_arg = t.V_B

            snippet_idxs, obj_val = self.greedy_opt(argument, context_arg)
            if type(snippet_idxs) == list:
                argument.excerpt_indices = snippet_idxs
                argument.excerpt = [argument.sentences[idx] for idx in snippet_idxs]
                objective_values[argument.arg_id] = obj_val
            else:
                self.log.error(
                    f'Inference failed. Greedy opt returns: {snippet_idxs} instead of indexes.')

        return objective_values

    def greedy_opt(self, argument: FeaturedArgument, context: FeaturedArgument):
        """
        Greedy optimization for sentences selection.

        :param argument: argument to summarize
        :param context: contextual argument
        :return: indices of sentences that are selected from argument
        """
        prototypes_idx = list()

        if argument.length <= self.snippet_length:
            # If an argument has less or equal many sentences than the desired snippet length,
            # use all of them.
            b = list(range(argument.length))
            self.log.warning(
                f'{argument.arg_id} has only {argument.length} many sentences :(\n Thus, '
                f'inference selects {b}.')
            return b

        for m in range(self.snippet_length):
            intermediate_results = list()
            for i in range(argument.length):
                prototypes_to_evaluate = prototypes_idx + [i]
                St = FeaturedArgument.only_snippet_argument_dummy(argument, prototypes_to_evaluate)
                objective_value = self.objective(argument, context, St, self.thetaA[0],
                                                 self.thetaB[0])
                intermediate_results.append(objective_value)

            largest_gain_idx = argmax(intermediate_results, prototypes_idx)
            self.log.debug(f'Lap {m}. Largest gain by adding sentence {largest_gain_idx}.')
            prototypes_idx.append(largest_gain_idx)

        # Recompute the objective
        snippet = FeaturedArgument.only_snippet_argument_dummy(argument, prototypes_idx)
        final_objective = self.objective(argument, context, snippet, self.thetaA[0], self.thetaB[0])

        return prototypes_idx, final_objective

    def objective(self, VAt, VBt, St, thetaA, thetaB):
        """
        Bista.2020 eq. 4.9
        """
        representativeness, contrastiveness = self.L_t_comp_fast(VAt, VBt, St, thetaA, thetaB)
        return (- representativeness) + self.param_lambda * contrastiveness
