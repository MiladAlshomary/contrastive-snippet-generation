import logging
import os
import pickle
from typing import List

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.nn import Parameter
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

from FeaturedArgument import FeaturedArgument
from MMDBase import MMDBase, DEVICE


class Trainer(MMDBase):

    def __init__(self, param_gamma, param_lambda, param_beta, model_size=6, epochs=20,
                 rollback=False):
        """

        :param param_gamma: weight for exponential kernel
        :param param_lambda: weight for the comparative part of the loss
        :param param_beta: weight for the regularization
        :param epochs: number of epochs
        :param rollback: if true a copy of the weights is maintained and if some step causes the
        weighs to be nan,
        they can be restored and optimization can continue with the next epoch
        """
        super().__init__(param_gamma, param_lambda)
        self.log = logging.getLogger(Trainer.__name__)
        self.param_beta = param_beta
        self.epochs = epochs

        torch.manual_seed(42)
        self.model = [Parameter(
            torch.randn(model_size, requires_grad=True, dtype=torch.float64, device=DEVICE))]
        self.rollback = rollback
        self.rollback_model = [p.clone().detach() for p in self.model]
        self.optimizer = Adam(self.model, 0.1)
        self.scheduler = ExponentialLR(self.optimizer, gamma=0.9)

        self.log.debug(
            f'Initialized: gamma={param_gamma}, lambda={param_lambda}, beta={param_beta}, '
            f'epochs={epochs}, rollback={rollback}, DEVICE={DEVICE}')

    def reg(self):
        _reg = 0
        for param in self.model[0]:
            _reg += (param ** 2).sum()
        return _reg

    def loss(self, VAt: FeaturedArgument, VBt: FeaturedArgument, St: FeaturedArgument, thetaA,
             thetaB):
        """
        Bista.2020 eq. 4.6
        """
        representativeness, contrastiveness = self.L_t_comp_fast(VAt, VBt, St, thetaA, thetaB)
        return representativeness - self.param_lambda * contrastiveness

    def train(self, training_data: List[MMDBase.triplet], plot_losses: bool = False, context='',
              cp_path=None):
        """
        Trains the model with the given data.
        :param training_data: data to train on
        :param plot_losses: whether losses should be plotted
        :param context: is shown in logs and on plot
        :return: model, losses
        """
        self.log.debug(f'initial parameters {self.model}')

        losses = list()
        for epoch in range(self.epochs):
            loss_accum = torch.DoubleTensor([0.0]).to(device=DEVICE)
            for t in training_data:
                loss = self.loss(t.V_A, t.V_B, t.S, self.model[0], self.model[0])
                loss_accum += loss
            loss_accum /= len(training_data)
            loss_accum += self.param_beta * self.reg()
            losses.append(float(loss_accum))
            self.optimizer.zero_grad()
            self.log.debug(f'epoch: {epoch}, loss: {loss_accum[0]}, lr: .1, '
                           f'parameter: {self.model}')
            loss_accum.backward()
            self.optimizer.step()
            # self.scheduler.step()

            np_model_params = self.model[0].cpu().detach().numpy()
            if np.isnan(np_model_params).any():
                context = set([t.V_A.query for t in training_data])
                self.log.error(f'Any of the parameters is nan: {self.model}, context: {context}')
                if self.rollback:
                    self.model = [p.clone().detach() for p in self.rollback_model]
                    self.log.info(f'Model was rolled back in epoch {epoch} of {self.epochs}.')
                else:
                    self.log.error(f'Excluded context due to causing nans: "{context}"')
                    break
            else:
                self.rollback_model = [p.clone().detach() for p in self.model]

            if cp_path is not None and epoch % 5 == 0:
                self.create_checkpoint(filename=f'{cp_path}-epoch-{epoch}.model')

        self.log.debug(f'Optimization yields {self.model}, losses: {losses}, context: {context}')
        if plot_losses:
            plt.plot(losses)
            plt.title(
                f'lambda={self.param_lambda}, beta={self.param_beta}, gamma={self.param_gamma}\n'
                f'{context}')
            plt.show()
        return np_model_params, losses

    def validate(self, val_data: List[MMDBase.triplet]):
        loss_accum = torch.DoubleTensor([0.0]).to(device=DEVICE)
        for t in val_data:
            loss = self.loss(t.V_A, t.V_B, t.S, self.model[0], self.model[0])
            loss_accum += loss
        loss_accum /= len(val_data)
        loss_accum += self.param_beta * self.reg()

        return float(loss_accum)

    def create_checkpoint(self, filename='mdd-trainer.model'):
        self.log.debug(f'Writing checkpoint to {filename}')
        cp = {
            'param_beta': self.param_beta,
            'param_gamma': self.param_gamma,
            'param_lambda': self.param_lambda,
            'epochs': self.epochs,
            'model': self.model,
            'rollback': self.rollback,
            'rollback_model': self.rollback_model,
        }
        dir = '.checkpoint'
        if not os.path.isdir(dir):
            os.mkdir(dir)
        with open(f'{dir}/{filename}', 'wb') as file:
            pickle.dump(cp, file)

        return f'{dir}/{filename}'

    @staticmethod
    def from_checkpoint(path):
        with open(path, 'rb') as file:
            cp = pickle.load(file)

        trainer = Trainer(
            param_gamma=cp['param_gamma'],
            param_lambda=cp['param_lambda'],
            param_beta=cp['param_beta'],
            rollback=cp['rollback'],
            model_size=6,
            epochs=cp['epochs'],
        )
        trainer.model = [Parameter(cp['model'][0])]
        trainer.rollback_model = [Parameter(cp['rollback_model'][0])]

        return trainer
