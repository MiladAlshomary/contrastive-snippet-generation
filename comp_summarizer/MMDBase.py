import logging
from collections import namedtuple

import torch
import torch.nn.functional as F

from FeaturedArgument import FeaturedArgument

# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE = torch.device('cpu')


class MMDBase:
    # V_A:  focal argument
    # V_B:  contextual argument
    # S:    snippet
    triplet = namedtuple('triplet', 'V_A V_B S')

    def __init__(self, param_gamma, param_lambda):
        self.param_gamma = param_gamma
        self.param_lambda = param_lambda
        self.kernel = lambda x, y, gamma: torch.exp(
            -gamma * torch.pow(torch.linalg.norm(torch.tensor(x) - torch.tensor(y)), 2))
        self.log = logging.getLogger(MMDBase.__name__)

    def cosine_kernel_matrix(self, a: torch.Tensor, b: torch.Tensor = None) -> torch.Tensor:
        """
        Computes pairwise cosine similarity kernel matrix of the given inputs. If b is not given
        similarity is computed between a and itself.

        https://stackoverflow.com/questions/48485373/pairwise-cosine-similarity-using-tensorflow
        :param a: input
        :param b: other input
        :return: similarity kernel matrix
        """
        normalized_a = F.normalize(a, p=2)
        if b is None:
            return torch.matmul(normalized_a, normalized_a.T)
        else:
            normalized_b = F.normalize(b, p=2)
            return torch.matmul(normalized_a, normalized_b.T)

    def L_t_comp_fast(self, VAt: FeaturedArgument, VBt: FeaturedArgument, St: FeaturedArgument,
                      thetaA, thetaB):
        """
        Computes loss for comparative summarization with sentence importance weights. (Bista.2020)
        :param VAt: focal argument that should be summarized
        :param VBt: context of the focal argument
        :param St: snippet
        :param thetaA: sentence importance function parameter for focal argument
        :param thetaB: sentence importance function parameter for contextual argument
        :return: representativeness term and contrastiveness term
        """
        f_theta_nt = F.softmax(torch.mv(VAt.surface_features.to(DEVICE), thetaA),
                               dim=0) * VAt.length
        f_theta_mt = F.softmax(torch.mv(VBt.surface_features.to(DEVICE), thetaB),
                               dim=0) * VBt.length
        f_theta_sA = F.softmax(torch.mv(St.surface_features.to(DEVICE), thetaA), dim=0) * St.length
        f_theta_sB = F.softmax(torch.mv(St.surface_features.to(DEVICE), thetaB), dim=0) * St.length

        VAt_dev = VAt.sentence_embeddings_tensor
        VAt_dev.to(device=DEVICE)
        VAt_kernel_matrix = torch.exp(
            self.param_gamma * self.cosine_kernel_matrix(VAt_dev).to(device=DEVICE))

        VBt_dev = VBt.sentence_embeddings_tensor
        VBt_dev.to(device=DEVICE)
        VBt_kernel_matrix = torch.exp(
            self.param_gamma * self.cosine_kernel_matrix(VBt_dev).to(device=DEVICE))

        St_dev = St.sentence_embeddings_tensor
        St_dev.to(device=DEVICE)
        St_kernel_matrix = torch.exp(
            self.param_gamma * self.cosine_kernel_matrix(St_dev).to(device=DEVICE))

        VAt_St_kernel_matrix = self.cosine_kernel_matrix(VAt_dev, St_dev).to(device=DEVICE)
        VBt_St_kernel_matrix = self.cosine_kernel_matrix(VBt_dev, St_dev).to(device=DEVICE)

        VAt_weighted_kernel = torch.dot(f_theta_nt,
                                        torch.mv(VAt_kernel_matrix.double(), f_theta_nt))
        VBt_weighted_kernel = torch.dot(f_theta_mt,
                                        torch.mv(VBt_kernel_matrix.double(), f_theta_mt))
        StA_weighted_kernel = torch.dot(f_theta_sA, torch.mv(St_kernel_matrix.double(), f_theta_sA))
        StB_weighted_kernel = torch.dot(f_theta_sB, torch.mv(St_kernel_matrix.double(), f_theta_sB))
        VAt_St_weighted_kernel = torch.dot(f_theta_nt,
                                           torch.mv(VAt_St_kernel_matrix.double(), f_theta_sA))
        VBt_St_weighted_kernel = torch.dot(f_theta_mt,
                                           torch.mv(VBt_St_kernel_matrix.double(), f_theta_sB))

        representativeness = VAt_weighted_kernel - 2. * VAt_St_weighted_kernel + StA_weighted_kernel
        contrastiveness = VBt_weighted_kernel - 2. * VBt_St_weighted_kernel + StB_weighted_kernel

        return torch.sqrt(representativeness), torch.sqrt(contrastiveness)

    def f_theta(self, theta, v):
        """

        :param theta: weight vector
        :param v: sentence feature values
        :return: raw importance value
        """
        return torch.inner(theta, v)

    def f_theta_xt(self, theta, v, Vt):
        """
        Computes values for f_theta^n_t and f_theta^m_t
        """
        enumerator = self.f_theta(theta, v)
        denominator = torch.sum(
            torch.tensor([self.f_theta(theta, v_prime) for v_prime in Vt.surface_features]), dim=0)
        return enumerator * torch.pow(denominator, -1) * Vt.length

    def L_t(self, Vt: FeaturedArgument, St: FeaturedArgument, theta):
        result = 0
        n = torch.tensor(Vt.length)
        m = torch.tensor(St.length)
        ONE = torch.tensor(1.)
        TWO = torch.tensor(2.)
        if n == 0 or m == 0:
            self.log.warning(f'Invalid number of elements: n = {n}, m = {m}')
            return 0

        for x_i in range(n):
            for x_j in range(n):
                result += (ONE / (n * n)) \
                          * self.f_theta_xt(theta, Vt.surface_features[x_i], Vt) \
                          * self.f_theta_xt(theta, Vt.surface_features[x_j], Vt) \
                          * self.kernel(Vt.sentence_embeddings[x_i], Vt.sentence_embeddings[x_j],
                                        self.param_gamma)

        for x_i in range(n):
            for y_j in range(m):
                result += - (TWO / (n * m)) \
                          * self.f_theta_xt(theta, Vt.surface_features[x_i], Vt) \
                          * self.f_theta_xt(theta, St.surface_features[y_j], St) \
                          * self.kernel(Vt.sentence_embeddings[x_i], St.sentence_embeddings[y_j],
                                        self.param_gamma)

        for y_i in range(m):
            for y_j in range(m):
                result += (ONE / (m * m)) \
                          * self.f_theta_xt(theta, St.surface_features[y_i], St) \
                          * self.f_theta_xt(theta, St.surface_features[y_j], St) \
                          * self.kernel(St.sentence_embeddings[y_i], St.sentence_embeddings[y_j],
                                        self.param_gamma)
        return result

    def L_t_comp(self, VAt: FeaturedArgument, VBt: FeaturedArgument, St: FeaturedArgument, thetaA,
                 thetaB):
        return self.L_t(VAt, St, thetaA) + self.param_lambda * self.L_t(VBt, St, thetaB)
