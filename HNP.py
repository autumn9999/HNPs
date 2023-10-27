import pdb

import torch
import torch.nn as nn

from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
import basic_model

class HNP(nn.Module):
    def __init__(self, config):
        super(HNP, self).__init__()

        self.dataset = config["dset_name"]
        self.num_task = config["num_task"]
        self.way_number = config["way_number"]
        self.shot_number = config["shot_number"]
        self.w_repeat = config["w_repeat"]
        self.z_repeat = config["z_repeat"]

        if self.dataset == "domainnet":
            self.x_feature = 512
        else:
            self.x_feature = 4096
        self.d_feature = self.x_feature
        config["d_feature"] = self.d_feature

        model = basic_model
        # task-wise and class-wise transformers
        self.task_probabilistic_encoder = model.ProbabilisticEncoder_theta(
            config, self.d_feature, self.num_task)
        self.class_probabilistic_encoder = model.ProbabilisticEncoder_phi(
            config, self.d_feature, self.way_number)

    @staticmethod
    def print_parameters(net):
        for name, parameters in net.named_parameters():
            print(name, ':', parameters.size())

    ######################################################################
    # Transformer inference for global latent representations
    ######################################################################
    def task_wise_transformer_inference(self, x_c_order, x_t_order):

        task_prior = x_c_order.view(self.num_task, -1, self.d_feature)
        task_posterior = x_t_order.view(self.num_task, -1, self.d_feature)

        # infer for prior
        z_pmu, z_psigma = self.task_probabilistic_encoder(task_prior)
        z_pdistirbution = Normal(z_pmu, z_psigma)
        z_psample = z_pdistirbution.rsample([self.a_z])

        # infer for posterior
        z_qmu, z_qsigma = self.task_probabilistic_encoder(task_posterior)
        z_qdistirbution = Normal(z_qmu, z_qsigma)
        z_qsample = z_qdistirbution.rsample([self.z_repeat])

        # kl_z
        kl_z = kl_divergence(z_qdistirbution, z_pdistirbution).sum(dim=1)
        return kl_z, z_psample, z_qsample


    ######################################################################
    # Transformer inference for local latent parameters
    ######################################################################
    def class_wise_transformer_inference(self, a_qsample, a_psample,
                                         x_c_order, x_t_order, x_t):

        kl_w = []
        output_prior_list = []
        output_posterior_list = []
        for num in range(self.num_task):

            if self.training:
                task_embedding = a_qsample[:, num, :]
            else:
                task_embedding = a_psample[:, num, :]

            class_prior = x_c_order
            class_posterior = x_t_order

            # infer for prior
            phi_pmu, phi_psigma = self.class_probabilistic_encoder(class_prior,
                                                                   task_embedding)
            phi_pdistirbution = Normal(phi_pmu, phi_psigma)

            # infer for posterior
            phi_qmu, phi_qsigma = self.class_probabilistic_encoder(class_posterior,
                                                                   task_embedding)
            phi_qdistirbution = Normal(phi_qmu, phi_qsigma)

            # task-specific kl_w
            task_specific_kl_w = kl_divergence(phi_qdistirbution,
                                               phi_pdistirbution).mean(dim=0).sum()
            kl_w.append(task_specific_kl_w.view(1))

            # perform prediction
            predict_samples = x_t[num].contiguous().view(-1, self.d_feature)
            repeat_predict_samples = predict_samples.unsqueeze(0)
            repeat_predict_samples = repeat_predict_samples.expand(
                self.w_repeat * self.z_repeat, predict_samples.shape[0],
                predict_samples.shape[1]).contiguous()

            if self.training:
                phi_qsample = phi_qdistirbution.rsample([self.w_repeat])

                phi_qsample = phi_qsample.transpose(0, 1)
                phi_qsample = phi_qsample.reshape(self.way_number,-1, self.d_feature)

                classifier_q = phi_qsample.transpose(0, 1).transpose(1, 2)

                phi_psample = phi_pdistirbution.rsample([self.w_repeat])

                phi_psample = phi_psample.transpose(0, 1)
                phi_psample = phi_psample.reshape(self.way_number,-1, self.d_feature)

                classifier_p = phi_psample.transpose(0, 1).transpose(1, 2)

            else:
                classifier_q = phi_qmu.unsqueeze(1).repeat(1, self.w_repeat, 1, 1)
                classifier_q = classifier_q.reshape(self.way_number, -1,
                                                    self.d_feature)
                classifier_q = classifier_q.transpose(0, 1).transpose(1, 2)

                classifier_p = phi_pmu.unsqueeze(1).repeat(1, self.w_repeat, 1, 1)
                classifier_p =classifier_p.reshape(self.way_number, -1,
                                                   self.d_feature)
                classifier_p = classifier_p.transpose(0, 1).transpose(1, 2)

            output_posterior = torch.bmm(repeat_predict_samples, classifier_q)
            output_posterior = output_posterior.unsqueeze(0)
            output_prior = torch.bmm(repeat_predict_samples, classifier_p)
            output_prior = output_prior.unsqueeze(0)
            output_posterior_list.append(output_posterior)
            output_prior_list.append(output_prior)

        kl_w = torch.cat(kl_w, 0)
        output_posterior_all = torch.cat(output_posterior_list, 0)
        output_prior_all = torch.cat(output_prior_list, 0)
        return kl_w, output_posterior_all, output_prior_all

    def forward(self, inputs_batch, labels_batch):
        '''
            inputs_batch 4 * 5 * 16 * 4096
            labels_batch 4 * 5 * 16 * 1
        '''

        # dividing the context and target samples
        label_all = labels_batch.squeeze(-1)
        _, indices = torch.sort(label_all[0, :, 0])

        x_c = inputs_batch[:, :, :self.shot_number, :]
        x_t = inputs_batch[:, :, self.shot_number:, :]

        x_c_order = x_c[:, indices, :, :]
        x_t_order = x_t[:, indices, :, :]

        # task-wise transformer
        kl_z, z_psample, z_qsample = self.task_wise_transformer_inference(
            x_c_order, x_t_order)
        kl_w, output_posterior_all, output_prior_all \
            = self.class_wise_transformer_inference(z_qsample, z_psample,
                                                    x_c_order, x_t_order, x_t)

        if self.training:
            return output_posterior_all, output_prior_all, kl_z, kl_w
        else:
            return output_prior_all


