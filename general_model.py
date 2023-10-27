import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np
from HNP import HNP


def log_string(file_out, out_str, print_out=True):
    file_out.write(out_str+'\n')
    file_out.flush()
    if print_out:
        print(out_str)

class general_model(object):
    def __init__(self, config):
        self.dataset = config["dset_name"]
        self.file_out = config["file_out"]
        self.optim_param = config["lr"]
        self.num_task = config["num_task"]
        self.way_number = config["way_number"]
        self.shot_number = config["shot_number"]
        self.w_repeat = config["w_repeat"]
        self.z_repeat = config["z_repeat"]
        self.label_repeat_times = self.w_repeat * self.z_repeat
        self.model = HNP(config).cuda()

        parameters = [{"params": self.model.parameters(), "lr": 1}]
        self.optimizer = optim.Adam(parameters, lr=1, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0005)
        self.criterion = nn.CrossEntropyLoss()

        self.iter_num = 1
        self.current_lr = 0.0
        self.beta = 0.0

        self.print_interval = 1
        self.train_cross_loss = 0.0
        self.train_kl_loss = 0.0
        self.train_kl_a_loss = 0.0
        self.train_kl_w_loss = 0.0
        self.train_total_loss = 0.0

    def one_train_iteration(self, inputs_batch, labels_batch):
        if self.optimizer.param_groups[0]["lr"] >= 0.000002:
            self.current_lr = self.optim_param["init_lr"] * (self.optim_param["gamma"] ** (self.iter_num // self.optim_param["stepsize"]))
        for component in self.optimizer.param_groups:
            component["lr"] = self.current_lr * 1.0

        before_iteration = time.time()

        self.model.train()
        output_posterior_all, output_prior_all, kl_a, kl_w = self.model(inputs_batch, labels_batch, self.z_repeat, self.w_repeat)

        # prepare  crepeat_label_target
        label_target = labels_batch[:, :, self.shot_number:, :].contiguous().view(self.num_task, -1)
        num_task, num_target= label_target.shape[0], label_target.shape[1]
        repeat_label_target = label_target.unsqueeze(1).expand(num_task, self.label_repeat_times, num_target).contiguous()

        # ce_loss
        mean_cls_loss_posterior = self.criterion(output_posterior_all.view(-1, self.way_number), repeat_label_target.view(-1))
        mean_cls_loss = mean_cls_loss_posterior

        _, output_predict_posterior = torch.max(output_posterior_all, -1)
        _, output_predict_prior = torch.max(output_prior_all, -1)

        task_right_number_posterior = torch.sum(torch.sum(output_predict_posterior.float() == repeat_label_target, -1), -1)
        task_accuracy_posterior = task_right_number_posterior / float(repeat_label_target.size()[1] * repeat_label_target.size()[2])

        task_right_number_prior = torch.sum(torch.sum(output_predict_prior.float() == repeat_label_target, -1), -1)
        task_accuracy_prior = task_right_number_prior / float(repeat_label_target.size()[1] * repeat_label_target.size()[2])

        # kl_loss
        mean_kl_a = kl_a.mean()
        mean_kl_w = kl_w.mean()
        mean_kl_loss = self.beta * (mean_kl_a + mean_kl_w)

        # all_loss
        mean_loss = mean_cls_loss + mean_kl_loss

        # back-propagation---------------------------------------
        self.optimizer.zero_grad()
        mean_loss.backward()
        self.optimizer.step()

        after_iteration = time.time()
        time_gap = after_iteration - before_iteration

        # recoding for print-------------------------------------
        self.train_cross_loss += mean_cls_loss.item()
        self.train_kl_loss += mean_kl_loss.item()
        self.train_kl_a_loss += mean_kl_a.item()
        self.train_kl_w_loss += mean_kl_w.item()
        self.train_total_loss += mean_loss.item()

        if self.iter_num % self.print_interval == 0:
            txt = "Iter {:05d}, lr:{:.6f}, Avg CE: {:.5f}; all_kl: {:.5f}; a_kl: {:.5f}; w_kl: {:.5f}; all_loss: {:.5f}"
            log_string(self.file_out,
                       txt.format(int(self.iter_num),
                                  self.current_lr,
                                  self.train_cross_loss / float(self.print_interval),
                                  self.train_kl_loss / float(self.print_interval),
                                  self.train_kl_a_loss / float(self.print_interval),
                                  self.train_kl_w_loss / float(self.print_interval),
                                  self.train_total_loss / float(self.print_interval)))
            self.file_out.flush()
            self.train_cross_loss = 0
            self.train_kl_loss = 0
            self.train_kl_a_loss = 0
            self.train_kl_w_loss = 0
            self.train_total_loss = 0

        if self.iter_num % (self.print_interval*5) == 0:
            log_string(self.file_out, "accuracy_posterior----{:.4f}, {:.4f}, {:.4f}, {:.4f}, average {:.4f}".format(
                task_accuracy_posterior[0],
                task_accuracy_posterior[1],
                task_accuracy_posterior[2],
                task_accuracy_posterior[3],
                task_accuracy_posterior.mean()))

            log_string(self.file_out, "accuracy_prior----{:.4f}, {:.4f}, {:.4f}, {:.4f}, average {:.4f}".format(
                task_accuracy_prior[0],
                task_accuracy_prior[1],
                task_accuracy_prior[2],
                task_accuracy_prior[3],
                task_accuracy_prior.mean()))

        # updating after finishing all tasks----------------------
        if self.iter_num % 10 == 0:
            self.beta += 1e-06
        self.iter_num += 1

        return time_gap

    def one_test_iteration(self, inputs_batch, labels_batch):

        self.model.eval()
        output = self.model(inputs_batch, labels_batch, self.z_repeat, self.w_repeat)
        _, output_predict = torch.max(output, -1)

        # prepare repeat_label_target
        label_target = labels_batch[:, :, self.shot_number:, :].contiguous().view(self.num_task, -1)
        num_task, num_target = label_target.shape[0], label_target.shape[1]
        repeat_label_target = label_target.unsqueeze(1).expand(num_task, self.label_repeat_times, num_target).contiguous()

        # computing task accuracy
        task_right_number = torch.sum(torch.sum(output_predict.float() == repeat_label_target, -1), -1)
        task_accuracy = task_right_number / float(repeat_label_target.size()[1] * repeat_label_target.size()[2])

        return np.array(task_accuracy.cpu())