import torch
import numpy as np
import  scipy.stats

from general_model import general_model

def log_string(file_out, out_str, print_out=True):
    file_out.write(out_str+'\n')
    file_out.flush()
    if print_out:
        print(out_str)

def mean_confidence_interval(accs, confidence=0.95):
    n = accs.shape[0]
    m, se = np.mean(accs, axis=0), scipy.stats.sem(accs, axis=0)
    h = se * scipy.stats.t._ppf((1 + confidence) / 2, n - 1)

    mean_accs = np.mean(accs, axis=1)
    mean, mean_se = np.mean(mean_accs), scipy.stats.sem(mean_accs, axis=0)
    mean_h = mean_se * scipy.stats.t._ppf((1 + confidence) / 2, n - 1)
    return m, h, mean, mean_h

def experiment(config):
    model = general_model(config)
    dset_loaders = config["loaders"]
    num_iter = config["num_iter"]
    test_interval = config["test_interval"]
    file_out = config["file_out"]

    num_task = config["num_task"]

    best_list = []
    best_acc = 0.0
    best_iter_num = 0

    train_loader = iter(dset_loaders["train"]) # many iterations in one epoch
    # if num_iter < len(train_loader):
    #     num_iter = len(train_loader)

    # True for ablation study
    print_training_inference_time = False
    time_gap_list = []

    for iter_num in range(1, num_iter + 1):
        # ---------------------------------------------
        # training processes
        data_list = train_loader.next()

        inputs_all = []
        labels_all = []
        for i in range(num_task):
            inputs = torch.cat(data_list[0][i], 0)  # way_number * (shot_number+15) * 4096
            labels = torch.cat(data_list[1][i], 0)  # way_number * (shot_number+15) * 1
            inputs = inputs.cuda()
            labels = labels.cuda()
            inputs_all.append(inputs.unsqueeze(0))
            labels_all.append(labels.unsqueeze(0))

        inputs_all = torch.cat(inputs_all, 0)  # num_task * way_number * (shot_number+5) * 4096
        labels_all = torch.cat(labels_all, 0)  # num_task * way_number * (shot_number+5) * 1
        time_gap = model.one_train_iteration(inputs_all, labels_all)

        if print_training_inference_time:
            time_gap_list.append(time_gap)

        # ---------------------------------------------
        # test process for each test_interval
        if num_iter > 100:
            if iter_num % test_interval == 0:
                accuracy_list = test(config, model)
                epoch_acc_list, epoch_acc_list_interval, average_acc, average_interval = mean_confidence_interval(accuracy_list)

                for i in range(num_task):
                    log_string(file_out, 'Iter {:05d} Acc on Task {:d}: {:.4f}, {:.4f}'.format(iter_num, i, epoch_acc_list[i], epoch_acc_list_interval[i]))

                # comparing the best performance
                if average_acc > best_acc:
                    best_acc = average_acc
                    best_acc_interval = average_interval
                    best_list = epoch_acc_list
                    best_list_interval = epoch_acc_list_interval
                    best_iter_num = iter_num

                    # saving models
                    save_dict = {}
                    save_dict["model"] = model.model.state_dict()
                    save_dict["iter_num"] = best_iter_num
                    torch.save(save_dict, config["log_name"] +"/best_model.pth.tar".format(iter_num))

                # print best valuation
                if config["dset_name"] == "domainnet":
                    log_string(file_out, 'Now- val Acc on Iter {:05d}: {:.4f}, {:.4f}'.format(iter_num, average_acc, average_interval))
                    log_string(file_out, 'Best val Acc on Iter {:05d}: {:.4f}, {:.4f}'.format(best_iter_num, best_acc, best_acc_interval))
                    log_string(file_out, 'Iter {:05d}: {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}'.format(best_iter_num,
                                                                                               best_list[0],
                                                                                               best_list[1],
                                                                                               best_list[2],
                                                                                               best_list[3],
                                                                                               best_list[4],
                                                                                               best_list[5]))
                    log_string(file_out, 'Iter {:05d}: {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}'.format(best_iter_num,
                                                                                               best_list_interval[0],
                                                                                               best_list_interval[1],
                                                                                               best_list_interval[2],
                                                                                               best_list_interval[3],
                                                                                               best_list_interval[4],
                                                                                               best_list_interval[5]))
                else:
                    log_string(file_out, 'Now- val Acc on Iter {:05d}: {:.4f}'.format(iter_num, np.mean(epoch_acc_list)))
                    log_string(file_out, 'Best val Acc on Iter {:05d}: {:.4f}, {:.4f}'.format(best_iter_num, best_acc, best_acc_interval))
                    log_string(file_out, 'Iter {:05d}: {:.4f}, {:.4f}, {:.4f}, {:.4f}'.format(best_iter_num,
                                                                                              best_list[0],
                                                                                              best_list[1],
                                                                                              best_list[2],
                                                                                              best_list[3]))
                    log_string(file_out, 'Iter {:05d}: {:.4f}, {:.4f}, {:.4f}, {:.4f}'.format(best_iter_num,
                                                                                               best_list_interval[0],
                                                                                               best_list_interval[1],
                                                                                               best_list_interval[2],
                                                                                               best_list_interval[3]))

def test(config, model):
    dset_loaders = config["loaders"]
    loader = dset_loaders["test"]
    num_task = config["num_task"]

    accuracy_list = []
    iter_val = iter(loader)   # many iterations in one epoch

    for i in range(len(loader)):
        data_list = iter_val.next()

        inputs_all = []
        labels_all = []
        for i in range(num_task):
            inputs = torch.cat(data_list[0][i], 0)  # way_number * (shot_number+15) * 4096
            labels = torch.cat(data_list[1][i], 0)  # way_number * (shot_number+15) * 1
            inputs = inputs.cuda()
            labels = labels.cuda()
            inputs_all.append(inputs.unsqueeze(0))
            labels_all.append(labels.unsqueeze(0))
        inputs_all = torch.cat(inputs_all, 0)  # num_task * way_number * (shot_number+5) * 4096
        labels_all = torch.cat(labels_all, 0)  # num_task * way_number * (shot_number+5) * 1
        task_accuracy_list = model.one_test_iteration(inputs_all, labels_all)
        accuracy_list.append(task_accuracy_list)  # size = 600 * task_num * (update_steps)
    return np.array(accuracy_list)