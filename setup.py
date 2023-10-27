import torch
import os
import argparse

from dataset import dataset
from experiment import experiment
import sys

project_path = os.getcwd()
torch.multiprocessing.set_sharing_strategy('file_system')


def log_string(file_out, out_str, print_out=True):
    file_out.write(out_str+'\n')
    file_out.flush()
    if print_out:
        print(out_str)

if __name__ == "__main__":

    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1', 'True'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0', 'False'):
            return False
        else:
            raise argparse.ArgumentTypeError('Unsupported value encountered.')

    parser = argparse.ArgumentParser(description='Transfer Learning')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--dset_name', type=str, nargs='?', default='office-home', help="dataset name")
    parser.add_argument('--way_number', type=int, nargs='?', default="5", help="way")
    parser.add_argument('--shot_number', type=int, nargs='?', default="1", help="shot")
    parser.add_argument('--log_name', type=str, nargs='?', default='log', help="log name")
    parser.add_argument('--num_iter', type=int, nargs='?', default=15000, help="number of iteration")
    parser.add_argument('--w_repeat', type=int, nargs='?', default="10",help="MC_samples")
    parser.add_argument('--z_repeat', type=int, nargs='?', default="5", help="MC_samples")
    args = parser.parse_args()


    config = {}
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    # basic arguments
    config["gpu_id"] = args.gpu_id
    config["dset_name"] = args.dset_name
    config["way_number"] = args.way_number
    config["shot_number"] = args.shot_number
    config["log_name"] = args.log_name
    config["w_repeat"] = args.w_repeat
    config["z_repeat"] = args.z_repeat
    config["lr"] = {"init_lr": 0.0001, "gamma": 0.5, "stepsize": 3000}
    config["num_iter"] = args.num_iter
    config["test_interval"] = 10

    # -------------------------------
    # saving all configs
    os.system("mkdir -p " + config["log_name"])
    config["file_out"] = open(config["log_name"] + "/train_log.txt", "w")

    os.system("mkdir -p " + config["log_name"] + '/files')
    os.system('cp %s %s' % ('model_pool/*.py', os.path.join(config["log_name"], 'files')))

    # print
    print(str(config) + '\n')
    config["file_out"].write(str(config) + '\n')
    st = ' '
    log_string(config["file_out"], st.join(sys.argv))

    if config["dset_name"] == "office-home":
        config["basenet"] = 'vgg16'
        task_name_list = ["Art", "Clipart", "Product", "Real_World"]
        num_class = 65
        meta_train_test_split = torch.load('meta_train_test_split/' + "office-home" + '_metatrain=40_metatest=25.pt')
    elif config["dset_name"] == "domainnet":
        config["basenet"] = 'resnet18'
        task_name_list = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]
        num_class = 345
        meta_train_test_split = torch.load('meta_train_test_split/' + "domainnet" + '_metatrain=276_metatest=69.pt')

    config["num_task"] = len(task_name_list)
    config["num_class"] = num_class

    # -------------------------------
    print("start loading data")
    dsets = {"train": [], "test": []}
    dsets["train"] = dataset(config["dset_name"], meta_train_test_split, task_name_list, way_number=config["way_number"], shot_number=config["shot_number"], state="training")
    dsets["test"] = dataset(config["dset_name"], meta_train_test_split, task_name_list, way_number=config["way_number"], shot_number=config["shot_number"], state="test")

    batch_size = {"train": 1, "test": 1}  #meta_batch_size
    dset_loaders = {"train": [], "test": []}
    dset_loaders["train"] = torch.utils.data.DataLoader(dsets["train"], batch_size=batch_size["train"], shuffle=True,num_workers=4)
    dset_loaders["test"] = torch.utils.data.DataLoader(dsets["test"], batch_size=batch_size["test"], shuffle=False, num_workers=4)
    config["loaders"] = dset_loaders

    experiment(config)
    print(config["log_name"])
    config["file_out"].close()