import numpy as np
import os
from numpy import random
class dataset(object):
    def __init__(self, dataset, meta_train_test_split, task_name_list, way_number=5, shot_number=1, state="training"):

        self.dataset = dataset
        self.way_number = way_number
        self.shot_number = shot_number
        self.num_task = len(task_name_list)

        if state == "training":
            meta_classes = meta_train_test_split[0]
            self.meta_number = 30000
        elif state == "test":
            meta_classes = meta_train_test_split[1]
            self.meta_number = 600

        self.meta_classes = meta_classes

        all_dataset = os.getcwd()
        all_dataset = all_dataset.split("project")[0]+"dataset/"

        self.task_images_class_list = []
        for i in task_name_list:
            images_class_list = []
            for j in self.meta_classes:
                if self.dataset == "office-home":
                    task_class_path = os.path.join(all_dataset, 'office-home_feature_vgg16/', i, j)
                elif self.dataset == "domainnet":
                    task_class_path = os.path.join(all_dataset, 'domainnet_feature_resnet18/', i, j)
                images_list = os.listdir(task_class_path)

                image_one_class_list = []
                for image_local_path in images_list:
                    image_global_path = os.path.join(task_class_path, image_local_path)
                    image_one_class_list.append(image_global_path)

                images_class_list.append(image_one_class_list)
            self.task_images_class_list.append(images_class_list)   # domain_number, classes_number, sample_number(not fixed)

    def __getitem__(self, index):
            train_labels = list(range(len(self.meta_classes)))
            random.seed(index)
            random.shuffle(train_labels)
            current_task_classes = train_labels[:self.way_number]

            features_list = []
            labels_list = []
            for i in range(self.num_task):  # for examples, ["Art", "Clipart", "Product", "Real_World"]
                features = []
                labels = []
                for j in range(len(current_task_classes)):
                    train_label = current_task_classes[j]
                    features_path = self.task_images_class_list[i][train_label]
                    random.seed(index)
                    random.shuffle(features_path)

                    while len(features_path) < self.shot_number+15:
                        features_path = features_path + features_path
                    current_task_class_paths = features_path[:self.shot_number+15] #15 for query set

                    class_features = []
                    class_labels = []
                    for m in current_task_class_paths:
                        feature = np.load(m)
                        label = j
                        class_features.append(feature)
                        class_labels.append(label)
                    class_features = np.vstack(class_features)
                    class_labels = np.vstack(class_labels)

                    features.append(class_features)
                    labels.append(class_labels)

                features_list.append(features)
                labels_list.append(labels)
            return features_list, labels_list

    def __len__(self):
        return self.meta_number