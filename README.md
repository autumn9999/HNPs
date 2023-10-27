# Episodic Multi-Task Learning with Heterogeneous Neural Processes
Code for paper “Episodic Multi-Task Learning with Heterogeneous Neural Processes” accepted by NeurIPS2023 as spotlight.

## Set Up
### Prerequisites
 - Python 3.6.9
 - Pytorch 1.1.0
 - GPU: an NVIDIA Tesla V100

### Getting Started
Inside this repository, we mainly conduct comprehensive experiments on Office-Home. Download the dataset from the following link. To split documents are obtained by randomly selecting 5\%, 10\%, and 20\% of samples from each task as the training set and use the remaining samples as the test set. The split documents used for the office-home dataset comes from [MRN](https://github.com/thuml/MTlearn). There are three training list files called train_5.txt, train_10.txt and train_20.txt and three test list files called test_5.txt, test_10.txt, test_20.txt which are corresponding to the training and test files of 5% data, 10% data and 20% data. More splits like(15%, 25%, ...) used for the ablation study in our paper will be open in the public repository.
- Office-home; [[link]](https://www.hemanthdv.org/officeHomeDataset.html)

To extract the input features based on VGG16 by using the following command:
```
python feature_extractor/feature_vgg16.py #gpu_id #split
```

The label spaces for meta-train and meta-test are proposed in the the file "/meta_train_test_split".


## Experiments
### Training
To train the proposed multi-task neural processes by running the command:

```
python setup.py --way_number 5 --shot_number 1
```

```
python setup.py --way_number 5 --shot_number 5
```

```
python setup.py --way_number 20 --shot_number 1
```

```
python setup.py --way_number 20 --shot_number 5
```

If you need to change the number of MC sampling:

```
python setup.py --z_repeat #N_z  --w_repeat #N_w
```
