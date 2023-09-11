import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import configargparse
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Sampler
from PIL import Image
import torch.nn.init as init
import time
import pickle
import numpy as np
import copy
from torch.utils.tensorboard import SummaryWriter
from sklearn import metrics
import os
from pathlib import Path
import pandas
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from enum import Enum


def build_configargparser(parser):
    parser.add_argument('--run_name', type=str, help='Run name to distinguish configs')
    parser.add_argument('--gpu_usg', default=0, type=int, help='gpu use, default False')
    parser.add_argument('--sequence_length', default=1, type=int, help='sequence length, default 10')
    parser.add_argument('--train_batch_size', default=100, type=int, help='train batch size, default 400')
    parser.add_argument('--val_batch_size', default=100, type=int, help='valid batch size, default 10')
    parser.add_argument('--optimizer_choice', default=0, type=str, help='0 for sgd 1 for adam, default 1')
    parser.add_argument('--multi_optim', default=1, type=int, help='0 for single opt, 1 for multi opt, default 1')
    parser.add_argument('--epochs', default=20, type=int, help='epochs to train and val, default 25')
    parser.add_argument('--workers', default=8, type=int, help='num of workers to use, default 4')
    parser.add_argument('--use_flip', default=1, type=int, help='0 for not flip, 1 for flip, default 0')
    parser.add_argument('--crop_type', default=1, type=int, help='0 rand, 1 cent, 5 five_crop, 10 ten_crop, default 1')
    parser.add_argument('--learning_rate', default=5e-4, type=float, help='learning rate for optimizer, default 5e-5')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum for sgd, default 0.9')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay for sgd, default 0')
    parser.add_argument('--dampening', default=0, type=float, help='dampening for sgd, default 0')
    parser.add_argument('--use_nesterov', default=1, type=int, help='nesterov momentum, default True')
    parser.add_argument('--sgd_adjust_lr', default=1, type=int, help='sgd method adjust lr 0 for step 1 for min, default 1')
    parser.add_argument('--sgd_step', default=5, type=int, help='number of steps to adjust lr for sgd, default 5')
    parser.add_argument('--sgd_gamma', default=0.1, type=float, help='gamma of steps to adjust lr for sgd, default 0.1')
    parser.add_argument('--test_dataset', default=0, type=int, help='presence of a test dataset')
    parser.add_argument('--path_for_saving', type=str, help='path name')
    parser.add_argument('--datafile', type=str, help='data filename')
    parser.add_argument('--adjusted_weights', default=0, type=int, help='adjusted weights flag')
    parser.add_argument('--inv_prop_weights', default=0, type=int, help='inversely proportional weights flag')
    parser.add_argument('--data_transformation_option_1', default=0, type=int, help='data transformation option 1')
    parser.add_argument('--data_transformation_option_2', default=0, type=int, help='data transformation option 2')
    parser.add_argument('--patience', default=50, type=int, help='maximum number of epochs for early stopping')
    parser.add_argument('--model', default='ResNet', type=str, help='model type')
    parser.add_argument('--resnet_weights', type=str, help='path for the resnet weights')
    parser.add_argument('--input_size', default=2048, type=int, help='input size')
    parser.add_argument('--hidden_size', default=256, type=int, help='hidden size')
    parser.add_argument('--num_classes', default=5, type=int, help='Number of classes')

    known_args, _ = parser.parse_known_args()
    return parser, known_args


class Implementation(Enum):
    ResNet = 'ResNet'
    StatefulLSTM = 'StatefulLSTM'


class DatasetObject(Dataset):
    def __init__(self, file_paths, file_labels, video_ids, transform_list=None, ):
        self.file_paths_dict = dict([(i, path) for i, path in enumerate(file_paths)])
        self.file_labels_dict = dict([(i, label) for i, label in enumerate(file_labels)])
        self.video_ids_dict = dict([(i, label) for i, label in enumerate(video_ids)])
        self.transform_dict = dict([(i, transform) for i, transform in enumerate(transform_list)])
        self.loader = self.pil_loader

    def pil_loader(self, path):
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')

    def __getitem__(self, index):
        img_names = self.file_paths_dict[index]
        labels_phase = self.file_labels_dict[index]
        video_id = self.video_ids_dict[index]
        imgs = self.loader(img_names)
        transform = self.transform_dict[index]
        if transform is not None:
            imgs = transform(imgs)

        return imgs, labels_phase, video_id

    def __len__(self):
        return len(self.file_paths_dict)


class resnetBaseline(torch.nn.Module):
    def __init__(self):
        super(resnetBaseline, self).__init__()
        resnet = models.resnet50(pretrained=True)
        self.share = torch.nn.Sequential()
        self.share.add_module("conv1", resnet.conv1)
        self.share.add_module("bn1", resnet.bn1)
        self.share.add_module("relu", resnet.relu)
        self.share.add_module("maxpool", resnet.maxpool)
        self.share.add_module("layer1", resnet.layer1)
        self.share.add_module("layer2", resnet.layer2)
        self.share.add_module("layer3", resnet.layer3)
        self.share.add_module("layer4", resnet.layer4)
        self.share.add_module("avgpool", resnet.avgpool)
        self.fc = nn.Sequential(nn.Linear(2048, 512),
                                nn.ReLU(),
                                nn.Dropout(0.5),
                                nn.Linear(512, 5))

    def forward(self, x, video_ids=None):
        x = self.share.forward(x)
        x = x.view(-1, 2048)
        y = self.fc(x)
        return y


class StatefulLSTM(torch.nn.Module):
    # def __init__(self, resnet_weights, input_size, hidden_size, num_classes):
    #     super(StatefulLSTM, self).__init__()

    def __init__(self, resnet_weights, input_size, hidden_size, num_classes):
        super(StatefulLSTM, self).__init__()
        # resnet_weights = '/Users/dorotheeduvaux 1/UCL CSML/MSc Project/Results/best_model_weights.pth'
        # input_size = 2048  # should correspond to x.view(x.size(0), -1)
        # # A higher hidden size should give capacity to learn more from the features
        # hidden_size = 256  # [128, 256 or 512]
        # num_classes = 5

        self.previous_video_id = 1
        # Load pre-trained resNet model without the final classification layer
        self.resnet = resnetBaseline()
        self.resnet.load_state_dict(torch.load(resnet_weights))
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        # Freeze the weights of that resnet model
        for param in self.resnet.parameters():
            param.requires_grad = False

        # Initialize LSTM layers
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

        # Hidden and cell states for stateful LSTM
        self.hidden_state = None
        self.cell_state = None

        # Apply Xavier initialization
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                init.xavier_normal_(param.data)
            elif 'weight_hh' in name:
                init.xavier_normal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)

        # Initialize fully connected layer using Xavier Uniform
        init.xavier_uniform_(self.fc.weight)
        init.zeros_(self.fc.bias)

    def reset_states(self):
        self.hidden_state = None
        self.cell_state = None

    def forward(self, x, video_ids):
        # Pass input frames through ResNet
        x = self.resnet(x)
        x = x.view(x.size(0), -1)  # shape is now [batch size = 20, 2048]

        # Reshape to treat the 20 images as a sequence
        x = x.unsqueeze(0)  # shape becomes [batch size = 1, sequence length = 20, 2048]

        # Need to reset the hidden and cell states if the sequence is for a new video.
        video_id = torch.unique(video_ids, dim=0)
        if len(video_id) > 1:
            raise ValueError("Something has gone wrong in the data loader")
        if video_id != self.previous_video_id:
            self.reset_states()
            self.previous_video_id = video_id

        # Pass the ResNet features through LSTM
        if self.hidden_state is None or self.cell_state is None:
            out, (self.hidden_state, self.cell_state) = self.lstm(x)
        else:
            # Pass the hidden and cell states from previous sequence to the next sequence.
            out, (self.hidden_state, self.cell_state) = self.lstm(x, (self.hidden_state.detach(),
                                                                      self.cell_state.detach()))

        # Pass the LSTM outputs through the classification layer
        out = self.fc(out.squeeze(0))  # Assuming out.shape is [1, 20, hidden_size], this makes it [20, hidden_size]

        return out


def get_indices(sequence_length, list_of_lengths):
    count = 0
    idx = []
    for i, length in enumerate(list_of_lengths):
        for j in range(count, count + (length + 1 - sequence_length)):
            idx.append(j)
        count += length
    return idx


def get_data(data_path, hparams):
    with open(data_path, 'rb') as f:
        train_test_paths_labels = pickle.load(f)

    train_paths = train_test_paths_labels[0]
    val_paths = train_test_paths_labels[1]

    train_labels = train_test_paths_labels[2]
    val_labels = train_test_paths_labels[3]

    train_num_each = train_test_paths_labels[4]
    val_num_each = train_test_paths_labels[5]

    test_paths = train_test_paths_labels[6]
    test_labels = train_test_paths_labels[7]
    test_num_each = train_test_paths_labels[8]


    try:
        train_ear_type = train_test_paths_labels[9]
        val_ear_type = train_test_paths_labels[10]
        test_ear_type = train_test_paths_labels[11]

        # train_video_ids = train_test_paths_labels[12]
        # val_video_ids = train_test_paths_labels[13]
        # test_video_ids = train_test_paths_labels[14]
    except Exception as e:
        print(e)
        train_ear_type = [None] * len(train_labels)
        val_ear_type = [None] * len(val_labels)
        test_ear_type = [None] * len(test_labels)

    print('train_paths  : {:6d}'.format(len(train_paths)))
    print('train_labels : {:6d}'.format(len(train_labels)))
    print('valid_paths  : {:6d}'.format(len(val_paths)))
    print('valid_labels : {:6d}'.format(len(val_labels)))
    print('test_paths  : {:6d}'.format(len(test_paths)))
    print('test_labels : {:6d}'.format(len(test_labels)))

    train_labels = np.asarray(train_labels, dtype=np.int64)
    val_labels = np.asarray(val_labels, dtype=np.int64)
    test_labels = np.asarray(test_labels, dtype=np.int64)

    train_video_ids = [i + 1 for i, count in enumerate(train_num_each) for _ in range(count)]
    val_video_ids = [i + 1 for i, count in enumerate(val_num_each) for _ in range(count)]
    test_video_ids = [i + 1 for i, count in enumerate(test_num_each) for _ in range(count)]

    print(set(train_labels))
    print(set(val_labels))
    print(set(test_labels))

    train_transforms_list = []
    test_transforms_list = []
    val_transforms_list = []

    if hparams.data_transformation_option_1:
        resizeCrop = 645
    elif hparams.data_transformation_option_2:
        resizeCrop = 224
    else:
        raise ValueError('Transformation not recognised')
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]
    print('resizing crop dimension: ', resizeCrop)

    train_transforms = [
        transforms.RandomResizedCrop(resizeCrop, scale=(0.8, 1.2)),
        # zooms image in our out by 20% and then crops to 224 or 645
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ]
    test_transforms = [
        transforms.CenterCrop(resizeCrop),
        # zooms image in our out by 20% and then crops to 224 or 645
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ]
    val_transforms = test_transforms

    for ear_type in train_ear_type:
        ttf = train_transforms
        if ear_type == 'L':
            ttf = train_transforms + [transforms.RandomHorizontalFlip(p=1.0)]
        composed_train_transforms = transforms.Compose(ttf)
        train_transforms_list.append(composed_train_transforms)
    for ear_type in test_ear_type:
        tetf = test_transforms
        if ear_type == 'L':
            tetf = test_transforms + [transforms.RandomHorizontalFlip(p=1.0)]
        composed_test_transforms = transforms.Compose(tetf)
        test_transforms_list.append(composed_test_transforms)
    for ear_type in val_ear_type:
        vtf = val_transforms
        if ear_type == 'L':
            vtf = val_transforms + [transforms.RandomHorizontalFlip(p=1.0)]
        composed_val_transforms = transforms.Compose(vtf)
        val_transforms_list.append(composed_val_transforms)

    print(set(train_labels))
    print(set(val_labels))
    print(set(test_labels))

    train_dataset = DatasetObject(train_paths, train_labels, train_video_ids, train_transforms_list)
    val_dataset = DatasetObject(val_paths, val_labels, val_video_ids, val_transforms_list)
    test_dataset = DatasetObject(test_paths, test_labels, test_video_ids, test_transforms_list)

    return train_dataset, train_num_each, val_dataset, val_num_each, test_dataset, test_num_each


class SeqSampler(Sampler):
    def __init__(self, data_source, idx):
        super().__init__(data_source)
        self.data_source = data_source
        self.idx = idx

    def __iter__(self):
        return iter(self.idx)

    def __len__(self):
        return len(self.idx)


class VideoBatchSampler(Sampler):
    def __init__(self, video_lengths, batch_size):
        self.video_lengths = video_lengths
        self.batch_size = batch_size
        self.video_starts = [0] + list(np.cumsum(video_lengths[:-1]))

    def __iter__(self):
        all_indices = []
        for start, length in zip(self.video_starts, self.video_lengths):
            indices = list(range(start, start + length))
            # split this video's indices into batches
            for i in range(0, len(indices), self.batch_size):
                all_indices.append(indices[i:i+self.batch_size])
        return iter(all_indices)

    def __len__(self):
        return sum((l + self.batch_size - 1) // self.batch_size for l in self.video_lengths)


def evaluate_model(model, data_loader, batch_size, indices, device, sequence_length):
    # Sets the module in evaluation mode.
    model.eval()
    total_loss = 0.0
    correct_steps = 0
    total_samples = 0
    progress = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data in data_loader:
            inputs, labels, video_ids = data
            inputs, labels, video_ids = inputs.to(device), labels.to(device), video_ids.to(device)

            labels = labels[(sequence_length - 1)::sequence_length]
            outputs = model.forward(inputs, video_ids)
            outputs = outputs[sequence_length - 1::sequence_length]

            _, predictions = torch.max(outputs.data, 1)
            # the cross entropy loss calculates by default the average loss over the batch
            loss_average = nn.CrossEntropyLoss()(outputs, labels)

            total_loss += loss_average.item() * inputs.size(0)
            correct_steps += torch.eq(predictions, labels.data).sum()
            total_samples += inputs.size(0)

            all_preds.extend(predictions.detach().cpu().tolist())
            all_labels.extend(labels.detach().cpu().tolist())

            progress += 1
            if progress * batch_size >= len(indices):
                percent = 100.0
                print('Val progress: %s [%d/%d]' % (str(percent) + '%', len(indices), len(indices)),
                      end='\n')
            else:
                percent = round(progress * batch_size / len(indices) * 100, 2)
                print('Val progress: %s [%d/%d]' % (str(percent) + '%', progress * batch_size,
                                                    len(indices)), end='\r')
    model.train()
    return total_loss, correct_steps, all_preds, all_labels, total_samples


def train_model(hparams, train_dataset, train_num_each, val_dataset, val_num_each):

    device = torch.device("cuda" if (torch.cuda.is_available() and hparams.gpu_usg) else "cpu")
    sequence_length = hparams.sequence_length

    # TensorBoard
    writer = SummaryWriter('../runs/baseline/{0}'.format(hparams.run_name))

    (train_dataset), (train_num_each), (val_dataset, test_dataset), (val_num_each, test_num_each) = \
        train_dataset, train_num_each, val_dataset, val_num_each

    # List the indices used from each dataset
    train_indices = get_indices(sequence_length, train_num_each)
    val_indices = get_indices(sequence_length, val_num_each)
    test_indices = get_indices(sequence_length, test_num_each)

    # if sequence_length != 1, then sequences of indices are being added
    # i.e. clips of consecutive images of length equal to the sequence_length
    # train_idx is deduced as part of each separate epoch.
    val_idx = [val_index + j for val_index in val_indices for j in range(sequence_length)]
    test_idx = [test_index + j for test_index in test_indices for j in range(sequence_length)]

    print('num of all train use:: {:6d}'.format(len(train_indices)))
    print('num of all valid use: {:6d}'.format(len(val_indices)))
    print('num of all test use: {:6d}'.format(len(test_indices)))

    # Possible implementations
    if hparams.model == Implementation.ResNet.value:
        # Note that the training loader is defined at every epoch
        val_sampler = SeqSampler(val_dataset, val_idx)
        val_loader = DataLoader(val_dataset, batch_size=hparams.val_batch_size, sampler=val_sampler,
                                num_workers=hparams.workers, pin_memory=False)

        test_sampler = SeqSampler(test_dataset, test_idx)
        test_loader = DataLoader(test_dataset, batch_size=hparams.val_batch_size, sampler=test_sampler,
                                 num_workers=hparams.workers, pin_memory=False)

        model = resnetBaseline()

    elif hparams.model == Implementation.StatefulLSTM.value:
        # These loaders are for the LSTM type of runs and are required to segment the data according to batch size
        # but also according to when a new video starts, since the hidden and cell states will need resetting.
        # Also training loader added here since no shuffling is possible and so it can be instantiated only once.
        batch_sampler = VideoBatchSampler(train_num_each, hparams.train_batch_size)
        train_loader = DataLoader(train_dataset, batch_sampler=batch_sampler,
                                  num_workers=hparams.workers, pin_memory=False)

        batch_sampler = VideoBatchSampler(val_num_each, hparams.val_batch_size)
        val_loader = DataLoader(val_dataset, batch_sampler=batch_sampler,
                                num_workers=hparams.workers, pin_memory=False)

        batch_sampler = VideoBatchSampler(test_num_each, hparams.val_batch_size)
        test_loader = DataLoader(test_dataset, batch_sampler=batch_sampler,
                                 num_workers=hparams.workers, pin_memory=False)

        model = StatefulLSTM(hparams.resnet_weights, hparams.input_size, hparams.hidden_size, hparams.num_classes)
    else:
        raise ValueError('This implementation does not exist')

    model.to(device)

    # Updating the parameters of the last fully connected layer at a different learning rate than the
    # rest of the network.
    if hparams.multi_optim == 1:
        optimizer = optim.SGD([
            {'params': model.share.parameters()},
            {'params': model.fc.parameters(), 'lr': hparams.learning_rate},
        ], lr=hparams.learning_rate, momentum=hparams.momentum, dampening=hparams.dampening,
            weight_decay=hparams.weight_decay, nesterov=hparams.use_nesterov)
    else:
        # Updating all the parameters with the same learning rate
        optimizer = optim.SGD(model.parameters(), lr=hparams.learning_rate, momentum=hparams.momentum,
                              dampening=hparams.dampening, weight_decay=hparams.weight_decay,
                              nesterov=hparams.use_nesterov)
    if hparams.sgd_adjust_lr == 1:
        exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', min_lr=1e-6)
    else:
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=hparams.sgd_step, gamma=hparams.sgd_gamma)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_f1 = 0
    best_epoch = None
    patience_counter = 0
    patience = hparams.patience

    for epoch in range(hparams.epochs):
        torch.cuda.empty_cache()

        num_train_all = len(train_indices)
        if hparams.model == Implementation.ResNet.value:
            # random shuffling of training indices for every epoch
            np.random.shuffle(train_indices)
            # if sequence_length != 1, then sequences of indices are being added
            # i.e. clips of consecutive images of length equal to the sequence_length
            train_idx = [train_index + j for train_index in train_indices for j in range(sequence_length)]
            # indices are being shuffled in the baseline run. Therefore need to instantiate the data loader for
            # each epoch.
            train_loader = DataLoader(
                train_dataset,
                batch_size=hparams.train_batch_size,
                sampler=SeqSampler(train_dataset, train_idx),
                num_workers=hparams.workers,
                pin_memory=False
            )

        # Sets the module in training mode.
        model.train()
        train_loss_epoch = 0.0
        train_corrects_phase = 0
        total_samples = 0
        batch_progress = 0.0
        train_start_time = time.time()
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            inputs, labels, video_ids = data
            weights = None

            # Weights > 1
            if hparams.adjusted_weights:
                weights = torch.zeros(5, dtype=torch.int64)
                counts = torch.bincount(labels)
                for idx, val in enumerate(counts):
                    weights[idx] += val
                weights = weights.sum()/weights
                weights = weights.to(device)

            # Weights inversely proportional
            if hparams.inv_prop_weights:
                weights = torch.zeros(5, dtype=torch.float)
                counts = torch.bincount(labels)
                for idx, val in enumerate(counts):
                    weights[idx] += 1/(val + torch.tensor(0.001))
                weights = weights/weights.sum()
                weights = weights.to(device)

            inputs, labels, video_ids = inputs.to(device), labels.to(device), video_ids.to(device)
            labels = labels[(sequence_length - 1)::sequence_length]
            outputs = model.forward(inputs, video_ids)
            outputs = outputs[sequence_length - 1::sequence_length]

            _, predictions = torch.max(outputs.data, 1)

            # the cross entropy loss calculates by default the average loss over the batch
            loss = nn.CrossEntropyLoss(weight=weights)(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss_epoch += loss.data.item() * inputs.size(0)
            total_samples += inputs.size(0)
            batch_corrects_phase = torch.eq(predictions, labels.data).sum()
            train_corrects_phase += batch_corrects_phase

            batch_progress += 1
            if batch_progress * hparams.train_batch_size >= num_train_all:
                percent = 100.0
                print('Batch progress: %s [%d/%d]' % (str(percent) + '%', num_train_all, num_train_all), end='\n')
            else:
                percent = round(batch_progress * hparams.train_batch_size / num_train_all * 100, 2)
                print('Batch progress: %s [%d/%d]' % (
                    str(percent) + '%', batch_progress * hparams.train_batch_size, num_train_all), end='\r')

        train_elapsed_time = time.time() - train_start_time
        train_accuracy = float(train_corrects_phase) / total_samples
        train_average_loss = train_loss_epoch / total_samples
        writer.add_scalar('training acc epoch',
                          float(train_accuracy), epoch)
        writer.add_scalar('training loss epoch ',
                          float(train_average_loss), epoch)

        #####################################
        # ### Testing on validation dataset
        #####################################
        val_start_time = time.time()
        validation_loss, validation_correct_steps, val_all_preds, val_all_labels, total_samples = \
            evaluate_model(model, val_loader, hparams.val_batch_size, val_indices, device, sequence_length)
        val_elapsed_time = time.time() - val_start_time
        val_accuracy = float(validation_correct_steps) / total_samples
        val_average_loss = validation_loss / total_samples
        val_recall = metrics.recall_score(val_all_labels, val_all_preds, average='macro')
        val_precision = metrics.precision_score(val_all_labels, val_all_preds, average='macro')
        val_f1 = metrics.f1_score(val_all_labels, val_all_preds, average='macro')
        writer.add_scalar('validation acc epoch',
                          float(val_accuracy), epoch)
        writer.add_scalar('validation f1 epoch',
                          float(val_f1), epoch)
        writer.add_scalar('validation loss epoch ',
                          float(val_average_loss), epoch)
        writer.add_scalar('validation precision epoch',
                          float(val_precision), epoch)
        writer.add_scalar('validation recall epoch',
                          float(val_recall), epoch)

        print('epoch: {:4d}'
              ' train in: {:2.0f}m{:2.0f}s'
              ' train loss : {:4.4f}'
              ' train accuracy : {:.4f}'
              ' valid in: {:2.0f}m{:2.0f}s'
              ' valid average loss: {:4.4f}'
              ' valid accuracy: {:.4f}'
              ' valid f1: {:.4f}'
              ' valid macro precision: {:4.4f}'
              ' valid macro recall: {:.4f}'
              .format(epoch,
                      train_elapsed_time // 60,
                      train_elapsed_time % 60,
                      train_average_loss,
                      train_accuracy,
                      val_elapsed_time // 60,
                      val_elapsed_time % 60,
                      val_average_loss,
                      val_accuracy,
                      val_f1,
                      val_precision,
                      val_recall
                      )
              )

        ###############################
        # ### Testing on test dataset
        ###############################
        if hparams.test_dataset:
            test_start_time = time.time()
            test_loss, test_correct_steps, test_all_preds, test_all_labels, total_samples = \
                evaluate_model(model, test_loader, hparams.val_batch_size, test_indices, device, sequence_length)
            test_elapsed_time = time.time() - test_start_time
            test_accuracy = float(test_correct_steps) / total_samples
            test_average_loss = test_loss / total_samples
            test_recall = metrics.recall_score(test_all_labels, test_all_preds, average='macro')
            test_precision = metrics.precision_score(test_all_labels, test_all_preds, average='macro')
            test_f1 = metrics.f1_score(test_all_labels, test_all_preds, average='macro')
            writer.add_scalar('test acc epoch',
                              float(test_accuracy), epoch)
            writer.add_scalar('test loss epoch ',
                              float(test_average_loss), epoch)
            writer.add_scalar('test precision epoch',
                              float(test_precision), epoch)
            writer.add_scalar('test recall epoch',
                              float(test_recall), epoch)
            writer.add_scalar('test f1 epoch',
                              float(test_f1), epoch)
            print(' test in: {:2.0f}m{:2.0f}s'
                  ' test loss(phase): {:4.4f}'
                  ' test accu(phase): {:.4f}'
                  ' test f1(phase): {:.4f}'
                  .format(test_elapsed_time // 60,
                          test_elapsed_time % 60,
                          test_average_loss,
                          test_accuracy,
                          test_f1
                          )
                  )

        if hparams.sgd_adjust_lr == 1:
            exp_lr_scheduler.step(val_average_loss)
        else:
            exp_lr_scheduler.step()

        # record the best model by the f1 score on the validation dataset
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model = model
            best_model_wts = copy.deepcopy(best_model.state_dict())
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
        # check if we need to stop early
        if patience_counter >= patience:
            print('Early stopping triggered')
            break

        print("best_epoch", str(best_epoch))
        print("last_epoch", str(epoch))

    # #### save last epoch model weights
    last_model = model
    last_model_wts = copy.deepcopy(last_model.state_dict())
    saving_path = os.path.join(hparams.path_for_saving, hparams.run_name)
    if not os.path.exists(saving_path):
        os.mkdir(saving_path)
    torch.save(best_model_wts, os.path.join(saving_path, "best_model_weights.pth"))
    # #### save best epoch model weights
    torch.save(last_model_wts, os.path.join(saving_path, "last_model_weights.pth"))

    return val_loader, test_loader


def create_metrics_table(prediction, actual, renaming_dict):

    df = pandas.DataFrame({'pred': prediction, 'actual': actual})
    df['count'] = 1
    grouped_df = df.groupby(['pred', 'actual'])['count'].sum().reset_index()
    output_df = grouped_df.pivot(index='pred', columns='actual', values='count').fillna(0)
    missing_steps = [item for item in range(5) if item not in output_df.index.values]
    for missing_step in missing_steps:
        new_row = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}
        output_df.loc[missing_step, :] = new_row
    output_df = output_df.sort_index()
    output_df = output_df.rename(index=renaming_dict)
    output_df = output_df.rename(columns=renaming_dict)
    precision = (np.diag(output_df)/np.sum(output_df, axis=1)).values
    recall = (np.diag(output_df)/np.sum(output_df, axis=0)).values
    f1 = 2 * precision * recall / (precision + recall)
    count = output_df.sum(axis=1).values
    final_results = pandas.DataFrame({'Categories': output_df.index,
                                      'Precision': precision,
                                      'Recall': recall,
                                      'F1': f1,
                                      'Count': count})
    return final_results


def produce_evaluation_plots(data_loader, device, hparams, type):

    saving_path = os.path.join(hparams.path_for_saving, hparams.run_name)
    weights_path = os.path.join(saving_path, f'{type}_weights.pth')
    if hparams.model == Implementation.ResNet.value:
        model = resnetBaseline()
    elif hparams.model == Implementation.StatefulLSTM.value:
        model = StatefulLSTM(hparams.resnet_weights, hparams.input_size, hparams.hidden_size, hparams.num_classes)
    else:
        raise ValueError('This implementation does not exist')
    model.load_state_dict(torch.load(weights_path))
    model = model.to(device)
    model.eval()

    saving_path = os.path.join(hparams.path_for_saving, hparams.run_name)
    if not os.path.exists(saving_path):
        os.mkdir(saving_path)
    filename = f'{type}'
    y_pred = []
    y_actual = []
    label_dict = {
        'tumour_debulking': 0,
        'dissection_medial': 1,
        'dissection_inferior': 2,
        'dissection_superior': 3,
        'dissection_lateral': 4
    }

    with torch.no_grad():
        # Iterate over the test data and generate predictions
        for i, data in enumerate(data_loader, 0):
            inputs, labels, video_ids = data
            inputs, labels, video_ids = inputs.to(device), labels.to(device), video_ids.to(device)
            outputs = model.forward(inputs, video_ids)
            # Set total and correct
            _, predicted = torch.max(outputs.data, 1)
            y_pred.extend(predicted.detach().cpu().tolist())
            y_actual.extend(labels.detach().cpu().tolist())

        """
        Save results to csv
        """
        df = pandas.DataFrame({'Actuals': y_actual, 'Predictions': y_pred})
        df.to_csv(os.path.join(saving_path, 'predictions_vs_actuals_{0}.csv'.format(filename)))

        """
        Metrics per category
        """
        renaming_dict = dict([(val, key) for key, val in label_dict.items()])
        metrics_table = create_metrics_table(y_pred, y_actual, renaming_dict)
        metrics_table = metrics_table.round(decimals=3)
        metrics_table = metrics_table.set_index('Categories')
        metrics_table.to_csv(os.path.join(saving_path, 'metrics_results_{0}.csv'.format(filename)))

        """
        Confusion matrix
        """
        cmatrix = confusion_matrix(y_actual, y_pred, labels=[0, 1, 2, 3, 4])
        print(cmatrix)
        fig, ax = plt.subplots()
        sns.heatmap(cmatrix, annot=True, ax=ax, cmap='Blues', fmt="d")
        ax.set_title('Confusion Matrix')
        ax.set_xlabel('Predicted Labels')
        ax.set_ylabel('True Labels')
        ax.xaxis.set_ticklabels(['debulk', 'medial', 'inferior', 'superior', 'lateral'])
        ax.yaxis.set_ticklabels(['debulk', 'medial', 'inferior', 'superior', 'lateral'])
        plt.savefig(os.path.join(saving_path, 'network_confusion_matrix_{0}.png'.format(filename)))

        """
        Overall metrics
        """
        metrics_columns = ['Overall accuracy',
                           'Macro average precision',
                           'Weighted average precision',
                           'Macro average recall',
                           'Weighted average recall',
                           'Macro average F1',
                           'Weighted average F1'
                           ]
        report = classification_report(y_actual, y_pred, output_dict=True)
        results_array = np.array([
            report['accuracy'],
            report['macro avg']['precision'],
            report['weighted avg']['precision'],
            report['macro avg']['recall'],
            report['weighted avg']['recall'],
            report['macro avg']['f1-score'],
            report['weighted avg']['f1-score'],
                                     ])
        results_df = pandas.DataFrame(data=results_array.round(decimals=3))
        results_df.index = metrics_columns
        results_df.columns = ['Value']
        results_df.to_csv(os.path.join(saving_path, 'results_df_{0}.csv'.format(filename)))


def main():
    # python3.9 train.py -c modules/cnn/config/config_feature_extract.yml
    # you are essentially giving the config file path to the "-c" argument.
    # If "-c" is not provided, the parser will look for it in the default_config_files
    parser = configargparse.ArgParser(default_config_files=[os.path.join(Path(__file__).parent.parent,
                                                                         'configs/config_train_embed.yml')],
                                      config_file_parser_class=configargparse.YAMLConfigFileParser)
    parser.add_argument('-c', is_config_file=True, help='config file path')
    parser, hparams = build_configargparser(parser)

    device = torch.device("cuda" if hparams.gpu_usg and torch.cuda.is_available() else "cpu")

    train_dataset, train_num, val_dataset, val_num, test_dataset, test_num = \
        get_data(os.path.join('../data_inputs', hparams.datafile), hparams)

    val_loader, test_loader = \
        train_model(hparams, (train_dataset), (train_num), (val_dataset, test_dataset), (val_num, test_num))

    # produce evaluation results on the best model
    produce_evaluation_plots(val_loader, device, hparams, type='best_model')
    # produce evaluation results on the last model
    produce_evaluation_plots(val_loader, device,  hparams, type='last_model')

    if hparams.test_dataset:
        produce_evaluation_plots(test_loader, device, hparams, type='test_results')


if __name__ == "__main__":
    main()

