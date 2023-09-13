import torch
import torch.nn as nn
import torch
import configargparse
from torch.utils.data import DataLoader
import os
from pathlib import Path
import pandas
from sklearn.metrics import classification_report
from enum import Enum
import numpy
from model_runner.model_runner import get_data, resnetBaseline, StatefulLSTM, get_indices, SeqSampler, \
    Implementation, VideoBatchSampler


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


def get_baseline_data_loaders(train_dataset, train_num, val_dataset, val_num,
                     test_dataset, test_num, hparams):

    sequence_length = hparams.sequence_length

    # List the indices used from each dataset
    train_indices = get_indices(sequence_length, train_num)
    val_indices = get_indices(sequence_length, val_num)
    test_indices = get_indices(sequence_length, test_num)

    # if sequence_length != 1, then sequences of indices are being added
    # i.e. clips of consecutive images of length equal to the sequence_length
    # train_idx is deduced as part of each separate epoch.
    train_idx = [train_index + j for train_index in train_indices for j in range(sequence_length)]
    val_idx = [val_index + j for val_index in val_indices for j in range(sequence_length)]
    test_idx = [test_index + j for test_index in test_indices for j in range(sequence_length)]

    val_loader = DataLoader(
        val_dataset,
        batch_size=hparams.val_batch_size,
        sampler=SeqSampler(val_dataset, val_idx),
        num_workers=hparams.workers,
        pin_memory=False
    )
    test_loader = DataLoader(
        val_dataset,
        batch_size=hparams.val_batch_size,
        sampler=SeqSampler(test_dataset, test_idx),
        num_workers=hparams.workers,
        pin_memory=False
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=hparams.train_batch_size,
        sampler=SeqSampler(train_dataset, train_idx),
        num_workers=hparams.workers,
        pin_memory=False
    )
    return train_loader, val_loader, test_loader


def predict_model(loader, hparams, weight_type):

    device = torch.device("cuda" if hparams.gpu_usg and torch.cuda.is_available() else "cpu")

    saving_path = os.path.join(hparams.path_for_saving, hparams.run_name)
    weights_path = os.path.join(saving_path, f'{weight_type}_model_weights.pth')
    if hparams.model == Implementation.ResNet.value:
        model = resnetBaseline()
    elif hparams.model == Implementation.StatefulLSTM.value:
        model = StatefulLSTM(hparams.resnet_weights, hparams.input_size, hparams.hidden_size, hparams.num_classes)
    else:
        raise ValueError('This implementation does not exist')
    model.load_state_dict(torch.load(weights_path))
    model = model.to(device)
    model.eval()

    y_pred = []
    y_actual = []
    all_video_ids = []
    # Iterate over the training data and retrieve a list of predictions and actuals
    with torch.no_grad():
        for i, data in enumerate(loader, 0):
            inputs, labels, video_ids = data
            inputs, labels, video_ids = inputs.to(device), labels.to(device), video_ids.to(device)
            outputs = model.forward(inputs, video_ids)
            # Set total and correct
            _, predicted = torch.max(outputs.data, 1)
            y_pred.extend(predicted.detach().cpu().tolist())
            y_actual.extend(labels.detach().cpu().tolist())
            all_video_ids.extend(video_ids.detach().cpu().tolist())

        df = pandas.DataFrame({'Actuals': y_actual, 'Predictions': y_pred, 'Video_ids': all_video_ids})

    return df


def main():
    # Pass in the config that was used for an actual run.
    parser = configargparse.ArgParser(default_config_files=[os.path.join(Path(__file__).parent.parent,
                                                                         'configs/config_temporal_smoothing_metrics.yml'
                                                                         )],
                                      config_file_parser_class=configargparse.YAMLConfigFileParser)
    parser.add_argument('-c', is_config_file=True, help='config file path')
    parser, hparams = build_configargparser(parser)

    device = torch.device("cuda" if hparams.gpu_usg and torch.cuda.is_available() else "cpu")

    train_dataset, train_num, val_dataset, val_num, test_dataset, test_num = \
        get_data(os.path.join('../data_inputs', hparams.datafile), hparams)

    if hparams.model == Implementation.ResNet.value:
        train_loader, val_loader, test_loader = get_baseline_data_loaders(train_dataset, train_num, val_dataset,
                                                                          val_num, test_dataset, test_num, hparams)
    elif hparams.model == Implementation.StatefulLSTM.value:
        batch_sampler = VideoBatchSampler(train_num, hparams.train_batch_size)
        train_loader = DataLoader(train_dataset, batch_sampler=batch_sampler,
                                  num_workers=hparams.workers, pin_memory=False)

        batch_sampler = VideoBatchSampler(val_num, hparams.val_batch_size)
        val_loader = DataLoader(val_dataset, batch_sampler=batch_sampler,
                                num_workers=hparams.workers, pin_memory=False)
    else:
        raise ValueError('This implementation does not exist')

    weight_types = ['best', 'last']

    saving_path = os.path.join(hparams.path_for_saving, hparams.run_name)
    if not os.path.exists(saving_path):
        raise ValueError('This path should already exist')
    for weight_type in weight_types:
        print('Starting training dataset predictions')
        training_predictions_df = predict_model(train_loader, hparams, weight_type)
        print(f'Successful prediction of training dataset for {weight_type} weights')
        training_predictions_df.to_csv(os.path.join(saving_path, f'training_predictions_{weight_type}.csv'))

        print('Starting validation dataset predictions')
        val_predictions_df = predict_model(val_loader, hparams, weight_type)
        print(f'Successful prediction of validation dataset for {weight_type} weights')
        val_predictions_df.to_csv(os.path.join(saving_path, f'validation_predictions_{weight_type}.csv'))


if __name__ == '__main__':
    main()