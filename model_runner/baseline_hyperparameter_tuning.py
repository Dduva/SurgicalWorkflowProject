import yaml
import sys
import configargparse
import os
from pathlib import Path
import torch
from new_model.baseline_resnet import get_data, train_model, Implementation, resnetBaseline, classification_report, \
    build_configargparser
import pandas
import numpy as np


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
    parser.add_argument('--learning_rate', default=5e-4, type=float, nargs='*', help='learning rate for optimizer, default 5e-5')
    parser.add_argument('--momentum', default=0.9, type=float, nargs='*', help='momentum for sgd, default 0.9')
    parser.add_argument('--weight_decay', default=5e-4, type=float, nargs='*', help='weight decay for sgd, default 0')
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


def retrieve_hyperparameter_results(data_loader, device, hparams, type):

    saving_path = os.path.join(hparams.path_for_saving, hparams.run_name)
    weights_path = os.path.join(saving_path, f'{type}_weights.pth')
    if hparams.model == Implementation.ResNet.value:
        model = resnetBaseline()
    else:
        raise ValueError('This implementation does not exist')
    model.load_state_dict(torch.load(weights_path))
    model = model.to(device)
    model.eval()

    saving_path = os.path.join(hparams.path_for_saving, hparams.run_name)
    if not os.path.exists(saving_path):
        os.mkdir(saving_path)
    y_pred = []
    y_actual = []

    with torch.no_grad():
        # Iterate over the test data and generate predictions
        for i, data in enumerate(data_loader, 0):
            inputs, labels, _ = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model.forward(inputs)
            # Set total and correct
            _, predicted = torch.max(outputs.data, 1)
            y_pred.extend(predicted.detach().cpu().tolist())
            y_actual.extend(labels.detach().cpu().tolist())

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
        results_df = results_df.transpose().reset_index().drop(columns='index')

    # Delete the weights once they have been used to calculate the results as they are no longer needed.
    if os.path.exists(weights_path):
        os.remove(weights_path)
        print(f"File {weights_path} has been deleted.")
    else:
        print(f"File {weights_path} does not exist.")

    return results_df.to_dict(orient='list')


def main():
    parser = configargparse.ArgParser(default_config_files=[os.path.join(Path(__file__).parent.parent,
                                                                         'configs/hyperparameter_tuning.yml')],
                                      config_file_parser_class=configargparse.YAMLConfigFileParser)
    parser.add_argument('-c', is_config_file=True, help='config file path')
    parser, hparams = build_configargparser(parser)

    train_dataset, train_num, val_dataset, val_num, test_dataset, test_num = \
        get_data(os.path.join('../data_inputs', hparams.datafile), hparams)

    device = torch.device("cuda" if hparams.gpu_usg and torch.cuda.is_available() else "cpu")

    learning_rates = hparams.learning_rate
    momentum_values = hparams.momentum
    weight_decay_values = hparams.weight_decay

    saving_path = os.path.join(hparams.path_for_saving, hparams.run_name)
    if not os.path.exists(saving_path):
        os.mkdir(saving_path)

    output_dfs = []

    # Iterating over the parameters that vary in the configuration. This needs changing if more hyperparameters are
    # tried. i.e. TODO: this could be made more flexible where the config yaml file is parsed.
    for learning_rate in learning_rates:
        for momentum in momentum_values:
            for weight_decay in weight_decay_values:
                hparams.learning_rate = learning_rate
                hparams.momentum = momentum
                hparams.weight_decay = weight_decay

                val_loader, test_loader = \
                    train_model(hparams, (train_dataset), (train_num), (val_dataset, test_dataset), (val_num, test_num))

                best_model_metrics_dict = retrieve_hyperparameter_results(val_loader, device, hparams, type='best_model')
                last_model_metrics_dict = retrieve_hyperparameter_results(val_loader, device, hparams, type='last_model')

                # delete weights after each iteration
                param_dict = {
                    'learning_rate': learning_rate,
                    'momentum': momentum,
                    'weight_decay': weight_decay,
                }
                param_dict.update({'type': 'best'})
                merged_best_dict = {**param_dict, **best_model_metrics_dict}
                param_dict.update({'type': 'last'})
                merged_last_dict = {**param_dict, **last_model_metrics_dict}

                output_dfs.append(pandas.DataFrame(merged_best_dict))
                output_dfs.append(pandas.DataFrame(merged_last_dict))

    final_results = pandas.concat(output_dfs).reset_index().drop(columns='index')
    final_results.to_csv(os.path.join(saving_path, f'{hparams.run_name}_results.csv'))


if __name__ == '__main__':
    main()



