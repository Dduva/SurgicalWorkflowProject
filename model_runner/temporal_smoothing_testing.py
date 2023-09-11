from sklearn.metrics import classification_report, confusion_matrix
import pandas
import os
import numpy


def custom_mode(series):
    # Find the mode(s)
    mode_values = series.mode()

    # If only one mode, return it
    if len(mode_values) == 1:
        return mode_values.iloc[0]
    # If more than one mode, return the mode from the closest row
    elif len(mode_values) > 1:
        return series[series.isin(mode_values)].iloc[-1]
    # If no mode, return None
    else:
        return None

def rolling_mode_by_group(group, window_size):
    return group['Predictions'].rolling(window_size + 1).apply(custom_mode, raw=False)

def calculate_column(group, n):
    for i in range(n, len(group)):
        if group['Predictions'].iloc[ i - n +1: i +1].nunique() == 1:
            group.loc[group.index[i], 'Threshold_predictions'] = group.loc[group.index[i], 'Predictions']
        else:
            group.loc[group.index[i], 'Threshold_predictions'] = group.loc[group.index[ i -1], 'Threshold_predictions']
    return group


def get_optimal_temporal_smoothing_function(saving_path, final_results_folder, folder_name, weight_type, total_window_size):
    training_filename = f'training_predictions_{weight_type}.csv'
    training_predictions_df = pandas.read_csv(os.path.join(saving_path, final_results_folder,
                                                           folder_name, training_filename))
    best_modal_f1 = 0
    best_threshold_f1 = 0
    optimal_modal_n = 0
    optimal_threshold_n = 0
    for window_size in range(1, total_window_size +1):
        print(window_size)
        # Modal function
        modal_predictions = training_predictions_df.groupby('Video_ids'). \
            apply(lambda g: rolling_mode_by_group(g, window_size)).reset_index(level=0, drop=True)
        training_predictions_df['modal_predictions'] = modal_predictions
        training_predictions_df['final_modal_predictions'] = numpy.where(
            training_predictions_df['modal_predictions'].isnull(), training_predictions_df['Predictions'],
            training_predictions_df['modal_predictions'])
        # Threshold function
        training_predictions_df['Threshold_predictions'] = training_predictions_df['Predictions']
        training_predictions_df = training_predictions_df.groupby('Video_ids').apply \
            (lambda g: calculate_column(g, window_size)).reset_index(level=0, drop=True)

        modal_f1 = classification_report(training_predictions_df['Actuals'],
                                         training_predictions_df['final_modal_predictions'],
                                         output_dict=True)['weighted avg']['f1-score']
        threshold_f1 = classification_report(training_predictions_df['Actuals'],
                                             training_predictions_df['Threshold_predictions'],
                                             output_dict=True)['weighted avg']['f1-score']
        if modal_f1 > best_modal_f1:
            best_modal_f1 = modal_f1
            optimal_modal_n = window_size
        if threshold_f1 > best_threshold_f1:
            best_threshold_f1 = threshold_f1
            optimal_threshold_n = window_size
        print('modal_f1', modal_f1)
        print('threshold_f1', threshold_f1)
        print('optimal_modal_n: ', optimal_modal_n)
        print('optimal_threshold_n: ', optimal_threshold_n)
    return optimal_modal_n, optimal_threshold_n


def compute_metrics(actuals, predictions):

    report = classification_report(actuals, predictions, output_dict=True)
    metrics_array = numpy.array([
        report['accuracy'],
        report['macro avg']['precision'],
        report['weighted avg']['precision'],
        report['macro avg']['recall'],
        report['weighted avg']['recall'],
        report['macro avg']['f1-score'],
        report['weighted avg']['f1-score'],
    ])
    return metrics_array


def compute_final_metrics(saving_path, final_results_folder, folder_name, weight_type, optimal_modal_n,
                          optimal_threshold_n):
    validation_filename = f'validation_predictions_{weight_type}.csv'
    validation_results_df = pandas.read_csv(
        os.path.join(saving_path, final_results_folder, folder_name, validation_filename))

    tsf_df = []
    # modal predictions
    modal_predictions = validation_results_df.groupby('Video_ids'). \
        apply(lambda g: rolling_mode_by_group(g, optimal_modal_n)).reset_index(level=0, drop=True)
    validation_results_df['modal_predictions'] = modal_predictions
    validation_results_df['final_modal_predictions'] = numpy.where(validation_results_df['modal_predictions'].isnull(),
                                                                   validation_results_df['Predictions'],
                                                                   validation_results_df['modal_predictions'])

    metrics_array = compute_metrics(validation_results_df['Actuals'], validation_results_df['final_modal_predictions'])
    new_row_df = pandas.DataFrame({
        'Run name': [folder_name],
        'Temporal function': ['modal'],
        'n': [optimal_modal_n],
        'Weighted F1 score': [metrics_array[0]],
        'Mean accuracy': [metrics_array[1]],
        'Weighted precision': [metrics_array[2]],
        'Weighted recall': [metrics_array[3]]
    })
    tsf_df.append(new_row_df)

    # threshold predictions
    validation_results_df['Threshold_predictions'] = validation_results_df['Predictions']
    validation_results_df = validation_results_df.groupby('Video_ids').apply(
        lambda g: calculate_column(g, optimal_threshold_n)).reset_index(level=0, drop=True)
    metrics_array = compute_metrics(validation_results_df['Actuals'], validation_results_df['Threshold_predictions'])
    new_row_df = pandas.DataFrame({
        'Run name': [folder_name],
        'Temporal function': ['threshold'],
        'n': [optimal_threshold_n],
        'Weighted F1 score': [metrics_array[0]],
        'Mean accuracy': [metrics_array[1]],
        'Weighted precision': [metrics_array[2]],
        'Weighted recall': [metrics_array[3]]
    })
    tsf_df.append(new_row_df)
    output_df = pandas.concat(tsf_df).reset_index().drop(columns='index')
    return output_df, validation_results_df


def main():
    saving_path = '/Users/dorotheeduvaux 1/UCL CSML/MSc Project'
    final_results_folder = 'FinalResults'
    weight_type = 'last'
    total_window_size = 20

    # folder_names = [
    #     'new_network_inv_weights_1fps_250',
    #     'new_network_inv_weights_refl_1fps_250',
    #     'new_nn_inv_weights_refl_1fps_250_fold2',
    #     'new_nn_inv_weights_refl_1fps_250_fold3',
    #     'new_nn_inv_weights_refl_1fps_250_fold4',
    #     'new_nn_inv_weights_refl_1fps_250_fold5',
    # ]

    folder_names = [
        'rLSTM_batch30_intprop_ear_1fp250_fold1',
        'rLSTM_batch30_intprop_ear_1fp250_fold2',
        'rLSTM_batch30_intprop_ear_1fp250_fold3',
        'rLSTM_batch30_intprop_ear_1fp250_fold4',
        'rLSTM_batch30_intprop_ear_1fp250_fold5'
    ]

    for folder_name in folder_names:
        print(folder_name)
        optimal_modal_n, optimal_threshold_n = get_optimal_temporal_smoothing_function(saving_path, final_results_folder,
                                                                                       folder_name, weight_type,
                                                                                       total_window_size)
        print('optimal_modal_n: ', optimal_modal_n)
        print('optimal_threshold_n: ', optimal_threshold_n)

        output_df, new_val_results = compute_final_metrics(saving_path, final_results_folder, folder_name, weight_type,
                                                           optimal_modal_n, optimal_threshold_n)
        print('Final metrics computed')
        print('Saving...')
        output_df.to_csv(os.path.join(saving_path, final_results_folder, folder_name, f'temporal_smoothing_metrics_{weight_type}.csv'))
        new_val_results.to_csv(
            os.path.join(saving_path, final_results_folder, folder_name, f'ts_adjusted_validation_predictions_{weight_type}.csv'))
        output_df = output_df.to_latex(index=False, float_format="%.3f")
        with open(os.path.join(saving_path, final_results_folder, folder_name,
                               f'temporal_smoothing_metrics_{weight_type}.tex'), 'w') as f:
            f.write(output_df)


if __name__ == '__main__':
    main()