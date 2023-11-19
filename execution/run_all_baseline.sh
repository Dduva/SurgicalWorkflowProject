#!/bin/bash

# Optional: Redirect stdout and stderr to a log file for monitoring
# exec > >(tee -i /home/dorothee/logs/logfile.log)
# exec 2>&1

# Optional: Print the date and time when the script starts
echo "Master script started on $(date)"

# Define a function to run the Python script with different config files
run_baseline_resnet() {
    CONFIG_FILE=$1
    echo "Starting model_runner.py with config: $CONFIG_FILE..."
    python3.9 model_runner.py -c $CONFIG_FILE

    if [ $? -eq 0 ]; then
        echo "Script finished successfully for config: $CONFIG_FILE"
    else
        echo "Script encountered an error for config: $CONFIG_FILE"
        # Optionally, exit if a script fails
        # exit 1
    fi
}

# Execute the function with different config files
# run_baseline_resnet "/home/dorothee/configs/baseline_new_network_inv_weights_eartype_1fps_250res_fold2.yml"
# run_baseline_resnet "/home/dorothee/configs/baseline_new_network_inv_weights_eartype_1fps_250res_fold3.yml"
# run_baseline_resnet "/home/dorothee/configs/baseline_new_network_inv_weights_eartype_1fps_250res_fold4.yml"
# run_baseline_resnet "/home/dorothee/configs/baseline_new_network_inv_weights_eartype_1fps_250res_fold5.yml"
# run_baseline_resnet "/home/dorothee/configs/config_train_original_runs_no_weights.yml"
# run_baseline_resnet "/home/dorothee/configs/config_train_original_runs_inv_weights.yml"
# run_baseline_resnet "/home/dorothee/configs/config_train_original_runs_adj_weights.yml"
# run_baseline_resnet "/home/dorothee/configs/resnet_LSTM_batch20_invprop_ear_1fp250.yml"
# run_baseline_resnet "/home/dorothee/configs/resnet_LSTM_batch40_invprop_ear_1fp250.yml"
# run_baseline_resnet "/home/dorothee/configs/resnet_LSTM_batch50_invprop_ear_1fp250.yml"
# run_baseline_resnet "/home/dorothee/configs/resnet_LSTM_batch100_invprop_ear_1fp250.yml"
run_baseline_resnet "/home/dorothee/configs/baseline_new_network_inv_weights_eartype_1fps_combined_250res.yml"
run_baseline_resnet "/home/dorothee/configs/baseline_new_network_inv_weights_eartype_1fps_combined_250res_fold2.yml"
run_baseline_resnet "/home/dorothee/configs/baseline_new_network_inv_weights_eartype_1fps_combined_250res_fold3.yml"
run_baseline_resnet "/home/dorothee/configs/baseline_new_network_inv_weights_eartype_1fps_combined_250res_fold4.yml"
run_baseline_resnet "/home/dorothee/configs/baseline_new_network_inv_weights_eartype_1fps_combined_250res_fold5.yml"

# ... Add more as needed

# Optional: Print the date and time when the script ends
echo "Master script finished on $(date)"
