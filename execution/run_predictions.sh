#!/bin/bash

# Optional: Redirect stdout and stderr to a log file for monitoring
exec > >(tee -i /home/dorothee/logs/logfile.log)
exec 2>&1

# Optional: Print the date and time when the script starts
echo "Master script started on $(date)"

# Define a function to run the Python script with different config files
run_computing_predictions() {
    CONFIG_FILE=$1
    echo "Starting computing_predictions.py with config: $CONFIG_FILE..."
    python3.9 computing_predictions.py -c $CONFIG_FILE

    if [ $? -eq 0 ]; then
        echo "Script finished successfully for config: $CONFIG_FILE"
    else
        echo "Script encountered an error for config: $CONFIG_FILE"
        # Optionally, exit if a script fails
        # exit 1
    fi
}

# Execute the function with different config files
run_computing_predictions "/home/dorothee/configs/baseline_new_network_inv_weights_1fps_250res.yml"
# run_computing_predictions "/home/dorothee/configs/baseline_new_network_inv_weights_eartype_1fps_250res.yml"
# run_computing_predictions "/home/dorothee/configs/baseline_new_network_inv_weights_eartype_1fps_250res_fold2.yml"
# run_computing_predictions "/home/dorothee/configs/baseline_new_network_inv_weights_eartype_1fps_250res_fold3.yml"
# run_computing_predictions "/home/dorothee/configs/baseline_new_network_inv_weights_eartype_1fps_250res_fold4.yml"
# run_computing_predictions "/home/dorothee/configs/baseline_new_network_inv_weights_eartype_1fps_250res_fold5.yml"
# run_computing_predictions "/home/dorothee/configs/resnet_LSTM_batch30_invprop_ear_1fp250_fold1.yml"
# run_computing_predictions "/home/dorothee/configs/resnet_LSTM_batch30_invprop_ear_1fp250_fold2.yml"
# run_computing_predictions "/home/dorothee/configs/resnet_LSTM_batch30_invprop_ear_1fp250_fold3.yml"
# run_computing_predictions "/home/dorothee/configs/resnet_LSTM_batch30_invprop_ear_1fp250_fold4.yml"
# run_computing_predictions "/home/dorothee/configs/resnet_LSTM_batch30_invprop_ear_1fp250_fold5.yml"
# ... Add more as needed

# Optional: Print the date and time when the script ends
echo "Master script finished on $(date)"

