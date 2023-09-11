
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os
import pandas as pd


def main():
    # Replace with the directory containing your TensorBoard logs
    log_dir = '../FinalTensorboard'

    # Initialize an empty list to collect all dataframes
    dfs = []

    for subdir, dirs, files in os.walk(log_dir):
        for file in files:
            # Only consider files with the 'tfevents' extension
            if "tfevents" in file:
                file_path = os.path.join(subdir, file)

                # Set up an event accumulator for this file
                event_acc = EventAccumulator(file_path)
                event_acc.Reload()

                # Get the run name (subdirectory name)
                run_name = os.path.relpath(subdir, log_dir)

                # Get scalar data
                for tag in event_acc.Tags()['scalars']:
                    scalar_events = event_acc.Scalars(tag)

                    # Extract wall times, step numbers, and values
                    w_times = [se.wall_time for se in scalar_events]
                    step_nums = [se.step for se in scalar_events]
                    vals = [se.value for se in scalar_events]

                    # Create a pandas dataframe from the scalar data
                    df = pd.DataFrame(data={'Wall time': w_times, 'Step': step_nums, 'Value': vals})

                    # Add columns for the run name and tag
                    df['Run'] = run_name
                    df['Tag'] = tag

                    # Add this dataframe to the list of all dataframes
                    dfs.append(df)

    # Combine all dataframes
    result_df = pd.concat(dfs, ignore_index=True)

    # Save the result as a CSV file
    result_df.to_csv('/Users/dorotheeduvaux 1/UCL CSML/MSc Project/FinalResults/tensorboard_data.csv', index=False)


if __name__ == '__main__':
    main()
