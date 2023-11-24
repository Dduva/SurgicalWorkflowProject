import pickle
import pandas
import os
import re

# OVERALL_PATH = r'/Users/dorotheeduvaux 1/UCL CSML/MSc Project/RS_data/pkl_files'
# OVERALL_PATH = r'/Users/dorotheeduvaux 1/UCL CSML/MSc Project/Video analytics'
OVERALL_PATH = r'/Users/dorotheeduvaux/PycharmProjects/SurgicalWorkflowProject/raw_data'


def retrieve_video_ear_type(input_file):
    df = pandas.read_excel(input_file, engine='openpyxl')
    df = df[~df['Video Title'].str.startswith('TL')]
    df['Video Title'] = df['Video Title'].str.replace('RS-', '')
    df['Left or Right'] = df['Left or Right'].map({'Right': 'R', 'Left': 'L'})
    df = df.set_index('Video Title')['Left or Right'].to_dict()
    return df


def compute_ear_type(paths, ear_dict):
    output_list = []
    video_id_list = []
    for path in paths:
        match = re.search(r'_frames\/([12]?\d)\/', path)
        if match:
            video = match.group(1)
            ear_type = ear_dict[video]
            output_list.append(ear_type)
            video_id_list.append(int(video))
        else:
            output_list.append(None)
            video_id_list.append(None)
            print(path, ": info not found")
    return output_list, video_id_list


def main():

    pkl_files = [
                 # 'server_train_val_paths_labels',
                 # 'high_res_60k_frames',
                 # 'server_train_val_paths_labels_fold2',
                 # 'server_train_val_paths_labels_fold3',
                 # 'server_train_val_paths_labels_fold4',
                 # 'server_train_val_paths_labels_fold5',
                 'server_train_val_paths_phases_labels',
                 'server_train_val_paths_phases_labels_fold2',
                 'server_train_val_paths_phases_labels_fold3',
                 'server_train_val_paths_phases_labels_fold4',
                 'server_train_val_paths_phases_labels_fold5',
                 ]

    # ear_type_dict = retrieve_video_ear_type(r'/Users/dorotheeduvaux 1/UCL CSML/MSc Project/RS_data/VS_side.xlsx')
    ear_type_dict = retrieve_video_ear_type(r'/Users/dorotheeduvaux/UCL CSML/MSc Project/RS_data/VS_side.xlsx')

    for pkl_file in pkl_files:
        with open(os.path.join(OVERALL_PATH, pkl_file + '.pkl'), 'rb') as file:
            data = pickle.load(file)
        train_paths = data[0]
        val_paths = data[1]
        test_paths = data[6]

        train_ear_type, train_video_id = compute_ear_type(train_paths, ear_type_dict)
        val_ear_type, val_video_id = compute_ear_type(val_paths, ear_type_dict)
        test_ear_type, test_video_id = compute_ear_type(test_paths, ear_type_dict)

        data.append(train_ear_type)
        data.append(val_ear_type)
        data.append(test_ear_type)

        data.append(train_video_id)
        data.append(val_video_id)
        data.append(test_video_id)

        with open(os.path.join(OVERALL_PATH, pkl_file + '_eartype_info.pkl'), 'wb') as f:
            pickle.dump(data, f)


if __name__ == '__main__':
    # path = '/Users/dorotheeduvaux/PycharmProjects/mscProject/data_inputs/server_train_val_paths_labels.pkl'
    # with open(path, 'rb') as file:
    #     data = pickle.load(file)
    main()


# path = '/Users/dorotheeduvaux/PycharmProjects/mscProject/data_inputs/server_train_val_paths_labels.pkl'
# with open(path, 'rb') as file:
#     data = pickle.load(file)
#
# train_paths = data[0]
# val_paths = data[1]
#
# train_labels = data[2]
# val_labels = data[3]
#
# train_num_each = data[4]
# val_num_each = data[5]
#
# test_paths = data[6]
# test_labels = data[7]
# test_num_each = data[8]
#
# train_paths = [item.replace('/home/dorothee/', '/Users/dorotheeduvaux 1/UCL CSML/MSc Project/') for item in
#                train_paths]
# val_paths = [item.replace('/home/dorothee/', '/Users/dorotheeduvaux 1/UCL CSML/MSc Project/') for item in
#                val_paths]
# test_paths = [item.replace('/home/dorothee/', '/Users/dorotheeduvaux 1/UCL CSML/MSc Project/') for item in
#                test_paths]
#
# new_data = [train_paths, val_paths, train_labels, val_labels, train_num_each, val_num_each, test_paths,
#             test_labels, test_num_each]
# with open('/Users/dorotheeduvaux/PycharmProjects/mscProject/data_inputs/desktop_server_train_val_paths_labels.pkl',
#           'wb') as f:
#     pickle.dump(new_data, f)
