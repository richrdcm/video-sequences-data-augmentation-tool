import os
import numpy as np
from vo_tools import T_list_to_angle, global2local, normalize_sequence
from parameters import par, flow_dic_par
import tensorflow as tf
import math
import pickle
import glob
from tqdm import tqdm
import cv2
import glob
import pandas as pd


def create_dataset_dictionaries():

    print("="*50)
    count = 0
    info_file = par.dict_dir + '/dictionaries_info.txt'

    if ('\n'.join("%s: %s" % item for item in vars(par).items())) in open(info_file).read():

        print("Dictionary already exist!, I'm not writing new one...")

    else:

        print("Creating dictionaries")

        # Create dictionaries to generate the dataset

        par_train_dict = {
            'path': par.data_dir,
            'sequences': par.train_sequence,
            'output': par.orientation_output,
            'sequence_length': par.sequence_length,
            'sequence_repeatability': par.sequence_repeatability,
            'change_order': par.forward_and_backwards,
            'max_step': par.max_seq_step,
            'min': par.train_boundaries[0],
            'max': par.train_boundaries[1]
        }
        train_dict, train_mean, train_std = dictionary_generator(**par_train_dict)

        par_val_dict = {
            'path': par.data_dir,
            'sequences': par.validation_sequence,
            'output': par.orientation_output,
            'sequence_length': par.sequence_length,
            'sequence_repeatability': 1,
            'change_order': False,
            'max_step': 1,
            'min': par.validation_boundaries[0],
            'max': par.validation_boundaries[1]
        }

        val_dict, val_mean, val_std = dictionary_generator(**par_val_dict)

        par_test_dict = {
            'path': par.data_dir,
            'sequences': par.test_sequence,
            'output': par.orientation_output,
            'sequence_length': par.sequence_length,
            'sequence_repeatability': 1,
            'change_order': False,
            'max_step': 1,
            'min': par.test_boundaries[0],
            'max': par.test_boundaries[1]
        }
        test_dict, test_mean, test_std = dictionary_generator(**par_test_dict)

        norm = {'train_mean': train_mean, 'train_std': train_std, 'val_mean': val_mean, 'val_std': val_std, 'test_mean': test_mean, 'test_std': test_std}
        df_norm = pd.DataFrame(norm, columns=['train_mean', 'train_std', 'val_mean', 'val_std', 'test_mean', 'test_std'])

        dictionaries = dict(train_dict=train_dict, val_dict=val_dict, test_dict=test_dict, norm=df_norm)  # test_dict=test_dict

        with open(info_file, "r") as f:
            for line in f:
                words = line.split()
                for i in words:
                    if i == "Sequence_number:":
                        count += 1

        with open(info_file, "a") as f:
            f.write('\n'+'='*50+'\n')
            f.write("Sequence_number: {}\n".format(count))
            f.write('\n'.join("%s: %s" % item for item in vars(par).items()))
            f.write('\n'+'='*50 + '\n')

        if not os.path.isdir(par.dict_dir + '/{}'.format(count) + '.dat'):
            with open(par.dict_dir + '/{}'.format(count) + '.dat', 'wb') as fout:
                pickle.dump(dictionaries, fout, protocol=pickle.HIGHEST_PROTOCOL)

        print("Training and validation dictionaries saved into '{}/{}.dat".format(par.log_dir, count))
    print("="*50)


def create_flow_dataset_dictionaries(record=True):

    print("="*50)
    count = 0
    info_file = par.dict_dir + '/dictionaries_info.txt'

    if ('\n'.join("%s: %s" % item for item in vars(flow_dic_par).items())) in open(info_file).read():
        print("Dictionary already exist!, I'm not writing new one...")

    else:
        print("Creating dictionaries")
        # Create dictionaries to generate the dataset

        par_train_dict = {
            'path': par.data_dir,
            'sequences': par.train_sequence,
            'output': par.orientation_output,
            'sequence_length': par.sequence_length,
            'sequence_repeatability': par.sequence_repeatability,
            'change_order': par.forward_and_backwards,
            'max_step': par.max_seq_step,
            'min': par.train_boundaries[0],
            'max': par.train_boundaries[1]
        }
        train_dict, train_mean, train_std = flow_dictionary_generator(**par_train_dict)

        par_val_dict = {
            'path': par.data_dir,
            'sequences': par.validation_sequence,
            'output': par.orientation_output,
            'sequence_length': par.sequence_length,
            'sequence_repeatability': 1,
            'change_order': False,
            'max_step': 1,
            'min': par.validation_boundaries[0],
            'max': par.validation_boundaries[1]
        }

        val_dict, val_mean, val_std = flow_dictionary_generator(**par_val_dict)

        par_test_dict = {
            'path': par.data_dir,
            'sequences': par.test_sequence,
            'output': par.orientation_output,
            'sequence_length': par.sequence_length,
            'sequence_repeatability': 1,
            'change_order': False,
            'max_step': 1,
            'min': par.test_boundaries[0],
            'max': par.test_boundaries[1]
        }
        test_dict, test_mean, test_std = flow_dictionary_generator(**par_test_dict)


        norm = {'train_mean': train_mean, 'train_std': train_std, 'val_mean': val_mean, 'val_std': val_std,
                'test_mean': test_mean, 'test_std': test_std}
        df_norm = pd.DataFrame(norm,
                               columns=['train_mean', 'train_std', 'val_mean', 'val_std', 'test_mean', 'test_std'])

        dictionaries = dict(train_dict=train_dict, val_dict=val_dict, test_dict=test_dict, norm=df_norm)  # test_dict=test_dict

        with open(info_file, "r") as f:
            for line in f:
                words = line.split()
                for i in words:
                    if i == "Sequence_number:":
                        count += 1

        with open(info_file, "a") as f:
            f.write('\n' + '=' * 50 + '\n')
            f.write("Sequence_number: {}\n".format(count))
            f.write('\n'.join("%s: %s" % item for item in vars(flow_dic_par).items()))
            f.write('\n' + '=' * 50 + '\n')

        if not os.path.isdir(par.dict_dir + '/{}'.format(count) + '.dat'):
            with open(par.dict_dir + '/{}'.format(count) + '.dat', 'wb') as fout:
                pickle.dump(dictionaries, fout, protocol=pickle.HIGHEST_PROTOCOL)

        print("Training and validation dictionaries saved into '{}/{}.dat".format(par.dict_dir, count))
    print("=" * 50)


def dictionary_generator(path, sequences, output, sequence_length, sequence_repeatability, change_order=False,
                         max_step=1, min=None, max=None):

    """ Generates a dictionary with IDs of the images that form the sequence, paths to the images in those IDs and
        the corresponding labels for training,
        output = representation (quaternion(unit q), euler or matrix(x12))
        order = ids couple are incremental or in decremental order
        mode = normal for a sequential order of ids, shuffle for random selecting the couples
        max_step = maximum separation between frames (this number is a randomly chosen)
        """
    sequence_list = []
    paths_list = []
    labels_list = []
    sequence_number_list = []

    for z in range(max_step):
        step = z+1  # random.randint(1, max_step)
        print("Frame step: {}".format(step))

        for s in sequences:

            min_len, max_len = sequence_length[0], sequence_length[1]

            img_path = path + '/sequences/%02d' % s + par.image_side + '/'
            pose_path = par.pose_dir + "/%02d.txt" % s

            n_frames = len(
                [name for name in os.listdir(img_path) if os.path.isfile(os.path.join(img_path, name))])
            fpaths = glob.glob('{}/*.png'.format(img_path))
            fpaths.sort()

            with open(pose_path) as f:
                all_poses = np.array([[float(x) for x in line.split()] for line in f])

            all_poses = T_list_to_angle(np.asarray(all_poses), output=par.orientation_output)



            if par.norm_output:
                all_poses, mean_, std_ = normalize_sequence(all_poses)
            else:
                mean_ = [0, 0, 0, 0, 0, 0]
                std_ = [1, 1, 1, 1, 1, 1]

            #all_poses = global2local(all_poses, input="euler")

            for t in range(0, sequence_repeatability):
                print("\nIterating over Sequence {0}, {1} / {2}".format(s, t+1, sequence_repeatability))

                start = 0

                while True:
                    n = np.random.randint(min_len, max_len)
                    if start + n < n_frames:
                        sequence_list.append(n)
                        paths_list.append(fpaths[start:start + n])
                        labels_list.append(all_poses[start:start + n])
                        sequence_number_list.append(s)
                    else:
                        print('Last %d frames is not used' % (start + n - n_frames))
                        break
                    start += n - 1

            if par.forward_and_backwards:
                for t in range(0, sequence_repeatability):
                    print("\nIterating backwards over Sequence {0}, {1} / {2}".format(s, t + 1, sequence_repeatability))

                    start = 0

                    while True:
                        n = np.random.randint(min_len, max_len)
                        if start + n < n_frames:
                            sequence_list.append(n)
                            paths_list.append(fpaths[start:start + n][::-1])
                            labels_list.append(all_poses[start:start + n][::-1])
                            sequence_number_list.append(s)
                        else:
                            print('Last %d frames is not used' % (start + n - n_frames))
                            break
                        start += n - 1

    if min is None and max is None:
        data = {'seq': sequence_number_list, 'seq_len': sequence_list, 'image_paths': paths_list, 'poses': labels_list}
    else:
        data = {'seq': sequence_number_list[min:max], 'seq_len': sequence_list[min:max],
                'image_paths': paths_list[min:max], 'poses': labels_list[min:max]}

    df = pd.DataFrame(data, columns=['seq', 'seq_len', 'image_paths', 'poses', 'out_mean', 'out_std'])

    # Shuffle through all videos
    if par.sort_dict_by_seq_len:
        df = df.sort_values(by=['seq_len'], ascending=False)

    return df, mean_, std_


def flow_dictionary_generator(path, sequences, output, sequence_length, sequence_repeatability, change_order=False,
                              max_step=1, min=None, max=None):

    """ Generates a dictionary with IDs of the images that form the sequence, paths to the images in those IDs and
        the corresponding labels for training,
        output = representation (quaternion(unit q), euler or matrix(x12))
        order = ids couple are incremental or in decremental order
        mode = normal for a sequential order of ids, shuffle for random selecting the couples
        max_step = maximum separation between frames (this number is a randomly chosen)
        """
    sequence_list = []
    paths_list = []
    labels_list = []
    sequence_number_list = []

    for z in range(max_step):
        step = z + 1  # random.randint(1, max_step)
        print("\nFrame step: {}".format(step))

        for s in sequences:

            min_len, max_len = sequence_length[0], sequence_length[1]

            img_path = path + '/%d_step_' % s + '%d_i_ffm_output' % step
            img_path_inv = path + '/%d_step_' % s + '%d_d_ffm_output' % step
            n_frames = len(
                [name for name in os.listdir(img_path) if os.path.isfile(os.path.join(img_path, name))])

            # Extracting the stepped frame
            print("\n***** Total frames: {}".format(n_frames))

            # Getting poses...
            pose_path = path + "/poses/%02d.txt" % s
            fpaths = glob.glob('{}/*.npy'.format(img_path))
            fpaths.sort()

            fpaths_i = glob.glob('{}/*.npy'.format(img_path_inv))
            fpaths_i.sort()

            with open(pose_path) as f:
                all_poses_ = np.array([[float(x) for x in line.split()] for line in f])

            indices_poses = np.arange(0, all_poses_.shape[0], step)
            all_poses = all_poses_[indices_poses]

            if par.norm_output:
                all_poses, mean_, std_ = normalize_sequence(all_poses)
            else:
                mean_ = [0]
                std_ = [1]

            all_poses_g = T_list_to_angle(all_poses, output=par.orientation_output)
            all_poses = global2local(all_poses_g, par.orientation_output)
            all_poses_i = global2local(all_poses_g, par.orientation_output, order="inverse")

            for t in range(0, sequence_repeatability):
                print("\nIterating over Sequence {0}, {1} / {2}".format(s, t + 1, sequence_repeatability))

                if min_len is None and max_len is None:
                    n = n_frames - 1
                    sequence_list.extend(np.full(n, 1))
                    paths_list.extend(fpaths)
                    # print("Path size: {}".format(len(fpaths)))
                    labels_list.extend(all_poses)
                    # print("label_size: {}".format(len(all_poses[start+2:start+2 + n])))
                    sequence_number_list.extend(np.full(n, s))
                    print('All %d frames used' % len(fpaths))
                    print('All %d poses used' % all_poses.shape[0])

                    if par.forward_and_backwards:
                        n = n_frames - 1
                        sequence_list.extend(np.full(n, 1))
                        paths_list.extend(fpaths_i)
                        # print("Path size: {}".format(len(fpaths)))
                        labels_list.extend(all_poses_i)
                        # print("label_size: {}".format(len(all_poses[start+2:start+2 + n])))
                        sequence_number_list.extend(np.full(n, s))
                        print('All %d frames used' % len(fpaths_i))
                        print('All %d poses used' % all_poses_i.shape[0])

                    break

                else:
                    start = 0
                    while True:

                            n = np.random.randint(min_len, max_len)

                            if start + n < n_frames:
                                sequence_list.append(n)
                                paths_list.append(fpaths[start:start + n])
                                #print("Path size: {}".format(len(fpaths[start:start + n])))
                                labels_list.append(all_poses[start+2:start+2 + n])
                                #print("label_size: {}".format(len(all_poses[start+2:start+2 + n])))
                                sequence_number_list.append(s)
                            else:

                                print('Last %d frames are not used' % (start + n - n_frames))
                                break
                            start += n - 1

    print("\nDatabase has the following members: \n seq: {0} \n seq_len:{1} \n image_paths:{2} \n poses:{3}".format(len(sequence_number_list), len(sequence_list), len(paths_list), len(labels_list)))

    if min is None and max is None:
        data = {'seq': sequence_number_list, 'seq_len': sequence_list, 'image_paths': paths_list, 'poses': labels_list}
    else:
        data = {'seq': sequence_number_list[min:max], 'seq_len': sequence_list[min:max],
                'image_paths': paths_list[min:max], 'poses': labels_list[min:max]}

    df = pd.DataFrame(data, columns=['seq', 'seq_len', 'image_paths', 'poses', 'out_mean', 'out_std'])

    # Shuffle through all videos
    if par.sort_dict_by_seq_len:
        df = df.sort_values(by=['seq_len'], ascending=False)

    return df, mean_, std_


def sanity_check_labels(dic):
    for i in dic:
        if np.isnan(i).all() or np.isinf(i).all():
            print("ERROR")


def sanity_check_image_inputs(dic):
    for i in dic:
        img_t1 = cv2.imread(i[0], 0)
        img_t2 = cv2.imread(i[1], 0)

        std1 = np.std(img_t1)
        std2 = np.std(img_t2)
        mean1 = np.mean(img_t1)
        mean2 = np.mean(img_t2)

        check1 = np.asarray([std1, mean1])
        check2 = np.asarray([std2, mean2])

        if np.isnan(check1).all() or np.isinf(check1).all() or std1 is 0 or mean1 is 0:
            print('Imagen erronea: {}'.format(i[0]))
        if np.isnan(check2).all() or np.isinf(check2).all() or std2 is 0 or mean2 is 0:
            print('Imagen erronea: {}'.format(i[1]))


def calculate_rgb_mean_std(image_path_list, minus_point_5=False):
    n_images = len(image_path_list)
    cnt_pixels = 0
    print('Numbers of frames in training dataset: {}'.format(n_images))

    mean_np = [0, 0, 0]
    mean_tensor = [0, 0, 0]
    std_tensor = [0, 0, 0]
    std_np = [0, 0, 0]

    for idx, img_path in enumerate(image_path_list):
        print('{} / {}'.format(idx, n_images), end='\r')
        img_as_img = tf.io.read_file(img_path)
        img_as_tensor = tf.image.decode_image(img_as_img)
        if minus_point_5:
            img_as_tensor = tf.to_float(img_as_tensor) - 0.5
        img_as_np = img_as_tensor.numpy()
        img_as_np = np.rollaxis(img_as_np, 2, 0)  # Why moving the pixels channels?
        cnt_pixels += img_as_np.shape[1] * img_as_np.shape[2]
        for c in range(3):
            mean_tensor[c] += float(tf.reduce_sum(img_as_tensor[c]))
            mean_np[c] += float(np.sum(img_as_np[c]))
            tmp = (img_as_tensor[c] - mean_tensor[c]) ** 2
            std_tensor[c] += float(tf.reduce_sum(tmp))
            tmp = (img_as_np[c] - mean_np[c]) ** 2
            std_np[c] += float(np.sum(tmp))
    mean_tensor = [v / cnt_pixels for v in mean_tensor]
    mean_np = [v / cnt_pixels for v in mean_np]
    std_tensor = [math.sqrt(v / cnt_pixels) for v in std_tensor]
    std_np = [math.sqrt(v / cnt_pixels) for v in std_np]

    print("=" * 50)
    print('mean_tensor = ', mean_tensor)
    print('mean_np = ', mean_np)
    print('std_tensor = ', std_tensor)
    print('std_np = ', std_np)

    img_params = dict(mean_tensor = np.asarray(mean_tensor),
                     mean_np = np.asarray(mean_np),
                     std_tensor = np.asarray(std_tensor),
                     std_np = np.asarray(std_np)
                     )
    with open(par.dict_dir + '/image_mean_std_' + par.dict_name + '.dat', 'wb') as fout:
        pickle.dump(img_params, fout, protocol=pickle.HIGHEST_PROTOCOL)

    print("Mean and std from database saven into: ./dataset_info/image_mean_std_{}".format(par.dataset_name))
    print("="*50)


def flow_paths_generator(path, sequence, order="incremental", max_step=4):

    """ Generates a dictionary with IDs of the images that form the sequence, paths to the images in those IDs and
        the corresponding labels for training,
        output = representation (quaternion(unit q), euler or matrix(x12))
        order = ids couple are incremental or in decremental order
        mode = normal for a sequential order of ids, shuffle for random selecting the couples
        max_step = maximum separation between frames (this number is a randomly chosen)
        """

    step = max_step  # random.randint(1, max_step)
    dic = {'ids': [], 'paths': []}

    ids_list = []
    paths_list = []

    img_path = path + '/sequences/%02d' % sequence + '/image_2_resized/'
    file_count = 0
    try:
        file_count = len([name for name in os.listdir(img_path) if os.path.isfile(os.path.join(img_path, name))])
    except:
        print("No path {}".format(img_path))
        print("Path should contain:\n -sequences \n --00 \n ---image_2 \n ----000000.png \n ---image_3 "
              "\n ----000000.png \n -poses \n --00.txt")

    ids = np.arange(start=0, stop=file_count, step=step)  # Size: file_count // step

    if order is "incremental":
        for j in range(0, len(ids)-1):
            ''' Here ids are generated '''
            current = ids[j]
            next = ids[j + 1]
            ids_list.append(('%s' % sequence)+'_'+('%d' % current) + '_' + ('%d' % next))

            ''' Here paths are generated '''
            paths_list.append([img_path + ('%06d' % current) + '.png', img_path + ('%06d' % next) + '.png'])

    if order is "decremental":

        for j in range(len(ids) - 1, 0, -1):
            current = ids[j]
            next = ids[j - 1]
            ids_list.append(('%s' % sequence)+'_'+('%d' % current) + '_' + ('%d' % next))

            ''' Here paths are generated '''
            paths_list.append([img_path + ('%06d' % current) + '.png', img_path + ('%06d' % next) + '.png'])

    dic['ids'] = np.asarray(ids_list)
    dic['paths'] = np.asarray(paths_list)

    return dic


if __name__ == '__main__':
    create_flow_dataset_dictionaries()
