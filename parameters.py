import os

class Parameters:
    def __init__(self):

        # Paths
        self.data_dir = os.path.join(os.environ['HOME'], 'rosbags', 'kitti_dataset')
        self.log_dir = os.path.join(os.environ['HOME'], 'rosbags', 'deep_vo')
        self.image_dir = self.data_dir + '/sequences/'
        self.pose_dir = self.data_dir + '/poses'
        self.dataset_info = self.log_dir + '/dataset_dictionaries'
        self.models_dir = self.log_dir + "/models"
        self.checkpoint_dir = self.log_dir + '/checkpoints'
        self.history_dir = self.log_dir + '/history'
        self.dict_dir = self.log_dir + '/dataset_dictionaries'
        self.eval = self.log_dir + '/evaluation'

        # Data Preprocessing
        self.dict_name = '8'
        self.rescale = 1. / 255.
        self.resize_mode = 'rescale'  # choice: 'crop' 'rescale' None
        self.img_w = 1226 #160  #1226 #640  #370  # 320  # original size is about 1226
        self.img_h = 370 #72  #370 #184  #1226  # 144  # original size is about 370
        self.img_d = 3
        self.minus_point_5 = True
	self.data_dir = os.path.join(os.environ['HOME'], 'rosbags', 'kitti_flow_feature_map', 'outputs', 'image_2')  #'/home/mi/carrillo/rosbags/kitti_flow/outputs'
        self.log_dir = os.path.join(os.environ['HOME'], 'rosbags', 'deep_vo')  #'/home/mi/carrillo/rosbags/deep_vo'
        self.pose_dir = os.path.join(os.environ['HOME'], 'rosbags', 'kitti_dataset', 'poses')  #'/media/carrillo/kasparov/kitti_dataset/poses' #'/home/mi/carrillo/rosbags/kitti_dataset/poses'
        self.dict_dir = self.log_dir + '/dataset_dictionaries/flow'

        # Dataset, rerun preprocess to generate the new dic if you change this
        self.train_sequence = [0]  # [0, 1, 2 ,8]
        self.train_boundaries = [None, None]  # or [None, None] We have to divide the number of elements with the sequence_length
        self.validation_sequence = [8]  # +[7,10]
        self.validation_boundaries = [None, None]  # or [None, None]
        self.test_sequence = [5]  # +[7,10]
        self.test_boundaries = [None, None]  # or [None, None]
        self.sequence_length = [None, None]  # ir [None, None] make unique sequence
        self.forward_and_backwards = False  # Generate database running backwards the data
        self.max_seq_step = 1  # Data augmentation through skipping frames from 0 to max_seq_step
        self.sequence_repeatability = 1
        self.orientation_output = 'euler'  # Selects the representation of the angles {'euler', 'quaternion'}
        self.norm_output = False
        self.sort_dict_by_seq_len = False
        self.dict_name = 'very simple learning'

par = Parameters()

