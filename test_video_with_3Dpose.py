import sys
import argparse
import cv2
import os 
import numpy as np
import torch
import torch.nn as nn
import glob
from tqdm import tqdm
import copy
import torch.nn.functional as F
sys.path.append(os.getcwd())
from lib.preprocess import h36m_coco_format, revise_kpts
from lib.hrnet.gen_kpts import gen_video_kpts as hrnet_pose
from common.model_poseformer import PoseTransformerV2 as Model
from common.camera import *

import matplotlib
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec

plt.switch_backend('agg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

from torch.utils.data import Dataset, DataLoader

from networks.model3d import EventDetector
save_model_name = 'pose_d3'


class SampleVideo(Dataset):
    def __init__(self, poses):
        self.poses = poses

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        # preprocess and return frames
        poses3D = self.poses
        labels = np.zeros(len(poses3D)) # only for compatibility with transforms
        sample = {'inputs': poses3D, 'labels': np.asarray(labels)}
        return sample

def get_pose2D(video_path):
    cap = cv2.VideoCapture(video_path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    print('\nGenerating 2D pose...')
    keypoints, scores = hrnet_pose(video_path, det_dim=416, num_peroson=1, gen_output=True)
    keypoints, scores, valid_frames = h36m_coco_format(keypoints, scores)
    scores = scores.reshape(scores.shape[0],scores.shape[1],scores.shape[2],1)
    kps_conf = np.concatenate([keypoints, scores], axis=3)
    # re_kpts = revise_kpts(keypoints, scores, valid_frames)
    print('Generating 2D pose successful!')

    return kps_conf, width, height

def get_pose3D(poses, width, height, model_name):
    args, _ = argparse.ArgumentParser().parse_known_args()
    args.embed_dim_ratio, args.depth, args.frames = 32, 4, 243
    args.number_of_kept_frames, args.number_of_kept_coeffs = 27, 27
    args.pad = (args.frames - 1) // 2
    args.previous_dir = 'checkpoint/'
    args.n_joints, args.out_joints = 17, 17

    ## Reload 
    model = nn.DataParallel(Model(args=args)).cuda()

    # Put the pretrained model of PoseFormerV2 in 'checkpoint/']
    model_path = sorted(glob.glob(os.path.join(args.previous_dir, model_name)))[0]

    pre_dict = torch.load(model_path)
    model.load_state_dict(pre_dict['model_pos'], strict=True)

    model.eval()

    ## input
    kps_conf = poses
    keypoints, scores = kps_conf[:,:,:,:2], kps_conf[:,:,:,2:]

    ## 3D
    print('\nGenerating 3D pose...')
    post_outs = []
    for i in tqdm(range(len(poses[0]))):

        ## input frames
        start = max(0, i - args.pad)
        end =  min(i + args.pad, len(keypoints[0])-1)

        input_2D_no = keypoints[0][start:end+1]
        
        left_pad, right_pad = 0, 0
        if input_2D_no.shape[0] != args.frames:
            if i < args.pad:
                left_pad = args.pad - i
            if i > len(keypoints[0]) - args.pad - 1:
                right_pad = i + args.pad - (len(keypoints[0]) - 1)

            input_2D_no = np.pad(input_2D_no, ((left_pad, right_pad), (0, 0), (0, 0)), 'edge')
        
        joints_left =  [4, 5, 6, 11, 12, 13]
        joints_right = [1, 2, 3, 14, 15, 16]

        # input_2D_no += np.random.normal(loc=0.0, scale=5, size=input_2D_no.shape)
        input_2D = normalize_screen_coordinates(input_2D_no, w=width, h=height)     # (243, 17, 2)

        input_2D_aug = copy.deepcopy(input_2D)
        input_2D_aug[ :, :, 0] *= -1
        input_2D_aug[ :, joints_left + joints_right] = input_2D_aug[ :, joints_right + joints_left]
        input_2D = np.concatenate((np.expand_dims(input_2D, axis=0), np.expand_dims(input_2D_aug, axis=0)), 0)
        # (2, 243, 17, 2)
        
        input_2D = input_2D[np.newaxis, :, :, :, :]     # (2, 243, 17, 2) -> (1, 2, 243, 17, 2)

        input_2D = torch.from_numpy(input_2D.astype('float32')).cuda()

        N = input_2D.size(0)

        ## estimation
        output_3D_non_flip = model(input_2D[:, 0]) 
        output_3D_flip     = model(input_2D[:, 1])
        # [1, 1, 17, 3]

        output_3D_flip[:, :, :, 0] *= -1
        output_3D_flip[:, :, joints_left + joints_right, :] = output_3D_flip[:, :, joints_right + joints_left, :] 

        output_3D = (output_3D_non_flip + output_3D_flip) / 2

        output_3D[:, :, 0, :] = 0
        post_out = output_3D[0, 0].cpu().detach().numpy()
        
        rot =  [0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088]
        rot = np.array(rot, dtype='float32')
        post_out = camera_to_world(post_out, R=rot, t=0)
        post_out[:, 2] -= np.min(post_out[:, 2])
        post_outs.append(post_out)

    output_3Dposes = np.array(post_outs)
    print('Generating 3D pose successful!')
    
    return output_3Dposes

event_names = {
    0: 'Address',
    1: 'Toe-up',
    2: 'backswing',
    3: 'Top',
    4: 'downswing',
    5: 'Impact',
    6: 'follow-through',
    7: 'Finish'
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, default='test_video.mp4', help='input video')   # test_video
    parser.add_argument('-s', '--seq-length', type=int, help='Number of frames to use per forward pass', default=64)
    parser.add_argument('--gpu', type=str, default='0', help='input video')
    parser.add_argument('--model_name', type=str, default='27_243_45.2.bin', help='input video')
    args = parser.parse_args()
    seq_length = args.seq_length
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    video_path = './demo/video/' + args.video
    video_name = video_path.split('/')[-1].split('.')[0]
    output_dir = './demo/output/' + video_name + '/'
    os.makedirs(output_dir, exist_ok=True)

    poses, width, height = get_pose2D(video_path)
    
    output_3Dposes = get_pose3D(poses, width, height, args.model_name)

    print('Generating demo successful!')


    ds = SampleVideo(output_3Dposes)
    dl = DataLoader(ds, batch_size=1, shuffle=False, drop_last=False)
    
    model = EventDetector(pretrain=True,
                          width_mult=1.,
                          lstm_layers=1,
                          lstm_hidden=256,
                          bidirectional=True,
                          dropout=False)
    
    load_model_root = 'models/{}'.format(save_model_name)
    load_model_name = sorted(glob.glob(load_model_root +'/*.tar'))[-1]
    try:
        save_dict = torch.load(load_model_name)
    except:
        print("Model weights not found. Download model weights and place in 'models' folder. See README for instructions")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    model.load_state_dict(save_dict['model_state_dict'])
    model.to(device)
    model.eval()
    print("Loaded model weights")
    
    print('Testing...')
    for sample in dl:
        inputs = sample['inputs']
        # full samples do not fit into GPU memory so evaluate sample in 'seq_length' batches
        batch = 0
        while batch * seq_length < inputs.shape[1]:
            if (batch + 1) * seq_length > inputs.shape[1]:
                input_batch = inputs[:, batch * seq_length:, :]
            else:
                input_batch = inputs[:, batch * seq_length:(batch + 1) * seq_length, :]
            logits = model(input_batch.cuda())
            if batch == 0:
                probs = F.softmax(logits.data, dim=1).cpu().numpy()
            else:
                probs = np.append(probs, F.softmax(logits.data, dim=1).cpu().numpy(), 0)
            batch += 1

    events = np.argmax(probs, axis=0)[:-1]
    print('Predicted event frames: {}'.format(events))
    cap = cv2.VideoCapture(video_path)

    confidence = []
    for i, e in enumerate(events):
        confidence.append(probs[e, i])
    print('Condifence: {}'.format([np.round(c, 3) for c in confidence]))
    thickness=0.5
    loc1=(20, 20)
    loc2=(20, 40)
    for i, e in enumerate(events):
        cap.set(cv2.CAP_PROP_POS_FRAMES, e)
        _, img = cap.read()
        if img.shape[0]>1000:
            cv2.namedWindow(event_names[i], cv2.WINDOW_NORMAL)
            cv2.resizeWindow(event_names[i], int(img.shape[0]*0.8), int(img.shape[0]*0.8))
            thickness=1
            loc1=(30, 30)
            loc2=(30, 70)
        cv2.putText(img, '{:.2f}%'.format(confidence[i]*100), (20, 30), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255))
        cv2.putText(img, '{}'.format(event_names[i]), (20, 10), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255))
        cv2.imshow(event_names[i], img)
        cv2.imwrite(output_dir+"{}.{}.jpg".format(i,event_names[i]), img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        

    print('Generating demo successful!')
    
    