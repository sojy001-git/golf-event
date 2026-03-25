import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
import numpy as np
from utils.util import correct_preds
import glob 
from tqdm import tqdm

'''
0: basic mode(image)

1: norm1+2Dpose
2: norm1+2Dpose+conf 
3: norm2+2Dpose
4: norm2+2Dpose+conf 

5: 1 + liftingNet
6: 2 + liftingNet
7: 3 + liftingNet
8: 4 + liftingNet

9: 3Dpose
'''

training_mode = 5

bs = 12  # batch size       # 22 -> 16 -> 12

if training_mode==0:
    from dataloaders.dataloader import GolfDB, Normalize, ToTensor
    from networks.model import EventDetector
    load_model_root = 'models/image'
    bs = 12  # batch size       # 22 -> 16 -> 12
elif training_mode==1:
    from dataloaders.dataloader_with_pose1 import GolfDB, Normalize, ToTensor
    from networks.model1 import EventDetector
    load_model_root = 'models/pose1'
    pose_dir='data/poses_160/'
    bs = 128  # batch size       # 22 -> 16 -> 12
elif training_mode==2:
    from dataloaders.dataloader_with_pose2 import GolfDB, Normalize, ToTensor
    from networks.model2 import EventDetector
    load_model_root = 'models/pose2'
    pose_dir='data/poses_160/'
    bs = 128  # batch size       # 22 -> 16 -> 12
elif training_mode==3:
    from dataloaders.dataloader_with_pose3 import GolfDB, Normalize, ToTensor
    from networks.model1 import EventDetector
    load_model_root = 'models/pose3'
    pose_dir='data/poses_160/'
    bs = 128  # batch size       # 22 -> 16 -> 12
elif training_mode==4:
    from dataloaders.dataloader_with_pose4 import GolfDB, Normalize, ToTensor
    from networks.model2 import EventDetector
    load_model_root = 'models/pose4'
    pose_dir='data/poses_160/'
    bs = 128  
elif training_mode==5:
    from dataloaders.dataloader_with_pose1 import GolfDB, Normalize, ToTensor
    from networks.model1_1 import EventDetector
    load_model_root = 'models/pose1_2'
    pose_dir='data/poses_160/'
    bs = 128  
elif training_mode==6:
    from dataloaders.dataloader_with_pose2 import GolfDB, Normalize, ToTensor
    from networks.model2_1 import EventDetector
    load_model_root = 'models/pose2_2'
    pose_dir='data/poses_160/'
    bs = 128  
elif training_mode==7:
    from dataloaders.dataloader_with_pose3 import GolfDB, Normalize, ToTensor
    from networks.model1_1 import EventDetector
    load_model_root = 'models/pose3_2'
    pose_dir='data/poses_160/'
    bs = 128  
elif training_mode==8:
    from dataloaders.dataloader_with_pose4 import GolfDB, Normalize, ToTensor
    from networks.model2_1 import EventDetector
    load_model_root = 'models/pose4_1'
    pose_dir='data/poses_160/'
    bs = 128  
else: 
    from dataloaders.dataloader_with_3Dpose import GolfDB, Normalize, ToTensor
    from networks.model3d import EventDetector
    pose_dir='data/poses3D_160/'
    load_model_root = 'models/pose_d3'
    bs = 128  


def eval(model, split, seq_length, n_cpu, disp):
    if training_mode==0:
        dataset = GolfDB(data_file='data/val_split_{}.pkl'.format(split),
                        vid_dir='data/videos_160/',
                        seq_length=seq_length,
                        transform=transforms.Compose([ToTensor(),
                                                    Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
                        train=False)
    else:
        dataset = GolfDB(data_file='data/val_split_{}.pkl'.format(split),
                        vid_dir='data/videos_160/',
                        pose_dir=pose_dir,
                        seq_length=seq_length,
                        transform=transforms.Compose([ToTensor(),
                                                    Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
                        train=False)

    data_loader = DataLoader(dataset,
                             batch_size=1,
                             shuffle=False,
                             num_workers=n_cpu,
                             drop_last=False)

    correct = []

    for i, sample in enumerate(tqdm(data_loader, 0)):
        inputs, labels = sample['inputs'], sample['labels']
        # full samples do not fit into GPU memory so evaluate sample in 'seq_length' batches
        batch = 0
        while batch * seq_length < inputs.shape[1]:
            if (batch + 1) * seq_length > inputs.shape[1]:
                if training_mode==0:
                    input_batch = inputs[:, batch * seq_length:, :, :, :]
                else:
                    input_batch = inputs[:, batch * seq_length:, :]
            else:
                if training_mode==0:
                    input_batch = inputs[:, batch * seq_length:(batch + 1) * seq_length, :, :, :]
                else:
                    input_batch = inputs[:, batch * seq_length:(batch + 1) * seq_length, :]
                
            logits = model(input_batch.cuda())
            if batch == 0:
                probs = F.softmax(logits.data, dim=1).cpu().numpy()
            else:
                probs = np.append(probs, F.softmax(logits.data, dim=1).cpu().numpy(), 0)
            batch += 1
        _, _, _, _, c = correct_preds(probs, labels.squeeze())
        if disp:
            print(i, c)
        correct.append(c)
    PCE = np.mean(correct)
    return PCE


if __name__ == '__main__':

    splits = [1,2,3,4]
    seq_length = 64
    n_cpu = 8

    model = EventDetector(pretrain=True,
                          width_mult=1.,
                          lstm_layers=1,
                          lstm_hidden=256,
                          bidirectional=True,
                          dropout=False)

    load_model_name = sorted(glob.glob(load_model_root +'/*.tar'))[-1]
    save_dict = torch.load(load_model_name)
    model.load_state_dict(save_dict['model_state_dict'])
    model.cuda()
    model.eval()
    PCEs = []
    for split in splits:
        PCE = eval(model, split, seq_length, n_cpu, False)   # True 
        PCEs.append(PCE)
    for i in range(len(PCEs)):
        print('Dataset Split {} => Average PCE: {}'.format(splits[i], PCEs[i]))
    
    print('Total Average PCEs: {:.4f}'.format(sum(PCEs)/len(PCEs)))


