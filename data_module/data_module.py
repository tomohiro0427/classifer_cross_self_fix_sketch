import os, logging
import random
import numpy as np

import torch
from torch.utils import data
from PIL import Image
from torchvision import transforms
from pathlib import Path
import pandas as pd
import cv2
import math
from data_module import point_operation



logger = logging.getLogger(__name__)

# taken from https://github.com/optas/latent_3d_points/blob/8e8f29f8124ed5fc59439e8551ba7ef7567c9a37/src/in_out.py
synsetid_to_cate = {
    '02691156': 'airplane', '02773838': 'bag', '02801938': 'basket',
    '02808440': 'bathtub', '02818832': 'bed', '02828884': 'bench',
    '02876657': 'bottle', '02880940': 'bowl', '02924116': 'bus',
    '02933112': 'cabinet', '02747177': 'can', '02942699': 'camera',
    '02954340': 'cap', '02958343': 'car', '03001627': 'chair',
    '03046257': 'clock', '03207941': 'dishwasher', '03211117': 'monitor',
    '04379243': 'table', '04401088': 'telephone', '02946921': 'tin_can',
    '04460130': 'tower', '04468005': 'train', '03085013': 'keyboard',
    '03261776': 'earphone', '03325088': 'faucet', '03337140': 'file',
    '03467517': 'guitar', '03513137': 'helmet', '03593526': 'jar',
    '03624134': 'knife', '03636649': 'lamp', '03642806': 'laptop',
    '03691459': 'speaker', '03710193': 'mailbox', '03759954': 'microphone',
    '03761084': 'microwave', '03790512': 'motorcycle', '03797390': 'mug',
    '03928116': 'piano', '03938244': 'pillow', '03948459': 'pistol',
    '03991062': 'pot', '04004475': 'printer', '04074963': 'remote_control',
    '04090263': 'rifle', '04099429': 'rocket', '04225987': 'skateboard',
    '04256520': 'sofa', '04330267': 'stove', '04530566': 'vessel',
    '04554684': 'washer', '02992529': 'cellphone',
    '02843684': 'birdhouse', '02871439': 'bookshelf',
    'all': 'all'
    # '02858304': 'boat', no boat in our dataset, merged into vessels
    # '02834778': 'bicycle', not in our taxonomy
}
cate_to_synsetid = {v: k for k, v in synsetid_to_cate.items()}

def get_class_id(class_name):
    return cate_to_synsetid[class_name]

def init_np_seed(worker_id):
    seed = torch.initial_seed()
    np.random.seed(seed % 4294967296)


class PointCloudPartSegWhithSketch(data.Dataset):
    def __init__(self, root, split = 'val', categories = ['chair'], get_images = ['shape_down']):
        if categories not in ['all']:
            synset_ids = [cate_to_synsetid[c] for c in categories]
            self.root_dir = Path(root, split, synset_ids[0] )
        else:
            #作り変える
            self.root_dir = Path(root)
        print(self.root_dir)

        self.input_dim = 3
        self.all_points = []
        self.all_labels = []
        self.pc_paths = self.find_npy_files()
        # Normalization
        self.all_points = np.concatenate(self.all_points)  # (N, 15000, 3)
        self.all_labels = np.concatenate(self.all_labels)  # (N, 15000)
        self.all_points, [self.per_points_shift, self.per_points_scale] = point_operation.normalize_point_cloud(self.all_points, verbose=True)

        self.data_paths = self.find_datas_dir()
        print(self.__len__())


    
    def find_npy_files(self):
        npy_paths = []
        npy_files = [npy_file for npy_file in self.root_dir.glob('**/*.npy') if any(npy_file.parent.iterdir()) and any(item.is_dir() for item in npy_file.parent.iterdir())]
        # for path_pc in self.root_dir.glob('**/*.npy'):
        for path_pc in npy_files:
            npy_paths.append(path_pc.stem)
            point_cloud = np.load(path_pc)
            self.all_points.append(point_cloud[np.newaxis, ...])

            parquet_data = pd.read_parquet(Path(path_pc.parent, f'{path_pc.stem}.parquet'))
            label_data =  parquet_data['label'].values

            self.all_labels.append(label_data[np.newaxis, ...])
        return npy_paths
    
    def find_datas_dir(self):
        dir_data_path = []
        for path in self.root_dir.rglob('edit_sketch.png'):
            dir_data_path.append(path.parent)
        if len(dir_data_path) == 0:
            print('No edit_sketch.png file found')
        return dir_data_path
    
    def get_image_data(self, path, name='edit_sketch.png'):
        
        image_path = Path(path, name)
        if not image_path.exists():
            print(f'File not found: {image_path}')
        else:
            image = cv2.imread(str(image_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            image = transforms.Resize((224, 224))(image)
            # zoom_range = 0.3
            # zoom_factor = random.uniform(1.0 - zoom_range, 1.0 + zoom_range)
            # if zoom_factor > 1:
            #     size = int(224*zoom_factor)
            #     image = image.resize((size, size))
            # else:
            #     size = int(224*zoom_factor)
            #     image = image.resize((size, size))
            #     image = transforms.Pad(padding=224-size, padding_mode="constant", fill=(255, 255, 255))(image)
            # image = transforms.CenterCrop((224, 224))(image)
            # # image = transforms.RandomAffine(degrees=(-0, 0),translate=(0.2, 0.1), scale=(1, 1), shear=0, fill=(255, 255, 255))(image)
            # image = transforms.functional.affine(image, angle=0, translate=(-20, -20),
            #                                scale=1, shear=0, fill=(255, 255, 255))
            image = transforms.ToTensor()(image)

        return image
    
    def get_masked_partPC(self, points_all,label,  edit_label,num_point = 2048):
        N, dim = points_all.shape
        E = num_point
        edit_msked_pc_datas = torch.empty(E, dim) # [B,E,3]
        fix_msked_pc_datas = torch.empty(E, dim) # [B,E,3]

        mask = torch.eq(edit_label, label)# [N] bool
        masked_points = points_all[mask]# [N,3] -> [E :3]
        masked_points = masked_points[torch.randperm(masked_points.shape[0])]
        # E以下なら　upsample それ以外はdownsample
        if E < masked_points.shape[0]:
            edit_msked_pc_datas = masked_points[:E,:] # fpsでサンプル
        else :
            masked_points = torch.cat([masked_points]*math.ceil(E/masked_points.shape[0]), dim=0)# [M,3] -> [E+A,3]
            edit_msked_pc_datas = masked_points[:E,:]# [1,E+A,3] -> [E,3]

        mask = ~mask
        masked_points = points_all[mask]# [N,3] -> [E :3]
        masked_points = masked_points[torch.randperm(masked_points.shape[0])]
        # E以下なら　upsample それ以外はdownsample
        if E < masked_points.shape[0]:
            fix_msked_pc_datas = masked_points[:E,:] # fpsでサンプル
        else :
            masked_points = torch.cat([masked_points]*math.ceil(E/masked_points.shape[0]), dim=0)# [M,3] -> [E+A,3]
            fix_msked_pc_datas = masked_points[:E,:]# [1,E+A,3] -> [E,3]

        return  edit_msked_pc_datas , fix_msked_pc_datas# [E,3]
    
  
    def get_standardize_stats(self, idx):
        shift = self.per_points_shift[idx].reshape(1, self.input_dim)
        scale = self.per_points_scale[idx].reshape(1, -1)
        return shift, scale  


    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        data_path = self.data_paths[idx]
        edit_sketch = self.get_image_data(data_path, name = 'edit_sketch.png')

        # point cloudのindexを取得
        index_pc = self.pc_paths.index(data_path.parents[1].stem)
        point_cloud = torch.tensor(self.all_points[index_pc])
        label = torch.tensor(self.all_labels[index_pc])
        #
        edit_label = int(data_path.stem.split('_')[-1])
        edit_label = torch.tensor(edit_label)
        edit_points,fix_points = self.get_masked_partPC(point_cloud,label,  edit_label = edit_label, num_point = 2048)

        fix_sample_path = Path(data_path, 'fix_sample.npy')
        if os.path.exists(fix_sample_path):
            fix_points = torch.from_numpy(np.load(fix_sample_path))
            ## normalization
            fix_points = (fix_points - self.per_points_shift[index_pc]) / self.per_points_scale[index_pc]
            print("fix_sample.npy is loaded")


        # 正規化の値を取得
        shift, scale = self.get_standardize_stats(index_pc)
        shift, scale = torch.from_numpy(np.asarray(shift)), torch.from_numpy(np.asarray(scale))

        # return辞書に他の固定的な値を追加
        return {
            'point_cloud': point_cloud,
            'edit_points': edit_points,
            'fix_points': fix_points,
            'edit_sketch': edit_sketch,
            'idx'   : idx,
            'label': label,
            'path': str(data_path),
            'shift': shift,
            'scale': scale,
            'name': data_path.parents[1].stem,
        }
    
class RandamPartialPointCloudWhithSketch(data.Dataset):
    def __init__(self, root, split = 'val', categories = ['chair'], get_images = ['shape_down']):
        if categories not in ['all']:
            synset_ids = [cate_to_synsetid[c] for c in categories]
            self.root_dir = Path(root, split, synset_ids[0] )
        else:
            #作り変える
            self.root_dir = Path(root)
        print(self.root_dir)

        self.input_dim = 3
        self.all_points = []
        # self.all_labels = []
        self.pc_paths = self.find_npy_files()
        # Normalization
        self.all_points = np.concatenate(self.all_points)  # (N, 15000, 3)
        # self.all_labels = np.concatenate(self.all_labels)  # (N, 15000)
        self.all_points, [self.per_points_shift, self.per_points_scale] = point_operation.normalize_point_cloud(self.all_points, verbose=True)

        self.data_paths = self.find_datas_dir()
        print(self.__len__())


    
    def find_npy_files(self):
        npy_paths = []
        for path_pc in self.root_dir.glob('**/*.npy'):
            npy_paths.append(path_pc.stem)
            point_cloud = np.load(path_pc)
            self.all_points.append(point_cloud[np.newaxis, ...])

            #同じディレクトリのparquetファイルを探す
            # parqeest_path = Path(path_pc.parent)
            # find_parquet = parqeest_path.glob('*.parquet')

            # parquet_data = pd.read_parquet(Path(path_pc.parent, f'{find_parquet[0]}'))
            # label_data =  parquet_data['label'].values

            # self.all_labels.append(label_data[np.newaxis, ...])
        return npy_paths
    
    def find_datas_dir(self):
        dir_data_path = []
        for path in self.root_dir.rglob('edit_sketch.png'):
            dir_data_path.append(path.parent)
        if len(dir_data_path) == 0:
            print('No edit_sketch.png file found')
        return dir_data_path
    
    def get_image_data(self, path, name='edit_sketch.png'):
        
        image_path = Path(path, name)
        if not image_path.exists():
            print(f'File not found: {image_path}')
        else:
            image = cv2.imread(str(image_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            image = transforms.Resize((224, 224))(image)
            # zoom_range = 0.3
            # zoom_factor = random.uniform(1.0 - zoom_range, 1.0 + zoom_range)
            # if zoom_factor > 1:
            #     size = int(224*zoom_factor)
            #     image = image.resize((size, size))
            # else:
            #     size = int(224*zoom_factor)
            #     image = image.resize((size, size))
            #     image = transforms.Pad(padding=224-size, padding_mode="constant", fill=(255, 255, 255))(image)
            # image = transforms.CenterCrop((224, 224))(image)
            # image = transforms.RandomAffine(degrees=(-0, 0),translate=(0.2, 0.1), scale=(1, 1), shear=0, fill=(255, 255, 255))(image)
            # image = transforms.functional.affine(image, angle=0, translate=(-20, -20),
            #                                scale=1, shear=0, fill=(255, 255, 255))
            image = transforms.ToTensor()(image)

        return image
    
    def get_masked_partPC(self, points_all,label,  edit_label,num_point = 2048):
        N, dim = points_all.shape
        E = num_point
        edit_msked_pc_datas = torch.empty(E, dim) # [B,E,3]
        fix_msked_pc_datas = torch.empty(E, dim) # [B,E,3]

        mask = torch.eq(edit_label, label)# [N] bool
        masked_points = points_all[mask]# [N,3] -> [E :3]
        masked_points = masked_points[torch.randperm(masked_points.shape[0])]
        # E以下なら　upsample それ以外はdownsample
        if E < masked_points.shape[0]:
            edit_msked_pc_datas = masked_points[:E,:] # fpsでサンプル
        else :
            masked_points = torch.cat([masked_points]*math.ceil(E/masked_points.shape[0]), dim=0)# [M,3] -> [E+A,3]
            edit_msked_pc_datas = masked_points[:E,:]# [1,E+A,3] -> [E,3]

        mask = ~mask
        masked_points = points_all[mask]# [N,3] -> [E :3]
        masked_points = masked_points[torch.randperm(masked_points.shape[0])]
        # E以下なら　upsample それ以外はdownsample
        if E < masked_points.shape[0]:
            fix_msked_pc_datas = masked_points[:E,:] # fpsでサンプル
        else :
            masked_points = torch.cat([masked_points]*math.ceil(E/masked_points.shape[0]), dim=0)# [M,3] -> [E+A,3]
            fix_msked_pc_datas = masked_points[:E,:]# [1,E+A,3] -> [E,3]

        return  edit_msked_pc_datas , fix_msked_pc_datas# [E,3]
    
    def get_sampled_points(self, points_all, num_point = 2048):
        N, dim = points_all.shape
        E = num_point
        randam_points = points_all[torch.randperm(points_all.shape[0])]
        sampled_points = randam_points[:E,:]# [1,E+A,3] -> [E,3]

        return sampled_points
        
    
  
    def get_standardize_stats(self, idx):
        shift = self.per_points_shift[idx].reshape(1, self.input_dim)
        scale = self.per_points_scale[idx].reshape(1, -1)
        return shift, scale  


    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        data_path = self.data_paths[idx]
        edit_sketch = self.get_image_data(data_path, name = 'edit_sketch.png')
        all_sketch = self.get_image_data(data_path.parent, name = 'all_sketch.png')

        #labelデータの取得
        parqeest_path = Path(data_path)
        parquet_list = [file for file in parqeest_path.glob("*.parquet")]
        parquet_data = pd.read_parquet( f'{parquet_list[0]}')
        # print(parquet_data)
        label =  torch.tensor(parquet_data['label'].values).int()


        # # point cloudのindexを取得
        index_pc = self.pc_paths.index(data_path.parents[1].stem)
        point_cloud = torch.tensor(self.all_points[index_pc])
        ## point_cloud randmu sampling
        sample_point_cloud = self.get_sampled_points(point_cloud, num_point = 2048)

 

        edit_label = torch.tensor(0)
        edit_points,fix_points = self.get_masked_partPC(point_cloud,label,  edit_label = edit_label, num_point = 2048)

        #view indexの取得
        view_index = torch.tensor(int(data_path.parents[0].stem))

        # 正規化の値を取得
        shift, scale = self.get_standardize_stats(index_pc)
        shift, scale = torch.from_numpy(np.asarray(shift)), torch.from_numpy(np.asarray(scale))

        # return辞書に他の固定的な値を追加
        return {
            'point_cloud': point_cloud,
            'sample_point_cloud': sample_point_cloud,
            'edit_points': edit_points,
            'fix_points': fix_points,
            'edit_sketch': edit_sketch,
            'all_sketch': all_sketch,
            'view_index': view_index,
            'idx'   : idx,
            'label': label,
            'path': str(data_path),
            'shift': shift,
            'scale': scale,
            'name': data_path.parents[1].stem,
        }
    


class Sketch_our(data.Dataset):
    def __init__(self, root):
        self.root_dir = Path(root)
        self.data_paths = self.find_datas_dir()
        self.input_dim = 3
        print(self.__len__())
    
    def find_datas_dir(self):
        dir_data_path = []
        for path in self.root_dir.rglob('**/edit_sketch.png'):
            dir_data_path.append(path)
        if len(dir_data_path) == 0:
            print('No edit_sketch.png file found')
        return dir_data_path
    
    def get_image_data(self, path, name=None):
        if name is None:
            image_path = Path(path)
        else: 
            image_path = Path(path, name)
        if not image_path.exists():
            print(f'File not found: {image_path}')
        else:
            image = cv2.imread(str(image_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            image = transforms.Resize((224, 224))(image)
            image = transforms.ToTensor()(image)

        return image
    

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        data_path = self.data_paths[idx]
        edit_sketch = self.get_image_data(data_path)

        # name = int(data_path.stem.split('-')[0])
        # pkl_path = Path(data_path.parents[2],'chairs_sym_hard', f'{name:04d}.pkl')
        # with open(pkl_path, 'rb') as f:
        #     data = pickle.load(f)
        # points = np.array(np.vstack(data['points']))
        # points = points[np.random.choice(points.shape[0], 2048, replace=False)]

        points = np.load(Path(data_path.parent.parent, '0_gen.npy'))
        parquet_data = pd.read_parquet(Path(data_path.parent, f'label.parquet'))
        label =  parquet_data['label'].values
        # label =  parquet_data['fix_bool'].values
        label = label[np.newaxis, ...]
        
        point_cloud, [shift, scale] = point_operation.normalize_point_cloud(points[np.newaxis, ...], verbose=True)
        fix_points = torch.tensor(point_cloud[label]).squeeze(0)
        # fix_points = torch.cat([fix_points]*2, dim=0)[2048:,:].squeeze(0)
        # fix_points = fix_points[:2048,:]

        point_cloud = torch.tensor(point_cloud).squeeze(0)
        shift, scale = torch.from_numpy(np.asarray(shift.reshape(self.input_dim))), torch.from_numpy(np.asarray(scale.reshape(1)))

        # return辞書に他の固定的な値を追加
        return {
            # 'point_cloud': point_cloud,
            'sample_points': point_cloud,
            # 'all_sketch': all_sketch,
            'edit_sketch': edit_sketch,
            'fix_points': fix_points,
            'idx'   : idx,
            'path': str(data_path),
            'shift': shift,
            'scale': scale,
            'name': data_path.stem,
            'path': str(data_path.parent),
        }
    
class add_Sketch_our(data.Dataset):
    def __init__(self, root):
        self.root_dir = Path(root)
        self.data_paths = self.find_datas_dir()
        self.input_dim = 3
        print(self.__len__())
    
    def find_datas_dir(self):
        dir_data_path = []
        for path in self.root_dir.rglob('**/edit_sketch.png'):
            dir_data_path.append(path)
        if len(dir_data_path) == 0:
            print('No edit_sketch.png file found')
        return dir_data_path
    
    def get_image_data(self, path, name=None):
        if name is None:
            image_path = Path(path)
        else: 
            image_path = Path(path, name)
        if not image_path.exists():
            print(f'File not found: {image_path}')
        else:
            image = cv2.imread(str(image_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            image = transforms.Resize((224, 224))(image)
            image = transforms.ToTensor()(image)

        return image
    

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        data_path = self.data_paths[idx]
        edit_sketch = self.get_image_data(data_path)

        # name = int(data_path.stem.split('-')[0])
        # pkl_path = Path(data_path.parents[2],'chairs_sym_hard', f'{name:04d}.pkl')
        # with open(pkl_path, 'rb') as f:
        #     data = pickle.load(f)
        # points = np.array(np.vstack(data['points']))
        # points = points[np.random.choice(points.shape[0], 2048, replace=False)]

        points = np.load(Path(data_path.parent, 'origin.npy'))
        # parquet_data = pd.read_parquet(Path(data_path.parent, f'label.parquet'))
        # label =  parquet_data['label'].values
        # label = label[np.newaxis, ...]
        points = points[:2048,:]
        
        point_cloud, [shift, scale] = point_operation.normalize_point_cloud(points[np.newaxis, ...], verbose=True)
        # fix_points = torch.tensor(point_cloud[label]).squeeze(0)
        # fix_points = torch.cat([fix_points]*2, dim=0)[2048:,:].squeeze(0)
        

        point_cloud = torch.tensor(point_cloud).squeeze(0)
        shift, scale = torch.from_numpy(np.asarray(shift.reshape(self.input_dim))), torch.from_numpy(np.asarray(scale.reshape(1)))

        # return辞書に他の固定的な値を追加
        return {
            # 'point_cloud': point_cloud,
            'sample_points': point_cloud,
            # 'all_sketch': all_sketch,
            'edit_sketch': edit_sketch,
            # 'fix_points': fix_points,
            'idx'   : idx,
            'path': str(data_path),
            'shift': shift,
            'scale': scale,
            'name': data_path.stem,
            'path': str(data_path.parent),
        }