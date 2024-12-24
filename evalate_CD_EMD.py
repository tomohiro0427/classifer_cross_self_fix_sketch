from pathlib import Path
import numpy as np
import torch
from metrics.evaluation_metrics import EMD_CD
import open3d as o3d

def get_part_point_with_gt(shape_all, shape_part, num_points = 1024):
    B = shape_all.size(0)
    N = shape_all.size(1)
    M = shape_part.size(1)
    extend_all  = shape_all.unsqueeze(2)# [B, N, 3] -> [B, N, 1, 3]
    extend_part  = shape_part.unsqueeze(1)# [B, M, 3] -> [B, 1, M, 3]
    distance_matrix = torch.norm(extend_part - extend_all , p =2 ,dim = -1) # [B, M, N, 3]->[B, N, NM]
    distance_matrix = distance_matrix.permute(0,2,1)
    idx = torch.argsort(distance_matrix,dim=-1, descending=False)
    idx_s = idx.reshape(M,N).T.ravel()
    # unique
    # idx_s = torch.unique(idx_s)[:1024] これは良くない
    unique_tensor = torch.tensor(list(dict.fromkeys(idx_s.tolist())))
    idx_s = unique_tensor[:num_points]

    zero_mask = torch.zeros([B,N])# [B, N]
    zero_mask.scatter_(1, idx_s.unsqueeze(0),1)#k近傍のインデックスを1にする
    mask_point = zero_mask == 1

    output = shape_all[mask_point]
    
    return output.squeeze(0)


result_cd_emd = {
    "03001627": {"cd": [], "emd": []},
    "04379243": {"cd": [], "emd": []},
    "02691156": {"cd": [], "emd": []}
}

dir_path = Path('./quantitative_our/02691156/')

# get gen.npy faile
gen_list_path =  list(dir_path.glob('**/*gen.npy'))
f_list_path =  list(dir_path.glob('**/*f.npy'))

# both list sort
gen_list_path.sort()
f_list_path.sort()

for gen_path, tgt_path in zip(gen_list_path, f_list_path):
    gen = np.load(gen_path)
    tgt = np.load(tgt_path)

    num_sample = 2048 - tgt.shape[0]

    # random smaple points
    gen = gen[np.random.choice(gen.shape[0], num_sample , replace=False)]
    gen = np.concatenate([gen, tgt], axis=0)
    # show points using open3d
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(gen)
    # o3d.visualization.draw_geometries([pcd])


    partial_edit = get_part_point_with_gt(torch.tensor(gen).unsqueeze(0), torch.tensor(tgt).unsqueeze(0), num_points=tgt.shape[0])

    # to calculate CD
    gt = torch.tensor(tgt).unsqueeze(0).float()
    partial = torch.tensor(partial_edit.detach().cpu().numpy()).unsqueeze(0).float()

    result = EMD_CD(gt, partial,1)

    result_cd_emd['02691156']["cd"].append(result['MMD-CD'].item())
    result_cd_emd['02691156']["emd"].append(result['MMD-EMD'].item())

    # break

dir_path = Path('./quantitative_our/03001627/')

# get gen.npy faile
gen_list_path =  list(dir_path.glob('**/*gen.npy'))
f_list_path =  list(dir_path.glob('**/*f.npy'))

# both list sort
gen_list_path.sort()
f_list_path.sort()

for gen_path, tgt_path in zip(gen_list_path, f_list_path):
    gen = np.load(gen_path)
    tgt = np.load(tgt_path)

    num_sample = 2048 - tgt.shape[0]

    # random smaple points
    gen = gen[np.random.choice(gen.shape[0], num_sample , replace=False)]
    gen = np.concatenate([gen, tgt], axis=0)
    # show points using open3d
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(gen)
    # o3d.visualization.draw_geometries([pcd])


    partial_edit = get_part_point_with_gt(torch.tensor(gen).unsqueeze(0), torch.tensor(tgt).unsqueeze(0), num_points=tgt.shape[0])

    # to calculate CD
    gt = torch.tensor(tgt).unsqueeze(0).float()
    partial = torch.tensor(partial_edit.detach().cpu().numpy()).unsqueeze(0).float()

    result = EMD_CD(gt, partial,1)

    result_cd_emd['03001627']["cd"].append(result['MMD-CD'].item())
    result_cd_emd['03001627']["emd"].append(result['MMD-EMD'].item())

    # break

dir_path = Path('./quantitative_our/04379243/')

# get gen.npy faile
gen_list_path =  list(dir_path.glob('**/*gen.npy'))
f_list_path =  list(dir_path.glob('**/*f.npy'))

# both list sort
gen_list_path.sort()
f_list_path.sort()

for gen_path, tgt_path in zip(gen_list_path, f_list_path):
    gen = np.load(gen_path)
    tgt = np.load(tgt_path)

    num_sample = 2048 - tgt.shape[0]

    # random smaple points
    gen = gen[np.random.choice(gen.shape[0], num_sample , replace=False)]
    gen = np.concatenate([gen, tgt], axis=0)
    # show points using open3d
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(gen)
    # o3d.visualization.draw_geometries([pcd])


    partial_edit = get_part_point_with_gt(torch.tensor(gen).unsqueeze(0), torch.tensor(tgt).unsqueeze(0), num_points=tgt.shape[0])

    # to calculate CD
    gt = torch.tensor(tgt).unsqueeze(0).float()
    partial = torch.tensor(partial_edit.detach().cpu().numpy()).unsqueeze(0).float()

    result = EMD_CD(gt, partial,1)

    result_cd_emd['04379243']["cd"].append(result['MMD-CD'].item())
    result_cd_emd['04379243']["emd"].append(result['MMD-EMD'].item())

    # break

for key in result_cd_emd.keys():
    print(key)
    print(f'cd: {np.mean(result_cd_emd[key]["cd"])}')
    print(f'emd: {np.mean(result_cd_emd[key]["emd"])}')