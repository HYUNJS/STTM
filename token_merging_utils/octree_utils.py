import sys
import os
## Get the absolute path of the project directory
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
## Add the project directory to sys.path
sys.path.append(project_dir)

import torch
import torch.nn.functional as F
import math
import pickle
import einops

from token_merging_utils.quadtree_builder import quadtree_build_video


def avgpool_to_even_side_feature_tyx(video_feature):
    stride = 2
    B, C, ori_t, ori_h, ori_w = video_feature.shape
    new_t, new_h, new_w = math.ceil(ori_t/2), math.ceil(ori_h/2), math.ceil(ori_w/2)
    device = video_feature.device

    if (ori_t % stride == 0) and (ori_h % stride == 0) and (ori_w % stride == 0): # even size
        video_feature_next_level = F.avg_pool3d(video_feature, (stride, stride, stride))
    else: # odd size
        ## Allocate an empty tensor for the pooled features.
        video_feature_next_level = torch.empty((B, C, new_t, new_h, new_w), device=device, dtype=video_feature.dtype)

        ## (1). First-top–left element: [[[0,0,0]]] -> [[[0, 0, 0]]]
        """
        [        
            [
                [[0,0,0]]
            ]
        ]
        -> [ [ [ [0,0,0] ] ] ]
        """
        ftls = video_feature[:, :, 0, 0, 0]
        video_feature_next_level[:, :, 0, 0, 0] = ftls

        ## (2). First-top–right element: [[[0,0,1], [0,0,2]]] -> [[[0,0,1]]]
        """
        [        
            [
                [[0,0,1], [0,0,2]]
            ]
        ]
        -> [ [ [ [0,0,1] ] ] ]
        """
        ftrs = video_feature[:, :, 0, 0, 1:]
        ftrs = ftrs.reshape(B, C, new_w-1, stride).mean(dim=-1)
        video_feature_next_level[:, :, 0, 0, 1:] = ftrs

        ## (3). First-bottom-left element:
        """
        [        
            [
                [[0,1,0]],
                [[0,2,0]]
            ]
        ]
        -> [ [ [ [0,1,0] ] ] ]
        """
        fbls = video_feature[:, :, 0, 1:, 0]
        fbls = fbls.reshape(B, C, new_h-1, stride).mean(dim=-1)
        video_feature_next_level[:, :, 0, 1:, 0] = fbls

        ## (4). First-bottom-right element:
        """
        [        
            [
                [[0,1,1], [0,1,2]],
                [[0,2,1], [0,2,2]]
            ]
        ]
        -> [ [ [ [0,1,1] ] ] ]
        """
        fbrs = F.avg_pool2d(video_feature[:, :, 0, 1:, 1:], kernel_size=(stride, stride))
        video_feature_next_level[:, :, 0, 1:, 1:] = fbrs

        ## (5). Last-top–left element:
        """
        [        
            [
                [[1,0,0]]
            ],
            [
                [[2,0,0]]
            ]
        ]
        -> [ [ [ [1,0,0] ] ] ]
        """
        ltls = video_feature[:, :, 1:, 0, 0]
        ltls = ltls.reshape(B, C, new_t-1, stride).mean(dim=-1)
        video_feature_next_level[:, :, 1:, 0, 0] = ltls

        ## (6). Last-top–right element:
        """
        [        
            [
                [[1,0,1], [1,0,2]]
            ],
            [
                [[2,0,1], [2,0,2]]
            ]
        ]
        -> [ [ [ [1,0,1] ] ] ]
        """
        ltrs = video_feature[:, :, 1:, 0, 1:]
        ltrs = F.avg_pool2d(ltrs, kernel_size=(stride, stride))
        video_feature_next_level[:, :, 1:, 0, 1:] = ltrs

        ## (7). Last-bottom-left element:
        """
        [        
            [
                [[1,1,0]],
                [[1,2,0]]
            ],
            [
                [[2,1,0]],
                [[2,2,1]]
            ]
        ]
        -> [ [ [ [1,1,0] ] ] ]
        """
        lbls = video_feature[:, :, 1:, 1:, 0]
        lbls = F.avg_pool2d(lbls, kernel_size=(2, 2))
        video_feature_next_level[:, :, 1:, 1:, 0] = lbls

        ## (8). Last-bottom-right element:
        """
        [        
            [
                [[1,1,1], [1,1,2]],
                [[1,2,1], [1,2,2]]
            ],
            [
                [[2,1,1], [2,1,2]],
                [[2,2,1], [2,2,2]]
            ]
        ]
        -> [ [ [ [1,1,1] ] ] ]
        """
        lbrs = F.avg_pool3d(video_feature[:, :, 1:, 1:, 1:], kernel_size=(stride, stride, stride))
        video_feature_next_level[:, :, 1:, 1:, 1:] = lbrs

    return video_feature_next_level

def avgpool_to_even_side_index_tyx(video_feature, prev_index_1d):
    ## Generate matched indices across feature map levels (for all frames)
    ## For this version, we do not pool across temporal axis. 
    stride = 2
    B, ori_t, ori_h, ori_w, _ = video_feature.shape
    device = video_feature.device
    new_t, new_h, new_w = math.ceil(ori_t/2), math.ceil(ori_h/2), math.ceil(ori_w/2)

    if (ori_t % stride == 0) and (ori_h % stride == 0) and (ori_w % stride == 0): # even size
        ## Build a grid of “first-top–left” indices for each 2x2×2 block:
        grid_b = torch.arange(0, B, device=device, dtype=torch.int32)
        grid_t = torch.arange(0, new_t, device=device, dtype=torch.int32) * stride
        grid_y = torch.arange(0, new_h, device=device, dtype=torch.int32) * stride
        grid_x = torch.arange(0, new_w, device=device, dtype=torch.int32) * stride

        ## Meshgrid: each output cell (i,j,k) will have first-top–left index (gt, gy, gx)
        gb, gt, gy, gx = torch.meshgrid(grid_b, grid_t, grid_y, grid_x, indexing='ij') # [B, new_t, new_h, new_w]
        first_top_left = torch.stack([gb, gt, gy, gx], dim=-1).unsqueeze(-2) # [B, new_t, new_h, new_w, 1, 4]

        ## Create a tensor of the fixed BxTxH×W offsets. B-axis offset is 0, restricting the octree scope within the same video snippet
        offsets = torch.tensor([[[0,0,0,0], [0,0,0,1], [0,0,1,0], [0,0,1,1], [0,1,0,0], [0,1,0,1], [0,1,1,0], [0,1,1,1]]], device=device, dtype=torch.int32)
        offsets = offsets.view(1, 1, 1, 8, 4)

        ## The child coordinates are just the first-top–left coordinates plus the offsets.
        child_btyx_coords = first_top_left + offsets # [B, new_t new_h, new_w, 8, 4]

        ## The children node's 1D index
        child_index_1d = prev_index_1d[:, 0::2, 0::2, 0::2] # [B, new_t, new_h, new_w]

        ## Generate Odd or Even mask
        child_valid_mask = torch.ones((B, new_t, new_h, new_w, 8), device=device, dtype=torch.bool)
    else: # odd size
        child_tyx_coords = torch.zeros((new_t, new_h, new_w, 8, 3), device=device, dtype=torch.int32) # will expand B-axis at the end
        child_index_1d = torch.zeros((B, new_t, new_h, new_w), device=device, dtype=torch.int32)
        child_valid_mask = torch.zeros((new_t, new_h, new_w, 8), device=device, dtype=torch.bool) # will expand B-axis at the end
        t_idx = torch.arange(0, new_t-1, device=device, dtype=torch.int32)
        h_idx = torch.arange(0, new_h-1, device=device, dtype=torch.int32)
        w_idx = torch.arange(0, new_w-1, device=device, dtype=torch.int32)
        t_start = 1 + 2 * t_idx
        h_start = 1 + 2 * h_idx
        w_start = 1 + 2 * w_idx

        ## (1). First-top-left element (t=0, h=0, w=0): 1x1x1 block
        child_index_1d[:, 0, 0, 0] = prev_index_1d[:, 0, 0, 0]
        child_valid_mask[0, 0, 0, 0] = True 
        
        ## (2). First-top-right element (t=0, h=0, w>=1): 1x1x2 block
        ftr_index = [0,1]
        child_tyx_coords[0, 0, 1:, ftr_index, 2] = torch.stack([w_start, w_start+1], dim=-1)
        child_index_1d[:, 0, 0, 1:] = prev_index_1d[:, 0, 0, 1::2]
        child_valid_mask[0, 0, 1:, ftr_index] = True

        ## (3). First-bottom-left (t=0, h>=1, w=0): 1x2x1 block
        fbl_index = [0,1]
        child_tyx_coords[0, 1:, 0, fbl_index, 1] = torch.stack([h_start, h_start+1], dim=-1)
        child_index_1d[:, 0, 1:, 0] = prev_index_1d[:, 0, 1::2, 0]
        child_valid_mask[0, 1:, 0, fbl_index] = True

        ## (4). First-bottom-right cells (t=0, h>=1, w>=1): 1x2x2 block
        fbr_index = [0,1,2,3]
        child_tyx_coords[0, 1:, 1:, fbr_index, 1] = torch.stack([*([h_start]*2), *([h_start+1]*2)], dim=-1).unsqueeze(1).repeat(1, new_w-1, 1)
        child_tyx_coords[0, 1:, 1:, fbr_index, 2] = torch.stack([w_start, w_start+1], dim=-1).repeat(1, 2).unsqueeze(0).repeat(new_h-1, 1, 1)
        child_index_1d[:, 0, 1:, 1:] = prev_index_1d[:, 0, 1::2, 1::2]
        child_valid_mask[0, 1:, 1:, fbr_index] = True
        
        ## (5). Last-top-left cells (t>=1, h=0, w=0): 2x1x1 block
        ltl_index = [0,4]
        child_tyx_coords[1:, 0, 0, ltl_index, 0] = torch.stack([t_start, t_start+1], dim=-1)
        child_index_1d[:, 1:, 0, 0] = prev_index_1d[:, 1::2, 0, 0]
        child_valid_mask[1:, 0, 0, ltl_index] = True

        ## (6). Last-top-right cells (t>=1, h=0, w>=1): 2x1x2 block
        ltr_index = [0,1,4,5]
        child_tyx_coords[1:, 0, 1:, ltr_index, 0] = torch.stack([*([t_start]*2),*([t_start+1]*2)], dim=-1).unsqueeze(1).repeat(1, new_w-1, 1)
        child_tyx_coords[1:, 0, 1:, ltr_index, 2] = torch.stack([w_start, w_start+1], dim=-1).repeat(1, 2).unsqueeze(0).repeat(new_t-1, 1, 1)
        child_index_1d[:, 1:, 0, 1:] = prev_index_1d[:, 1::2, 0, 1::2]
        child_valid_mask[1:, 0, 1:, ltr_index] = True

        ## (7). Last-bottom-left cells (t>=1, h>=1, w=0): 2x2x1 block
        lbl_index = [0,2,4,6]
        child_tyx_coords[1:, 1:, 0, lbl_index, 0] = torch.stack([*([t_start]*2),*([t_start+1]*2)], dim=-1).unsqueeze(1).repeat(1, new_h-1, 1)
        child_tyx_coords[1:, 1:, 0, lbl_index, 1] = torch.stack([h_start, h_start+1], dim=-1).repeat(1, 2).unsqueeze(0).repeat(new_t-1, 1, 1)
        child_index_1d[:, 1:, 1:, 0] = prev_index_1d[:, 1::2, 1::2, 0]
        child_valid_mask[1:, 1:, 0, lbl_index] = True

        ## (8). Last-bottom-right cells (t>=1, h>=1, w>=1): 2x2x2 block
        lbr_index = [0,1,2,3,4,5,6,7]
        child_tyx_coords[1:, 1:, 1:, lbr_index, 0] = torch.stack([*([t_start]*4), *([t_start+1]*4)], dim=-1).unsqueeze(1).unsqueeze(2).repeat(1, new_h-1, new_w-1, 1)
        child_tyx_coords[1:, 1:, 1:, lbr_index, 1] = torch.stack([*([h_start]*2),*([h_start+1]*2)], dim=-1).repeat(1, 2).unsqueeze(0).unsqueeze(2).repeat(new_t-1, 1, new_w-1, 1)
        child_tyx_coords[1:, 1:, 1:, lbr_index, 2] = torch.stack([w_start, w_start+1], dim=-1).repeat(1, 4).unsqueeze(0).unsqueeze(1).repeat(new_t-1, new_h-1, 1, 1)
        child_index_1d[:, 1:, 1:, 1:] = prev_index_1d[:, 1::2, 1::2, 1::2]
        child_valid_mask[1:, 1:, 1:, lbr_index] = True

        ## Last. Expand batch dimension
        b_idx = torch.arange(0, B, device=device, dtype=torch.int32)
        b_coords = b_idx.reshape(-1, 1, 1, 1, 1, 1).repeat(1, new_t, new_h, new_w, 8, 1)
        _child_tyx_coords = child_tyx_coords.unsqueeze(0).repeat(B, 1, 1, 1, 1, 1) # [B, new_t, new_h, new_w, 8, 3]
        child_btyx_coords = torch.cat([b_coords, _child_tyx_coords], dim=-1)
        child_valid_mask = child_valid_mask.unsqueeze(0).repeat(B, 1, 1, 1, 1)
    
    return child_btyx_coords, child_valid_mask, child_index_1d

def octree_build_iteration(parent_btyx_coords_4d,
                             video_features_per_level, node_index_1d_per_level,
                             child_btyx_coords_per_level, child_valid_mask_per_level,
                             octree_features_list, octree_index_1d_list,
                             curr_lvl, n_level, threshold):
    
    ## Step1: indexing the current level node data using parent_btyx_coords
    p_b, p_t, p_y, p_x = parent_btyx_coords_4d.T # [N]
    parent_features = video_features_per_level[curr_lvl] # [B, T_lvl_1, H_lvl_1, W_lvl_1, C]
    parent_node_index_1d = node_index_1d_per_level[curr_lvl] # [B, T_lvl_1, H_lvl_1, W_lvl_1]
    tgt_parent_features = parent_features[p_b, p_t, p_y, p_x, :]  # [N, C]
    tgt_parent_node_index_1d = parent_node_index_1d[p_b, p_t, p_y, p_x]  # [N]

    if curr_lvl == n_level-1:
        octree_features_list.append(tgt_parent_features)
        octree_index_1d_list.append(tgt_parent_node_index_1d)
        return None

    ## Step2: indexing the child data
    child_features = video_features_per_level[curr_lvl+1] # [B, T_lvl_2, H_lvl_2, W_lvl_2, C]
    child_btyx_coords = child_btyx_coords_per_level[curr_lvl] # [B, T_lvl_1, H_lvl_1, W_lvl_1, 8, 4]
    child_valid_mask = child_valid_mask_per_level[curr_lvl] # [B, T_lvl_1, H_lvl_1, W_lvl_1, 8]
    tgt_child_btyx_coords = child_btyx_coords[p_b, p_t, p_y, p_x] # [N, 8, 4]
    tgt_child_valid_mask = child_valid_mask[p_b, p_t, p_y, p_x] # [N, 8]
    tgt_child_btyx_coords_4d = tgt_child_btyx_coords.flatten(0, 1)  # [N*8, 4]
    c_b, c_t, c_y, c_x = tgt_child_btyx_coords_4d.T  # [N*8]
    tgt_child_features = einops.rearrange(child_features[c_b, c_t, c_y, c_x], "(n s) c -> n s c", s=8) # [N, 8, C]
    
    ## Step3: Compute criterion metric and decide spliting node
    # upcast to float32 for precise computation``
    similarity = F.cosine_similarity(tgt_parent_features.unsqueeze(1).float(), tgt_child_features.float(), dim=-1) # [N, 8]
    stop_mask = (similarity >= threshold).all(dim=-1) # [N]
    split_node_mask = torch.logical_and((~stop_mask).unsqueeze(1), tgt_child_valid_mask).flatten(0, 1)
    parent_btyx_coords_4d = tgt_child_btyx_coords_4d[split_node_mask] # [N, 4]
    
    ## Step4: Save the output data
    octree_features_list.append(tgt_parent_features[stop_mask])
    octree_index_1d_list.append(tgt_parent_node_index_1d[stop_mask])

    return parent_btyx_coords_4d # dividing node coords

def octree_build(_video_feature, threshold, root_level=0):
    ## sppliting video into multiple snippets to fit cube size 
    _video_feature # [T, C, H, W]
    T, _, H, W = _video_feature.shape
    snippet_size = W # Cube size: [t, h, w], where t=h=w
    num_snippet = T // snippet_size

    ## TODO. How to handle the remainder frames
    num_frame_pad = snippet_size - T % snippet_size
    num_frame_drop = T % snippet_size
    # if num_frame_pad > 0:
    #     _video_feature = torch.cat([_video_feature, _video_feature[-1:].repeat(num_frame_pad, 1, 1, 1)], dim=0)
    #     num_snippet += 1
    if num_snippet == 0:
        return quadtree_build_video(_video_feature, threshold, root_level=root_level)[0]

    if num_frame_drop > 0:
        remainder_frame_features = _video_feature[-num_frame_drop:]
        _video_feature = _video_feature[0:-num_frame_drop]
        
    w = _video_feature.size(-1)
    size_per_level = [w]
    while w != 2:
        w = math.ceil(w/2)
        size_per_level.insert(0, w)

    snippet_features = einops.rearrange(_video_feature, "(B T) C H W -> B C T H W", B=num_snippet) # [B, C, T, H, W]
    _video_feature_lv0 = snippet_features # [B, C, T, H, W]
    device = _video_feature_lv0.device

    ## Build multi-level features via pooling (or interpolation)
    video_features_per_level = [_video_feature_lv0]
    _video_feature_nxt_lvl = _video_feature_lv0
    while video_features_per_level[0].size(-1) != size_per_level[root_level]:
        _video_feature_nxt_lvl = avgpool_to_even_side_feature_tyx(_video_feature_nxt_lvl) # [B, C, T, H, W]
        video_features_per_level.insert(0, _video_feature_nxt_lvl)

    ## Reshape feature maps [B, C, T, H, W] -> [B, T, H, W, C]
    n_level = len(video_features_per_level)
    for i in range(n_level):
        video_features_per_level[i] = video_features_per_level[i].permute(0, 2, 3, 4, 1)

    ## Build 1D flattened node index & child node linked 3D coordinates (T, H, W) and valid mask
    B, T, H, W, _ = video_features_per_level[-1].shape
    grid_b = torch.arange(0, B, device=device, dtype=torch.int32)
    grid_t = torch.arange(0, T, device=device, dtype=torch.int32)
    grid_y = torch.arange(0, H, device=device, dtype=torch.int32)
    grid_x = torch.arange(0, W, device=device, dtype=torch.int32)
    node_index_1d_lv0_yx = ((grid_y.unsqueeze(1) * W) + grid_x)
    node_index_1d_lv0_tyx = grid_t.reshape(-1, 1, 1) * (H*W) + node_index_1d_lv0_yx.unsqueeze(0)
    node_index_1d_lv0_btyx = grid_b.reshape(-1, 1, 1, 1) * (T*H*W) + node_index_1d_lv0_tyx.unsqueeze(0)

    node_index_1d_nxt_lvl = node_index_1d_lv0_btyx
    node_index_1d_per_level, child_btyx_coords_per_level, child_valid_mask_per_level = [node_index_1d_nxt_lvl], [], []
    for i in range(n_level-1):
        tgt_lvl = n_level-1-i
        child_btyx_coords_nxt_lvl, child_valid_mask_nxt_lvl, node_index_1d_nxt_lvl = avgpool_to_even_side_index_tyx(video_features_per_level[tgt_lvl], node_index_1d_nxt_lvl)

        node_index_1d_per_level.insert(0, node_index_1d_nxt_lvl)
        child_btyx_coords_per_level.insert(0, child_btyx_coords_nxt_lvl)
        child_valid_mask_per_level.insert(0, child_valid_mask_nxt_lvl)


    ## build first level coord
    B, T_lvl0, H_lvl0, W_lvl0, _ = video_features_per_level[0].shape
    grid_b_lvl0 = torch.arange(0, B, device=device, dtype=torch.int32)
    grid_t_lvl0 = torch.arange(0, T_lvl0, device=device, dtype=torch.int32)
    grid_y_lvl0 = torch.arange(0, H_lvl0, device=device, dtype=torch.int32)
    grid_x_lvl0 = torch.arange(0, W_lvl0, device=device, dtype=torch.int32)
    gb, gt, gy, gx = torch.meshgrid(grid_b_lvl0, grid_t_lvl0, grid_y_lvl0, grid_x_lvl0, indexing='ij') # [B, T_lvl0, H_lvl0, W_lvl0]
    parent_btyx_coords = torch.stack([gb, gt, gy, gx], dim=-1) # [B, T_lvl0, H_lvl0, W_lvl0, 4]
    parent_btyx_coords_4d = parent_btyx_coords.flatten(0, 3)  # [N, 4]

    ## iterate depth to build octree
    octree_features_list, octree_index_1d_list = [], []
    for curr_lvl in range(n_level): 
        parent_btyx_coords_4d = octree_build_iteration(parent_btyx_coords_4d, 
                                        video_features_per_level, node_index_1d_per_level,
                                        child_btyx_coords_per_level, child_valid_mask_per_level,
                                        octree_features_list, octree_index_1d_list,
                                        curr_lvl, n_level, threshold)
    
    ## flatten octree features into correct order of 1D sequence
    octree_features = torch.cat(octree_features_list) # [N, C]
    octree_index_1d = torch.cat(octree_index_1d_list) # [N]
    sorted_idx = torch.argsort(octree_index_1d)  # ascending order
    octree_features = octree_features[sorted_idx]

    if num_frame_drop > 0: ## TODO. how to handle remainder frames
        remainder_frame_quadtree_features = quadtree_build_video(remainder_frame_features, threshold, root_level=root_level)[0]
        octree_features = torch.cat([octree_features, remainder_frame_quadtree_features], dim=0)
        octree_features
    
    return octree_features

def get_octree_features(_video_feature, threshold, root_level=0):
    octree_features = octree_build(_video_feature, threshold, root_level)
    return octree_features


if __name__ == "__main__":
    vid = "fFjv93ACGo8" # first qid
    feat_filepath = f"datasets/videomme/preprocess_data/llava-video-7b-qwen2-video-only/F-128_fps-1/features/{vid}.pt"
    _video_feature = torch.load(feat_filepath, weights_only=True)
    meta_filepath = f"datasets/videomme/preprocess_data/llava-video-7b-qwen2-video-only/F-128_fps-1/metadata/{vid}.pkl"
    with open(meta_filepath, "rb") as fp:
        metadata = pickle.load(fp)
    
    # _video_feature = torch.rand((60, 24*24, 256))
    _video_feature = _video_feature.to("cuda")

    T, HW, D = _video_feature.shape
    H = int(math.sqrt(HW))
    W = H # H, W = 27, 27
    device = _video_feature.device
    stride = 2
    scaled_shape = [math.ceil(H / stride), math.ceil(W / stride)]
    _video_feature = _video_feature.view(T, H, W, -1).permute(0, 3, 1, 2).contiguous() # [T C H W] 27x27
    _video_feature = F.interpolate(_video_feature, scaled_shape) # 14x14

    octree_features = octree_build(_video_feature, 0.9)
    octree_features

    # from quadtree_utils import quadtree_build_video
    # quadtree_features = quadtree_build_video(_video_feature, 0.9)