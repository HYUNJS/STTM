import sys
import os
## Get the absolute path of the project directory
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
## Add the project directory to sys.path
sys.path.append(project_dir)

import torch
import torch.nn.functional as F
import math
import pickle
import einops

from token_merging_utils.quadtree_spatial_merger import avgpool_to_even_side_feature, sumpool_to_even_side_feature, pool_to_even_side_index_video, avgpool_to_even_side_pos_embs, sumpool_to_even_side_pos_embs
from token_merging_utils.quadtree_temporal_merger import cross_frame_node_merging_fast, cross_frame_node_merging_slow


def quadtree_build_iteration_video(parent_tyx_coords_3d,
                            video_features_per_level, node_tyxyx_tlbr_per_level,
                            child_tyx_coords_per_level, child_valid_mask_per_level,
                            quadtree_features_list, quadtree_tyxyx_tlbr_list,
                            curr_lvl, n_level, threshold, head_dim=None,
                            pos_embs_cos_per_level=None, pos_embs_sin_per_level=None, 
                            quadtree_pos_embs_cos_list=None, quadtree_pos_embs_sin_list=None):
    
    if curr_lvl == n_level-1:
        last_node_features = video_features_per_level[curr_lvl]
        last_node_tyxyx_tlbr = node_tyxyx_tlbr_per_level[curr_lvl]
        p_t, p_y, p_x = parent_tyx_coords_3d.T # [N]
        quadtree_features_list.append(last_node_features[p_t, p_y, p_x])
        quadtree_tyxyx_tlbr_list.append(last_node_tyxyx_tlbr[p_t, p_y, p_x])
        if quadtree_pos_embs_cos_list is not None:
            last_node_pos_embs_cos = pos_embs_cos_per_level[curr_lvl]
            last_node_pos_embs_sin = pos_embs_sin_per_level[curr_lvl]
            quadtree_pos_embs_cos_list.append(last_node_pos_embs_cos[p_t, p_y, p_x])
            quadtree_pos_embs_sin_list.append(last_node_pos_embs_sin[p_t, p_y, p_x])
        return None

    ## Step1: indexing the parent and child data using parent_tyx_coords
    ## parent data
    parent_features = video_features_per_level[curr_lvl] # [T, H_lvl_1, W_lvl_1, C]
    parent_node_tyxyx_tlbr = node_tyxyx_tlbr_per_level[curr_lvl] # [T, H_lvl_1, W_lvl_1, 5]
    p_t, p_y, p_x = parent_tyx_coords_3d.T # [N]
    tgt_parent_features = parent_features[p_t, p_y, p_x, :]  # [N, C]
    tgt_parent_node_tyxyx_tlbr = parent_node_tyxyx_tlbr[p_t, p_y, p_x] # [N, 5]
    ## child data
    child_features = video_features_per_level[curr_lvl+1] # [T, H_lvl_2, W_lvl_2, C]
    child_tyx_coords = child_tyx_coords_per_level[curr_lvl] # [T, H_lvl_1, W_lvl_1, 4, 3]
    child_valid_mask = child_valid_mask_per_level[curr_lvl] # [H_lvl_1, W_lvl_1, 4]
    tgt_child_tyx_coords = child_tyx_coords[p_t, p_y, p_x] # [N, 4, 3]
    tgt_child_valid_mask = child_valid_mask[p_t, p_y, p_x] # [N, 4]
    tgt_child_tyx_coords_3d = tgt_child_tyx_coords.flatten(0, 1)  # [N*4, 3]
    c_t, c_y, c_x = tgt_child_tyx_coords_3d.T  # [N*4]
    tgt_child_features = einops.rearrange(child_features[c_t, c_y, c_x], "(n s) c -> n s c", s=4) # [N, 4, C]
    
    ## Step2: Compute criterion metric and decide spliting node
    # upcast to float32 for precise computation
    tgt_parent_features_fp32 = tgt_parent_features.unsqueeze(1).float()
    tgt_child_features_fp32 = tgt_child_features.float()
    if head_dim is None:
        similarity = F.cosine_similarity(tgt_parent_features_fp32, tgt_child_features_fp32, dim=-1) # [N, 4]
    else:
        tgt_parent_features_fp32_per_head = einops.rearrange(tgt_parent_features_fp32, "n s (n_head d_head) -> n s n_head d_head", d_head=head_dim)
        tgt_child_features_fp32_per_head = einops.rearrange(tgt_child_features_fp32, "n s (n_head d_head) -> n s n_head d_head", d_head=head_dim)
        similarity_per_head = F.cosine_similarity(tgt_parent_features_fp32_per_head, tgt_child_features_fp32_per_head, dim=-1) # [N, 4, n_head]
        similarity = similarity_per_head.mean(dim=-1)
    
    stop_mask = (similarity >= threshold).all(dim=-1) # [N]
    split_node_mask = torch.logical_and((~stop_mask).unsqueeze(1), tgt_child_valid_mask).flatten(0, 1)
    parent_tyx_coords_3d = tgt_child_tyx_coords_3d[split_node_mask] # [N, 3]
    
    ## Step3: Save the output data
    quadtree_features_list.append(tgt_parent_features[stop_mask])
    quadtree_tyxyx_tlbr_list.append(tgt_parent_node_tyxyx_tlbr[stop_mask])
    if quadtree_pos_embs_cos_list is not None:
        parent_pos_embs_cos = pos_embs_cos_per_level[curr_lvl] # [T, H_lvl_1, W_lvl_1, C_pos]
        parent_pos_embs_sin = pos_embs_sin_per_level[curr_lvl] # [T, H_lvl_1, W_lvl_1, C_pos]
        tgt_parent_pos_embs_cos =  parent_pos_embs_cos[p_t, p_y, p_x, :] # [N, C_pos]
        tgt_parent_pos_embs_sin =  parent_pos_embs_sin[p_t, p_y, p_x, :] # [N, C_pos]
        quadtree_pos_embs_cos_list.append(tgt_parent_pos_embs_cos[stop_mask])
        quadtree_pos_embs_sin_list.append(tgt_parent_pos_embs_sin[stop_mask])

    return parent_tyx_coords_3d # dividing node coords

def quadtree_build_video(_video_feature, threshold, temporal_thresh=-1.0, root_level=0, weighted_avg=False, slow_ver=False, head_dim=None, pos_embs=None, pos_emb_weighted_avg=False):
    _video_feature_lv0 = _video_feature # [T, C, H, W]
    device = _video_feature_lv0.device
    quadtree_num_patches_per_node = None
    pool_to_even_side_feature = sumpool_to_even_side_feature if weighted_avg else avgpool_to_even_side_feature
    cross_frame_node_merging = cross_frame_node_merging_slow if slow_ver else cross_frame_node_merging_fast
    
    quadtree_pos_embs_cos, quadtree_pos_embs_sin, pos_embs_cos_per_level, pos_embs_sin_per_level = None, None, None, None
    if pos_embs is not None:
        pool_to_even_side_pos_embs = sumpool_to_even_side_pos_embs if pos_emb_weighted_avg else avgpool_to_even_side_pos_embs
        _pos_embs_cos_lv0, _pos_embs_sin_lv0 = pos_embs # Tuple([T, C, H, W])
        pos_embs_cos_per_level = [_pos_embs_cos_lv0]
        pos_embs_sin_per_level = [_pos_embs_sin_lv0]
        _pos_embs_cos_nxt_lvl = _pos_embs_cos_lv0
        _pos_embs_sin_nxt_lvl = _pos_embs_sin_lv0

    t, _, h, w = _video_feature.shape
    size_per_level = [(h, w)]
    while h != 2 and w != 2:
        w = math.ceil(w/2)
        h = math.ceil(h/2)
        size_per_level.insert(0, (h, w))

    ## Build multi-level features via pooling (or interpolation)
    video_features_per_level = [_video_feature_lv0]
    _video_feature_nxt_lvl = _video_feature_lv0
    while video_features_per_level[0].size(-1) != size_per_level[root_level][1]:
        _video_feature_nxt_lvl = pool_to_even_side_feature(_video_feature_nxt_lvl) # [T, C, H, W]
        video_features_per_level.insert(0, _video_feature_nxt_lvl)
        if pos_embs is not None:
            _pos_embs_cos_nxt_lvl, _pos_embs_sin_nxt_lvl = pool_to_even_side_pos_embs(_pos_embs_cos_nxt_lvl, _pos_embs_sin_nxt_lvl) # [T, C, H, W]
            pos_embs_cos_per_level.insert(0, _pos_embs_cos_nxt_lvl)
            pos_embs_sin_per_level.insert(0, _pos_embs_sin_nxt_lvl)
    
    ## Reshape feature maps [T, C, H, W] -> [T, H, W, C]
    n_level = len(video_features_per_level)
    for i in range(n_level):
        video_features_per_level[i] = video_features_per_level[i].permute(0, 2, 3, 1)
        if pos_embs is not None:
            pos_embs_cos_per_level[i] = pos_embs_cos_per_level[i].permute(0, 2, 3, 1)
            pos_embs_sin_per_level[i] = pos_embs_sin_per_level[i].permute(0, 2, 3, 1)

    ## Build 1D flattened node index & child node linked 2D coordinates and valid mask
    T, H, W, _ = video_features_per_level[-1].shape
    grid_t = torch.arange(0, T, device=device, dtype=torch.int32)
    grid_y = torch.arange(0, H, device=device, dtype=torch.int32)
    grid_x = torch.arange(0, W, device=device, dtype=torch.int32)
    gt_lv0, gy_lv0, gx_lv0 = torch.meshgrid(grid_t, grid_y, grid_x, indexing='ij') # [T, H, W]
    node_tyxyx_tlbr_lv0 = torch.stack([gt_lv0, gy_lv0, gx_lv0, gy_lv0+1, gx_lv0+1], dim=-1) # [T, H, W, 3]
    
    node_tyxyx_tlbr_nxt_lvl = node_tyxyx_tlbr_lv0
    node_tyxyx_tlbr_per_level = [node_tyxyx_tlbr_nxt_lvl]
    child_tyx_coords_per_level, child_valid_mask_per_level = [], []
    for i in range(n_level-1):
        tgt_lvl = n_level-1-i
        child_tyx_coords_nxt_lvl, child_valid_mask_nxt_lvl, node_tyxyx_tlbr_nxt_lvl = pool_to_even_side_index_video(video_features_per_level[tgt_lvl], node_tyxyx_tlbr_nxt_lvl)
        node_tyxyx_tlbr_per_level.insert(0, node_tyxyx_tlbr_nxt_lvl)
        child_tyx_coords_per_level.insert(0, child_tyx_coords_nxt_lvl)
        child_valid_mask_per_level.insert(0, child_valid_mask_nxt_lvl)

    ## no feature pyramid to build quadtree
    if n_level == 1:
        quadtree_features_video = video_features_per_level[0].flatten(0, 2)
        ## metadata
        quadtree_tyxyx_tlbr = node_tyxyx_tlbr_lv0.flatten(0, 2)
        node_height = quadtree_tyxyx_tlbr[:, 3] - quadtree_tyxyx_tlbr[:, 1]
        node_width = quadtree_tyxyx_tlbr[:, 4] - quadtree_tyxyx_tlbr[:, 2]
        quadtree_num_patches_per_node = node_height * node_width # [N]
        if pos_embs is not None:
            quadtree_pos_embs_cos = pos_embs_cos_per_level[0].flatten(0, 2)
            quadtree_pos_embs_sin = pos_embs_sin_per_level[0].flatten(0, 2)
        
        if temporal_thresh > 0:
            merged_results = cross_frame_node_merging(quadtree_features_video, quadtree_tyxyx_tlbr, temporal_thresh, quadtree_num_patches_per_node, weighted_avg, head_dim,
                                                                                                                quadtree_pos_embs_cos, quadtree_pos_embs_sin, pos_emb_weighted_avg)
            quadtree_features_video, quadtree_num_patches_per_node, quadtree_tyxyx_tlbr = merged_results.get('feature', None), merged_results.get('num_patch', None), merged_results.get('tlbr', None)
            if pos_embs is not None:
                pos_embs_cos, pos_embs_sin = merged_results.get('pos_embs_cos', None), merged_results.get('pos_embs_sin', None)
        
        if temporal_thresh <= 0 and weighted_avg:
            quadtree_features_video /= quadtree_num_patches_per_node.unsqueeze(1)
        
        if temporal_thresh <= 0 and pos_embs is not None and pos_emb_weighted_avg:
            pos_embs_cos = quadtree_pos_embs_cos / quadtree_num_patches_per_node.unsqueeze(1)
            pos_embs_sin = quadtree_pos_embs_sin / quadtree_num_patches_per_node.unsqueeze(1)
        
        if pos_embs is not None:
            return quadtree_features_video, quadtree_num_patches_per_node, quadtree_tyxyx_tlbr, (pos_embs_cos, pos_embs_sin)
        else:
            return quadtree_features_video, quadtree_num_patches_per_node, quadtree_tyxyx_tlbr

    ## build first level coord
    T, H_lvl0, W_lvl0, _ = video_features_per_level[0].shape
    grid_t_lvl0 = torch.arange(0, T, device=device, dtype=torch.int32)
    grid_y_lvl0 = torch.arange(0, H_lvl0, device=device, dtype=torch.int32)
    grid_x_lvl0 = torch.arange(0, W_lvl0, device=device, dtype=torch.int32)
    gt, gy, gx = torch.meshgrid(grid_t_lvl0, grid_y_lvl0, grid_x_lvl0, indexing='ij') # [T, H_lvl_0, W_lvl_0]
    parent_tyx_coords = torch.stack([gt, gy, gx], dim=-1) # [T, H_lvl_0, W_lvl_0, 3]
    parent_tyx_coords_3d = parent_tyx_coords.flatten(0, 2)  # [N, 3]

    ## iterate depth to build quadtree
    quadtree_features_list, quadtree_tyxyx_tlbr_list = [], []
    quadtree_pos_embs_cos_list, quadtree_pos_embs_sin_list = ([], []) if pos_embs is not None else (None, None)
    for curr_lvl in range(n_level): 
        parent_tyx_coords_3d = quadtree_build_iteration_video(parent_tyx_coords_3d, 
                                        video_features_per_level, node_tyxyx_tlbr_per_level,
                                        child_tyx_coords_per_level, child_valid_mask_per_level,
                                        quadtree_features_list, quadtree_tyxyx_tlbr_list,
                                        curr_lvl, n_level, threshold, head_dim,
                                        pos_embs_cos_per_level, pos_embs_sin_per_level, 
                                        quadtree_pos_embs_cos_list, quadtree_pos_embs_sin_list)
    
    ## flatten quadtree features into correct order of 1D sequence
    quadtree_features_video = torch.cat(quadtree_features_list) # [N, C]
    quadtree_tyxyx_tlbr = torch.cat(quadtree_tyxyx_tlbr_list) # [N, 5]
    tyx_offsets = torch.tensor([H*W, W, 1], device=device, dtype=torch.int32)
    quadtree_1d_index = (quadtree_tyxyx_tlbr[:, :3] * tyx_offsets.unsqueeze(0)).sum(dim=-1)
    sorted_idx = torch.argsort(quadtree_1d_index)  # ascending order
    quadtree_features_video = quadtree_features_video[sorted_idx]
    
    ## metadata
    quadtree_tyxyx_tlbr = quadtree_tyxyx_tlbr[sorted_idx]
    node_height = quadtree_tyxyx_tlbr[:, 3] - quadtree_tyxyx_tlbr[:, 1]
    node_width = quadtree_tyxyx_tlbr[:, 4] - quadtree_tyxyx_tlbr[:, 2]
    quadtree_num_patches_per_node = node_height * node_width # [N]

    if pos_embs is not None:
        quadtree_pos_embs_cos = torch.cat(quadtree_pos_embs_cos_list) # [N, C_pos]
        quadtree_pos_embs_sin = torch.cat(quadtree_pos_embs_sin_list) # [N, C_pos]
        quadtree_pos_embs_cos = quadtree_pos_embs_cos[sorted_idx]
        quadtree_pos_embs_sin = quadtree_pos_embs_sin[sorted_idx]

    if temporal_thresh > 0:
        merged_results = cross_frame_node_merging(quadtree_features_video, quadtree_tyxyx_tlbr, temporal_thresh, 
                                                                                        quadtree_num_patches_per_node, weighted_avg, head_dim,
                                                                                        quadtree_pos_embs_cos, quadtree_pos_embs_sin, pos_emb_weighted_avg)
        quadtree_features_video, quadtree_num_patches_per_node, quadtree_tyxyx_tlbr = merged_results.get('feature', None), merged_results.get('num_patch', None), merged_results.get('tlbr', None)
        if pos_embs is not None:
            pos_embs_cos, pos_embs_sin = merged_results.get('pos_embs_cos', None), merged_results.get('pos_embs_sin', None)
        
    if temporal_thresh <= 0 and weighted_avg:
        quadtree_features_video /= quadtree_num_patches_per_node.unsqueeze(1)
    
    if temporal_thresh <= 0 and pos_embs is not None and pos_emb_weighted_avg:
        pos_embs_cos = quadtree_pos_embs_cos / quadtree_num_patches_per_node.unsqueeze(1)
        pos_embs_sin = quadtree_pos_embs_sin / quadtree_num_patches_per_node.unsqueeze(1)
    
    if pos_embs is not None:
        return quadtree_features_video, quadtree_num_patches_per_node, quadtree_tyxyx_tlbr, (pos_embs_cos, pos_embs_sin)
    else:
        return quadtree_features_video, quadtree_num_patches_per_node, quadtree_tyxyx_tlbr


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
    
    # quadtree_features_video_1 = quadtree_build_video(_video_feature, threshold=0.80, root_level=1)
    # quadtree_features_video_2 = quadtree_build_video(_video_feature, threshold=0.80, temporal_thresh=0.60, root_level=1)
    quadtree_features_video_2 = quadtree_build_video(_video_feature, threshold=0.80, temporal_thresh=0.60, root_level=1)
    # quadtree_features_video_3 = quadtree_build_video(_video_feature, threshold=1.00, root_level=-1)
    # quadtree_features_video_4 = quadtree_build_video(_video_feature, threshold=1.00, temporal_thresh=0.9, root_level=-1)
    _video_feature
    