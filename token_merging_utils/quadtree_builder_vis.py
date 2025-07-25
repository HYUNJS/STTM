import torch
import torch.nn.functional as F
import math
import pickle
import einops

from token_merging_utils.quadtree_spatial_merger import avgpool_to_even_side_feature, sumpool_to_even_side_feature, pool_to_even_side_index_video
from token_merging_utils.quadtree_temporal_merger import cross_frame_node_merging_vis
from token_merging_utils.quadtree_builder import quadtree_build_iteration_video


def quadtree_build_video_vis(_video_feature, threshold, temporal_thresh=-1.0, root_level=0, weighted_avg=False, slow_ver=False):
    assert temporal_thresh > 0, "Visualization only suppports spatio-temporal merging version"
    assert not slow_ver, "Visualization only suppports fast version"
    
    _video_feature_lv0 = _video_feature # [T, C, H, W]
    device = _video_feature_lv0.device
    quadtree_num_patches_per_node = None
    pool_to_even_side_feature = sumpool_to_even_side_feature if weighted_avg else avgpool_to_even_side_feature
    cross_frame_node_merging = cross_frame_node_merging_vis

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
    
    ## Reshape feature maps [T, C, H, W] -> [T, H, W, C]
    n_level = len(video_features_per_level)
    for i in range(n_level):
        video_features_per_level[i] = video_features_per_level[i].permute(0, 2, 3, 1)

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
        if temporal_thresh > 0:
            merged_results, node_metadata = cross_frame_node_merging(quadtree_features_video, quadtree_tyxyx_tlbr, temporal_thresh, quadtree_num_patches_per_node, weighted_avg)
            quadtree_features_video, quadtree_num_patches_per_node, quadtree_tyxyx_tlbr = merged_results.get('feature', None), merged_results.get('num_patch', None), merged_results.get('tlbr', None)
        
        return quadtree_features_video, quadtree_num_patches_per_node, quadtree_tyxyx_tlbr, node_metadata

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
    for curr_lvl in range(n_level): 
        parent_tyx_coords_3d = quadtree_build_iteration_video(parent_tyx_coords_3d, 
                                        video_features_per_level, node_tyxyx_tlbr_per_level,
                                        child_tyx_coords_per_level, child_valid_mask_per_level,
                                        quadtree_features_list, quadtree_tyxyx_tlbr_list,
                                        curr_lvl, n_level, threshold)
    
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

    if temporal_thresh > 0:
        merged_results, node_metadata = cross_frame_node_merging(quadtree_features_video, quadtree_tyxyx_tlbr, temporal_thresh, quadtree_num_patches_per_node, weighted_avg)
        quadtree_features_video, quadtree_num_patches_per_node, quadtree_tyxyx_tlbr = merged_results.get('feature', None), merged_results.get('num_patch', None), merged_results.get('tlbr', None)
    
    return quadtree_features_video, quadtree_num_patches_per_node, quadtree_tyxyx_tlbr, node_metadata


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
    
    quadtree_features_video_2 = quadtree_build_video_visualization(_video_feature, threshold=0.80, temporal_thresh=0.60, root_level=1)
    _video_feature
    