import torch
import torch.nn.functional as F
import math
import pickle
import einops


def get_cross_frame_node_pairs_fast(quadtree_tyxyx_tlbr):
    device = quadtree_tyxyx_tlbr.device
    N_node = quadtree_tyxyx_tlbr.size(0)

    ## Obtain metadata
    new_frame_idx_list = (torch.nonzero(quadtree_tyxyx_tlbr[0:-1, 0] != quadtree_tyxyx_tlbr[1:, 0]).squeeze(1)+1).tolist()
    new_frame_idx_list = [0, *new_frame_idx_list, len(quadtree_tyxyx_tlbr)]
    new_frame_idx = torch.tensor(new_frame_idx_list, device=device, dtype=torch.int32) # [T+1] boundary index of new frame
    num_nodes_per_frame = new_frame_idx[1:] - new_frame_idx[:-1]
    max_num_node = num_nodes_per_frame.max().item() # [M]
    N_frame = len(new_frame_idx_list) - 1

    ## Compute mapping index for 1D (N) flatten nodes into 2D (TxM) batched nodes
    node_idx = torch.arange(N_node, device=device, dtype=torch.int32)
    frame_ids = torch.bucketize(node_idx, new_frame_idx[1:-1], out_int32=True, right=True) # [N]
    frame_starts = new_frame_idx[frame_ids]
    local_idx = node_idx - frame_starts
    src2tgt_index = local_idx + frame_ids * max_num_node

    ## Scatter the 1D tlbr into 2D batched data. Build valid node mask
    # Note: bool type is not natively supported in index_add_
    tlbr_padded = torch.zeros(N_frame*max_num_node, 4, device=device, dtype=torch.int32)
    valid_mask_padded = torch.zeros(N_frame*max_num_node, device=device, dtype=torch.int32)
    tlbr_padded.index_add_(0, src2tgt_index, quadtree_tyxyx_tlbr[:, 1:])
    valid_mask_padded.index_add_(0, src2tgt_index, torch.ones(N_node, device=device, dtype=torch.int32)) > 0
    tlbr_padded = tlbr_padded.reshape(N_frame, max_num_node, 4)
    valid_mask_padded = valid_mask_padded.reshape(N_frame, max_num_node)
    
    ## Condition 1: current node fully contains next node.
    ## Condition 2: next node fully contains current node.
    ## condition 3: valid node mask (not padding node)
    cur_nodes_tlbr, nxt_nodes_tlbr = tlbr_padded[:-1], tlbr_padded[1:]
    cur_valid_mask, nxt_valid_mask = valid_mask_padded[:-1], valid_mask_padded[1:]
    diff = cur_nodes_tlbr.unsqueeze(2) - nxt_nodes_tlbr.unsqueeze(1)
    cur_contain_nxt = ((diff[..., :2] <= 0).all(dim=-1)) & ((diff[..., 2:] >= 0).all(dim=-1))
    nxt_contain_cur = ((diff[..., :2] >= 0).all(dim=-1)) & ((diff[..., 2:] <= 0).all(dim=-1))
    valid_node_mask = cur_valid_mask.unsqueeze(2) & nxt_valid_mask.unsqueeze(1)
    pair_mask = (cur_contain_nxt | nxt_contain_cur) & valid_node_mask  # [T-1, M, M]

    ## Compute the paired node indices of consecutive frames
    pair_indices = torch.nonzero(pair_mask)  # [L, 3]
    b_idx, cur_idx, nxt_idx = pair_indices.T
    cur_idx_offset = new_frame_idx[:-2]
    nxt_idx_offset = new_frame_idx[1:-1]
    cur_idx = cur_idx + cur_idx_offset[b_idx]
    nxt_idx = nxt_idx + nxt_idx_offset[b_idx]
    pair_idxs = torch.stack([cur_idx, nxt_idx], dim=1) # [L, 2]

    return pair_idxs

def filter_cross_frame_node_pairs(quadtree_features_video, pair_idxs, temporal_thresh, head_dim=None):
    ## Compute cosine similarity and filtering by threshold
    quadtree_features_video_float32 = quadtree_features_video.float()
    if head_dim is None:
        quadtree_features_norm = (quadtree_features_video_float32 / (quadtree_features_video_float32.norm(dim=-1, keepdim=True) + 1e-8))
        pair_sim = (quadtree_features_norm[pair_idxs[:, 0]] * quadtree_features_norm[pair_idxs[:, 1]]).sum(dim=-1)
    else:
        quadtree_features_video_float32_per_head = einops.rearrange(quadtree_features_video_float32, "N (head_num head_dim) -> N head_num head_dim", head_dim=head_dim)
        quadtree_features_norm_per_head = (quadtree_features_video_float32_per_head / (quadtree_features_video_float32_per_head.norm(dim=-1, keepdim=True) + 1e-8))
        pair_sim_per_head = (quadtree_features_norm_per_head[pair_idxs[:, 0]] * quadtree_features_norm_per_head[pair_idxs[:, 1]]).sum(dim=-1)
        pair_sim = pair_sim_per_head.mean(dim=-1)
    
    merging_mask = pair_sim >= temporal_thresh
    pair_idxs_for_merging = pair_idxs[merging_mask] # [L']
    
    return pair_idxs_for_merging

def get_cross_frame_node_pairs_slow(quadtree_features_video, quadtree_tyxyx_tlbr, temporal_thresh):
    device = quadtree_features_video.device

    pair_idxs_list = []
    new_frame_idx_list = (torch.nonzero(quadtree_tyxyx_tlbr[0:-1, 0] != quadtree_tyxyx_tlbr[1:, 0]).squeeze(1)+1).tolist()
    new_frame_idx_list = [0, *new_frame_idx_list, len(quadtree_tyxyx_tlbr)]
    
    quadtree_features_video_float32 = quadtree_features_video.float()
    quadtree_features_norm = (quadtree_features_video_float32 / (quadtree_features_video_float32.norm(dim=-1, keepdim=True) + 1e-8))
    for i in range(len(new_frame_idx_list)-2):
        cur_frame_nodes_tlbr = quadtree_tyxyx_tlbr[new_frame_idx_list[i]:new_frame_idx_list[i+1], 1:] # [N1, 4]
        nxt_frame_nodes_tlbr = quadtree_tyxyx_tlbr[new_frame_idx_list[i+1]:new_frame_idx_list[i+2], 1:] # [N2, 4]
        
        ## Compute spatial relationship to obtain pairs
        tlbr_diff = cur_frame_nodes_tlbr.unsqueeze(1) - nxt_frame_nodes_tlbr.unsqueeze(0) # [n1, n2, 4]
        first_larger_grid = torch.logical_and((tlbr_diff[:, :, 0:2] <= 0).all(dim=-1), (tlbr_diff[:, :, 2:4] >= 0).all(dim=-1)) # [n1, n2]
        last_larger_grid = torch.logical_and((tlbr_diff[:, :, 0:2] >= 0).all(dim=-1), (tlbr_diff[:, :, 2:4] <= 0).all(dim=-1))  # [n1, n2]
        pair_idxs_i = torch.nonzero(torch.logical_or(first_larger_grid, last_larger_grid)).to(torch.int32) # [m, 2]
        pair_idxs_i[:, 0] += new_frame_idx_list[i]
        pair_idxs_i[:, 1] += new_frame_idx_list[i+1]

        pair_sim_i = (quadtree_features_norm[pair_idxs_i[:, 0]] * quadtree_features_norm[pair_idxs_i[:, 1]]).sum(dim=-1)
        merging_pairs_i = torch.nonzero(pair_sim_i >= temporal_thresh).squeeze(1)
        pair_sim_for_merging_i = pair_sim_i[merging_pairs_i]
        pair_idxs_for_merging_i = pair_idxs_i[merging_pairs_i]
        pair_idxs_i # [N, 2] - (dst, src)
        merging_pairs_i # [L]
        pair_idxs_for_merging_i # [L, 2]

        # Step 1: Sort srcs by descending similarity
        dsts, srcs = pair_idxs_for_merging_i.T
        sorted_sims, sort_idx = pair_sim_for_merging_i.sort(descending=True)
        sorted_srcs, sorted_dsts = srcs[sort_idx], dsts[sort_idx]

        # Step 2: Get mask for first occurrence of each src
        keep_mask = torch.ones_like(sorted_srcs, dtype=torch.bool, device=device)
        keep_mask[1:] = sorted_srcs[1:] != sorted_srcs[:-1]

        # Step 3: Apply mask
        final_srcs = sorted_srcs[keep_mask]
        final_dsts = sorted_dsts[keep_mask]
        filtered_pairs = torch.stack([final_dsts, final_srcs], dim=1)  # [M, 2]
        pair_idxs_list.append(filtered_pairs)
    
    pair_idxs_for_merging = torch.cat(pair_idxs_list).to(torch.long) # [M, 2]

    return pair_idxs_for_merging

def agg_feature_and_metadata(quadtree_features_video, quadtree_num_patches_per_node, quadtree_tyxyx_tlbr, 
                            final_representative, weighted_avg,
                            quadtree_pos_embs_cos=None, quadtree_pos_embs_sin=None, pos_emb_weighted_avg=False):
    device = quadtree_features_video.device
    N_node = quadtree_features_video.size(0)

    ## aggregate features
    feature_accum = torch.zeros_like(quadtree_features_video) # [N, C]
    feature_count = torch.zeros(N_node, device=device, dtype=torch.int32) # [N]
    feature_accum.index_add_(0, final_representative, quadtree_features_video)
    feature_count.index_add_(0, final_representative, torch.ones_like(feature_count))
    survived_node_mask = feature_count > 0

    ## aggregate metadata
    agg_num_patches_per_node = torch.zeros(N_node, device=device, dtype=torch.int32)
    agg_num_patches_per_node.index_add_(0, final_representative, quadtree_num_patches_per_node)
    agg_num_patches_per_node = agg_num_patches_per_node[survived_node_mask]
    agg_tyxyx_tlbr = quadtree_tyxyx_tlbr[survived_node_mask]

    if weighted_avg:
        agg_features = feature_accum[survived_node_mask] / agg_num_patches_per_node.unsqueeze(1) # [N', C]
    else:
        agg_features = feature_accum[survived_node_mask] / feature_count[survived_node_mask].unsqueeze(-1) # [N', C]
    
    merged_results = {
                    'feature': agg_features,
                    'num_patch': agg_num_patches_per_node,
                    'tlbr': agg_tyxyx_tlbr,
                    }
    
    if quadtree_pos_embs_cos is not None:
        pos_embs_cos_accum = torch.zeros_like(quadtree_pos_embs_cos) # [N, C_pos]
        pos_embs_sin_accum = torch.zeros_like(quadtree_pos_embs_sin) # [N, C_pos]
        pos_embs_cos_accum.index_add_(0, final_representative, quadtree_pos_embs_cos)
        pos_embs_sin_accum.index_add_(0, final_representative, quadtree_pos_embs_sin)
            
        if pos_emb_weighted_avg:
            agg_pos_embs_cos = pos_embs_cos_accum[survived_node_mask] / agg_num_patches_per_node.unsqueeze(1) # [N', C_pos]
            agg_pos_embs_sin = pos_embs_sin_accum[survived_node_mask] / agg_num_patches_per_node.unsqueeze(1) # [N', C_pos]
        else:
            agg_pos_embs_cos = pos_embs_cos_accum[survived_node_mask] / feature_count[survived_node_mask].unsqueeze(-1) # [N', C_pos]
            agg_pos_embs_sin = pos_embs_sin_accum[survived_node_mask] / feature_count[survived_node_mask].unsqueeze(-1) # [N', C_pos]
        
        merged_results.update({
            'pos_embs_cos': agg_pos_embs_cos,
            'pos_embs_sin': agg_pos_embs_sin,
        })
    
    return merged_results

def get_merge_dst_idx_unsafe(pair_idxs_for_merging, N_node):

    """
    This function is variant of vectorized union-find algorithm. 
    How this algorithm works with example:
    Take input pair_idxs_for_merging shape (L, 2): [[0, 85], [85, 170], [170, 252], [1, 86], ...]
    We expect the output final_representative shape (N): [0 (0), 1 (1), 2 (2), ..., 0 (85), 1 (86), 2 (87), ..., 0 (170), 1 (171), 2 (172), ...], where each position (i) indicates the destination index

    dst_idx: [0, 85, 170, 1, ...]
    src_idx: [85, 170, 252, 86, ...]
    1st iteration
        final_representative: [0, 1, 2, 3, ...] # [N]
        min_rep: [0, 85, 170, 1, ...]
        final_representative[[0, 85, 170, 1, ...]] = [0, 85, 170, 1, ...] # dst_idx
        final_representative[[85, 170, 252, 86, ...]] = [0, 85, 170, 1, ...] # src_idx

        final_representative = final_representative[[0, 1, ..., 0, 1, ..., 85, ... 170, ...]]
        
        Stop if final_representative == final_representative[[0, 1, ..., 0, 1, ..., 85, ... 170, ...]]
    """

    device = pair_idxs_for_merging.device
    final_representative = torch.arange(N_node, device=device, dtype=torch.int32) # [N]
    while True: # iterate until the maximum level of merging depth 

        ## Gather current representatives
        dst_idx = pair_idxs_for_merging[:, 0] # [L]
        src_idx = pair_idxs_for_merging[:, 1] # [L]
        dst_rep = final_representative[dst_idx]
        src_rep = final_representative[src_idx]
        min_rep = torch.minimum(src_rep, dst_rep)
        
        ## TODO. Safely assign minimum values
        # # Scatter minimum values safely using scatter_reduce_
        # final_representative.scatter_reduce_(0, src_idx, min_rep, reduce="amin")
        # final_representative.scatter_reduce_(0, dst_idx, min_rep, reduce="amin")

        ## Scatter the minimum back to all paired nodes
        final_representative[dst_idx] = min_rep
        final_representative[src_idx] = min_rep

        ## Propagate the minimum over all nodes - vectorized version of path compression in union-find algorithm
        final_representative = final_representative[final_representative] # going up one-level in the linked chain towards first-top-left (min_rep)

        ## Check for convergence - no more depth to go deeper
        if torch.equal(final_representative, final_representative[final_representative]):
            break

    return final_representative

def get_merge_dst_idx_safe(pair_idxs_for_merging, N_node):

    """
    This function is variant of vectorized union-find algorithm. 
    How this algorithm works with example:
    Take input pair_idxs_for_merging shape (L, 2): [[0, 85], [85, 170], [170, 252], [1, 86], ...]
    We expect the output final_representative shape (N): [0 (0), 1 (1), 2 (2), ..., 0 (85), 1 (86), 2 (87), ..., 0 (170), 1 (171), 2 (172), ...], where each position (i) indicates the destination index

    dst_idx: [0, 85, 170, 1, ...]
    src_idx: [85, 170, 252, 86, ...]
    1st iteration
        final_representative: [0, 1, 2, 3, ...] # [N]
        min_rep: [0, 85, 170, 1, ...]
        final_representative[[0, 85, 170, 1, ...]] = [0, 85, 170, 1, ...] # dst_idx
        final_representative[[85, 170, 252, 86, ...]] = [0, 85, 170, 1, ...] # src_idx

        final_representative = final_representative[[0, 1, ..., 0, 1, ..., 85, ... 170, ...]]
        
        Stop if final_representative == final_representative[[0, 1, ..., 0, 1, ..., 85, ... 170, ...]]
    """

    device = pair_idxs_for_merging.device
    final_representative = torch.arange(N_node, device=device, dtype=torch.int32) # [N]
    n_iter = 0
    while True: # iterate until the maximum level of merging depth 

        ## Gather current representatives
        dst_idx = pair_idxs_for_merging[:, 0] # [L]
        src_idx = pair_idxs_for_merging[:, 1] # [L]
        dst_rep = final_representative[dst_idx]
        src_rep = final_representative[src_idx]
        min_rep = torch.minimum(src_rep, dst_rep)

        ## Scatter the minimum back to all paired nodes: scatter_reduce_ is cuda-safe operation 
        final_representative.scatter_reduce_(0, dst_idx, min_rep, reduce="amin")
        final_representative.scatter_reduce_(0, src_idx, min_rep, reduce="amin")
        
        ## Propagate the minimum over all nodes - vectorized version of path compression in union-find algorithm
        final_representative = final_representative[final_representative] # going up one-level in the linked chain towards first-top-left (min_rep)

        ## Check for convergence - no more depth to go deeper
        n_iter += 1
        if torch.equal(final_representative, final_representative[final_representative]):
            break
    # print(f"# iteration: {n_iter}")

    return final_representative

def cross_frame_node_merging_fast(quadtree_features_video, quadtree_tyxyx_tlbr, temporal_thresh, 
                                quadtree_num_patches_per_node, weighted_avg=False, head_dim=None,
                                quadtree_pos_embs_cos=None, quadtree_pos_embs_sin=None, pos_emb_weighted_avg=False):
    """
        Merge the consecutive frames' tokens within the same spatial layout and showing high similarity
        quadtree_features_video: [N, C]
        quadtree_tyxyx_tlbr: [N, 5]
    """
    N_node = quadtree_features_video.size(0)

    pair_idxs = get_cross_frame_node_pairs_fast(quadtree_tyxyx_tlbr)
    pair_idxs_for_merging = filter_cross_frame_node_pairs(quadtree_features_video, pair_idxs, temporal_thresh, head_dim)
    final_representative = get_merge_dst_idx_safe(pair_idxs_for_merging, N_node) # [N]
    merged_results = agg_feature_and_metadata(quadtree_features_video, quadtree_num_patches_per_node, quadtree_tyxyx_tlbr, 
                                            final_representative, weighted_avg, quadtree_pos_embs_cos, quadtree_pos_embs_sin, pos_emb_weighted_avg)
    
    return merged_results

def cross_frame_node_merging_slow(quadtree_features_video, quadtree_tyxyx_tlbr, temporal_thresh, 
                                quadtree_num_patches_per_node, weighted_avg=False, head_dim=None,
                                quadtree_pos_embs_cos=None, quadtree_pos_embs_sin=None, pos_emb_weighted_avg=False):
    N_node = quadtree_features_video.size(0)

    pair_idxs_for_merging = get_cross_frame_node_pairs_slow(quadtree_features_video, quadtree_tyxyx_tlbr, temporal_thresh)
    final_representative = get_merge_dst_idx_safe(pair_idxs_for_merging, N_node) # [N]
    merged_results = agg_feature_and_metadata(quadtree_features_video, quadtree_num_patches_per_node, quadtree_tyxyx_tlbr, 
                                            final_representative, weighted_avg, quadtree_pos_embs_cos, quadtree_pos_embs_sin, pos_emb_weighted_avg)
    
    return merged_results

def cross_frame_node_merging_vis(quadtree_features_video, quadtree_tyxyx_tlbr, temporal_thresh, 
                                        quadtree_num_patches_per_node, weighted_avg=False, head_dim=None,
                                        quadtree_pos_embs_cos=None, quadtree_pos_embs_sin=None, pos_emb_weighted_avg=False):
    N_node = quadtree_features_video.size(0)

    pair_idxs = get_cross_frame_node_pairs_fast(quadtree_tyxyx_tlbr)
    pair_idxs_for_merging = filter_cross_frame_node_pairs(quadtree_features_video, pair_idxs, temporal_thresh)
    final_representative = get_merge_dst_idx_safe(pair_idxs_for_merging, N_node) # [N]
    merged_results = agg_feature_and_metadata(quadtree_features_video, quadtree_num_patches_per_node, quadtree_tyxyx_tlbr, 
                                            final_representative, weighted_avg, quadtree_pos_embs_cos, quadtree_pos_embs_sin, pos_emb_weighted_avg)
    
    ## Get unique destination node ids (if they aren’t sorted, you can sort them if needed)
    node_ids = torch.unique(final_representative)
    node_metadata = {}
    for node_id in node_ids:
        ## Create a boolean mask for rows with the current node_id
        mask = (final_representative == node_id)
        node_metadata[int(node_id.item())] = quadtree_tyxyx_tlbr[mask].tolist()
    
    return merged_results, node_metadata
    

if __name__ == "__main__":
    quadtree_features_video = torch.load("quadtree_features_video.pt")
    quadtree_tyxyx_tlbr = torch.load("quadtree_tyxyx_tlbr.pt")

    quadtree_features_video = quadtree_features_video.to("cuda")
    quadtree_tyxyx_tlbr = quadtree_tyxyx_tlbr.to("cuda")
    cross_frame_node_merging_slow(quadtree_features_video, quadtree_tyxyx_tlbr, 0.6)