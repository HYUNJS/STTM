import torch
import torch.nn.functional as F
import math
import einops


stride = 2

def avgpool_to_even_side_feature(video_feature):
    T, C, ori_h, ori_w = video_feature.shape
    new_h, new_w = math.ceil(ori_h/2), math.ceil(ori_w/2)
    device = video_feature.device
    even_height_flag, even_width_flag = ori_h % stride == 0, ori_w % stride == 0
    if even_height_flag and even_width_flag: # even size
        video_feature_next_level = F.avg_pool2d(video_feature, (stride, stride))
    else: # odd size
        ## Allocate an empty tensor for the pooled features.
        video_feature_next_level = torch.empty((T, C, new_h, new_w), device=device,  dtype=video_feature.dtype)

        if not even_height_flag and even_width_flag:
            top = video_feature[:, :, 0, :]
            top = top.reshape(T, C, new_w, stride).mean(dim=-1)
            video_feature_next_level[:, :, 0, :] = top

            bottom = video_feature[:, :, 1:, :]
            bottom = F.avg_pool2d(bottom, kernel_size=(stride, stride))
            video_feature_next_level[:, :, 1:, :] = bottom
            
        if even_height_flag and not even_width_flag:
            left = video_feature[:, :, :, 0]
            left = left.reshape(T, C, new_h, stride).mean(dim=-1)
            video_feature_next_level[:, :, :, 0] = left

            right = video_feature[:, :, :, 1:]
            right = F.avg_pool2d(right, kernel_size=(stride, stride))
            video_feature_next_level[:, :, :, 1:] = right

        if not even_height_flag and not even_width_flag:
            ## (a). Top–left element (remains unchanged):
            video_feature_next_level[:, :, 0, 0] = video_feature[:, :, 0, 0]

            ## (b). Top–right stripes (pooling along row 0):
            trs = video_feature[:, :, 0, 1:] # extract first row (without first col)
            trs = trs.reshape(T, C, new_w-1, stride).mean(dim=-1)
            video_feature_next_level[:, :, 0, 1:] = trs

            # (c). Left–column stripes (pooling along column 0):
            bls = video_feature[:, :, 1:, 0] # extract first col (without first row)
            bls = bls.reshape(T, C, new_h-1, stride).mean(dim=-1)
            video_feature_next_level[:, :, 1:, 0] = bls

            # (d). Bottom–right region (full 2×2 pooling):
            brs = F.avg_pool2d(video_feature[:, :, 1:, 1:], kernel_size=(stride, stride))
            video_feature_next_level[:, :, 1:, 1:] = brs

    return video_feature_next_level

def sumpool_to_even_side_feature(video_feature):
    T, C, ori_h, ori_w = video_feature.shape
    new_h, new_w = math.ceil(ori_h/2), math.ceil(ori_w/2)
    device = video_feature.device

    if (ori_h % stride == 0) and (ori_w % stride == 0): # even size
        video_feature_next_level = F.lp_pool2d(video_feature, 1, kernel_size=(stride, stride))
    else: # odd size
        ## Allocate an empty tensor for the pooled features.
        video_feature_next_level = torch.empty((T, C, new_h, new_w), device=device,  dtype=video_feature.dtype)

        ## (a). Top–left element (remains unchanged):
        video_feature_next_level[:, :, 0, 0] = video_feature[:, :, 0, 0]

        ## (b). Top–right stripes (pooling along row 0):
        trs = video_feature[:, :, 0, 1:] # extract first row (without first col)
        trs = trs.reshape(T, C, new_w-1, stride).sum(dim=-1)
        video_feature_next_level[:, :, 0, 1:] = trs

        # (c). Left–column stripes (pooling along column 0):
        bls = video_feature[:, :, 1:, 0] # extract first col (without first row)
        bls = bls.reshape(T, C, new_h-1, stride).sum(dim=-1)
        video_feature_next_level[:, :, 1:, 0] = bls

        # (d). Bottom–right region (full 2×2 pooling):
        brs = F.lp_pool2d(video_feature[:, :, 1:, 1:], 1, kernel_size=(stride, stride))
        video_feature_next_level[:, :, 1:, 1:] = brs

    return video_feature_next_level

def avgpool_to_even_side_pos_embs(pos_embs_cos, pos_embs_sin):
    T, C_pos, ori_h, ori_w = pos_embs_cos.shape
    new_h, new_w = math.ceil(ori_h/2), math.ceil(ori_w/2)
    device = pos_embs_cos.device
    even_height_flag, even_width_flag = ori_h % stride == 0, ori_w % stride == 0
    if even_height_flag and even_width_flag: # even size
        pos_embs_cos_next_level = F.avg_pool2d(pos_embs_cos, (stride, stride))
        pos_embs_sin_next_level = F.avg_pool2d(pos_embs_sin, (stride, stride))
    else: # odd size
        ## Allocate an empty tensor for the pooled features.
        ## Assume square-shape only - not supporting qwen2vl, but only llava-video and llava-onevision
        C_pos = pos_embs_cos.size(1)
        pos_embs_cos_next_level = torch.empty([T, C_pos, new_h, new_w], device=device, dtype=pos_embs_cos.dtype)
        pos_embs_sin_next_level = torch.empty([T, C_pos, new_h, new_w], device=device, dtype=pos_embs_cos.dtype)
        
        pos_embs_cos_next_level[:, :, 0, 0] = pos_embs_cos[:, :, 0, 0]
        pos_embs_sin_next_level[:, :, 0, 0] = pos_embs_sin[:, :, 0, 0]

        trs_pos_cos, trs_pos_sin = [pos_embs[:, :, 0, 1:] for pos_embs in [pos_embs_cos, pos_embs_sin]]
        trs_pos_cos, trs_pos_sin = [trs_pos.reshape(T, C_pos, new_w-1, stride).mean(dim=-1) for trs_pos in [trs_pos_cos, trs_pos_sin]]
        pos_embs_cos_next_level[:, :, 0, 1:] = trs_pos_cos
        pos_embs_sin_next_level[:, :, 0, 1:] = trs_pos_sin
        
        bls_pos_cos, bls_pos_sin = [pos_embs[:, :, 1:, 0] for pos_embs in [pos_embs_cos, pos_embs_sin]]
        bls_pos_cos, bls_pos_sin = [bls_pos.reshape(T, C_pos, new_h-1, stride).mean(dim=-1) for bls_pos in [bls_pos_cos, bls_pos_sin]]
        pos_embs_cos_next_level[:, :, 1:, 0] = bls_pos_cos
        pos_embs_sin_next_level[:, :, 1:, 0] = bls_pos_sin
        
        brs_pos_cos, brs_pos_sin = [F.avg_pool2d(pos_embs[:, :, 1:, 1:], kernel_size=(stride, stride)) for pos_embs in [pos_embs_cos, pos_embs_sin]]
        pos_embs_cos_next_level[:, :, 1:, 1:] = brs_pos_cos
        pos_embs_sin_next_level[:, :, 1:, 1:] = brs_pos_sin

    return pos_embs_cos_next_level, pos_embs_sin_next_level

def sumpool_to_even_side_pos_embs(pos_embs_cos, pos_embs_sin):
    T, C_pos, ori_h, ori_w = pos_embs_cos.shape
    new_h, new_w = math.ceil(ori_h/2), math.ceil(ori_w/2)
    device = pos_embs_cos.device

    if (ori_h % stride == 0) and (ori_w % stride == 0): # even size
        pos_embs_cos_next_level = F.lp_pool2d(pos_embs_cos, 1, kernel_size=(stride, stride))
        pos_embs_sin_next_level = F.lp_pool2d(pos_embs_sin, 1, kernel_size=(stride, stride))
    else: # odd size
        ## Allocate an empty tensor for the pooled features.
        ## Assume square-shape only - not supporting qwen2vl, but only llava-video and llava-onevision
        pos_embs_cos_next_level = torch.empty([T, C_pos, new_h, new_w], device=device, dtype=pos_embs_cos.dtype)
        pos_embs_sin_next_level = torch.empty([T, C_pos, new_h, new_w], device=device, dtype=pos_embs_cos.dtype)
        
        pos_embs_cos_next_level[:, :, 0, 0] = pos_embs_cos[:, :, 0, 0]
        pos_embs_sin_next_level[:, :, 0, 0] = pos_embs_sin[:, :, 0, 0]
        
        trs_pos_cos, trs_pos_sin = [pos_embs[:, :, 0, 1:] for pos_embs in [pos_embs_cos, pos_embs_sin]]
        trs_pos_cos, trs_pos_sin = [trs_pos.reshape(T, C_pos, new_w-1, stride).sum(dim=-1) for trs_pos in [trs_pos_cos, trs_pos_sin]]
        pos_embs_cos_next_level[:, :, 0, 1:] = trs_pos_cos
        pos_embs_sin_next_level[:, :, 0, 1:] = trs_pos_sin

        bls_pos_cos, bls_pos_sin = [pos_embs[:, :, 1:, 0] for pos_embs in [pos_embs_cos, pos_embs_sin]]
        bls_pos_cos, bls_pos_sin = [bls_pos.reshape(T, C_pos, new_h-1, stride).sum(dim=-1) for bls_pos in [bls_pos_cos, bls_pos_sin]]
        pos_embs_cos_next_level[:, :, 1:, 0] = bls_pos_cos
        pos_embs_sin_next_level[:, :, 1:, 0] = bls_pos_sin
        
        brs_pos_cos, brs_pos_sin = [F.lp_pool2d(pos_embs[:, :, 1:, 1:], 1, kernel_size=(stride, stride)) for pos_embs in [pos_embs_cos, pos_embs_sin]]
        pos_embs_cos_next_level[:, :, 1:, 1:] = brs_pos_cos
        pos_embs_sin_next_level[:, :, 1:, 1:] = brs_pos_sin

    return pos_embs_cos_next_level, pos_embs_sin_next_level

def pool_to_even_side_index_video(video_feature, prev_tyxyx_tlbr):
    ## Generate matched indices across feature map levels (for all frames)
    ## For this version, we do not pool across temporal axis.
    ## assign top-left 

    T, ori_h, ori_w, _ = video_feature.shape
    device = video_feature.device
    new_h, new_w = math.ceil(ori_h/2), math.ceil(ori_w/2)
    even_height_flag, even_width_flag = ori_h % stride == 0, ori_w % stride == 0
    if even_height_flag and even_width_flag: # even size
        ## Build a grid of “top–left” indices for each 2×2 block:
        grid_t = torch.arange(0, T, device=device, dtype=torch.int32)
        grid_y = torch.arange(0, new_h, device=device, dtype=torch.int32) * 2
        grid_x = torch.arange(0, new_w, device=device, dtype=torch.int32) * 2

        ## Meshgrid: each output cell (i,j) will have top–left index (gt, gy, gx)
        gt, gy, gx = torch.meshgrid(grid_t, grid_y, grid_x, indexing='ij') # [T, new_h, new_w]
        top_left = torch.stack([gt, gy, gx], dim=-1).unsqueeze(-2) # [T, new_h, new_w, 1, 3]

        ## Create a tensor of the fixed 2x2×2 offsets. T-axis offset is 0 for image-only quadtree mode
        offsets = torch.tensor([[0,0,0], [0,0,1], [0,1,0], [0,1,1]], device=device, dtype=torch.int32)
        offsets = offsets.view(1, 1, 1, 4, 3)

        ## The child coordinates are just the top–left coordinates plus the offsets.
        child_tyx_coords = top_left + offsets # [T, new_h, new_w, 4, 3]

        ## child nodes' original tlbr coords in tyxyx format
        child_tyxyx_tlbr = torch.zeros((T, new_h, new_w, 5), device=device, dtype=torch.int32)
        child_tyxyx_tlbr[:, :, :, 0] = prev_tyxyx_tlbr[:, 0::2, 0::2, 0] # t_idx
        child_tyxyx_tlbr[:, :, :, 1:3] = prev_tyxyx_tlbr[:, 0::2, 0::2, 1:3] # yx_tl
        child_tyxyx_tlbr[:, :, :, 3:5] = prev_tyxyx_tlbr[:, 1::2, 1::2, 3:5] # yx_br

        ## Generate Odd or Even mask
        child_valid_mask = torch.ones((T, new_h, new_w, 4), device=device, dtype=torch.bool)
    else: # odd size
        child_yx_coords = torch.zeros((new_h, new_w, 4, 2), device=device, dtype=torch.int32) # will expand T-axis at the end
        child_valid_mask = torch.zeros((new_h, new_w, 4), device=device, dtype=torch.bool) # will expand T-axis at the end
        child_tyxyx_tlbr = torch.zeros((T, new_h, new_w, 5), device=device, dtype=torch.int32)

        if not even_height_flag and even_width_flag:
            h_start = 1 + 2 * torch.arange(0, new_h-1, device=device, dtype=torch.int32) # row_start
            w_start = 2 * torch.arange(0, new_w, device=device, dtype=torch.int32) # col_start
            ## (a) h=0: 1x2 block
            child_yx_coords[0, :, [0,1], 1] = torch.stack([w_start, w_start+1], dim=-1)
            child_valid_mask[0, :, [0,1]] = True
            child_tyxyx_tlbr[:, 0, :, 0] = prev_tyxyx_tlbr[:, 0, 0::2, 0]
            child_tyxyx_tlbr[:, 0, :, 1:3] = prev_tyxyx_tlbr[:, 0, 0::2, 1:3]
            child_tyxyx_tlbr[:, 0, :, 3:5] = prev_tyxyx_tlbr[:, 0, 1::2, 3:5]

            ## (b) h>=1: 2x2 block
            child_yx_coords[1:, :, [0,1,2,3], 0] = torch.stack([*([h_start]*2),*([h_start+1]*2)], dim=-1).unsqueeze(1).repeat(1, new_w, 1)
            child_yx_coords[1:, :, [0,1,2,3], 1] = torch.stack([w_start, w_start+1], dim=-1).repeat(1, 2).unsqueeze(0).repeat(new_h-1, 1, 1)
            child_valid_mask[1:, :, [0,1,2,3]] = True
            child_tyxyx_tlbr[:, 1:, :, 0] = prev_tyxyx_tlbr[:, 1::2, 0::2, 0]
            child_tyxyx_tlbr[:, 1:, :, 1:3] = prev_tyxyx_tlbr[:, 1::2, 0::2, 1:3]
            child_tyxyx_tlbr[:, 1:, :, 3:5] = prev_tyxyx_tlbr[:, 2::2, 1::2, 3:5]
        
        if even_height_flag and not even_width_flag:
            h_start = 2 * torch.arange(0, new_h, device=device, dtype=torch.int32) # row_start
            w_start = 1 + 2 * torch.arange(0, new_w-1, device=device, dtype=torch.int32) # col_start
            
            ## (a) w=0: 2x1 block
            child_yx_coords[:, 0, [0,2], 0] = torch.stack([h_start, h_start+1], dim=-1)
            child_valid_mask[:, 0, [0,2]] = True
            child_tyxyx_tlbr[:, :, 0, 0] = prev_tyxyx_tlbr[:, 0::2, 0, 0]
            child_tyxyx_tlbr[:, :, 0, 1:3] = prev_tyxyx_tlbr[:, 0::2, 0, 1:3]
            child_tyxyx_tlbr[:, :, 0, 3:5] = prev_tyxyx_tlbr[:, 1::2, 0, 3:5]

            ## (b) w>=1: 2x2 block
            child_yx_coords[:, 1:, [0,1,2,3], 0] = torch.stack([*([h_start]*2),*([h_start+1]*2)], dim=-1).unsqueeze(1).repeat(1, new_w-1, 1)
            child_yx_coords[:, 1:, [0,1,2,3], 1] = torch.stack([w_start, w_start+1], dim=-1).repeat(1, 2).unsqueeze(0).repeat(new_h, 1, 1)
            child_valid_mask[:, 1:, [0,1,2,3]] = True
            child_tyxyx_tlbr[:, :, 1:, 0] = prev_tyxyx_tlbr[:, 0::2, 1::2, 0]
            child_tyxyx_tlbr[:, :, 1:, 1:3] = prev_tyxyx_tlbr[:, 0::2, 1::2, 1:3]
            child_tyxyx_tlbr[:, :, 1:, 3:5] = prev_tyxyx_tlbr[:, 1::2, 2::2, 3:5]
        
        if not even_height_flag and not even_width_flag:
            h_start = 1 + 2 * torch.arange(0, new_h-1, device=device, dtype=torch.int32) # row_start
            w_start = 1 + 2 * torch.arange(0, new_w-1, device=device, dtype=torch.int32) # col_start

            ## (a). Top-left element (h=0, w=0): 1x1 block
            child_valid_mask[0, 0, 0] = True  # only the first slot is valid
            child_tyxyx_tlbr[:, 0, 0] = prev_tyxyx_tlbr[:, 0, 0]

            ## (b). Top-right element (h=0, w>=1): 1x2 block
            tr_index = [0,1]
            child_yx_coords[0, 1:, tr_index, 1] = torch.stack([w_start, w_start+1], dim=-1)
            child_valid_mask[0, 1:, tr_index] = True
            child_tyxyx_tlbr[:, 0, 1:, 0] = prev_tyxyx_tlbr[:, 0, 1::2, 0]
            child_tyxyx_tlbr[:, 0, 1:, 1:3] = prev_tyxyx_tlbr[:, 0, 1::2, 1:3]
            child_tyxyx_tlbr[:, 0, 1:, 3:5] = prev_tyxyx_tlbr[:, 0, 2::2, 3:5]
        
            # (c). Bottom-left element (h>=1, w=0): 2x1 block
            bl_index = [0,2]
            child_yx_coords[1:, 0, bl_index, 0] = torch.stack([h_start, h_start+1], dim=-1)
            child_valid_mask[1:, 0, bl_index] = True
            child_tyxyx_tlbr[:, 1:, 0, 0] = prev_tyxyx_tlbr[:, 1::2, 0, 0]
            child_tyxyx_tlbr[:, 1:, 0, 1:3] = prev_tyxyx_tlbr[:, 1::2, 0, 1:3]
            child_tyxyx_tlbr[:, 1:, 0, 3:5] = prev_tyxyx_tlbr[:, 2::2, 0, 3:5]
        
            # (d). Bottom-right element (h>=1, w>=1): 2×2 block
            br_index = [0,1,2,3]
            child_yx_coords[1:, 1:, br_index, 0] = torch.stack([*([h_start]*2),*([h_start+1]*2)], dim=-1).unsqueeze(1).repeat(1, new_w-1, 1)
            child_yx_coords[1:, 1:, br_index, 1] = torch.stack([w_start, w_start+1], dim=-1).repeat(1, 2).unsqueeze(0).repeat(new_h-1, 1, 1)
            child_valid_mask[1:, 1:, br_index] = True
            child_tyxyx_tlbr[:, 1:, 1:, 0] = prev_tyxyx_tlbr[:, 1::2, 1::2, 0]
            child_tyxyx_tlbr[:, 1:, 1:, 1:3] = prev_tyxyx_tlbr[:, 1::2, 1::2, 1:3]
            child_tyxyx_tlbr[:, 1:, 1:, 3:5] = prev_tyxyx_tlbr[:, 2::2, 2::2, 3:5]
        
        ## (e). Expand temporal dimension
        t_idx = torch.arange(0, T, device=device, dtype=torch.int32)
        t_coords = t_idx.reshape(-1, 1, 1, 1, 1).repeat(1, new_h, new_w, 4, 1)
        _child_yx_coords = child_yx_coords.unsqueeze(0).repeat(T, 1, 1, 1, 1)
        child_tyx_coords = torch.cat([t_coords, _child_yx_coords], dim=-1)
        child_valid_mask = child_valid_mask.unsqueeze(0).repeat(T, 1, 1, 1)
    
    return child_tyx_coords, child_valid_mask, child_tyxyx_tlbr
