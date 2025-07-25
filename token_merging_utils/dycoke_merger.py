import torch
import math
import pickle
import torch.nn.functional as F
import einops


def dycoke_ttm(image_feature, num_frames, prune_ratio=0.7):

    # Split frames into tokens
    num_tokens_per_frame = image_feature.shape[0] // num_frames
    keep_ratio = 1 - prune_ratio
    # Calculate similarities between adjacent even frames
    similarities = []
    for i in range(0, num_frames - 1, 2):
        # Get tokens for adjacent frames
        frame1_tokens = image_feature[i * num_tokens_per_frame: (i + 1) * num_tokens_per_frame]
        frame2_tokens = image_feature[(i + 1) * num_tokens_per_frame: (i + 2) * num_tokens_per_frame]

        similarity = F.cosine_similarity(frame1_tokens, frame2_tokens, dim=1)
        similarities.append(similarity)

    # similarities = torch.stack([torch.tensor(similarity) for similarity in similarities])
    similarities = torch.stack(similarities)
    
    patch_idx_frame = torch.arange(num_tokens_per_frame, device=image_feature.device)
    # Process even frames
    modified_image_feature = []
    token_indices = []
    for i in range(0, num_frames - 1, 2):
        frame1_start_idx, frame1_end_idx = i*num_tokens_per_frame, (i+1)*num_tokens_per_frame
        frame2_start_idx, frame2_end_idx = (i+1)*num_tokens_per_frame, (i+2)*num_tokens_per_frame
        frame1_tokens = image_feature[frame1_start_idx:frame1_end_idx]
        frame2_tokens = image_feature[frame2_start_idx:frame2_end_idx]
        
        avg_similarity = similarities[i // 2]
        num_tokens_to_keep = int(keep_ratio * num_tokens_per_frame)
        tokens_to_keep = avg_similarity.topk(num_tokens_to_keep, largest=False).indices
        
        modified_image_feature.append(frame1_tokens)
        modified_image_feature.append(frame2_tokens[tokens_to_keep])

        token_indices.append(patch_idx_frame + frame1_start_idx)
        token_indices.append(patch_idx_frame[tokens_to_keep] + frame2_start_idx)

    if len(modified_image_feature) < num_frames:
        ## when odd-number frames are given, the last frame is omitted in the original code
        i = len(modified_image_feature)
        frame_start_idx, frame_end_idx = i*num_tokens_per_frame, (i+1)*num_tokens_per_frame
        modified_image_feature.append(image_feature[frame_start_idx:frame_end_idx])
        token_indices.append(patch_idx_frame + frame_start_idx)
    
    # Process odd frames
    odd_similarities = []
    for i in range(0, num_frames - 4, 4):
        frame1_tokens = image_feature[i * num_tokens_per_frame: (i + 1) * num_tokens_per_frame]
        frame2_tokens = image_feature[(i + 2) * num_tokens_per_frame: (i + 3) * num_tokens_per_frame]
        
        similarity = F.cosine_similarity(frame1_tokens, frame2_tokens, dim=1)
        odd_similarities.append(similarity)

    # odd_similarities = torch.stack([torch.tensor(similarity) for similarity in odd_similarities])
    odd_similarities = torch.stack(odd_similarities)

    for i in range(0, num_frames - 4, 4):
        frame1_start_idx, frame1_end_idx = i*num_tokens_per_frame, (i+1)*num_tokens_per_frame
        frame2_start_idx, frame2_end_idx = (i+2)*num_tokens_per_frame, (i+3)*num_tokens_per_frame
        frame1_tokens = image_feature[i * num_tokens_per_frame: (i + 1) * num_tokens_per_frame]
        frame2_tokens = image_feature[(i + 2) * num_tokens_per_frame: (i + 3) * num_tokens_per_frame]
        
        avg_similarity = odd_similarities[i // 4]
        num_tokens_to_keep = int(keep_ratio * num_tokens_per_frame)
        tokens_to_keep = avg_similarity.topk(num_tokens_to_keep, largest=False).indices
        
        modified_image_feature[i] = frame1_tokens
        modified_image_feature[i + 2] = frame2_tokens[tokens_to_keep]

        token_indices[i+2] = token_indices[i+2][tokens_to_keep]

    # Combine all tokens
    combined_tokens = torch.cat(modified_image_feature, dim=0)
    combined_indices = torch.cat(token_indices, dim=0)
    return combined_tokens, combined_indices


if __name__ == "__main__":
    vid = "fFjv93ACGo8" # first qid
    feat_filepath = f"datasets/videomme/preprocess_data/llava-video-7b-qwen2-video-only/F-128_fps-1/features/{vid}.pt"
    _video_feature = torch.load(feat_filepath, weights_only=True)
    meta_filepath = f"datasets/videomme/preprocess_data/llava-video-7b-qwen2-video-only/F-128_fps-1/metadata/{vid}.pkl"
    with open(meta_filepath, "rb") as fp:
        metadata = pickle.load(fp)
    
    # _video_feature = torch.rand((60, 24*24, 256))
    # _video_feature = _video_feature.to("cuda")

    T, HW, D = _video_feature.shape
    H = int(math.sqrt(HW))
    W = H # H, W = 27, 27
    device = _video_feature.device
    stride = 2
    scaled_shape = [math.ceil(H / stride), math.ceil(W / stride)]
    _video_feature = _video_feature.view(T, H, W, -1).permute(0, 3, 1, 2).contiguous() # [T C H W] 27x27
    _video_feature = F.interpolate(_video_feature, scaled_shape) # 14x14

    video_feature = einops.rearrange(_video_feature, "t c h w -> (t h w) c")

    # merging_ratio=0.3 -> 47.2%
    # merging_ratio=0.075 -> 30.4%
    # merging_ratio=0.005 -> 25%
    ttm_features_video, ttm_indices = dycoke_ttm(video_feature, T, 0.7)
    _video_feature
    