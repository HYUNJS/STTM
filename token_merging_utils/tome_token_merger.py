from typing import Callable, Tuple
import torch
import torch.nn.functional as F
import math
import einops
import pickle


def do_nothing(x, mode=None):
    return x


def bipartite_soft_matching(
    metric: torch.Tensor,
    r: int, 
) -> Tuple[Callable, Callable]:
    """
    Applies ToMe with a balanced matching set (50%, 50%).

    Input size is [batch, tokens, channels].
    r indicates the number of tokens to remove (max 50% of tokens).
    """
    protected = 0

    t = metric.shape[1]
    r = min(r, (t - protected) // 2)

    if r <= 0:
        return do_nothing, do_nothing

    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = metric[..., ::2, :], metric[..., 1::2, :]
        scores = a @ b.transpose(-1, -2)

        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
        src_idx = edge_idx[..., :r, :]  # Merged Tokens
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)

    def merge(x: torch.Tensor, token_idx: torch.Tensor = None, mode="mean") -> torch.Tensor:
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        n, t1, c = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src = src.gather(dim=-2, index=src_idx.expand(n, r, c)) # 
        dst = dst.scatter_add(-2, dst_idx.expand(n, r, c), src) # , reduce=mode)
        merged_feats = torch.cat([unm, dst], dim=1)

        if token_idx is not None:
            src_token_idx, dst_token_idx = token_idx[..., ::2, :], token_idx[..., 1::2, :]
            unm_token_idx = src_token_idx.gather(dim=-2, index=unm_idx.expand(n, t1 - r, 1))
            merged_token_idx = torch.cat([unm_token_idx, dst_token_idx], dim=1)
            return merged_feats, merged_token_idx
        else:
            return merged_feats

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        n, _, c = unm.shape

        src = dst.gather(dim=-2, index=dst_idx.expand(n, r, c))

        out = torch.zeros(n, metric.shape[1], c, device=x.device, dtype=x.dtype)

        out[..., 1::2, :] = dst
        out.scatter_(dim=-2, index=(2 * unm_idx).expand(n, unm_len, c), src=unm)
        out.scatter_(dim=-2, index=(2 * src_idx).expand(n, r, c), src=src)

        return out

    return merge, unmerge


def merge_wavg(
    merge: Callable, x: torch.Tensor, token_idx: torch.Tensor, size: torch.Tensor = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies the merge function by taking a weighted average based on token size.
    Returns the merged tensor and the new token sizes.
    """
    if size is None:
        size = torch.ones_like(x[..., 0, None])

    x, _token_idx = merge(x * size, token_idx, mode="sum")
    size = merge(size, mode="sum")

    x = x / size
    return x, _token_idx, size


# def merge_tokens(x, r_merge_list=[1280, 640, 320, 160, 80, 40, 10], clips=10):
#     size = None
#     head = 16
#     b, p, c = x.shape
#     dim = c // head
#     x = x.reshape(b//clips, clips*p, -1) # [nc, ws*H*W, C] , where T=nc*ws
#     bclips, p, c = x.shape
#     for r in r_merge_list:
#         metric = x.reshape(bclips, p, head, dim).mean(2) # [b, p, c//head]
#         merge, _ = bipartite_soft_matching(
#             metric, 
#             r
#         )
#         x, size = merge_wavg(merge, x, size)
#         _, p, _ = x.shape # [nc, ws*H*W/2, C]
#     x = x.reshape(-1, c)  # 
#     return x

def tome_per_frame(x, prune_ratio, n_head=1):
    ## Suppose each snippet spanning one frame
    size = None
    x = einops.rearrange(x, "t c h w -> t (h w) c")
    num_snippet, num_tokens_per_snippet, c = x.shape
    dim_head = c // n_head
    tgt_num_tokens = math.ceil(num_tokens_per_snippet * (1-prune_ratio))
    curr_num_tokens = num_tokens_per_snippet
    num_iter = 0
    token_idx = torch.arange(curr_num_tokens, device=x.device).reshape(1, -1, 1)
    while num_iter == 0 or curr_num_tokens > tgt_num_tokens:
        metric = x.reshape(num_snippet, curr_num_tokens, n_head, dim_head).mean(2) # [num_snippet, num_tokens_per_snippet, head_dim]
        num_remove_tokens = curr_num_tokens - tgt_num_tokens
        merge, _ = bipartite_soft_matching(metric, num_remove_tokens)
        x, token_idx, size = merge_wavg(merge, x, token_idx, size)
        _, curr_num_tokens, _ = x.shape
        num_iter += 1
    x = x.reshape(-1, c)
    token_idx_final = token_idx.reshape(-1)
    return x, token_idx_final

def tome_per_video(x, prune_ratio, n_head=1):
    ## Suppose each snippet spanning whole video
    size = None
    x = einops.rearrange(x, "t c h w -> 1 (t h w) c")
    num_snippet, num_tokens_per_snippet, c = x.shape
    dim_head = c // n_head
    tgt_num_tokens = math.ceil(num_tokens_per_snippet * (1-prune_ratio))
    curr_num_tokens = num_tokens_per_snippet
    num_iter = 0
    token_idx = torch.arange(curr_num_tokens, device=x.device).reshape(1, -1, 1)
    while num_iter == 0 or curr_num_tokens > tgt_num_tokens:
        metric = x.reshape(num_snippet, curr_num_tokens, n_head, dim_head).mean(2) # [num_snippet, num_tokens_per_snippet, head_dim]
        num_remove_tokens = curr_num_tokens - tgt_num_tokens
        merge, _ = bipartite_soft_matching(metric, num_remove_tokens)
        x, token_idx, size = merge_wavg(merge, x, token_idx, size)
        _, curr_num_tokens, _ = x.shape
        num_iter += 1
    x = x.reshape(-1, c)
    token_idx_final = token_idx.reshape(-1)
    return x, token_idx_final

def tome_per_snippet(x, prune_ratio, n_head=1, snippet_size=1):
    ## TODO. THis is not yet implemented
    return None


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
    # _video_feature = einops.rearrange(_video_feature, "t c h w -> t (h w) c")
    # _video_feature = einops.rearrange(_video_feature, "t c h w -> 1 (t h w) c")
    # x = tome_per_frame(_video_feature, merge_ratio_list=[0.25], snippet_size=1, n_head=16)
    x, token_idx_final = tome_per_video(_video_feature, prune_ratio=0.80, n_head=1)
    
    _video_feature
    