from token_merging_utils.dycoke_merger import get_quadtree_features

from token_merging_utils.tome_token_merger import tome_per_frame, tome_per_video, tome_per_snippet

def get_dycoke_features(_video_feature, prune_ratio, tome_ver, n_head=1):
    if tome_ver == "frame":
        return tome_per_frame(_video_feature, prune_ratio, n_head)
    elif tome_ver == "video":
        return tome_per_video(_video_feature, prune_ratio, n_head)
    elif tome_ver == "snippet":
        return tome_per_snippet(_video_feature, prune_ratio, n_head, snippet_size=4)
