from token_merging_utils.quadtree_builder import quadtree_build_video
from token_merging_utils.quadtree_builder_vis import quadtree_build_video_vis


def get_quadtree_features(_video_feature, threshold, temporal_thresh=-1.0, root_level=0, weighted_avg=False, 
                        vis_flag=False, slow_ver=False, head_dim=None, pos_embs=None, pos_emb_weighted_avg=False):
    if vis_flag:
        return quadtree_build_video_vis(_video_feature, threshold, temporal_thresh, root_level, 
                                    weighted_avg=weighted_avg)
    else:
        return quadtree_build_video(_video_feature, threshold, temporal_thresh, root_level, 
                                    weighted_avg=weighted_avg, slow_ver=slow_ver, head_dim=head_dim,
                                    pos_embs=pos_embs, pos_emb_weighted_avg=pos_emb_weighted_avg)
