## for llava-video and llava-onevision
from token_merging_monkey_patch.quadtree_attn_monkey_patch import replace_qwen2_with_quadtree_attn
from token_merging_monkey_patch.quadtree_attn_monkey_patch_for_abl_pos import replace_qwen2_with_quadtree_attn_for_abl_pos
from token_merging_monkey_patch.quadtree_attn_monkey_patch_for_vis import replace_qwen2_with_quadtree_attn_for_vis
from token_merging_monkey_patch.pyrd_attn_monkey_patch import replace_qwen2_with_pyrd_attn
from token_merging_monkey_patch.octree_attn_monkey_patch import replace_qwen2_with_octree_attn
from token_merging_monkey_patch.tome_attn_monkey_patch import replace_qwen2_with_tome_attn
from token_merging_monkey_patch.dycoke_attn_monkey_patch import replace_qwen2_with_dycoke_attn
from token_merging_monkey_patch.dycoke_stage1_attn_monkey_patch import replace_qwen2_with_dycoke_stage1_attn

## for qwen2vl
from token_merging_qwen2vl_monkey_patch.quadtree_attn_monkey_patch import replace_qwen2vl_with_quadtree_attn
from token_merging_qwen2vl_monkey_patch.tome_attn_monkey_patch import replace_qwen2vl_with_tome_attn
from token_merging_qwen2vl_monkey_patch.dycoke_stage1_attn_monkey_patch import replace_qwen2vl_with_dycoke_stage1_attn


def replace_qwen2_by_sparse_attn(pattern_name, **kwargs):
    if pattern_name == "quadtree":
        replace_qwen2_with_quadtree_attn(**kwargs)
        replace_qwen2vl_with_quadtree_attn(**kwargs)
    elif pattern_name == "quadtree-abl-pos":
        replace_qwen2_with_quadtree_attn_for_abl_pos(**kwargs)
    elif pattern_name == "octree":
        replace_qwen2_with_octree_attn(**kwargs)
    elif pattern_name == "pyrd":
        replace_qwen2_with_pyrd_attn(**kwargs)
    elif pattern_name == "quadtree_vis":
        replace_qwen2_with_quadtree_attn_for_vis(**kwargs)
    elif pattern_name == "tome":
        replace_qwen2_with_tome_attn(**kwargs)
        replace_qwen2vl_with_tome_attn(**kwargs)
    elif pattern_name == "dycoke":
        replace_qwen2_with_dycoke_attn(**kwargs)
    elif pattern_name == "dycoke-stage1":
        replace_qwen2_with_dycoke_stage1_attn(**kwargs)
        replace_qwen2vl_with_dycoke_stage1_attn(**kwargs)
    else:
        raise NotImplementedError(f"{pattern_name} is not yet implemented")
