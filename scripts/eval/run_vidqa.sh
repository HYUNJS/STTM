##### Start: model cfg
llava_video_only_7b_with_grid="--model_name_or_path ckpts/LLaVA-Video-7B-Qwen2-Video-Only/ --mm_newline_position grid"
llava_video_72b_with_grid="--model_name_or_path ckpts/LLaVA-Video-72B-Qwen2/ --mm_newline_position grid"
llava_ov_7b_with_grid="--model_name_or_path ckpts/llava-onevision-qwen2-7b-ov/ --mm_newline_position grid"

llava_video_7b="--model_name_or_path ckpts/LLaVA-Video-7B-Qwen2-Video-Only/ --mm_newline_position no_token"
llava_video_72b="--model_name_or_path ckpts/LLaVA-Video-72B-Qwen2/ --mm_newline_position no_token"
llava_ov_7b="--model_name_or_path ckpts/llava-onevision-qwen2-7b-ov/ --mm_newline_position no_token"
qwen2vl_7b="--model_name_or_path ckpts/Qwen2-VL-7B-Instruct/"
##### End

##### Start: data loader cfg for llava-video and llava-onevision
vmme_f128_cfg=" --dataset_name videomme --frames_upbound 128 --tgt_video_fps 1 --mm_spatial_pool_stride 2 --rope_scaling_factor 1.0 "
vnb_f180_cfg=" --dataset_name vnbench --frames_upbound 180 --tgt_video_fps 1 --mm_spatial_pool_stride 2 --rope_scaling_factor 2.0 "
egos_f128_cfg=" --dataset_name egoschema --frames_upbound 128 --tgt_video_fps 1 --mm_spatial_pool_stride 2 --rope_scaling_factor 1.0 "
lvb_f128_cfg=" --dataset_name lvb-val --frames_upbound 128 --tgt_video_fps 1 --mm_spatial_pool_stride 2 --rope_scaling_factor 1.0 "
next_f128_cfg=" --dataset_name nextqa-mcq --frames_upbound 128 --tgt_video_fps 1 --mm_spatial_pool_stride 2 --rope_scaling_factor 1.0 "
mlvu_f128_cfg=" --dataset_name mlvu-mcq --frames_upbound 128 --tgt_video_fps 1 --mm_spatial_pool_stride 2 --rope_scaling_factor 1.0 "
##### End

##### Start: data loader cfg for qwen2vl
vmme_f256_cfg=" --dataset_name videomme --frames_upbound 256 --tgt_video_fps 2"
vnb_f360_cfg=" --dataset_name vnbench --frames_upbound 360 --tgt_video_fps 2"
egos_f256_cfg=" --dataset_name egoschema --frames_upbound 256 --tgt_video_fps 2"
lvb_f256_cfg=" --dataset_name lvb-val --frames_upbound 256 --tgt_video_fps 2"
next_f256_cfg=" --dataset_name nextqa-mcq --frames_upbound 256 --tgt_video_fps 2"
mlvu_f256_cfg=" --dataset_name mlvu-mcq --frames_upbound 256 --tgt_video_fps 2"
##### #nd

##### Start: other token reduction methods
fastv_cfg_50="--sa_pattern fastv --sa_start_layer_idx 2 --sa_fastv_evict_ratio 0.50"
fastv_cfg_30="--sa_pattern fastv --sa_start_layer_idx 2 --sa_fastv_evict_ratio 0.70"
fastv_cfg_20="--sa_pattern fastv --sa_start_layer_idx 2 --sa_fastv_evict_ratio 0.80"
fastv_cfg_15="--sa_pattern fastv --sa_start_layer_idx 2 --sa_fastv_evict_ratio 0.85"

framefusion_cfg_50="--sa_pattern framefusion --sa_framefusion_cost 0.50"
framefusion_cfg_30="--sa_pattern framefusion --sa_framefusion_cost 0.30"
framefusion_cfg_20="--sa_pattern framefusion --sa_framefusion_cost 0.20"
framefusion_cfg_15="--sa_pattern framefusion --sa_framefusion_cost 0.15"

dycoke_cfg_50="--sa_pattern dycoke --sa_start_layer_idx 0 --sa_prune_ratio 0.7 --dycoke_l 3 --dycoke_p 0.8"
dycoke_cfg_30="--sa_pattern dycoke --sa_start_layer_idx 0 --sa_prune_ratio 0.925 --dycoke_l 3 --dycoke_p 0.8"

tome_cfg_50="--sa_pattern tome --sa_start_layer_idx 2 --sa_prune_ratio 0.50 --sa_tome_ver video"
tome_cfg_30="--sa_pattern tome --sa_start_layer_idx 2 --sa_prune_ratio 0.70 --sa_tome_ver video"
tome_cfg_15="--sa_pattern tome --sa_start_layer_idx 2 --sa_prune_ratio 0.85 --sa_tome_ver video"

dycoke_stage1_cfg_2_50="--sa_pattern dycoke-stage1 --sa_start_layer_idx 2 --sa_prune_ratio 0.7"
dycoke_stage1_cfg_2_30="--sa_pattern dycoke-stage1 --sa_start_layer_idx 2 --sa_prune_ratio 0.925"
dycoke_stage1_cfg_2_25="--sa_pattern dycoke-stage1 --sa_start_layer_idx 2 --sa_prune_ratio 1.0"
##### End

##### Start: STTM token reduction method
sttm_common_cfg="--sa_pattern quadtree --sa_start_layer_idx 2 --sa_tree_root_level 1 "
## llava-video-qwen2-7b-video-only
sttm_cfg_50_vnb_llavavideo_7b="${sttm_common_cfg} --sa_tree_thresh 0.85 --sa_tree_temporal_thresh 0.65"
sttm_cfg_30_vnb_llavavideo_7b="${sttm_common_cfg} --sa_tree_thresh 0.80 --sa_tree_temporal_thresh 0.60"
sttm_cfg_50_vmme_llavavideo_7b="${sttm_common_cfg} --sa_tree_thresh 0.85 --sa_tree_temporal_thresh 0.55"
sttm_cfg_30_vmme_llavavideo_7b="${sttm_common_cfg} --sa_tree_thresh 0.80 --sa_tree_temporal_thresh 0.50"
sttm_cfg_50_egos_llavavideo_7b="${sttm_common_cfg} --sa_tree_thresh 0.85 --sa_tree_temporal_thresh 0.55"
sttm_cfg_30_egos_llavavideo_7b="${sttm_common_cfg} --sa_tree_thresh 0.80 --sa_tree_temporal_thresh 0.60"
sttm_cfg_50_next_llavavideo_7b="${sttm_common_cfg} --sa_tree_thresh 0.85 --sa_tree_temporal_thresh 0.65"
sttm_cfg_30_next_llavavideo_7b="${sttm_common_cfg} --sa_tree_thresh 0.80 --sa_tree_temporal_thresh 0.65"
sttm_cfg_50_lvb_llavavideo_7b="${sttm_common_cfg} --sa_tree_thresh 0.85 --sa_tree_temporal_thresh 0.60"
sttm_cfg_30_lvb_llavavideo_7b="${sttm_common_cfg} --sa_tree_thresh 0.80 --sa_tree_temporal_thresh 0.55"
sttm_cfg_50_mlvu_llavavideo_7b="${sttm_common_cfg} --sa_tree_thresh 0.85 --sa_tree_temporal_thresh 0.55"
sttm_cfg_30_mlvu_llavavideo_7b="${sttm_common_cfg} --sa_tree_thresh 0.80 --sa_tree_temporal_thresh 0.55"
## llava-onevision-7b
sttm_cfg_50_vnb_llavaov_7b="${sttm_common_cfg} --sa_tree_thresh 0.90 --sa_tree_temporal_thresh 0.65"
sttm_cfg_30_vnb_llavaov_7b="${sttm_common_cfg} --sa_tree_thresh 0.85 --sa_tree_temporal_thresh 0.65"
sttm_cfg_50_vmme_llavaov_7b="${sttm_common_cfg} --sa_tree_thresh 0.85 --sa_tree_temporal_thresh 0.65"
sttm_cfg_30_vmme_llavaov_7b="${sttm_common_cfg} --sa_tree_thresh 0.85 --sa_tree_temporal_thresh 0.55"
sttm_cfg_50_egos_llavaov_7b="${sttm_common_cfg} --sa_tree_thresh 0.90 --sa_tree_temporal_thresh 0.60"
sttm_cfg_30_egos_llavaov_7b="${sttm_common_cfg} --sa_tree_thresh 0.85 --sa_tree_temporal_thresh 0.60"
sttm_cfg_50_next_llavaov_7b="${sttm_common_cfg} --sa_tree_thresh 0.95 --sa_tree_temporal_thresh 0.65"
sttm_cfg_30_next_llavaov_7b="${sttm_common_cfg} --sa_tree_thresh 0.90 --sa_tree_temporal_thresh 0.60"
sttm_cfg_50_lvb_llavaov_7b="${sttm_common_cfg} --sa_tree_thresh 0.90 --sa_tree_temporal_thresh 0.60"
sttm_cfg_30_lvb_llavaov_7b="${sttm_common_cfg} --sa_tree_thresh 0.85 --sa_tree_temporal_thresh 0.60"
sttm_cfg_50_mlvu_llavaov_7b="${sttm_common_cfg} --sa_tree_thresh 0.90 --sa_tree_temporal_thresh 0.55"
sttm_cfg_30_mlvu_llavaov_7b="${sttm_common_cfg} --sa_tree_thresh 0.85 --sa_tree_temporal_thresh 0.60"
## qwen2vl-7b
sttm_cfg_50_vnb_qwen2vl_7b="${sttm_common_cfg} --sa_tree_thresh 0.80 --sa_tree_temporal_thresh 0.65"
sttm_cfg_30_vnb_qwen2vl_7b="${sttm_common_cfg} --sa_tree_thresh 0.75 --sa_tree_temporal_thresh 0.50"
sttm_cfg_50_vmme_qwen2vl_7b="${sttm_common_cfg} --sa_tree_thresh 0.85 --sa_tree_temporal_thresh 0.60"
sttm_cfg_30_vmme_qwen2vl_7b="${sttm_common_cfg} --sa_tree_thresh 0.80 --sa_tree_temporal_thresh 0.60"
sttm_cfg_50_lvb_qwen2vl_7b="${sttm_common_cfg} --sa_tree_thresh 0.85 --sa_tree_temporal_thresh 0.65"
sttm_cfg_30_lvb_qwen2vl_7b="${sttm_common_cfg} --sa_tree_thresh 0.80 --sa_tree_temporal_thresh 0.60"
## llava-video-qwen2-72b
sttm_cfg_50_vmme_llavavideo_72b="--sa_pattern quadtree --sa_start_layer_idx 0 --sa_tree_root_level 1 --sa_tree_thresh 0.94 --sa_tree_temporal_thresh 0.82"
sttm_cfg_30_vmme_llavavideo_72b="--sa_pattern quadtree --sa_start_layer_idx 0 --sa_tree_root_level 1 --sa_tree_thresh 0.90 --sa_tree_temporal_thresh 0.90"
##### End

script_llava="python llava/eval/eval_vidqa_by_feat_llavavideo.py "
script_qwen2vl="python llava/eval/eval_vidqa_by_feat_qwen2vl.py "

## Please follow the given structure
## python llava/eval/eval_vidqa_by_feat.py --reproduce ${<data_loader_cfg>} ${llava_video_only_7b_with_grid} ${<token_reduction_cfg>}

# example for running llava-video-7b & vnbench & STTM & 50 percent token reduction
device=0
CUDA_VISIBLE_DEVICES=${device} ${script_llava} --reproduce ${vnb_f180_cfg} ${model_path_llava_video_only_7b} ${sttm_cfg_50_vnb_llavavideo_7b}
