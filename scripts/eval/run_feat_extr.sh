##### Start: Model Config
llava_video_only_7b_cfg="--model_name_or_path ckpts/LLaVA-Video-7B-Qwen2-Video-Only/"
llava_video_72b_cfg="--model_name_or_path ckpts/LLaVA-Video-72B-Qwen2/"
llava_ov_7b_cfg="--model_name_or_path ckpts/llava-onevision-qwen2-7b-ov/"
qwen2vl_7b_cfg="--model_name_or_path ckpts/Qwen2-VL-7B-Instruct/"
##### End

###### Start: Dataset Config
## for llava-based models
vmme_f128_fps1_p2_cfg="--dataset_name videomme --frames_upbound 128 --tgt_video_fps 1 --mm_spatial_pool_stride 2 "
vnb_f180_fps1_p2_cfg="--dataset_name vnbench --frames_upbound 180 --tgt_video_fps 1 --mm_spatial_pool_stride 2 "
egos_f128_fps1_p2_cfg="--dataset_name egoschema --frames_upbound 128 --tgt_video_fps 1 --mm_spatial_pool_stride 2 "
lvb_f128_fps1_p2_cfg="--dataset_name lvb-val --frames_upbound 128 --tgt_video_fps 1 --mm_spatial_pool_stride 2 "
nextqa_f128_fps1_p2_cfg="--dataset_name nextqa-mcq --frames_upbound 128 --tgt_video_fps 1 --mm_spatial_pool_stride 2 "
mlvu_f128_fps1_p2_cfg="--dataset_name mlvu-mcq --frames_upbound 128 --tgt_video_fps 1 --mm_spatial_pool_stride 2 "

## for qwen2vl
vmme_f256_fps2_cfg="--dataset_name videomme --frames_upbound 256 --tgt_video_fps 2 "
vnb_f360_fps2_cfg="--dataset_name vnbench --frames_upbound 360 --tgt_video_fps 2 "
egos_f256_fps2_cfg="--dataset_name egoschema --frames_upbound 256 --tgt_video_fps 2 "
lvb_f256_fps2_cfg="--dataset_name lvb-val --frames_upbound 256 --tgt_video_fps 2 "
nextqa_f256_fps2_cfg="--dataset_name nextqa-mcq --frames_upbound 256 --tgt_video_fps 2 "
mlvu_f256_fps2_cfg="--dataset_name mlvu-mcq --frames_upbound 256 --tgt_video_fps 2 "
##### End


##### Start: llava-video-only-7B
CUDA_VISIBLE_DEVICES=0 python llava/eval/video_feat_llavavideo.py ${vmme_f128_fps1_p2_cfg} ${llava_video_only_7b_cfg}
CUDA_VISIBLE_DEVICES=0 python llava/eval/video_feat_llavavideo.py ${vnb_f180_fps1_p2_cfg} ${llava_video_only_7b_cfg}
CUDA_VISIBLE_DEVICES=0 python llava/eval/video_feat_llavavideo.py ${egos_f128_fps1_p2_cfg} ${llava_video_only_7b_cfg}
CUDA_VISIBLE_DEVICES=0 python llava/eval/video_feat_llavavideo.py ${lvb_f128_fps1_p2_cfg} ${llava_video_only_7b_cfg}
CUDA_VISIBLE_DEVICES=0 python llava/eval/video_feat_llavavideo.py ${nextqa_f128_fps1_p2_cfg} ${llava_video_only_7b_cfg}
CUDA_VISIBLE_DEVICES=0 python llava/eval/video_feat_llavavideo.py ${mlvu_f128_fps1_p2_cfg} ${llava_video_only_7b_cfg}
##### End

##### Start: llava-video-72B
CUDA_VISIBLE_DEVICES=0,1,2,3 python llava/eval/video_feat_llavavideo.py ${vmme_f128_fps1_p2_cfg} ${llava_video_72b_cfg}
CUDA_VISIBLE_DEVICES=0,1,2,3 python llava/eval/video_feat_llavavideo.py ${vnb_f180_fps1_p2_cfg} ${llava_video_72b_cfg}
CUDA_VISIBLE_DEVICES=0,1,2,3 python llava/eval/video_feat_llavavideo.py ${egos_f128_fps1_p2_cfg} ${llava_video_72b_cfg}
CUDA_VISIBLE_DEVICES=0,1,2,3 python llava/eval/video_feat_llavavideo.py ${lvb_f128_fps1_p2_cfg} ${llava_video_72b_cfg}
CUDA_VISIBLE_DEVICES=0,1,2,3 python llava/eval/video_feat_llavavideo.py ${nextqa_f128_fps1_p2_cfg} ${llava_video_72b_cfg}
CUDA_VISIBLE_DEVICES=0,1,2,3 python llava/eval/video_feat_llavavideo.py ${mlvu_f128_fps1_p2_cfg} ${llava_video_72b_cfg}
##### End

##### Start: llava-onevision-7B
CUDA_VISIBLE_DEVICES=0 python llava/eval/video_feat_llavavideo.py ${vmme_f128_fps1_p2_cfg} ${llava_ov_7b_cfg}
CUDA_VISIBLE_DEVICES=0 python llava/eval/video_feat_llavavideo.py ${vnb_f180_fps1_p2_cfg} ${llava_ov_7b_cfg}
CUDA_VISIBLE_DEVICES=0 python llava/eval/video_feat_llavavideo.py ${egos_f128_fps1_p2_cfg} ${llava_ov_7b_cfg}
CUDA_VISIBLE_DEVICES=0 python llava/eval/video_feat_llavavideo.py ${lvb_f128_fps1_p2_cfg} ${llava_ov_7b_cfg}
CUDA_VISIBLE_DEVICES=0 python llava/eval/video_feat_llavavideo.py ${nextqa_f128_fps1_p2_cfg} ${llava_ov_7b_cfg}
CUDA_VISIBLE_DEVICES=0 python llava/eval/video_feat_llavavideo.py ${mlvu_f128_fps1_p2_cfg} ${llava_ov_7b_cfg}
##### End

##### Start: qwen2-vl-7B
CUDA_VISIBLE_DEVICES=0 python llava/eval/video_feat_qwen2vl.py ${vmme_f256_fps2_cfg} ${qwen2vl_7b_cfg}
CUDA_VISIBLE_DEVICES=0 python llava/eval/video_feat_qwen2vl.py ${vnb_f360_fps2_cfg} ${qwen2vl_7b_cfg}
CUDA_VISIBLE_DEVICES=0 python llava/eval/video_feat_qwen2vl.py ${egos_f256_fps2_cfg} ${qwen2vl_7b_cfg}
CUDA_VISIBLE_DEVICES=0 python llava/eval/video_feat_qwen2vl.py ${lvb_f256_fps2_cfg} ${qwen2vl_7b_cfg}
CUDA_VISIBLE_DEVICES=0 python llava/eval/video_feat_qwen2vl.py ${nextqa_f256_fps2_cfg} ${qwen2vl_7b_cfg}
CUDA_VISIBLE_DEVICES=0 python llava/eval/video_feat_qwen2vl.py ${mlvu_f256_fps2_cfg} ${qwen2vl_7b_cfg}
##### End
