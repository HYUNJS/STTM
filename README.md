# Multi-Granular Spatio-Temporal Token Merging for Training-Free Acceleration of Video LLMs

[arXiv](https://arxiv.org/abs/2507.07990) | [Project Page](https://www.jshyun.me/projects/sttm)

## TL;DR

STTM is a training-free spatio-temporal token merging method that supports KV-cache reuse.
It operates in two steps: (1) Spatial merging based on a quadtree structure; (2) Temporal merging of multi-granular spatial tokens;

STTM is validated using three models: LLaVA-Video-7B/72B, LLaVA-OneVision-7B, and Qwen2VL-7B. Evaluation is conducted across six video QA benchmarks:
* NIAH: VNBench
* Long videos: Video-MME; LongVideoBench; MLVU
* Short videos: NExT-QA; EgoSchema


## Environment Setup

```bash
git clone https://github.com/HYUNJS/STTM.git
cd STTM

## (1) Option with conda. I used virtualenv for experimentation.
conda create -n sttm python=3.10 -y
conda activate sttm

pip install -e ".[train]" --extra-index-url https://download.pytorch.org/whl/cu121  # for cu121 - default is cu124
pip install flash-attn==2.7.3 --no-build-isolation # compatible version with torch==2.5.1
```


## 🗂️ Dataset Setup
Please prepare the checkpoints in `./ckpts/` folder.

The datasets are organized as follows:
```
datasets/
├── egoschema/
├── longvideobench/
├── mlvu/
├── nextqa/
├── videomme/
└── vnbench/
    ├── annotations/
    ├── videos/ (Optional) for feature extraction and visualization
    └── preprocess_data/
        ├── {model_name}/
        │   └── {frame_sampling_name}/
        │       ├── features/
        │       │   └── {vid}.pt
        │       └── metadata/
        │           └── {vid}.pkl
        └── llava-video-7b-qwen2-video-only/
            └── F-180_fps-1/
                ├── features/
                │   └── 10109006686_cnt_edit1.pt
                └── metadata/
                    └── 10109006686_cnt_edit1.pkl
```

* Each benchmark (e.g., egoschema, longvideobench, etc.) has its own folder.
* `videos/`: Raw video files (can be removed after feature extraction).
* `annotations/`: Contains annotation files (some are reformatted) for the benchmark. We format some benchmarks and save in `sttm_annotations/` folder. Please copy it to setup the datasets.
* `preprocess_data/`: Stores preprocessed features and metadata. 
* Model-specific preprocessed data is stored in the `{model_name}/` folder. `llava-video-7b-qwen2-video-only/` is example of model directory.
* `{frame_sampling_name}/`: Name of frame sampling strategy used for feature extraction (e.g., F-128_fps-1 or F-180_fps-1).
* `features/`: Extracted video features ({vid.pt}).
* `metadata/`: Associated metadata ({vid.pkl}).

To help you get started easily, we provide preprocessed feature data for Video-MME and VNBench on HuggingFace.
Each dataset includes multiple frame sampling setups (e.g., F-64_fps-1, F-128_fps-1).
Please use the Hugging Face Hub API to selectively download only the configurations you need.
* Video-MME: https://huggingface.co/datasets/js-hyun/preprocess-videomme-data
* VNBench: https://huggingface.co/datasets/js-hyun/preprocess-vnbench-data

## 📁 File Structure

The project is organized into modular components for token merging, model adaptation, and evaluation. Below is a brief overview of the key directories and scripts:

- `token_merging_utils/`  
  Core implementations of the token merging algorithms.

- `token_merging_monkey_patch/`  
  Monkey patch files for injecting token merging into intermediate LLM layers of LLaVA-Video and LLaVA-OneVision models.

- `token_merging_qwen2vl_monkey_patch/`  
  Monkey patch files tailored for the Qwen2VL model.

- `llava/eval/video_feat_{model_name}.py`  
  Video feature extractor script.  
  ➤ Example: `video_feat_llavavideo.py`

- `llava/eval/eval_vidqa_by_feat_{model_name}.py`  
  Video QA evaluation using pre-extracted features.

- `llava/eval/eval_vidqa_by_video_{model_name}.py`  
  Video QA evaluation directly from raw video input.

- `llava/eval/metric_{dataset_name}.py`  
  Metric computation scripts specific to each dataset.  
  ➤ Example: `metric_vnbench.py`, `metric_videomme.py`

## 🏃‍♂️ How to Run
### 🔹 Frame Extraction

To extract video frames and features, refer to the following script:

- `scripts/eval/run_feat_extr.sh` – Example commands for running feature extraction.

### 🔹 Reproducible Evaluation

For reproducible results, we provide a `--reproduce` flag that sets a fixed random seed and enables deterministic CUDA operations.

- `scripts/eval/run_vidqa.sh` – Contains example commands for video QA evaluation with reproducibility enabled.

Frame extraction. Please refer to `scripts/eval/run_feat_extr.sh` for the commands.

For getting reproducible result, we support `--reproduce` option which sets fixed random seed and makes deterministic CUDA operation. Please refer to `scripts/eval/run_vidqa.sh` for the commands.
The basic format is:
```bash
CUDA_VISIBLE_DEVICES=${device} \
python llava/eval/eval_vidqa_by_feat_{model_name}.py \
  --reproduce \
  ${<data_loader_cfg>} \
  ${<model_cfg>} \
  ${<token_reduction_cfg>}
```

## Citation

If you find this project helpful for your research or applications, please cite our paper:

```bibtex
@article{hyun2025multi,
  title={Multi-Granular Spatio-Temporal Token Merging for Training-Free Acceleration of Video LLMs},
  author={Hyun, Jeongseok and Hwang, Sukjun and Han, Su Ho and Kim, Taeoh and Lee, Inwoong and Wee, Dongyoon and Lee, Joon-Young and Kim, Seon Joo and Shim, Minho},
  journal={arXiv preprint arXiv:2507.07990},
  year={2025}
}
```

## Acknowledgement 🙏
We would like to thank the authors of the following projects for their valuable contributions, which our work builds upon or references:
- [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT): We use its codebase for the LLaVA architecture, including the llava-video and llava-onevision models.
- [ToMe](https://github.com/facebookresearch/ToMe), [DyCoke](https://github.com/KD-TAO/DyCoke), and [FrameFusion](https://github.com/thu-nics/FrameFusion): These codebases are used as references for our baseline experiments.
