from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from transformers import Qwen2_5_VLForConditionalGeneration
from PIL import Image
import requests
import torch
from token_merging_gemma3_monkey_patch.quadtree_attn_monkey_patch import replace_gemma3_with_quadtree_attn


# torch._dynamo.disable()

if __name__  == "__main__":
    # frames = ["https://url/to/frame1.jpg", ..., "https://url/to/frame8.jpg"]
    frames = [
        "datasets/RVOSJoint/davis17/valid/JPEGImages/soapbox/00000.jpg",
        "datasets/RVOSJoint/davis17/valid/JPEGImages/soapbox/00010.jpg",
        "datasets/RVOSJoint/davis17/valid/JPEGImages/soapbox/00020.jpg",
        "datasets/RVOSJoint/davis17/valid/JPEGImages/soapbox/00030.jpg",
        "datasets/RVOSJoint/davis17/valid/JPEGImages/soapbox/00040.jpg",
        "datasets/RVOSJoint/davis17/valid/JPEGImages/soapbox/00050.jpg",
        "datasets/RVOSJoint/davis17/valid/JPEGImages/soapbox/00060.jpg",
        "datasets/RVOSJoint/davis17/valid/JPEGImages/soapbox/00070.jpg",
    ]
    system_prompt = (
        "You are a professional video description model.\n"
        "\n"
        "**Objective:**\n"
        "Process the following sequence of images as consecutive video frames. Analyze temporal relationships, motion patterns, and contextual changes across frames to understand dynamic scenes.\n"
        "\n"
        "**Key Instructions:**\n"
        "1. **Temporal awareness**\n"
        "   - Treat frames as ordered timesteps (e.g., Frame 1 → Frame 2 → ... → Frame N)\n"
        "   - Identify changes between frames (e.g., object motion, lighting shifts, scene transitions)\n"
        "\n"
        "2. **Video-centric analysis**\n"
        '   - Focus on **action continuity** (e.g., "The person begins waving")\n'
        '   - Track entities across frames (e.g., "The car moves left-to-right")\n'
        "\n"
        "3. **Output structure**\n"
        "   - Follow user's request, output only the result\n"
        '   - Use temporal markers, only if necessary (e.g., "Initially," "Subsequently," "Finally")\n'
        '   - Highlight cause-effect relationships if exists (e.g., "Because X happened, Y occurs")\n'
    )
    
    # ckpt_path = "ckpts/gemma-3-27b-it"
    gemma_ckpt_path = "ckpts/gemma-3-4b-it"
    qwen2_5vl_ckpt_path = "ckpts/Qwen2.5-VL-3B-Instruct"
    dtype = torch.bfloat16
    # model = Gemma3ForConditionalGeneration.from_pretrained(ckpt_path, torch_dtype=dtype, low_cpu_mem_usage=True, attn_implementation="flash_attention_2", device_map="auto")
    # model = Gemma3ForConditionalGeneration.from_pretrained(ckpt_path, torch_dtype=dtype, attn_implementation="flash_attention_2", device_map="auto")
    
    # model = Gemma3ForConditionalGeneration.from_pretrained(gemma_ckpt_path, torch_dtype=dtype, attn_implementation="flash_attention_2", device_map="cuda:0")
    model = Gemma3ForConditionalGeneration.from_pretrained(gemma_ckpt_path, torch_dtype=dtype, attn_implementation="eager", device_map="cuda:0")
    processor = AutoProcessor.from_pretrained(gemma_ckpt_path)
    
    # model = Qwen2_5_VLForConditionalGeneration.from_pretrained(qwen2_5vl_ckpt_path, torch_dtype=dtype, attn_implementation="flash_attention_2", device_map="cuda:0")
    # processor = AutoProcessor.from_pretrained(qwen2_5vl_ckpt_path)
    
    # model = model.to(dtype).cuda()
    print("Device: ", model.device)

    messages=[
        # {
        #     "role": "system",
        #     "content": [{"type": "text", "text": system_prompt}]
        # },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": frames[0]},
                {"type": "image", "image": frames[1]},
                {"type": "image", "image": frames[2]},
                {"type": "image", "image": frames[3]},
                # {"type": "image", "image": frames[4]},
                # {"type": "image", "image": frames[5]},
                # {"type": "image", "image": frames[6]},
                # {"type": "image", "image": frames[7]},
                {"type": "text", "text": "Describe the video in 2 sentences, depict only what you see."},
            ],
        }
    ]

    # messages = [
    #     {
    #         "role": "system",
    #         "content": [{"type": "text", "text": "You are a helpful assistant."}]
    #     },
    #     {
    #         "role": "user",
    #         "content": [
    #             {"type": "image", "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"},
    #             {"type": "text", "text": "Describe this image in detail."}
    #         ]
    #     }
    # ]


    inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt")
    inputs = inputs.to(model.device, dtype=dtype)

    input_len = inputs["input_ids"].shape[-1]
    input_len

    ##### Start (JS). TODD. Change depends on the input number of frames
    image_token_id = processor.image_token_id
    image_token_idxs = (inputs["input_ids"][0] == image_token_id).nonzero()[:, 0]
    model.model.language_model.image_token_start_index = image_token_idxs[0].item()
    model.model.language_model.image_token_length = image_token_idxs.size(0)
    model.model.language_model.num_frame = 2
    model.model.language_model.image_H = 16
    model.model.language_model.image_W = 16
    
    sa_start_layer_idx = 0
    sa_tree_thresh = 0.95
    sa_tree_temporal_thresh = 0.70
    sa_tree_root_level = 1
    replace_gemma3_with_quadtree_attn(sa_start_layer_idx, sa_tree_thresh, sa_tree_temporal_thresh, sa_tree_root_level)
    ##### End (JS)

    with torch.inference_mode():
        generation = model.generate(**inputs, max_new_tokens=100, do_sample=False, logits_to_keep=1)
        # generation = model.generate(**inputs, max_new_tokens=500, do_sample=True)
    
    generation = generation[0][input_len:]

    decoded = processor.decode(generation, skip_special_tokens=True)
    print(decoded)

