import math
import torch
import einops
import warnings
from typing import List, Optional, Tuple, Union

import transformers
from transformers.cache_utils import Cache, DynamicCache, DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_flash_attention_utils import _flash_attention_forward
from llava.model.qwen2vl.modeling_qwen2vl import logger, apply_multimodal_rotary_pos_emb, repeat_kv

from token_merging_utils.quadtree_interface import get_quadtree_features



# def Qwen2VLFlashAttention2_forward(
#     self,
#     hidden_states: torch.Tensor,
#     attention_mask: Optional[torch.Tensor] = None,
#     position_ids: Optional[torch.LongTensor] = None,
#     past_key_value: Optional[Cache] = None,
#     output_attentions: bool = False,
#     use_cache: bool = False,
#     cache_position: Optional[torch.LongTensor] = None,
#     position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
# ):
#     bsz, q_len, _ = hidden_states.size()

#     query_states = self.q_proj(hidden_states)
#     key_states = self.k_proj(hidden_states)
#     value_states = self.v_proj(hidden_states)

#     query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
#     key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
#     value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

#     kv_seq_len = key_states.shape[-2]
#     if past_key_value is not None:
#         if self.layer_idx is None:
#             raise ValueError(
#                 f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
#                 "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
#                 "with a layer index."
#             )
#         kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

#     # Because the input can be padded, the absolute sequence length depends on the max position id.
#     if position_embeddings is None:
#         logger.warning_once(
#             "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
#             "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
#             "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
#             "removed and `position_embeddings` will be mandatory."
#         )
#         cos, sin = self.rotary_emb(value_states, position_ids)
#     else:
#         cos, sin = position_embeddings

#     query_states, key_states = apply_multimodal_rotary_pos_emb(
#         query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
#     )

#     if past_key_value is not None:
#         # Activate slicing cache only if the config has a value `sliding_windows` attribute
#         cache_has_contents = past_key_value.get_seq_length(self.layer_idx) > 0
#         if (
#             getattr(self.config, "sliding_window", None) is not None
#             and kv_seq_len > self.config.sliding_window
#             and cache_has_contents
#         ):
#             slicing_tokens = 1 - self.config.sliding_window

#             past_key = past_key_value[self.layer_idx][0]
#             past_value = past_key_value[self.layer_idx][1]

#             past_key = past_key[:, :, slicing_tokens:, :].contiguous()
#             past_value = past_value[:, :, slicing_tokens:, :].contiguous()

#             if past_key.shape[-2] != self.config.sliding_window - 1:
#                 raise ValueError(
#                     f"past key must have a shape of (`batch_size, num_heads, self.config.sliding_window-1, head_dim`), got"
#                     f" {past_key.shape}"
#                 )

#             if attention_mask is not None:
#                 attention_mask = attention_mask[:, slicing_tokens:]
#                 attention_mask = torch.cat([attention_mask, torch.ones_like(attention_mask[:, -1:])], dim=-1)

#         cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
#         key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

#     # repeat k/v heads if n_kv_heads < n_heads
#     key_states = repeat_kv(key_states, self.num_key_value_groups)
#     value_states = repeat_kv(value_states, self.num_key_value_groups)
#     dropout_rate = 0.0 if not self.training else self.attention_dropout

#     # In PEFT, usually we cast the layer norms in float32 for training stability reasons
#     # therefore the input hidden states gets silently casted in float32. Hence, we need
#     # cast them back in float16 just to be sure everything works as expected.
#     input_dtype = query_states.dtype
#     if input_dtype == torch.float32:
#         if torch.is_autocast_enabled():
#             target_dtype = torch.get_autocast_gpu_dtype()
#         # Handle the case where the model is quantized
#         elif hasattr(self.config, "_pre_quantization_dtype"):
#             target_dtype = self.config._pre_quantization_dtype
#         else:
#             target_dtype = self.q_proj.weight.dtype

#         logger.warning_once(
#             f"The input hidden states seems to be silently casted in float32, this might be related to"
#             f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
#             f" {target_dtype}."
#         )

#         query_states = query_states.to(target_dtype)
#         key_states = key_states.to(target_dtype)
#         value_states = value_states.to(target_dtype)

#     # Reashape to the expected shape for Flash Attention
#     query_states = query_states.transpose(1, 2)
#     key_states = key_states.transpose(1, 2)
#     value_states = value_states.transpose(1, 2)

#     if (
#         self.config.use_sliding_window
#         and getattr(self.config, "sliding_window", None) is not None
#         and self.layer_idx >= self.config.max_window_layers
#     ):
#         sliding_window = self.config.sliding_window
#     else:
#         sliding_window = None

#     attn_output = _flash_attention_forward(
#         query_states,
#         key_states,
#         value_states,
#         attention_mask,
#         q_len,
#         dropout=dropout_rate,
#         sliding_window=sliding_window,
#         is_causal=self.is_causal,
#         use_top_left_mask=self._flash_attn_uses_top_left_mask,
#     )

#     attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
#     attn_output = self.o_proj(attn_output)

#     if not output_attentions:
#         attn_weights = None

#     return attn_output, attn_weights, past_key_value

def Qwen2VLDecoderLayer_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: Optional[bool] = False,
    use_cache: Optional[bool] = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
    **kwargs,
) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:

    residual = hidden_states

    ##### start (JS)
    hidden_states = self.input_layernorm(hidden_states) # v2

    if self.do_merging:
        sys_features = hidden_states[:, :self.visual_tok_start_idx]
        inst_features = hidden_states[:, self.visual_tok_end_idx:]
        visual_features = hidden_states[:, self.visual_tok_start_idx:self.visual_tok_end_idx]
        video_features = einops.rearrange(visual_features[0], "(T H W) C -> T C H W", T=self.visual_T, H=self.visual_H)
        # quadtree_features, _, quadtree_tyxyx_tlbr = get_quadtree_features(video_features, self.sa_tree_thresh, self.sa_tree_temporal_thresh, self.sa_tree_root_level)
        quadtree_features, _, quadtree_tyxyx_tlbr, quadtree_idx_mapper = get_quadtree_features(video_features, self.sa_tree_thresh, self.sa_tree_temporal_thresh, self.sa_tree_root_level, get_idx_mapper=True)
        quadtree_ts, quadtree_ys, quadtree_xs = quadtree_tyxyx_tlbr[:, 0:3].T
        merged_token_1d_idx = quadtree_ts*self.visual_H*self.visual_W + quadtree_ys*self.visual_W + quadtree_xs
        merged_hidden_states = torch.cat([sys_features, quadtree_features.unsqueeze(0), inst_features], dim=1)
        hidden_states = merged_hidden_states
        len_merged_vis_prompt = quadtree_features.size(0)
        ## replace position embeddings and ids
        len_inst_prompt = inst_features.size(1)
        device = video_features.device
        sys_idxs = torch.arange(self.visual_tok_start_idx, dtype=torch.int32, device=device)
        vis_idxs = merged_token_1d_idx + self.visual_tok_start_idx
        inst_idxs = torch.arange(len_inst_prompt, dtype=torch.int32, device=device) + self.visual_tok_end_idx
        new_idxs = torch.cat([sys_idxs, vis_idxs, inst_idxs])
        pos_emb_cos, pos_emb_sin = position_embeddings
        new_position_embeddings = tuple([pos_emb_cos[:, :, new_idxs, :], pos_emb_sin[:, :, new_idxs, :]])
        position_embeddings = new_position_embeddings
        position_ids = position_ids[:, :, new_idxs]
        ## cache_position
    ##### end (JS)

    # hidden_states = self.input_layernorm(hidden_states) # v1. merging 

    # Self Attention
    hidden_states, self_attn_weights, present_key_value = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
        cache_position=cache_position,
        position_embeddings=position_embeddings,
    )

    ##### start (JS)
    if self.do_merging:
        vis_start_idx, vis_end_idx = self.visual_tok_start_idx, self.visual_tok_start_idx + len_merged_vis_prompt
        sys_features_attn = hidden_states[:, :vis_start_idx]
        inst_features_attn = hidden_states[:, vis_end_idx:]
        visual_features_attn = hidden_states[:, vis_start_idx:vis_end_idx]
        scattered_visual_features = visual_features_attn.index_select(dim=1, index=quadtree_idx_mapper)  # [1, THW, C]
        merged_states_attn = torch.cat([sys_features_attn, scattered_visual_features, inst_features_attn], dim=1)
        hidden_states = merged_states_attn
    ##### end (JS)
    
    hidden_states = residual + hidden_states

    # Fully Connected
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states

    outputs = (hidden_states,)

    if output_attentions:
        outputs += (self_attn_weights,)

    if use_cache:
        outputs += (present_key_value,)

    return outputs

def Qwen2VLModel_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
) -> Union[Tuple, BaseModelOutputWithPast]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError(
            "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
        )

    if self.gradient_checkpointing and self.training:
        if use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

    # kept for BC (non `Cache` `past_key_values` inputs)
    return_legacy_cache = False
    if use_cache and not isinstance(past_key_values, Cache):
        return_legacy_cache = True
        if past_key_values is None:
            past_key_values = DynamicCache()
        else:
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            logger.warning_once(
                "We detected that you are passing `past_key_values` as a tuple of tuples. This is deprecated and "
                "will be removed in v4.47. Please convert your cache or use an appropriate `Cache` class "
                "(https://huggingface.co/docs/transformers/kv_cache#legacy-cache-format)"
            )

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )
    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    causal_mask = self._update_causal_mask(
        attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
    )

    hidden_states = inputs_embeds

    # create position embeddings to be shared across the decoder layers
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = None
    
    ##### start (JS):
    is_prefilling = past_key_values is None or past_key_values.get_seq_length() == 0
    if is_prefilling:
        visual_tok_start_idx = self.image_token_start_index.item()
        num_visual_tokens = self.image_token_length.item()
        visual_tok_end_idx = visual_tok_start_idx + num_visual_tokens
        T = self.num_frame.item()
        H = self.image_H.item()
        W = self.image_W.item()
        for idx in range(len(self.layers)):
            layer = self.layers[idx]
            layer.visual_tok_start_idx = visual_tok_start_idx
            layer.visual_tok_end_idx = visual_tok_end_idx
            layer.num_visual_tokens = num_visual_tokens
            layer.visual_T = T
            layer.visual_H = H
            layer.visual_W = W
            layer.sa_tree_thresh = self.sa_tree_thresh
            layer.sa_tree_temporal_thresh = self.sa_tree_temporal_thresh
            layer.sa_tree_root_level = self.sa_tree_root_level
    ##### end (JS)

    for decoder_layer in self.layers:
        decoder_layer.do_merging = is_prefilling and decoder_layer.self_attn.layer_idx >= self.sa_start_layer_idx
        # if is_prefilling and decoder_layer.self_attn.layer_idx == self.sa_start_layer_idx:
        #     ### start token division with QuadTree algorithm
        #     visual_tok_start_idx = self.image_token_start_index.item()
        #     num_visual_tokens = self.image_token_length.item()
        #     visual_tok_end_idx = visual_tok_start_idx + num_visual_tokens
        #     sys_features = hidden_states[:, :visual_tok_start_idx]
        #     inst_features = hidden_states[:, visual_tok_end_idx:]
        #     visual_features = hidden_states[:, visual_tok_start_idx:visual_tok_end_idx]
        #     T = self.num_frame.item()
        #     H = self.image_H.item()
        #     W = self.image_W.item()
        #     video_features = einops.rearrange(visual_features[0], "(T H W) C -> T C H W", T=T, H=H)
        #     quadtree_features, _, quadtree_tyxyx_tlbr, _ = get_quadtree_features(video_features, self.sa_tree_thresh, self.sa_tree_temporal_thresh, self.sa_tree_root_level,
        #                                                 self.sa_tree_weighted_avg, slow_ver=self.sttm_slow_ver)
        #     quadtree_ts, quadtree_ys, quadtree_xs = quadtree_tyxyx_tlbr[:, 0:3].T
        #     merged_token_1d_idx = quadtree_ts*H*W + quadtree_ys*W + quadtree_xs
        #     merged_hidden_states = torch.cat([sys_features, quadtree_features.unsqueeze(0), inst_features], dim=1)
            
        #     ## Adjust positional embeddings with new hidden_states
        #     hidden_states = merged_hidden_states
        #     merged_num_tokens = merged_hidden_states.size(1)
        #     sys_position_ids = position_ids[:, :, :visual_tok_start_idx]
        #     inst_position_ids = position_ids[:, :, visual_tok_end_idx:]
        #     visual_position_ids = position_ids[:, :, visual_tok_start_idx:visual_tok_end_idx][:, :, merged_token_1d_idx]
        #     new_position_ids = torch.cat([sys_position_ids, visual_position_ids, inst_position_ids], dim=-1)
        #     new_position_embeddings = self.rotary_emb(hidden_states, new_position_ids)

        #     len_inst_prompt = inst_features.size(1)
        #     device = inst_features.device
        #     merged_token_1d_idx + visual_tok_start_idx
        #     sys_idxs = torch.arange(visual_tok_start_idx, dtype=torch.int32, device=device)
        #     vis_idxs = merged_token_1d_idx + visual_tok_start_idx
        #     inst_idxs = torch.arange(len_inst_prompt, dtype=torch.int32, device=device) + visual_tok_end_idx
        #     new_idxs = torch.cat([sys_idxs, vis_idxs, inst_idxs])
        #     pos_emb_cos, pos_emb_sin = position_embeddings
        #     new_position_embeddings_v2 = tuple([pos_emb_cos[:, :, new_idxs, :], pos_emb_sin[:, :, new_idxs, :]])

        #     cache_position = torch.arange(merged_num_tokens, device=self.device, dtype=torch.int)
        #     ### end token division with QuadTree algorithm
        
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                decoder_layer.__call__,
                hidden_states,
                causal_mask,
                position_ids,
                past_key_values,
                output_attentions,
                use_cache,
                cache_position,
                position_embeddings,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache = layer_outputs[2 if output_attentions else 1]

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = next_decoder_cache if use_cache else None
    if return_legacy_cache:
        next_cache = next_cache.to_legacy_cache()

    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )

def replace_qwen2vl_with_quadtree_residual_attn(sa_start_layer_idx=0, sa_tree_thresh=0.90, sa_tree_temporal_thresh=-1.0, sa_tree_root_level=0, 
                                    sa_tree_weighted_avg=False, sttm_slow_ver=False, **kwargs):
    print("Replace Qwen2VL Flash Attention2 by QuadTree Attn")
    import llava
    llava.model.qwen2vl.modeling_qwen2vl.Qwen2VLModel.sa_start_layer_idx = sa_start_layer_idx
    llava.model.qwen2vl.modeling_qwen2vl.Qwen2VLModel.sa_tree_thresh = sa_tree_thresh
    llava.model.qwen2vl.modeling_qwen2vl.Qwen2VLModel.sa_tree_temporal_thresh = sa_tree_temporal_thresh
    llava.model.qwen2vl.modeling_qwen2vl.Qwen2VLModel.sa_tree_root_level = sa_tree_root_level
    llava.model.qwen2vl.modeling_qwen2vl.Qwen2VLModel.sa_tree_weighted_avg = sa_tree_weighted_avg
    llava.model.qwen2vl.modeling_qwen2vl.Qwen2VLModel.sttm_slow_ver = sttm_slow_ver
    llava.model.qwen2vl.modeling_qwen2vl.Qwen2VLModel.forward = Qwen2VLModel_forward
    llava.model.qwen2vl.modeling_qwen2vl.Qwen2VLDecoderLayer.forward = Qwen2VLDecoderLayer_forward
    # llava.model.qwen2vl.modeling_qwen2vl.Qwen2VLFlashAttention2.forward = Qwen2VLFlashAttention2_forward
