import math
import torch
import einops
import warnings
from typing import List, Optional, Tuple, Union

import transformers
from transformers.cache_utils import Cache, DynamicCache
from transformers.models.gemma3.modeling_gemma3 import (
    Unpack, TransformersKwargs, logger, # general utils
    create_causal_mask, create_sliding_window_causal_mask, # masking utils
    BaseModelOutputWithPast, Cache, DynamicCache # output and cache utils
) 

from token_merging_utils.quadtree_interface import get_quadtree_features

def Gemma3TextModel_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[TransformersKwargs],
) -> BaseModelOutputWithPast:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if self.gradient_checkpointing and self.training and use_cache:
        logger.warning_once(
            "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
        )
        use_cache = False

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if use_cache and past_key_values is None and not self.training:
        past_key_values = DynamicCache()

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens,
            past_seen_tokens + inputs_embeds.shape[1],
            device=inputs_embeds.device,
        )

    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    # It may already have been prepared by e.g. `generate`
    if not isinstance(causal_mask_mapping := attention_mask, dict):
        # Prepare mask arguments
        mask_kwargs = {
            "config": self.config,
            "input_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "cache_position": cache_position,
            "past_key_values": past_key_values,
            "position_ids": position_ids,
        }
        # Create the masks
        causal_mask_mapping = {
            "full_attention": create_causal_mask(**mask_kwargs),
            "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
        }

    # embed positions
    hidden_states = inputs_embeds

    # create position embeddings to be shared across the decoder layers
    position_embeddings_global = self.rotary_emb(hidden_states, position_ids)
    position_embeddings_local = self.rotary_emb_local(hidden_states, position_ids)

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    
    ##### Start (JS)
    is_prefilling = past_key_values is None or past_key_values.get_seq_length() == 0
    unmerged_cache_position = cache_position
    merged_cache_position = cache_position
    ##### End (JS)
    for decoder_layer in self.layers[: self.config.num_hidden_layers]:
        ##### Start (JS)
        if is_prefilling and decoder_layer.self_attn.layer_idx == self.sa_start_layer_idx:
            ### start token division with QuadTree algorithm
            visual_tok_start_idx = self.image_token_start_index
            num_visual_tokens = self.image_token_length
            visual_tok_end_idx = visual_tok_start_idx + num_visual_tokens
            sys_features = hidden_states[:, :visual_tok_start_idx]
            inst_features = hidden_states[:, visual_tok_end_idx:]
            visual_features = hidden_states[:, visual_tok_start_idx:visual_tok_end_idx]
            T, H, W = self.num_frame, self.image_H, self.image_W
            video_features = einops.rearrange(visual_features[0], "(T H W) C -> T C H W", T=T, H=H)
            quadtree_features, _, quadtree_tyxyx_tlbr = get_quadtree_features(video_features, self.sa_tree_thresh, self.sa_tree_temporal_thresh, self.sa_tree_root_level,
                                                        self.sa_tree_weighted_avg, slow_ver=self.sttm_slow_ver)
            quadtree_ts, quadtree_ys, quadtree_xs = quadtree_tyxyx_tlbr[:, 0:3].T
            merged_token_1d_idx = quadtree_ts*H*W + quadtree_ys*W + quadtree_xs
            merged_hidden_states = torch.cat([sys_features, quadtree_features.unsqueeze(0), inst_features], dim=1)
            
            ## Adjust positional embeddings with new hidden_states
            hidden_states = merged_hidden_states
            merged_num_tokens = merged_hidden_states.size(1)
            ## pos_emb: survival
            # sys_position_ids = position_ids[:, :visual_tok_start_idx]
            # inst_position_ids = position_ids[:, visual_tok_end_idx:]
            # visual_position_ids = position_ids[:, visual_tok_start_idx:visual_tok_end_idx][:, merged_token_1d_idx]
            # position_ids = torch.cat([sys_position_ids, visual_position_ids, inst_position_ids], dim=-1)
            ## pos_emb: reassignment
            position_ids = position_ids[:, :merged_num_tokens]
            ## recopmute rotary embedding
            position_embeddings_global = self.rotary_emb(hidden_states, position_ids)
            position_embeddings_local = self.rotary_emb_local(hidden_states, position_ids)
            merged_cache_position = torch.arange(merged_num_tokens, device=self.device, dtype=torch.long)
            ### end token division with QuadTree algorithm
        # cache_position = unmerged_cache_position if decoder_layer.self_attn.layer_idx < self.sa_start_layer_idx else merged_cache_position
        ##### End (JS)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        layer_outputs = decoder_layer(
            hidden_states,
            position_embeddings_global=position_embeddings_global,
            position_embeddings_local=position_embeddings_local,
            attention_mask=causal_mask_mapping[decoder_layer.attention_type],
            position_ids=position_ids,
            past_key_value=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = layer_outputs[0]

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )

def replace_gemma3_with_quadtree_attn(sa_start_layer_idx=0, sa_tree_thresh=0.90, sa_tree_temporal_thresh=-1.0, sa_tree_root_level=0, 
                                    sa_tree_weighted_avg=False, sttm_slow_ver=False, **kwargs):
    print("Replace Gemma3 Flash Attention2 by QuadTree Attn")
    transformers.models.gemma3.modeling_gemma3.Gemma3TextModel.sa_start_layer_idx = sa_start_layer_idx
    transformers.models.gemma3.modeling_gemma3.Gemma3TextModel.sa_tree_thresh = sa_tree_thresh
    transformers.models.gemma3.modeling_gemma3.Gemma3TextModel.sa_tree_temporal_thresh = sa_tree_temporal_thresh
    transformers.models.gemma3.modeling_gemma3.Gemma3TextModel.sa_tree_root_level = sa_tree_root_level
    transformers.models.gemma3.modeling_gemma3.Gemma3TextModel.sa_tree_weighted_avg = sa_tree_weighted_avg
    transformers.models.gemma3.modeling_gemma3.Gemma3TextModel.sttm_slow_ver = sttm_slow_ver
    transformers.models.gemma3.modeling_gemma3.Gemma3TextModel.forward = Gemma3TextModel_forward