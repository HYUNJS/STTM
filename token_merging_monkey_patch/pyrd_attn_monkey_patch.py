import math
import torch
import einops
import warnings
from typing import List, Optional, Tuple, Union
import torch.nn.functional as F

import transformers
from transformers.cache_utils import Cache, DynamicCache, DynamicCache
from transformers.models.qwen2.modeling_qwen2 import repeat_kv,apply_rotary_pos_emb, logger, QWEN2_INPUTS_DOCSTRING
from transformers.modeling_outputs import BaseModelOutputWithPast


def Qwen2Model_forward(
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
    
    is_prefilling = past_key_values is None or past_key_values.get_seq_length() == 0
    for decoder_layer in self.layers:
        if is_prefilling and decoder_layer.self_attn.layer_idx in self.sa_pyrd_llm_idxs:
            ### start token merging
            visual_tok_start_idx = self.image_token_start_index.item()
            num_visual_tokens = self.image_token_length.item()
            visual_tok_end_idx = visual_tok_start_idx + num_visual_tokens
            sys_features = hidden_states[:, :visual_tok_start_idx]
            inst_features = hidden_states[:, visual_tok_end_idx:]
            visual_features = hidden_states[:, visual_tok_start_idx:visual_tok_end_idx]
            T = self.num_frame.item()
            H = int(math.sqrt(num_visual_tokens // T))
            video_features = einops.rearrange(visual_features[0], "(T H W) C -> T C H W", T=T, H=H)
            tgt_size = self.sa_pyrd_idx2size[decoder_layer.self_attn.layer_idx]
            resized_video_features = F.interpolate(video_features, size=(tgt_size, tgt_size))
            resized_video_features = einops.rearrange(resized_video_features, "T C H W -> (T H W) C")
            self.image_token_length = torch.tensor(resized_video_features.size(0))

            merged_hidden_states = torch.cat([sys_features, resized_video_features.unsqueeze(0), inst_features], dim=1)
            
            ## Adjust positional embeddings with new hidden_states
            num_merged_tokens = merged_hidden_states.size(1)
            hidden_states = merged_hidden_states
            position_ids = position_ids[:, :num_merged_tokens]
            position_embeddings = self.rotary_emb(hidden_states, position_ids)
            ### end token merging
        
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

def replace_qwen2_with_pyrd_attn(sa_pyrd_loc_list=[2], sa_pyrd_size_list=[10]):
    print("Replace Qwen2 Flash Attention2 by Pyramid token merging")
    assert len(sa_pyrd_loc_list) == len(sa_pyrd_size_list)
    pyrd_idx2size = {sa_pyrd_loc_list[i]: sa_pyrd_size_list[i] for i in range(len(sa_pyrd_loc_list))}
    transformers.models.qwen2.modeling_qwen2.Qwen2Model.sa_pyrd_llm_idxs = sa_pyrd_loc_list
    transformers.models.qwen2.modeling_qwen2.Qwen2Model.sa_pyrd_idx2size = pyrd_idx2size
    transformers.models.qwen2.modeling_qwen2.Qwen2Model.forward = Qwen2Model_forward
