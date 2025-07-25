import math
import torch
import einops
import warnings
from typing import List, Optional, Tuple, Union
import torch.nn.functional as F

import transformers
from transformers.cache_utils import Cache, DynamicCache
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv, _flash_attention_forward, logger, Qwen2FlashAttention2, Qwen2Attention
from transformers.modeling_outputs import BaseModelOutputWithPast

from token_merging_utils.dycoke_merger import dycoke_ttm


class DycokeConfigs():
    def __init__(self):
        self.dycoke_layer_idx = 3
        self.dycoke_radio = 0.8
        self.image_token_start_index = None
        self.image_token_length = None
        # self.similarity = None
        # self.attention_score = None

class PrunableDynamicCache(DynamicCache):
    def __init__(self) -> None:
        self._seen_tokens = 0  # Used in `generate` to keep tally of how many tokens the cache has seen
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self.kv_cache = None
        self.similarity = None
        self.attention_score = None
    
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        # Update the cache
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

        if self.kv_cache is None:
            return self.key_cache[layer_idx], self.value_cache[layer_idx]
        else:
            k_device, v_device = self.key_cache[layer_idx].device, self.value_cache[layer_idx].device
            k_dim0, k_dim1, k_dim3 = self.key_cache[layer_idx].size(0), self.key_cache[layer_idx].size(1), self.key_cache[layer_idx].size(3)
            v_dim0, v_dim1, v_dim3 = self.value_cache[layer_idx].size(0), self.value_cache[layer_idx].size(1), self.value_cache[layer_idx].size(3)
            new_key_cache = torch.gather(self.key_cache[layer_idx], dim=2, index=torch.tensor(self.kv_cache, device=k_device).view(1, 1, -1, 1).expand(k_dim0, k_dim1, -1, k_dim3))
            new_value_cache = torch.gather(self.value_cache[layer_idx], dim=2, index=torch.tensor(self.kv_cache, device=v_device).view(1, 1, -1, 1).expand(v_dim0, v_dim1, -1, v_dim3))         
            return new_key_cache, new_value_cache
    
    def update_cache(self, image_attention, config):
        # Pre-calculate values to avoid repeated computation
        start_idx = config.image_token_start_index
        img_len = config.image_token_length
        num_keep = int(img_len * (1 - config.dycoke_radio))
        
        # Get top indices in one operation
        top_indices = torch.topk(image_attention, num_keep, sorted=False)[1] + start_idx
        
        # Create ranges efficiently using single arange call
        device = image_attention.device
        full_range = torch.arange(config.seq_length_with_past, device=device)
        keep_indexs = torch.cat([full_range[:start_idx], top_indices, full_range[start_idx + img_len:]])

        # Convert to list once at end
        self.kv_cache = keep_indexs.tolist()

    def dycoke_pruning(self, attn, layer_idx, config):
        attention_avg = attn[1].mean(1)[0, -1]
        start_idx = config.image_token_start_index
        img_len = config.image_token_length
        image_attention = attention_avg[start_idx:start_idx + img_len]
        
        # if config.attention_score is not None:
        #     config.similarity = F.cosine_similarity(image_attention, config.attention_score, dim=0)
        # else:
        #     config.similarity = 0
        # config.attention_score = image_attention
                
        # if config.similarity < 0.9:
        #     self.update_cache(image_attention, config)
        if self.attention_score is not None:
            self.similarity = F.cosine_similarity(image_attention, self.attention_score, dim=0)
        else:
            self.similarity = 0
        self.attention_score = image_attention
                
        if self.similarity < 0.9:
            self.update_cache(image_attention, config)

class Qwen2FlashAttn_with_eager_mode(Qwen2FlashAttention2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            return Qwen2Attention.forward(self, hidden_states, attention_mask, position_ids, past_key_value, True, use_cache, cache_position, position_embeddings)
        else:
            return super().forward(hidden_states, attention_mask, position_ids, past_key_value, False, use_cache, cache_position, position_embeddings)

def Qwen2FlashAttn_with_eager_mode_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
):
    if output_attentions:
        # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
        logger.warning_once(
            "Qwen2Model is using Qwen2SdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
            'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
        )
        return Qwen2Attention.forward(
            self,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    if position_embeddings is None:
        logger.warning_once(
            "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
            "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
            "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
            "removed and `position_embeddings` will be mandatory."
        )
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        # Activate slicing cache only if the config has a value `sliding_windows` attribute
        cache_has_contents = past_key_value.get_seq_length(self.layer_idx) > 0
        kv_seq_len = key_states.shape[-2] + cache_position[0]
        if (
            getattr(self.config, "sliding_window", None) is not None
            and kv_seq_len > self.config.sliding_window
            and cache_has_contents
        ):
            slicing_tokens = 1 - self.config.sliding_window

            past_key = past_key_value[self.layer_idx][0]
            past_value = past_key_value[self.layer_idx][1]

            past_key = past_key[:, :, slicing_tokens:, :].contiguous()
            past_value = past_value[:, :, slicing_tokens:, :].contiguous()

            if past_key.shape[-2] != self.config.sliding_window - 1:
                raise ValueError(
                    f"past key must have a shape of (`batch_size, num_heads, self.config.sliding_window-1, head_dim`), got"
                    f" {past_key.shape}"
                )

            if attention_mask is not None:
                attention_mask = attention_mask[:, slicing_tokens:]
                attention_mask = torch.cat([attention_mask, torch.ones_like(attention_mask[:, -1:])], dim=-1)

        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    # repeat k/v heads if n_kv_heads < n_heads
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)
    dropout_rate = 0.0 if not self.training else self.attention_dropout

    # In PEFT, usually we cast the layer norms in float32 for training stability reasons
    # therefore the input hidden states gets silently casted in float32. Hence, we need
    # cast them back in float16 just to be sure everything works as expected.
    input_dtype = query_states.dtype
    if input_dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        # Handle the case where the model is quantized
        elif hasattr(self.config, "_pre_quantization_dtype"):
            target_dtype = self.config._pre_quantization_dtype
        else:
            target_dtype = self.q_proj.weight.dtype

        logger.warning_once(
            f"The input hidden states seems to be silently casted in float32, this might be related to"
            f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
            f" {target_dtype}."
        )

        query_states = query_states.to(target_dtype)
        key_states = key_states.to(target_dtype)
        value_states = value_states.to(target_dtype)

    # Reashape to the expected shape for Flash Attention
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    if (
        self.config.use_sliding_window
        and getattr(self.config, "sliding_window", None) is not None
        and self.layer_idx >= self.config.max_window_layers
    ):
        sliding_window = self.config.sliding_window
    else:
        sliding_window = None

    attn_output = _flash_attention_forward(
        query_states,
        key_states,
        value_states,
        attention_mask,
        q_len,
        position_ids=position_ids,
        dropout=dropout_rate,
        sliding_window=sliding_window,
        is_causal=self.is_causal,
        use_top_left_mask=self._flash_attn_uses_top_left_mask,
    )

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value

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


    if use_cache:
        use_legacy_cache = not isinstance(past_key_values, Cache)
        if use_legacy_cache and past_key_values is None:
            past_key_values = PrunableDynamicCache.from_legacy_cache(past_key_values)
        elif use_legacy_cache:
            past_key_values = PrunableDynamicCache.from_legacy_cache(past_key_values)
        else:
            new_past_key_values = PrunableDynamicCache()
            new_past_key_values.key_cache = past_key_values.key_cache
            new_past_key_values.value_cache = past_key_values.value_cache
            new_past_key_values._seen_tokens = past_key_values._seen_tokens
            past_key_values = new_past_key_values
        if input_ids is not None:
            seq_length = input_ids.size(1)
        elif inputs_embeds is not None:
            seq_length = inputs_embeds.size(1)
        past_key_values_length = past_key_values.get_usable_length(seq_length)

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
    for layer_idx, decoder_layer in enumerate(self.layers):
        if is_prefilling and decoder_layer.self_attn.layer_idx == self.sa_start_layer_idx:
            ### start token division with QuadTree algorithm
            visual_tok_start_idx = self.image_token_start_index.item()
            num_visual_tokens = self.image_token_length.item()
            visual_tok_end_idx = visual_tok_start_idx + num_visual_tokens
            sys_features = hidden_states[:, :visual_tok_start_idx]
            inst_features = hidden_states[:, visual_tok_end_idx:]
            visual_features = hidden_states[:, visual_tok_start_idx:visual_tok_end_idx]
            T = self.num_frame.item()
            dycoke_features, merged_token_1d_idx = dycoke_ttm(visual_features[0], T, self.sa_prune_ratio)
            merged_hidden_states = torch.cat([sys_features, dycoke_features.unsqueeze(0), inst_features], dim=1)
            
            ##### Start: update Dycoke config
            self.DycokeConfig.image_token_start_index = visual_tok_start_idx
            self.DycokeConfig.image_token_length = num_visual_tokens
            ##### End

            ## Adjust positional embeddings with new hidden_states
            merged_num_tokens = merged_hidden_states.size(1)
            hidden_states = merged_hidden_states
            # sys_position_ids = position_ids[:, :visual_tok_start_idx]
            # inst_position_ids = position_ids[:, visual_tok_end_idx:]
            # visual_position_ids = position_ids[:, visual_tok_start_idx:visual_tok_end_idx][:, merged_token_1d_idx]
            # position_ids = torch.cat([sys_position_ids, visual_position_ids, inst_position_ids], dim=-1)
            position_ids = position_ids[:, :merged_num_tokens]
            position_embeddings = self.rotary_emb(hidden_states, position_ids)
            cache_position = torch.arange(merged_num_tokens, device=self.device, dtype=torch.int)
            ### end token division with QuadTree algorithm
        
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
            seq_length = hidden_states.size(1)
            past_key_values_length = past_key_values.get_usable_length(seq_length)
            self.DycokeConfig.seq_length_with_past = seq_length + past_key_values_length
            if layer_idx < self.dycoke_l:
                past_key_values.kv_cache = None
            elif layer_idx == self.dycoke_l and past_key_values.kv_cache is None and position_ids.shape[1] == 1: # decoding stage
                past_key_values.dycoke_pruning(layer_outputs, layer_idx, self.DycokeConfig)
        
            if layer_idx == self.dycoke_l-1 and past_key_values.kv_cache is None and position_ids.shape[1] == 1:
                output_attentions = True
            
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

        output_attentions = False
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

def replace_qwen2_with_dycoke_attn(sa_start_layer_idx=0, sa_prune_ratio=0.7, dycoke_l=3, dycoke_p=0.8):
    print("Replace Qwen2 Flash Attention2 by DyCoke Attn")
    dycoke_configs = DycokeConfigs()
    dycoke_configs.dycoke_layer_idx = dycoke_l
    dycoke_configs.dycoke_radio = dycoke_p
    dycoke_configs.image_token_start_index = -1
    dycoke_configs.image_token_length = -1

    transformers.models.qwen2.modeling_qwen2.Qwen2Model.sa_start_layer_idx = sa_start_layer_idx
    transformers.models.qwen2.modeling_qwen2.Qwen2Model.sa_prune_ratio = sa_prune_ratio
    transformers.models.qwen2.modeling_qwen2.Qwen2Model.DycokeConfig = dycoke_configs
    transformers.models.qwen2.modeling_qwen2.Qwen2Model.dycoke_l = dycoke_l
    transformers.models.qwen2.modeling_qwen2.Qwen2Model.dycoke_p = dycoke_p
    transformers.models.qwen2.modeling_qwen2.Qwen2Model.forward = Qwen2Model_forward
    # transformers.models.qwen2.modeling_qwen2.Qwen2FlashAttention2 = Qwen2FlashAttn_with_eager_mode
    # transformers.models.qwen2.modeling_qwen2.Qwen2FlashAttention2.forward = Qwen2FlashAttn_with_eager_mode_forward
