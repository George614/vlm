import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from typing import Optional, Tuple, List
import math
from modeling_siglip import SiglipVisionConfig, SiglipVisionModel


class GemmaConfig:
        
    def __init__(
        self,
        vocab_size: int = 256000,
        hidden_size: int = 2048,
        intermediate_size: int = 8192,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 16,  # for query
        num_key_value_heads: int = 4,
        head_dim: int = 256,
        max_position_embeddings: int = 8192,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 10000.0,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        pad_token_id: Optional[int] = None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.pad_token_id = pad_token_id
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout


class PaliGemmaConfig:
    
    def __init__(
        self,
        vision_config: SiglipVisionConfig = None,
        text_config: GemmaConfig = None,
        ignore_index: int = -100,
        image_token_index: int = 256000,
        vocab_size: int = 257152,
        projection_dim: int = 2048,
        hidden_size: int = 2048,
        pad_token_id: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.image_token_index = image_token_index
        self.vocab_size = vocab_size
        self.projection_dim = projection_dim
        self.hidden_size = hidden_size
        self.pad_token_id = pad_token_id
        self.is_encoder_decoder = False

        self.vision_config = SiglipVisionConfig(**vision_config)
        self.text_config = GemmaConfig(**text_config, pad_token_id=pad_token_id)
        
        self.vocab_size = self.text_config.vocab_size
        self.text_config.num_image_tokens = (self.vision_config.image_size // self.vision_config.patch_size) ** 2 + 1
        self.vision_config.projeciton_dim = self.projection_dim   


class GemmaRMSNorm(nn.Module):
        
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.zeros(hidden_size))
        
    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float())
        output = output * (1.0 + self.scale.float())
        return output.type_as(x)


class GemmaMLP(nn.Module):
        
    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # [batch_size, seq_length, hidden_size]
        return self.down_proj(F.gelu(self.gate_proj(hidden_states), approximate="tanh") * self.up_proj(hidden_states))


class GemmaAttention(nn.Module):
    
    def __init__(self, config: GemmaConfig, layer_idx: Optional[int] = None) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        
        assert self.hidden_size % self.num_heads == 0, "hidden_size must be divisible by num_heads"
        
        # Grouped query attention, note the number of heads for key and value are smaller than the number of heads for query
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        

class GemmaDecoderLayer(nn.Module):
    
    def __init__(self, config: GemmaConfig, layer_idx: int):
        super().__init__()
        self.hideen_size = config.hidden_size
        self.self_attn = GemmaAttention(config=config, layer_idx=layer_idx)
        self.mlp = GemmaMLP(config=config)
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        # [batch_size, seq_length, hidden_size]
        hidden_states = self.input_layernorm(hidden_states)
        # [batch_size, seq_length, hidden_size]
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            kv_cache=kv_cache,
        )
        hidden_states = hidden_states + residual
        residual = hidden_states
        # [batch_size, seq_length, hidden_size]
        hidden_states = self.post_attention_layernorm(hidden_states)
        # [batch_size, seq_length, hidden_size]
        hidden_states = self.mlp(hidden_states) 
        hidden_states = hidden_states + residual
        
        return hidden_states
    

class GemmaModel(nn.Module):
    
    def __init__(self, config: GemmaConfig) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=self.padding_idx)
        self.layers = nn.ModuleList(
            [GemmaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
    def get_input_embeddings(self):
        return self.embed_tokens

    def forward(
        self,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> torch.FloatTensor:
        # [batch_size, seq_length, hidden_size]
        hidden_states = inputs_embeds
        # [batch_size, seq_length, hidden_size]
        normalizer = torch.tensor(self.config.hidden_size ** 0.5, dtype=hidden_states.dtype, device=hidden_states.device)
        hidden_states = hidden_states * normalizer  # scale the input embeddings??
        
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                kv_cache=kv_cache,
            )
        # [batch_size, seq_length, hidden_size]
        hidden_states = self.norm(hidden_states)
        return hidden_states
        

class GemmaForCausalLM(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = GemmaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def get_input_embeddings(self):
        return self.model.embed_tokens
    
    def tie_weights(self):
        self.lm_head.weight = self.model.embed_tokens.weight
        
    def forward(
        self,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple:
        # input_embeds : [batch_size, seq_length, hidden_size]
        # outputs : [batch_size, seq_length, hidden_size]
        outputs = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            kv_cache=kv_cache,
        )
        
        hidden_states = outputs
        lm_logits = self.lm_head(hidden_states)
        lm_logits = lm_logits.float()
        
        return_data = {"logits": lm_logits}
        
        if kv_cache is not None:
            return_data["kv_cache"] = kv_cache
            
        return return_data

class PaliGemmaMultiModalProjector(nn.Module):
    
    def __init__(self, config:PaliGemmaConfig):
        super().__init__()
        self.linear = nn.Linear(config.vision_config.hidden_size, config.vision_config.projection_dim)
    
    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        # [batch_size, num_patches, hidden_size] -> [batch_size, num_patches, projection_dim]
        return self.linear(image_features)


class PaliGemmaForConditionalGeneration(nn.Module):

    def __init__(self, config: PaliGemmaConfig) -> None:
        super().__init__()
        self.config = config
        self.vision_tower = SiglipVisionModel(config.vision_config)
        self.multi_modal_projector = PaliGemmaMultiModalProjector(config)
        self.vocab_size = config.vocab_size

        self.language_model = GemmaForCausalLM(config.text_config)
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1

    def tie_weights(self) -> None:
        return self.language_model.tie_weights()

    def _merge_input_ids_with_image_features(
        self,
        image_features: torch.Tensor,
        inputs_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
    ):
        _, _, embed_dim = image_features.shape
        batch_size, seq_length = input_ids.shape
        dtype, device = inputs_embeds.dtype, inputs_embeds.device
        # shape (batch_size, sequence_length, hidden_size)
        scaled_image_features = image_features / (self.config.hidden_size ** 0.5)

        # combine the embeddings of the text and the image tokens and mask out the padding tokens
        final_embedding = torch.zeros(batch_size, seq_length, embed_dim, dtype=dtype, device=device)
        # shape [batch_size, seq_length]
        text_mask = (input_ids != self.config.image_token_index) & (input_ids != self.pad_token_id)
        # shape [batch_size, seq_length]
        image_mask = input_ids == self.config.image_token_index
        # shape [batch_size, seq_length]
        pad_mask = input_ids == self.pad_token_id

        # expand the masks to the embedding dimension to use them with torch.where
        text_mask = text_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        pad_mask = pad_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        image_mask = image_mask.unsqueeze(-1).expand(-1, -1, embed_dim)

        # add the text embededdings to the final embedding
        final_embedding = torch.where(text_mask, inputs_embeds, final_embedding)
        # add the image embeddings to the final embedding. Note that sequence length of 
        # scaled_image_features is not same as seq_length of final_embedding
        final_embedding = final_embedding.masked_scatter(image_mask, scaled_image_features)
        # zero out padding tokens
        final_embedding = torch.where(pad_mask, torch.zeros_like(final_embedding), final_embedding)
        
        ## create the attention mask ##
        dtype, device = inputs_embeds.dtype, inputs_embeds.device
        min_dtype = torch.finfo(dtype).min
        q_len = inputs_embeds.shape[1]
        
        if kv_cache is None or kv_cache.num_items() == 0:
            # do not mask any token since we are in the prefill phase NOTE only works with no padding
            causal_mask = torch.full(
                (batch_size, q_len, q_len), fill_value=0, dtype=dtype, device=device
            )
        else:
            # since we are generating the next token, the query length is 1
            assert q_len == 1
            kv_len = kv_cache.num_items() + q_len
            # no need to mask anything, since each query shuold attend to all previous tokens
            # NOTE this only works when we have no padding
            causal_mask = torch.full(
                (batch_size, q_len, kv_len), fill_value=0, dtype=dtype, device=device
            )
            
        # add the head dimension to the mask
        # [batch_size, q_len, kv_len] -> [batch_size, num_heads, q_len, kv_len]
        causal_mask = causal_mask.unsqueeze(1)
        
        if kv_cache is not None and kv_cache.num_items() > 0:
            # the position of the query is the last position in the key-value cache
            position_ids = attention_mask.cumsum(dim=-1)[:, -1]
            if position_ids.dim() == 1:
                position_ids = position_ids.unsqueeze(0)
        else:
            # create a position_ids based on the size of the attention_mask
            # for masked tokens, use the number 1 as position
            position_ids = (attention_mask.cumsum(dim=-1)).masked_fill(~attention_mask, 1).to(device)
            
        return final_embedding, causal_mask, position_ids
            

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple:
        assert torch.all(attention_mask == 1), "The input cannot be padded"

        # extract the input embeddings
        # shape (batch_size, sequence_length, hidden_size)
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        # merge text and image embeddings
        # [batch_size, channel, height, width] -> [batch_size, num_patches, embed_dim]
        selected_image_feature = self.vision_tower(pixel_values.to(inputs_embeds.dtype))
        # [batch_size, num_patches, embed_dim] -> [batch_size, num_patches, hidden_size]
        image_feature = self.multi_modal_projector(selected_image_feature)
        # merge the embeddings of the text and the image tokens
        inputs_embeds, attention_mask, position_ids = (
            self._merge_input_ids_with_image_features(
                image_feature, inputs_embeds, input_ids, attention_mask, kv_cache
            )
        )

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            kv_cache=kv_cache,
        )

        return outputs
