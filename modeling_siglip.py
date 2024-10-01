from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class SiglipVisionConfig:
    
    def __init__(
        self,
        hidden_size=768,         # vector embedding size
        intermediate_size=3072,  # linear layer size
        num_hidden_layers=12,    # vision transformer
        num_attention_heads=12,  # vision transformer
        num_channels=3,          # image channels
        image_size=224,
        patch_size=16,
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        num_image_tokens: int = None,
        **kwargs
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self.num_image_tokens = num_image_tokens


class SiglipVisionEmbeddings(nn.Module):
    
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        
        self.patch_embeddings = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",  # no padding
        )
        
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        # learned embeddings instead of fixed embeddings
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer(  # fixed embeddings
            "position_ids",
            torch.arange(self.num_positions),
            persistent=False,
        )
    
    def forward(self, pixel_values: torch.FloatTensor) -> torch.FloatTensor:
        _, _, height, width = pixel_values.shape  # (B, C, H, W)
        # output size of patch_embeddings: (B, embed_dim, num_patches, num_patches)
        patch_embeddings = self.patch_embeddings(pixel_values)
        patch_embeddings = patch_embeddings.flatten(2)  # (B, embed_dim, num_patches ** 2)
        patch_embeddings = patch_embeddings.transpose(1, 2)  # (B, num_patches ** 2, embed_dim)
        
        # add position embeddings
        position_embeddings = self.position_embedding(self.position_ids)
        embeddings = patch_embeddings + position_embeddings
        
        return embeddings
        

class SiglipMLP(nn.Module):
    
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        
    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        # (B, num_patches, embed_dim) -> (B, num_patches, intermediate_size)
        hidden_states = self.fc1(hidden_states)
        hidden_states = F.gelu(hidden_states)
        # (B, num_patches, intermediate_size) -> (B, num_patches, embed_dim)
        hidden_states = self.fc2(hidden_states)
        return hidden_states
    

class SiglipAttention(nn.Module):
    
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim ** -0.5  # 1 / sqrt(self.head_dim)
        self.dropout = config.attention_dropout
        
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)
        
    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
    
    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        # residual: [batch_size, num_patches, embed_dim]
        batch_size, seq_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        # query_states after transpose: [batch_size, num_heads, num_patches, head_dim]
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # calculate the attention using formula Q*K.T / sqrt(d_k). attn_weights: [batch_sizes, num_heads, num_patches, num_patches]
        atten_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scale
        
        if atten_weights.size() != (batch_size, self.num_heads, seq_len, seq_len):
            raise ValueError(
                f"Attention weights should be of size {(batch_size, self.num_heads, seq_len, seq_len)}, \
                but is {atten_weights.size()}"
            )
        
        # apply softmax row-wise. attn_weights: [batch_size, num_head, num_patches, num_patches]
        atten_weights = F.softmax(atten_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        # apply dropout only during training
        atten_weights = F.dropout(atten_weights, p=self.dropout, training=self.training)
        # multiply the attention weights aby the value states. attn_output [batch_suze, num_head, num_patches, head_dim]
        atten_output = torch.matmul(atten_weights, value_states)
        
        if atten_output.size() != (batch_size, self.num_heads, seq_len, self.head_dim):
            raise ValueError(
                f"atten_output should be of size {(batch_size, self.num_heads, seq_len, self.head_dim)}, \
                but is {atten_output.size()}"
            )
            
        # transpose back to [batch_size, num_patches, num_heads, head_dim]
        atten_output = atten_output.transpose(1, 2).contiguous()
        # reshape to [batch_size, num_patches, embed_dim]
        atten_output = atten_output.reshape(batch_size, seq_len, self.embed_dim)
        # [batch_size, num_patches, embed_dim]
        atten_output = self.out_proj(atten_output)
        
        return atten_output, atten_weights


class SiglipEncoderLayer(nn.Module):
    
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.self_atten = SiglipAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)
        
    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        # residual: [batch_size, num_patches, embed_dim]
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)  # (B, num_patches, embed_dim)
        hidden_states, _ = self.self_atten(hidden_states)  # (B, num_patches, embed_dim)
        hidden_states = hidden_states + residual  # (B, num_patches, embed_dim)
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)  # (B, num_patches, embed_dim)
        hidden_states = self.mlp(hidden_states)  # (B, num_patches, embed_dim)
        hidden_states = hidden_states + residual  # (B, num_patches, embed_dim)
        return hidden_states
        


class SiglipEncoder(nn.Module):
    
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [SiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        
    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        # input_embeds: [batch_size, num_patches, embed_dim]
        hidden_states = inputs_embeds
        
        # [batch_size, num_patches, embed_dim] -> [batch_size, num_patches, embed_dim]
        for encoder_layer in self.layers:
            hidden_states = encoder_layer(hidden_states)
            
        return hidden_states

class SiglipVisionTransformer(nn.Module):
    
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        
        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipEncoder(config)
        self.post_layernorm = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        
    def forward(self, pixel_values: torch.FloatTensor) -> Tuple[torch.FloatTensor]:
        # [batch_size, channels, height, width] -> [batch_size, num_patches, embedding_size]
        embeddings = self.embeddings(pixel_values)
        encoder_outputs = self.encoder(inputs_embeds=embeddings)
        last_hidden_state = self.post_layernorm(encoder_outputs)
        return last_hidden_state
        
        
        
class SiglipVisionModel(nn.Module):
    
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.vision_model = SiglipVisionTransformer(config)
        
    def forward(self, pixel_values: torch.FloatTensor) -> Tuple[torch.FloatTensor]:
        # [batch_size, channels, height, width] -> [batch_size, num_patches, embedding_size]
        return self.vision_model(pixel_values=pixel_values)
    

