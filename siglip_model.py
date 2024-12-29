from typing import Optional, Tuple
import torch
import torch.nn as nn

class SiglipVisionConfig:

    def __init__(
        self,
        embd_dim=768,
        intermediate_dim=3072,
        num_layers=12,
        num_attention_heads=12,
        num_channels=3,
        image_size=224,
        patch_size=16,
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        num_image_tokens: int = None,
        **kwargs
    ):
        super().__init__()

        self.embd_dim = embd_dim
        self.intermediate_dim = intermediate_dim
        self.num_layers = num_layers
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
        
        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=config.embd_dim,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            padding="valid"
        )

        self.num_patches = (config.image_size // config.patch_size) ** 2
        self.position_embedding = nn.Embedding(self.num_patches, config.embd_dim)
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_patches).expand((1, -1)),
            persistent=False,
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # [batch, channel, height, width] -> [batch, embd_dim, num_patches_H, num_patches_W]
        # num_patches_H = height // patch_size, num_patches_W = width // patch_size
        patch_embds = self.patch_embedding(pixel_values)
        # [batch, embd_dim, num_patches_H, num_patches_W] -> [batch, embd_dim, num_patches]
        # num_patches = num_patches_H * num_patches_W
        embeddings = patch_embds.flatten(start_dim=2) # [batch, embd_dim, num_patches]
        embeddings = embeddings.transpose(1, 2) # [batch, num_patches, embd_dim]
        embeddings = embeddings + self.position_embedding(self.position_ids) # [batch, num_patches, embd_dim]
        return embeddings # [batch, num_patches, embd_dim]


class SiglipAttention(nn.Module):

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embd_dim = config.embd_dim
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embd_dim // self.num_heads
        self.scale = self.head_dim ** -0.5  # 1 / sqrt(head_dim)
        self.dropout = config.attention_dropout

        self.q_proj = nn.Linear(self.embd_dim, self.embd_dim)
        self.k_proj = nn.Linear(self.embd_dim, self.embd_dim)
        self.v_proj = nn.Linear(self.embd_dim, self.embd_dim)
        self.out_proj = nn.Linear(self.embd_dim, self.embd_dim)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # hidden_states: [batch, num_patches, embd_dim]
        batch_size, seq_len, _ = hidden_states.size()
        
        # [batch, num_patches, embd_dim]
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # [batch, num_patches, num_heads, head_dim] -> [batch, num_heads, num_patches, head_dim]
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # [batch, num_heads, num_patches, num_patches]
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scale
        
        if attn_weights.size() != (batch_size, self.num_heads, seq_len, seq_len):
            raise ValueError(
                f"Attention weights should be of size {(batch_size, self.num_heads, seq_len, seq_len)}, "
                f"but is {attn_weights.size()}"
            )

        # [batch, num_heads, num_patches, num_patches]
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        
        # [batch, num_heads, num_patches, head_dim]
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (batch_size, self.num_heads, seq_len, self.head_dim):
            raise ValueError(
                f"Attention output should be of size {(batch_size, self.num_heads, seq_len, self.head_dim)}, "
                f"but is {attn_output.size()}"
            )

        # [batch, num_heads, num_patches, head_dim] -> [batch, num_patches, num_heads, head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous()
        # [batch, num_patches, embd_dim]
        attn_output = attn_output.reshape(batch_size, seq_len, self.embd_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights


class SiglipMLP(nn.Module):

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(config.embd_dim, config.intermediate_dim)
        self.fc2 = nn.Linear(config.intermediate_dim, config.embd_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states: [batch, num_patches, embd_dim]
        hidden_states = self.fc1(hidden_states) # [batch, num_patches, intermediate_dim]
        hidden_states = nn.functional.gelu(hidden_states, approximate="tanh") # [batch, num_patches, intermediate_dim]
        hidden_states = self.fc2(hidden_states) # [batch, num_patches, embd_dim]
        return hidden_states # [batch, num_patches, embd_dim]


class SiglipEncoderLayer(nn.Module):

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.layer_norm1 = nn.LayerNorm(config.embd_dim, eps=config.layer_norm_eps)
        self.self_attn = SiglipAttention(config)
        self.layer_norm2 = nn.LayerNorm(config.embd_dim, eps=config.layer_norm_eps) 
        self.mlp = SiglipMLP(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states # [batch, num_patches, embd_dim]
        hidden_states = self.layer_norm1(hidden_states) # [batch, num_patches, embd_dim]
        hidden_states, _ = self.self_attn(hidden_states=hidden_states) # [batch, num_patches, embd_dim]
        hidden_states = residual + hidden_states # [batch, num_patches, embd_dim]

        residual = hidden_states # [batch, num_patches, embd_dim]
        hidden_states = self.layer_norm2(hidden_states) # [batch, num_patches, embd_dim]
        hidden_states = self.mlp(hidden_states) # [batch, num_patches, embd_dim]
        hidden_states = residual + hidden_states # [batch, num_patches, embd_dim]
        
        return hidden_states # [batch, num_patches, embd_dim]


class SiglipEncoder(nn.Module):

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [SiglipEncoderLayer(config) for _ in range(config.num_layers)]
        )

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        # [batch, num_patches, embd_dim]
        hidden_states = inputs_embeds

        for encoder_layer in self.layers:
            hidden_states = encoder_layer(hidden_states)

        return hidden_states


class SiglipVisionTransformer(nn.Module):
    
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipEncoder(config)
        self.post_layernorm = nn.LayerNorm(config.embd_dim, eps=config.layer_norm_eps)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # [batch, channel, height, width] -> [batch, num_patches, embd_dim]
        img_embeddings = self.embeddings(pixel_values)
        output = self.encoder(inputs_embeds=img_embeddings)
        output = self.post_layernorm(output)
        return output


class SiglipVisionModel(nn.Module):

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.vision_model = SiglipVisionTransformer(config)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # [batch, channel, height, width] -> [batch, num_patches, embd_dim]
        return self.vision_model(pixel_values=pixel_values)