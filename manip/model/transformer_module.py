import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_sinusoidal_encoding_table(num_positions, hidden_dim, padding_idx=None):
    positions = torch.arange(num_positions, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, hidden_dim, 2, dtype=torch.float32)
        * (-math.log(10000.0) / hidden_dim)
    )

    encoding_table = torch.zeros(num_positions, hidden_dim, dtype=torch.float32)
    encoding_table[:, 0::2] = torch.sin(positions * div_term)
    encoding_table[:, 1::2] = torch.cos(positions * div_term[: encoding_table[:, 1::2].shape[1]])

    if padding_idx is not None:
        encoding_table[padding_idx] = 0.0

    return encoding_table


def build_causal_attention_mask(token_positions):
    batch_size, sequence_length = token_positions.size()
    attention_mask = torch.triu(
        torch.ones((sequence_length, sequence_length), device=token_positions.device, dtype=torch.bool),
        diagonal=1,
    )
    return attention_mask.unsqueeze(0).expand(batch_size, -1, -1)


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, model_dim, key_dim, value_dim):
        super().__init__()

        self.num_heads = num_heads
        self.model_dim = model_dim
        self.key_dim = key_dim
        self.value_dim = value_dim

        self.query_projection = nn.Linear(model_dim, num_heads * key_dim)
        self.key_projection = nn.Linear(model_dim, num_heads * key_dim)
        self.value_projection = nn.Linear(model_dim, num_heads * value_dim)
        nn.init.normal_(self.query_projection.weight, mean=0, std=math.sqrt(2.0 / (model_dim + key_dim)))
        nn.init.normal_(self.key_projection.weight, mean=0, std=math.sqrt(2.0 / (model_dim + key_dim)))
        nn.init.normal_(self.value_projection.weight, mean=0, std=math.sqrt(2.0 / (model_dim + value_dim)))

        self.temperature = math.sqrt(key_dim)
        self.attention_dropout = nn.Dropout(0.1)

        self.output_projection = nn.Linear(num_heads * value_dim, model_dim)
        nn.init.xavier_normal_(self.output_projection.weight)
        self.layer_norm = nn.LayerNorm(model_dim)
        self.output_dropout = nn.Dropout(0.1)

    def forward(self, query, key, value, mask=None):
        batch_size, query_length, _ = query.shape
        _, key_length, _ = key.shape
        _, value_length, _ = value.shape

        assert key_length == value_length

        residual = query

        query = self.query_projection(query).view(batch_size, query_length, self.num_heads, self.key_dim)
        query = query.permute(2, 0, 1, 3).contiguous().view(-1, query_length, self.key_dim)
        key = self.key_projection(key).view(batch_size, key_length, self.num_heads, self.key_dim)
        key = key.permute(2, 0, 1, 3).contiguous().view(-1, key_length, self.key_dim)
        value = self.value_projection(value).view(batch_size, value_length, self.num_heads, self.value_dim)
        value = value.permute(2, 0, 1, 3).contiguous().view(-1, value_length, self.value_dim)

        attention_scores = torch.bmm(query, key.transpose(1, 2)) / self.temperature

        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask.repeat(self.num_heads, 1, 1), -torch.inf)

        attention_weights = F.softmax(attention_scores, dim=2)
        attention_weights = self.attention_dropout(attention_weights)
        attention_output = torch.bmm(attention_weights, value)

        attention_output = attention_output.view(self.num_heads, batch_size, query_length, self.value_dim)
        attention_output = attention_output.permute(1, 2, 0, 3).contiguous().view(batch_size, query_length, -1)
        attention_output = self.output_dropout(self.output_projection(attention_output))
        attention_output = self.layer_norm(attention_output + residual)

        return attention_output, attention_weights


class PositionwiseFeedForward(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()

        self.input_projection = nn.Conv1d(input_dim, hidden_dim, 1)
        self.output_projection = nn.Conv1d(hidden_dim, input_dim, 1)
        self.layer_norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, hidden_states):
        residual = hidden_states
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.output_projection(F.relu(self.input_projection(hidden_states)))
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.dropout(hidden_states)
        return self.layer_norm(hidden_states + residual)


class CrossAttentionDecoderLayer(nn.Module):
    def __init__(self, model_dim, num_heads, key_dim, value_dim):
        super().__init__()

        self.self_attention = MultiHeadAttention(num_heads, model_dim, key_dim, value_dim)
        self.cross_attention = MultiHeadAttention(num_heads, model_dim, key_dim, value_dim)
        self.feed_forward = PositionwiseFeedForward(model_dim, model_dim)

    def forward(self, hidden_states, self_attention_mask, padding_mask, context_embedding=None):
        hidden_states, self_attention = self.self_attention(
            hidden_states,
            hidden_states,
            hidden_states,
            mask=self_attention_mask,
        )
        hidden_states = hidden_states * padding_mask.unsqueeze(-1).float()

        if context_embedding is not None:
            hidden_states, _ = self.cross_attention(
                hidden_states,
                context_embedding,
                context_embedding,
            )
            hidden_states = hidden_states * padding_mask.unsqueeze(-1).float()

        hidden_states = self.feed_forward(hidden_states)
        hidden_states = hidden_states * padding_mask.unsqueeze(-1).float()

        return hidden_states, self_attention


class MotionTransformerDecoder(nn.Module):
    def __init__(
        self,
        input_dim,
        model_dim,
        num_layers,
        num_heads,
        key_dim,
        value_dim,
        max_timesteps,
        use_full_attention=False,
    ):
        super().__init__()

        self.input_projection = nn.Conv1d(input_dim, model_dim, 1)
        self.position_embedding = nn.Embedding.from_pretrained(
            build_sinusoidal_encoding_table(max_timesteps + 1, model_dim, padding_idx=0),
            freeze=True,
        )
        self.layers = nn.ModuleList(
            [
                CrossAttentionDecoderLayer(model_dim, num_heads, key_dim, value_dim)
                for _ in range(num_layers)
            ]
        )
        self.use_full_attention = use_full_attention

    def forward(
        self,
        decoder_input,
        padding_mask,
        decoder_position_ids,
        obj_embedding=None,
        language_embedding=None,
    ):
        attention_maps = []

        padding_mask = padding_mask.squeeze(1)
        decoder_position_ids = decoder_position_ids.squeeze(1)

        hidden_states = self.input_projection(decoder_input).transpose(1, 2)
        if obj_embedding is not None:
            hidden_states = torch.cat((obj_embedding, hidden_states), dim=1)

        position_embedding = self.position_embedding(decoder_position_ids)
        if self.use_full_attention:
            self_attention_mask = None
        else:
            self_attention_mask = build_causal_attention_mask(decoder_position_ids)

        hidden_states = hidden_states + position_embedding
        for layer in self.layers:
            hidden_states, self_attention = layer(
                hidden_states,
                self_attention_mask=self_attention_mask,
                padding_mask=padding_mask,
                context_embedding=language_embedding,
            )
            attention_maps.append(self_attention)

        return hidden_states, attention_maps


def maybe_apply_spectral_norm(module: nn.Module, use_spectral_norm: bool):
    return nn.utils.spectral_norm(module) if use_spectral_norm else module


class TemporalConvResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=1, dropout=0.0, use_spectral_norm=False):
        super().__init__()
        padding = (kernel_size - 1) // 2 * dilation

        def make_conv():
            return maybe_apply_spectral_norm(
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    padding=padding,
                    dilation=dilation,
                ),
                use_spectral_norm,
            )

        self.conv1 = make_conv()
        self.conv2 = make_conv()
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states):
        residual = hidden_states
        hidden_states = hidden_states.permute(0, 2, 1)
        hidden_states = self.conv1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)
        hidden_states = hidden_states.permute(0, 2, 1)
        return self.activation(residual + hidden_states)
