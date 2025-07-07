import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Autoformer_EncDec import series_decomp
from layers.Embed import DataEmbedding_wo_pos
from layers.StandardNorm import Normalize
from layers.ChebyKANLayer import ChebyKANLinear
from layers.TimeDART_EncDec import Diffusion


class GlobalSTAR(nn.Module):
    """全局STAR模块 - 在原始序列上进行全局信息聚合"""

    def __init__(self, seq_len, channels, d_core=None):
        super().__init__()
        self.seq_len = seq_len
        self.channels = channels
        self.d_core = d_core if d_core is not None else channels // 2

        # 核心表示生成网络 - 作用于通道维度
        self.core_gen = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, self.d_core)
        )

        # 融合网络
        self.fusion_net = nn.Sequential(
            nn.Linear(channels + self.d_core, channels),
            nn.GELU(),
            nn.Dropout(0.1)
        )

    def stochastic_pooling(self, x):
        """SOFTS风格的随机池化 - 在通道维度进行聚合"""
        # x: [B, T, d_core]
        if self.training:
            # 训练时：按概率随机采样
            probs = F.softmax(x, dim=2)  # [B, T, d_core]

            # 对每个时间步独立采样
            sampled_values = []
            for t in range(x.shape[1]):
                # 对当前时间步的所有batch进行采样
                t_probs = probs[:, t, :]  # [B, d_core]
                t_values = x[:, t, :]  # [B, d_core]

                # 为每个batch样本采样一个核心值
                sampled_indices = torch.multinomial(t_probs, 1).squeeze(-1)  # [B]
                batch_indices = torch.arange(x.shape[0], device=x.device)
                sampled_t = t_values[batch_indices, sampled_indices].unsqueeze(1)  # [B, 1]
                sampled_values.append(sampled_t)

            core = torch.cat(sampled_values, dim=1).unsqueeze(-1)  # [B, T, 1]
        else:
            # 测试时：加权平均
            weights = F.softmax(x, dim=2)  # [B, T, d_core]
            core = torch.sum(x * weights, dim=2, keepdim=True)  # [B, T, 1]

        return core

    def forward(self, x):
        """
        x: [B, T, C] - 原始时序数据
        输出: [B, T, C] - 全局信息增强后的序列
        """
        B, T, C = x.shape

        # 生成核心表示候选
        core_candidates = self.core_gen(x)  # [B, T, d_core]

        # 随机池化生成全局核心 - 每个时间步一个核心值
        global_core = self.stochastic_pooling(core_candidates)  # [B, T, 1]

        # 将全局核心扩展到所有通道
        global_core_expanded = global_core.expand(B, T, self.d_core)  # [B, T, d_core]

        # 融合原始特征和全局核心
        fused_input = torch.cat([x, global_core_expanded], dim=-1)  # [B, T, C + d_core]
        fused_output = self.fusion_net(fused_input)  # [B, T, C]

        # 残差连接
        return x + fused_output


class LightweightDiffusion(nn.Module):
    """轻量级扩散模块"""

    def __init__(self, time_steps=20, device='cuda', scheduler='linear'):
        super().__init__()
        self.diffusion = Diffusion(time_steps=time_steps, device=device, scheduler=scheduler)

    def forward(self, x, apply_noise=True):
        if apply_noise and self.training:
            return self.diffusion(x)
        else:
            return x, None, None

class ResidualCNNProcessor(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.conv = nn.Conv1d(configs.d_model, configs.d_model, kernel_size=3, padding=1, groups=1)
        self.norm = nn.LayerNorm(configs.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(configs.d_model, configs.d_ff),
            nn.GELU(),
            nn.Linear(configs.d_ff, configs.d_model),
            nn.Dropout(configs.dropout)
        )

    def forward(self, x):
        # x: [B, T, C]
        x_conv = self.conv(x.transpose(1, 2)).transpose(1, 2)  # [B, T, C]
        x = x + x_conv  # 残差连接
        x = self.norm(x)
        return self.ffn(x)




class AdaptiveKANMixer(nn.Module):
    """自适应KAN混合器"""

    def __init__(self, d_model, component_type='trend'):
        super().__init__()
        # 根据分量类型选择KAN阶数
        order_map = {'trend': 6, 'seasonal': 4, 'residual': 8}
        order = order_map.get(component_type, 4)

        self.kan_layer = ChebyKANLinear(d_model, d_model, order)
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        B, T, C = x.shape
        x_kan = self.kan_layer(x.reshape(B * T, C)).reshape(B, T, C)
        x_conv = self.conv(x.transpose(1, 2)).transpose(1, 2)
        return self.norm(x + x_kan + x_conv)


class ComponentProcessor(nn.Module):
    """分量处理器 - 保持原有的KAN设计"""

    def __init__(self, configs, component_type):
        super().__init__()
        self.component_type = component_type

        if component_type == 'trend':
            self.processor = nn.Sequential(
                AdaptiveKANMixer(configs.d_model, 'trend'),
                nn.Linear(configs.d_model, configs.d_model),
                nn.GELU(),
                nn.Dropout(configs.dropout)
            )
        elif component_type == 'seasonal':
            self.diffusion = LightweightDiffusion(time_steps=20, device=configs.device)
            self.processor = AdaptiveKANMixer(configs.d_model, 'seasonal')
        else:  # residual
            self.processor = ResidualCNNProcessor(configs)

    def forward(self, x):
        if self.component_type == 'seasonal' and self.training:
            x_noise, noise, t = self.diffusion(x, apply_noise=True)
            return self.processor(x_noise)
        else:
            return self.processor(x)


class Model(nn.Module):
    """全局STAR + 分解KAN融合模型"""

    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        # 全局STAR模块 - 在分解之前进行全局信息聚合
        self.global_star = GlobalSTAR(
            seq_len=configs.seq_len,
            channels=configs.enc_in,
            d_core=configs.enc_in // 2
        )

        # 分解模块
        self.decomposition = series_decomp(configs.moving_avg)

        # 嵌入层
        if configs.channel_independence == 1:
            self.enc_embedding = DataEmbedding_wo_pos(1, configs.d_model, configs.embed, configs.freq, configs.dropout)
        else:
            self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                      configs.dropout)

        # 分量处理器（保持你的创新设计）
        self.trend_processor = ComponentProcessor(configs, 'trend')
        self.seasonal_processor = ComponentProcessor(configs, 'seasonal')
        self.residual_processor = ComponentProcessor(configs, 'residual')

        # 归一化
        self.normalize_layers = torch.nn.ModuleList([
            Normalize(configs.enc_in, affine=True, non_norm=True if configs.use_norm == 0 else False)
            for i in range(configs.down_sampling_layers + 1)
        ])

        # 预测层
        self.trend_predictor = nn.Linear(configs.seq_len, configs.pred_len)
        self.seasonal_predictor = nn.Linear(configs.seq_len, configs.pred_len)
        self.residual_predictor = nn.Linear(configs.seq_len, configs.pred_len)

        # 输出投影
        if configs.channel_independence == 1:
            self.projection_layer = nn.Linear(configs.d_model, 1, bias=True)
        else:
            self.projection_layer = nn.Linear(configs.d_model, configs.c_out, bias=True)

        # 可学习融合权重
        self.fusion_weights = nn.Parameter(torch.tensor([0.4, 0.2, 0.4]))  # [trend, seasonal, residual]

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast':
            return self.forecast(x_enc, x_mark_enc)
        else:
            raise ValueError('Only long_term_forecast implemented')

    def forecast(self, x_enc, x_mark_enc=None):
        B, T, N = x_enc.size()

        # 归一化
        x_enc = self.normalize_layers[0](x_enc, 'norm')

        # 🔥 关键改进：先进行全局STAR信息聚合，再分解
        x_star_enhanced = self.global_star(x_enc)  # [B, T, N] - 全局信息增强

        # 在增强后的序列上进行分解
        seasonal, trend = self.decomposition(x_star_enhanced)
        residual = x_star_enhanced - seasonal - trend

        # 通道独立性处理
        if self.configs.channel_independence == 1:
            seasonal = seasonal.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
            trend = trend.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
            residual = residual.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)

        # 嵌入
        if self.configs.channel_independence == 1 and x_mark_enc is not None:
            x_mark_enc_expanded = x_mark_enc.repeat(N, 1, 1)  # [B*N, T, mark_dim]
        else:
            x_mark_enc_expanded = x_mark_enc

        seasonal_emb = self.enc_embedding(seasonal, x_mark_enc_expanded)
        trend_emb = self.enc_embedding(trend, x_mark_enc_expanded)
        residual_emb = self.enc_embedding(residual, x_mark_enc_expanded)

        # 分量处理（保持你的KAN创新）
        seasonal_out = self.seasonal_processor(seasonal_emb)
        trend_out = self.trend_processor(trend_emb)
        residual_out = self.residual_processor(residual_emb)

        # 时序预测
        seasonal_pred = self.seasonal_predictor(seasonal_out.permute(0, 2, 1)).permute(0, 2, 1)
        trend_pred = self.trend_predictor(trend_out.permute(0, 2, 1)).permute(0, 2, 1)
        residual_pred = self.residual_predictor(residual_out.permute(0, 2, 1)).permute(0, 2, 1)

        # 投影
        seasonal_pred = self.projection_layer(seasonal_pred)
        trend_pred = self.projection_layer(trend_pred)
        residual_pred = self.projection_layer(residual_pred)

        # 加权融合
        weights = F.softmax(self.fusion_weights, dim=0)
        dec_out = (weights[0] * trend_pred +
                   weights[1] * seasonal_pred +
                   weights[2] * residual_pred)

        # 输出重塑
        if self.configs.channel_independence == 1:
            dec_out = dec_out.reshape(B, N, self.pred_len, -1)
            if dec_out.shape[-1] == 1:
                dec_out = dec_out.squeeze(-1)
            dec_out = dec_out.permute(0, 2, 1).contiguous()

        if dec_out.shape[-1] > self.configs.c_out:
            dec_out = dec_out[..., :self.configs.c_out]

        # 反归一化
        dec_out = self.normalize_layers[0](dec_out, 'denorm')
        return dec_out