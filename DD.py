import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Autoformer_EncDec import series_decomp
from layers.Embed import DataEmbedding_wo_pos
from layers.StandardNorm import Normalize
from layers.ChebyKANLayer import ChebyKANLinear
from layers.TimeDART_EncDec import Diffusion


class AdaptiveDiffusion(nn.Module):
    """改进的自适应扩散模块"""

    def __init__(self, time_steps=20, device='cuda', scheduler='cosine'):
        super().__init__()
        self.time_steps = time_steps
        self.device = device

        # 原有的Diffusion保持不变，确保兼容性
        self.diffusion = Diffusion(time_steps=time_steps, device=device, scheduler=scheduler)

        # 新增：可学习的噪声权重（为三个分量分别设置）
        self.noise_weights = nn.Parameter(torch.ones(3))

        # 余弦调度器
        if scheduler == 'cosine':
            self.alpha_schedule = self._cosine_schedule().to(device)
        else:
            # 保持原有的线性调度器作为备选
            self.alpha_schedule = None

    def _cosine_schedule(self):
        """余弦噪声调度器 - 根据TimeDART论文优化"""
        steps = torch.arange(self.time_steps + 1, dtype=torch.float32)
        alpha_bar = torch.cos((steps / self.time_steps + 0.008) / 1.008 * torch.pi / 2) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]
        return alpha_bar

    def forward(self, x, component_type='seasonal', apply_noise=True):
        if not apply_noise or not self.training:
            return x, None, None

        # 如果使用余弦调度器，使用改进的噪声添加逻辑
        if self.alpha_schedule is not None:
            return self._enhanced_diffusion(x, component_type)
        else:
            # 回退到原有的diffusion逻辑
            return self.diffusion(x)

    def _enhanced_diffusion(self, x, component_type):
        """增强的扩散过程"""
        batch_size, seq_len, features = x.shape

        # 分量特定的噪声权重
        component_weights = F.softmax(self.noise_weights, dim=0)
        if component_type == 'trend':
            noise_scale = component_weights[0] * 0.5  # 趋势用较小噪声
        elif component_type == 'seasonal':
            noise_scale = component_weights[1] * 1.0  # 季节性用中等噪声
        else:  # residual
            noise_scale = component_weights[2] * 1.5  # 残差用较大噪声

        # 独立采样时间步长（TimeDART的独立噪声策略）
        t = torch.randint(0, self.time_steps, (seq_len,), device=self.device)

        # 生成噪声
        noise = torch.randn_like(x)

        # 根据时间步长和分量类型添加噪声
        alpha_bar_t = self.alpha_schedule[t].to(self.device)
        alpha_bar_t = alpha_bar_t.view(1, -1, 1)  # [1, seq_len, 1]

        # 应用分量特定的噪声缩放
        effective_alpha = alpha_bar_t * noise_scale + (1 - noise_scale) * 0.8
        effective_alpha = torch.clamp(effective_alpha, min=0.1, max=0.99)

        x_noisy = torch.sqrt(effective_alpha) * x + torch.sqrt(1 - effective_alpha) * noise

        return x_noisy, noise, t


class CrossAttentionDenoiser(nn.Module):
    """轻量级交叉注意力去噪器"""

    def __init__(self, d_model, num_heads=4, num_layers=1):
        super().__init__()

        # 保持轻量级，只用一层注意力
        self.attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # 简单的前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model)
        )

        # 时间嵌入（简化版本）
        self.time_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model)
        )

    def sinusoidal_position_encoding(self, timesteps, dim):
        """为时间步生成正弦位置编码"""
        if timesteps.numel() == 0:
            return torch.zeros(0, dim, device=timesteps.device)

        half_dim = dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1))
        return emb

    def forward(self, noisy_input, clean_context, timesteps=None):
        """
        noisy_input: [B, T, D] 噪声输入
        clean_context: [B, T, D] 清洁的上下文
        timesteps: [T] 时间步长（可选）
        """
        x = noisy_input

        # 如果有时间步信息，添加时间嵌入
        if timesteps is not None and len(timesteps) > 0:
            time_emb = self.sinusoidal_position_encoding(timesteps, x.shape[-1])
            if time_emb.shape[0] == x.shape[1]:  # 确保维度匹配
                time_emb = self.time_mlp(time_emb).unsqueeze(0)  # [1, T, D]
                x = x + time_emb

        # 交叉注意力：用清洁上下文作为key和value
        attn_out, _ = self.attention(x, clean_context, clean_context)
        x = self.norm1(x + attn_out)

        # 前馈网络
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return x


class AdaptiveKANMixer(nn.Module):
    """自适应KAN混合器"""

    def __init__(self, d_model, component_type='trend', order=None):
        super().__init__()
        # 如果没有指定order，使用默认映射
        if order is None:
            order_map = {'trend': 3, 'seasonal': 5, 'residual': 4}
            order = order_map.get(component_type, 4)

        self.kan_layer = ChebyKANLinear(d_model, d_model, order)
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        B, T, C = x.shape
        x_kan = self.kan_layer(x.reshape(B * T, C)).reshape(B, T, C)
        x_conv = self.conv(x.transpose(1, 2)).transpose(1, 2)
        return self.norm(x + x_kan + x_conv)


class EnhancedComponentProcessor(nn.Module):
    """增强的分量处理器"""

    def __init__(self, configs, component_type):
        super().__init__()
        self.component_type = component_type
        self.configs = configs

        # 从configs中读取配置，提供默认值
        diffusion_steps = getattr(configs, 'diffusion_steps', 20)
        diffusion_scheduler = getattr(configs, 'diffusion_scheduler', 'cosine')
        use_enhanced_diffusion = getattr(configs, 'use_enhanced_diffusion', True)
        use_cross_attention_denoiser = getattr(configs, 'use_cross_attention_denoiser', True)
        denoiser_num_heads = getattr(configs, 'denoiser_num_heads', 4)
        denoiser_num_layers = getattr(configs, 'denoiser_num_layers', 1)

        # KAN阶数配置
        kan_order_trend = getattr(configs, 'kan_order_trend', 3)
        kan_order_seasonal = getattr(configs, 'kan_order_seasonal', 5)
        kan_order_residual = getattr(configs, 'kan_order_residual', 4)

        # 使用改进的自适应扩散
        if use_enhanced_diffusion:
            self.diffusion = AdaptiveDiffusion(
                time_steps=diffusion_steps,
                device=configs.device,
                scheduler=diffusion_scheduler
            )
        else:
            # 使用原始扩散模块
            from layers.TimeDART_EncDec import Diffusion
            self.diffusion = Diffusion(
                time_steps=diffusion_steps,
                device=configs.device,
                scheduler=diffusion_scheduler
            )

        # 去噪器（根据配置决定是否在季节性分量中使用）
        if component_type == 'seasonal' and use_cross_attention_denoiser:
            self.denoiser = CrossAttentionDenoiser(
                d_model=configs.d_model,
                num_heads=denoiser_num_heads,
                num_layers=denoiser_num_layers
            )

        # 根据分量类型选择KAN阶数
        if component_type == 'trend':
            kan_order = kan_order_trend
        elif component_type == 'seasonal':
            kan_order = kan_order_seasonal
        else:  # residual
            kan_order = kan_order_residual

        # 保持原有的处理器结构，但使用配置化的KAN阶数
        if component_type == 'trend':
            self.processor = nn.Sequential(
                AdaptiveKANMixer(configs.d_model, 'trend', order=kan_order),
                nn.Linear(configs.d_model, configs.d_model),
                nn.GELU(),
                nn.Dropout(configs.dropout)
            )
        elif component_type == 'seasonal':
            self.processor = AdaptiveKANMixer(configs.d_model, 'seasonal', order=kan_order)
        else:  # residual
            self.processor = nn.Sequential(
                nn.Linear(configs.d_model, configs.d_ff),
                nn.GELU(),
                nn.Linear(configs.d_ff, configs.d_model),
                nn.Dropout(configs.dropout)
            )

    def forward(self, x, causal_context=None):
        """
        x: 输入分量嵌入
        causal_context: 因果上下文（未来可能会用到）
        """
        if self.component_type == 'seasonal' and self.training:
            # 应用扩散过程
            if hasattr(self.diffusion, '_enhanced_diffusion'):
                # 使用增强扩散
                x_noise, noise, timesteps = self.diffusion(
                    x, component_type=self.component_type, apply_noise=True
                )
            else:
                # 使用原始扩散
                x_noise, noise, timesteps = self.diffusion(x)

            # 如果有去噪器，进行去噪处理
            if hasattr(self, 'denoiser') and causal_context is not None:
                x_denoised = self.denoiser(x_noise, causal_context, timesteps)
                return self.processor(x_denoised)
            else:
                return self.processor(x_noise)
        else:
            return self.processor(x)


class Model(nn.Module):
    """融合时序模型主体"""

    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        # 分解模块
        self.decomposition = series_decomp(configs.moving_avg)

        # 嵌入层
        if configs.channel_independence == 1:
            self.enc_embedding = DataEmbedding_wo_pos(1, configs.d_model, configs.embed, configs.freq, configs.dropout)
        else:
            self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                      configs.dropout)

        # 使用增强的分量处理器
        self.trend_processor = EnhancedComponentProcessor(configs, 'trend')
        self.seasonal_processor = EnhancedComponentProcessor(configs, 'seasonal')
        self.residual_processor = EnhancedComponentProcessor(configs, 'residual')

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

        # 融合机制配置
        self.use_dynamic_fusion = getattr(configs, 'use_dynamic_fusion', False)

        if self.use_dynamic_fusion:
            # 动态权重生成器
            self.dynamic_fusion = nn.Sequential(
                nn.Linear(configs.d_model * 3, configs.d_model),
                nn.GELU(),
                nn.Linear(configs.d_model, 3),
                nn.Softmax(dim=-1)
            )
            print("✅ 启用动态融合权重生成器")
        else:
            # 静态可学习融合权重
            self.fusion_weights = nn.Parameter(torch.ones(3) / 3)
            print("✅ 使用静态可学习融合权重")

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast':
            return self.forecast(x_enc, x_mark_enc)
        else:
            raise ValueError('Only long_term_forecast implemented')

    def forecast(self, x_enc, x_mark_enc=None):
        B, T, N = x_enc.size()

        # 归一化
        x_enc = self.normalize_layers[0](x_enc, 'norm')

        # 分解
        seasonal, trend = self.decomposition(x_enc)
        residual = x_enc - seasonal - trend

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

        # 分量处理（传递上下文信息给季节性处理器）
        trend_out = self.trend_processor(trend_emb)
        seasonal_out = self.seasonal_processor(seasonal_emb, causal_context=trend_out)  # 传递趋势上下文
        residual_out = self.residual_processor(residual_emb)

        # 时序预测
        seasonal_pred = self.seasonal_predictor(seasonal_out.permute(0, 2, 1)).permute(0, 2, 1)
        trend_pred = self.trend_predictor(trend_out.permute(0, 2, 1)).permute(0, 2, 1)
        residual_pred = self.residual_predictor(residual_out.permute(0, 2, 1)).permute(0, 2, 1)

        # 投影
        seasonal_pred = self.projection_layer(seasonal_pred)
        trend_pred = self.projection_layer(trend_pred)
        residual_pred = self.projection_layer(residual_pred)

        # 融合预测结果
        if self.use_dynamic_fusion:
            # 动态权重融合
            combined_features = torch.cat([trend_pred, seasonal_pred, residual_pred], dim=-1)
            dynamic_weights = self.dynamic_fusion(combined_features)  # [B, T, 3]

            dec_out = (dynamic_weights[..., 0:1] * trend_pred +
                       dynamic_weights[..., 1:2] * seasonal_pred +
                       dynamic_weights[..., 2:3] * residual_pred)
        else:
            # 静态权重融合（原有逻辑）
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