from models.hunyuan_video import HunyuanVideoPipeline, get_rotary_pos_embed
from models.hunyuan_video import InitialLayer, DoubleBlock, concatenate_hidden_states, SingleBlock, OutputLayer
from models.hunyuan_video import init_empty_weights, load_model, load_safetensors, _convert_state_dict_keys, load_state_dict, set_module_tensor_to_device
import torch
from utils.common import AUTOCAST_DTYPE
import torch.nn.functional as F

def get_cond_latents(
    latents
):  
    batch_size, num_channels_latents, num_frames, height, width  = latents.shape

    first_frame_latents = latents[:, :, 0, ...]

    device, dtype = latents.device, latents.dtype

    padding_shape = (
        batch_size,
        num_channels_latents,
        num_frames - 1,
        height,
        width
    )

    latent_padding = torch.zeros(padding_shape, device=device, dtype=dtype)
    cond_image_latents = torch.cat([first_frame_latents, latent_padding], dim=2)
    return cond_image_latents


class SkyReelOutputLayer(OutputLayer):
    def __init__(self, transformer, is_i2v):
        super().__init__(transformer)
        self.is_i2v = is_i2v

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        x, vec, cu_seqlens, max_seqlen, freqs_cos, freqs_sin, txt_seq_len, img_seq_len, unpatchify_args, target = inputs
        img = x[:, :img_seq_len.item(), ...]

        # ---------------------------- Final layer ------------------------------
        img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)

        tt, th, tw = (arg.item() for arg in unpatchify_args)
        output = self.transformer[0].unpatchify(img, tt, th, tw)

        with torch.autocast('cuda', enabled=False):
            output = output.to(torch.float32)
            target = target.to(torch.float32)

            if self.is_i2v:
                num_channels = output.shape[1] // 2
                loss = F.mse_loss(output[:, :num_channels, ...], target)
            else:
                loss = F.mse_loss(output, target)
                
        return loss


class SkyreelsVideoPipeline(HunyuanVideoPipeline):
    def __init__(self, config):
        super().__init__(config)
        self.is_i2v = self.model_config["is_i2v"]

    # delay loading transformer to save RAM
    def load_diffusion_model(self):
        transformer_dtype = self.model_config.get('transformer_dtype', self.model_config['dtype'])
        # Device needs to be cuda here or we get an error. We initialize the model with empty weights so it doesn't matter, and
        # then directly load the weights onto CPU right after.
        factor_kwargs = {"device": 'cuda', "dtype": transformer_dtype}

        # only modify the following two lines comparing to hunyuan video 
        in_channels = self.args.latent_channels * 2 if self.is_i2v else self.args.latent_channels
        out_channels = self.args.latent_channels * 2 if self.is_i2v else self.args.latent_channels
        
        with init_empty_weights():
            transformer = load_model(
                self.args,
                in_channels=in_channels,
                out_channels=out_channels,
                factor_kwargs=factor_kwargs,
            )
        if transformer_path := self.model_config.get('transformer_path', None):
            state_dict = load_safetensors(transformer_path)
            state_dict = _convert_state_dict_keys(transformer.state_dict(), state_dict)
        else:
            state_dict = load_state_dict(self.args, self.args.model_base)
        params_to_keep = {"norm", "bias", "time_in", "vector_in", "guidance_in", "txt_in", "img_in"}
        base_dtype = self.model_config['dtype']
        for name, param in transformer.named_parameters():
            dtype_to_use = base_dtype if any(keyword in name for keyword in params_to_keep) else transformer_dtype
            set_module_tensor_to_device(transformer, name, device='cpu', dtype=dtype_to_use, value=state_dict[name])

        self.diffusers_pipeline.transformer = transformer
        self.transformer.train()
        # We'll need the original parameter name for saving, and the name changes once we wrap modules for pipeline parallelism,
        # so store it in an attribute here. Same thing below if we're training a lora and creating lora weights.
        for name, p in self.transformer.named_parameters():
            p.original_name = name
 

    def prepare_inputs(self, inputs, timestep_quantile=None):
        latents = inputs['latents'].float()
        bs, channels, num_frames, h, w = latents.shape

        prompt_embeds_1 = inputs['prompt_embeds_1']
        prompt_attention_mask_1 = inputs['prompt_attention_mask_1']
        prompt_embeds_2 = inputs['prompt_embeds_2']

        guidance_expand = torch.tensor(
            [self.model_config.get('guidance', 1.0)] * bs,
            dtype=torch.float32,
        ) * 1000

        timestep_sample_method = self.model_config.get('timestep_sample_method', 'logit_normal')

        if timestep_sample_method == 'logit_normal':
            dist = torch.distributions.normal.Normal(0, 1)
        elif timestep_sample_method == 'uniform':
            dist = torch.distributions.uniform.Uniform(0, 1)
        else:
            raise NotImplementedError()

        if timestep_quantile is not None:
            t = dist.icdf(torch.full((bs,), timestep_quantile, device=latents.device))
        else:
            t = dist.sample((bs,)).to(latents.device)

        if timestep_sample_method == 'logit_normal':
            sigmoid_scale = self.model_config.get('sigmoid_scale', 1.0)
            t = t * sigmoid_scale
            t = torch.sigmoid(t)

        if shift := self.model_config.get('shift', None):
            t = (t * shift) / (1 + (shift - 1) * t)

        x_1 = latents
        x_0 = torch.randn_like(x_1)
        t_expanded = t.view(-1, 1, 1, 1, 1)

        x_t = (1 - t_expanded) * x_1 + t_expanded * x_0

        if self.is_i2v:
            cond_image_latents = get_cond_latents(x_t)
            x_t = torch.cat([x_t, cond_image_latents], dim=1)

        target = x_0 - x_1 
 
        video_length = (num_frames - 1) * 4 + 1
        video_height = h * 8
        video_width = w * 8
        freqs_cos, freqs_sin = get_rotary_pos_embed(
            self.transformer, video_length, video_height, video_width
        )
        freqs_cos = freqs_cos.expand(bs, -1, -1)
        freqs_sin = freqs_sin.expand(bs, -1, -1)

        # timestep input to model needs to be in range [0, 1000]
        t = t * 1000

        return (
            x_t,
            t,
            prompt_embeds_1,
            prompt_attention_mask_1,
            prompt_embeds_2,
            freqs_cos,
            freqs_sin,
            guidance_expand,
            target,
        )

    def to_layers(self):
        transformer = self.transformer
        layers = [InitialLayer(transformer)]
        for block in transformer.double_blocks:
            layers.append(DoubleBlock(block))
        layers.append(concatenate_hidden_states)
        for block in transformer.single_blocks:
            layers.append(SingleBlock(block))
        layers.append(SkyReelOutputLayer(transformer, self.is_i2v))
        return layers
