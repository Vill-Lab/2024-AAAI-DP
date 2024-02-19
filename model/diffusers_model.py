import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T
from typing import Any, Optional, Tuple, Union, Dict, List, Callable
import types

from diffusers import AutoencoderKL, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.models.attention_processor import Attention, AttnProcessor, AttnProcessor2_0

from transformers import CLIPTextModel
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.clip.modeling_clip import CLIPTextTransformer, CLIPPreTrainedModel, CLIPModel, _expand_mask

import gc

class DiversePersonModel(nn.Module):
    def __init__(self, text_encoder, image_encoder, vae, unet, args):
        super().__init__()
        self.pretrained_model_hf = args.pretrained_model_hf
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        self.vae = vae
        self.unet = unet
        self.revision = args.revision
        self.attribute_loc = args.attribute_loc
        self.attribute_loc_weight = args.attribute_loc_weight
        
        # Initializing reference attention dictionary
        if self.attribute_loc:
            self.reference_attention = {}
            self.unet = get_reference_from_attation(self.unet, self.reference_attention, self.loc_latyers)
            # Initializing attribute localization loss function from BalancedL1Loss
            self.attribute_localization_loss_fn = BalancedL1Loss(1.0, args.attribute_loc_norm)
        
        self.loc_latyers = args.loc_latyers
        self.mask_loss = args.mask_loss
        embedding_dim = text_encoder.config.hidden_size
        self.controller = DiversePersonController(embedding_dim)
        
        
    @staticmethod
    def load_pretrained_model_hf(args):
        text_encoder = DiversePersonTextEncoder.load_pretrained_model_hf(args.pretrained_model_hf, subfolder="text_encoder", revision=args.revision)
        vae = AutoencoderKL.from_pretrained(args.pretrained_model_hf, subfolder="vae", revision=args.revision)
        unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_hf, subfolder="unet")
        image_encoder = UniversalCLIPImageEncoder.load_pretrained_model_hf(args.image_encoder_hf)
        
        return DiversePersonModel(text_encoder, image_encoder, vae, unet, args)
     
     
    def convert_model(self):
        convertor = StableDiffusionPipeline.from_pretrained(self.pretrained_model_hf, revision=self.revision, text_encoder=self.text_encoder, vae=self.vae, unet=self.unet)
        convertor.safety_checker = None
        convertor.image_encoder = self.image_encoder
        convertor.controller = self.controller
        
        return convertor
    
    
    def forward(self, batch, noise_sched):
        # Retrieving batch items
        pix_v = batch["pix_v"]
        input_ids = batch["input_ids"]
        img_token_mask = batch["img_token_mask"]
        attribute_v = batch["attribute_v"]
        reference_attribute_num = batch["reference_attribute_num"]
        
        # Encoding latent vectors using VAE
        latents = self.vae.encode(pix_v.to(self.vae.parameters().__next__().dtype)).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor
        
        # Adding noise to latent vectors
        noise = torch.randn_like(latents)
        time_steps = torch.randint(0, noise_sched.num_train_timesteps, (latents.shape[0],), device=latents.device)
        time_steps = time_steps.long()
        noisy_latents = noise_sched.add_noise(latents, noise, time_steps)
        
        attribute_embedding = self.image_encoder(attribute_v)
        encoder_hidden_states = self.text_encoder(input_ids, img_token_mask, attribute_embedding, reference_attribute_num)[0]  
        # Controlling the encoder hidden states
        encoder_hidden_states = self.controller(encoder_hidden_states,attribute_embedding,img_token_mask,reference_attribute_num,)
        
        if noise_sched.config.prediction_type == "epsilon":
            target = noise
        elif noise_sched.config.prediction_type == "v_prediction":
            target = noise_sched.get_velocity(latents, noise, time_steps)
        else:
            raise ValueError(f"Unsupported prediction type {noise_sched.config.prediction_type}")
        
        # Generating predictions using UNet
        predition = self.unet(noisy_latents, time_steps, encoder_hidden_states).sample
        
        if self.mask_loss and torch.rand(1) < 0.73:
            attribute_seg_maps = batch["attribute_seg_maps"]
            mask = (attribute_seg_maps.sum(dim=1) > 0).float()
            mask = F.interpolate(mask.unsqueeze(1),size=(predition.shape[-2], predition.shape[-1]),mode="bilinear",align_corners=False,)
            predition = predition * mask
            target = target * mask
        
        denoise_loss = F.mse_loss(predition.float(), target.float(), reduction="mean")
        loss_dict = {"denoise_loss": denoise_loss}
        
        if self.attribute_loc:
            attribute_seg_maps = batch["attribute_seg_maps"]
            img_token_idx = batch["img_token_idx"]
            img_token_idx_mask = batch["img_token_idx_mask"]
            
            localization_loss = get_attribute_localization_loss(self.reference_attention,attribute_seg_maps,img_token_idx,img_token_idx_mask,self.attribute_localization_loss_fn,)
            loss_dict["localization_loss"] = localization_loss
            loss = self.attribute_loc_weight * localization_loss + denoise_loss
            
            if hasattr(self, "reference_attention"):
                keys = list(self.reference_attention.keys())
                for k in keys:
                    del self.reference_attention[k]
            gc.collect()
            
        else:
            loss = denoise_loss
            
        loss_dict["loss"] = loss
        
        return loss_dict
    
    
class DiversePersonController(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.mlp1 = MLP(embedding_dim * 2, embedding_dim, embedding_dim, use_residual=False)
        self.mlp2 = MLP(embedding_dim, embedding_dim, embedding_dim, use_residual=True)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        
        
    def fuse_fn(self, text_embeddings, attribute_embedding):
        text_attribute_embeddings = torch.cat([text_embeddings, attribute_embedding], dim=-1)
        text_attribute_embeddings = self.mlp1(text_attribute_embeddings) + text_embeddings
        text_attribute_embeddings = self.mlp2(text_attribute_embeddings)
        text_attribute_embeddings = self.layer_norm(text_attribute_embeddings)
        
        return text_attribute_embeddings
    
    
    def forward(self,text_embeddings,attribute_embedding,img_token_mask,reference_attribute_num,)->torch.Tensor:
        text_attribute_embeddings = fuse_attribute_embeddings(text_embeddings,img_token_mask,attribute_embedding,reference_attribute_num,self.fuse_fn,)
        
        return text_attribute_embeddings
    
    
class DiversePersonTextEncoder(CLIPPreTrainedModel):
    _build_causal_attention_mask = CLIPTextTransformer._build_causal_attention_mask
    
    
    @staticmethod
    def load_pretrained_model_hf(model_name_or_path, **kwargs):
        model = CLIPTextModel.from_pretrained(model_name_or_path, **kwargs)
        text_model = model.text_model
        
        return DiversePersonTextEncoder(text_model)
    
    
    def __init__(self, text_model):
        super().__init__(text_model.config)
        self.config = text_model.config
        self.final_layer_norm = text_model.final_layer_norm
        self.embeddings = text_model.embeddings
        self.encoder = text_model.encoder
        
        
    def forward(self,input_ids,img_token_mask=None,attribute_embedding=None,reference_attribute_num=None,attention_mask: Optional[torch.Tensor] = None,output_attentions: Optional[bool] = None,output_hidden_states: Optional[bool] = None,loss_dict: Optional[bool] = None,)->Union[Tuple,BaseModelOutputWithPooling]:
        output_attentions = (output_attentions if output_attentions is not None else self.config.output_attentions)
        output_hidden_states = (output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states)
        
        loss_dict = (loss_dict if loss_dict is not None else self.config.use_return_dict)
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        
        hidden_states = self.embeddings(input_ids)
        bsz, seq_len = input_shape
        causal_attention_mask = self._build_causal_attention_mask(bsz, seq_len, hidden_states.dtype).to(hidden_states.device)
        
        if attention_mask is not None:
            attention_mask = _expand_mask(attention_mask, hidden_states.dtype)
        
        encoder_outputs = self.encoder(inputs_embeddings=hidden_states,attention_mask=attention_mask,causal_attention_mask=causal_attention_mask,output_attentions=output_attentions,output_hidden_states=output_hidden_states,loss_dict=loss_dict,)
        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.final_layer_norm(last_hidden_state)
        pooled_output = last_hidden_state[torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),input_ids.to(dtype=torch.int, device=last_hidden_state.device).argmax(dim=-1),]
        
        if not loss_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]
        
        return BaseModelOutputWithPooling(last_hidden_state=last_hidden_state,pooler_output=pooled_output,hidden_states=encoder_outputs.hidden_states,attentions=encoder_outputs.attentions,)


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, use_residual=True):
        super().__init__()
        if use_residual:
            assert in_dim == out_dim
            
        self.layernorm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.use_residual = use_residual
        self.act_fn = nn.GELU()
        
        
    def forward(self, x):
        residual = x
        x = self.layernorm(x)
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.fc2(x)
        
        if self.use_residual:
            x = x + residual
        return x
    
    
class UniversalCLIPImageEncoder(CLIPPreTrainedModel):
    @staticmethod
    def load_pretrained_model_hf(model_hf,):
        model = CLIPModel.from_pretrained(model_hf)
        vision_model = model.vision_model
        visual_projection = model.visual_projection
        vision_processor = T.Normalize((0.48145466, 0.4578275, 0.40821073),(0.26862954, 0.26130258, 0.27577711),)
        
        return UniversalCLIPImageEncoder(vision_model,visual_projection,vision_processor,)
    
    def __init__(self,vision_model,visual_projection,vision_processor,):
        super().__init__(vision_model.config)
        self.vision_model = vision_model
        self.visual_projection = visual_projection
        self.vision_processor = vision_processor
        self.image_size = vision_model.config.image_size
        
        
    def forward(self,attribute_v):
        b, reference_attribute_num, c, h, w = attribute_v.shape
        attribute_v = attribute_v.view(b * reference_attribute_num, c, h, w)
        
        if h != self.image_size or w != self.image_size:
            h, w = self.image_size, self.image_size
            attribute_v = F.interpolate(attribute_v, (h, w), mode="bilinear", antialias=True,)
        
        attribute_v = self.vision_processor(attribute_v)
        attribute_embedding = self.vision_model(attribute_v)[1]
        attribute_embedding = self.visual_projection(attribute_embedding)
        attribute_embedding = attribute_embedding.view(b, reference_attribute_num, 1, -1)
        
        return attribute_embedding
    

class BalancedL1Loss(nn.Module):
    def __init__(self, thres=1.0, norm=False,):
        super().__init__()
        self.thres = thres
        self.norm = norm
    def forward(self, attribute_token_attn_prob, attribute_seg_maps,):
        if self.norm:
            attribute_token_attn_prob = attribute_token_attn_prob / (attribute_token_attn_prob.max(dim=2, keepdim=True)[0] + 1e-5,)
        
        background_seg_maps = 1 - attribute_seg_maps
        background_seg_maps_sum = background_seg_maps.sum(dim=2) + 1e-5
        attribute_seg_maps_sum = attribute_seg_maps.sum(dim=2) + 1e-5
        background_loss = (attribute_token_attn_prob * background_seg_maps).sum(dim=2,) / background_seg_maps_sum
        attribute_loss = (attribute_token_attn_prob * attribute_seg_maps).sum(dim=2,) / attribute_seg_maps_sum
        
        return background_loss - attribute_loss
    

def scatter_attribute_embeddings(inputs_embeddings,img_token_mask,attribute_embedding,reference_attribute_num,image_embedding_transform=None,):
    attribute_embedding = attribute_embedding.to(inputs_embeddings.dtype)
    
    batch_size, MAX_REFERENCE_ATTRIBUTE_NUM = attribute_embedding.shape[:2]
    seq_length = inputs_embeddings.shape[1]
    
    flat_attribute_embeddings = attribute_embedding.view(-1, attribute_embedding.shape[-2], attribute_embedding.shape[-1])
    valid_attribute_mask = (torch.arange(MAX_REFERENCE_ATTRIBUTE_NUM, device=flat_attribute_embeddings.device)[None, :] < reference_attribute_num[:, None],)
    valid_attribute_embeddings = flat_attribute_embeddings[valid_attribute_mask.flatten()]
    
    if image_embedding_transform is not None:
        valid_attribute_embeddings = image_embedding_transform(valid_attribute_embeddings)
    
    inputs_embeddings = inputs_embeddings.view(-1, inputs_embeddings.shape[-1])
    img_token_mask = img_token_mask.view(-1)
    valid_attribute_embeddings = valid_attribute_embeddings.view(-1, valid_attribute_embeddings.shape[-1])
    
    inputs_embeddings.masked_scatter_(img_token_mask[:, None], valid_attribute_embeddings)
    inputs_embeddings = inputs_embeddings.view(batch_size, seq_length, -1)
    
    return inputs_embeddings


def fuse_attribute_embeddings(inputs_embeddings,img_token_mask,attribute_embedding,reference_attribute_num,fuse_fn=torch.add,):
    attribute_embedding = attribute_embedding.to(inputs_embeddings.dtype)
    batch_size, MAX_REFERENCE_ATTRIBUTE_NUM = attribute_embedding.shape[:2]
    seq_length = inputs_embeddings.shape[1]
    
    flat_attribute_embeddings = attribute_embedding.view(-1, attribute_embedding.shape[-2], attribute_embedding.shape[-1])
    
    valid_attribute_mask = (torch.arange(MAX_REFERENCE_ATTRIBUTE_NUM, device=flat_attribute_embeddings.device)[None, :] < reference_attribute_num[:, None],)
    valid_attribute_embeddings = flat_attribute_embeddings[valid_attribute_mask.flatten()]
    
    inputs_embeddings = inputs_embeddings.view(-1, inputs_embeddings.shape[-1])
    img_token_mask = img_token_mask.view(-1)
    valid_attribute_embeddings = valid_attribute_embeddings.view(-1, valid_attribute_embeddings.shape[-1])
    image_token_embeddings = inputs_embeddings[img_token_mask]
    valid_attribute_embeddings = fuse_fn(image_token_embeddings, valid_attribute_embeddings)
    
    inputs_embeddings.masked_scatter_(img_token_mask[:, None], valid_attribute_embeddings)
    inputs_embeddings = inputs_embeddings.view(batch_size, seq_length, -1)
    
    return inputs_embeddings


def get_reference_from_attation(unet,attention_scores,layers=5,):
    unet_layer_list = ["down_blocks.0","down_blocks.1","down_blocks.2","mid_block","up_blocks.1","up_blocks.2","up_blocks.3",]
    start_layer = (len(unet_layer_list) - layers) // 2
    end_layer = start_layer + layers
    applicable_layers = unet_layer_list[start_layer:end_layer]
    
    def update_attention(name,):
        def get_attention(module, query, key, attention_mask=None,):
            attention_probs = module.old_get_attention_scores(query, key, attention_mask,)
            attention_scores[name] = attention_probs
            return attention_probs
        return get_attention
    
    for name, module in unet.named_modules():
        if isinstance(module, Attention) and "attn2" in name:
            if not any(layer in name for layer in applicable_layers):
                continue
            if isinstance(module.processor, AttnProcessor2_0):
                module.set_processor(AttnProcessor(),)
            module.old_get_attention_scores = module.get_attention_scores
            module.get_attention_scores = types.MethodType(update_attention(name), module,)
    
    
    return unet


    
def get_attribute_localization_loss_for_one_layer(reference_attention,attribute_seg_maps,attribute_token_idx,attribute_token_idx_mask,loss_fn,):
    bxh, num_noise_latents, num_text_tokens = reference_attention.shape
    b, MAX_REFERENCE_ATTRIBUTE_NUM = attribute_seg_maps.shape[:2]
    
    size = int(num_noise_latents**0.5)
    
    attribute_seg_maps = F.interpolate(attribute_seg_maps, size=(size, size), mode="bilinear", antialias=True,)
    attribute_seg_maps = attribute_seg_maps.view(b, MAX_REFERENCE_ATTRIBUTE_NUM, -1,)
    num_heads = bxh // b
    
    reference_attention = reference_attention.view(b, num_heads, num_noise_latents, num_text_tokens,)
    attribute_token_attn_prob = torch.gather(reference_attention,dim=3,index=attribute_token_idx.view(b, 1, 1, MAX_REFERENCE_ATTRIBUTE_NUM).expand(b, num_heads, num_noise_latents, MAX_REFERENCE_ATTRIBUTE_NUM,),)
    attribute_seg_maps = (attribute_seg_maps.permute(0, 2, 1).unsqueeze(1).expand(b, num_heads, num_noise_latents, MAX_REFERENCE_ATTRIBUTE_NUM),)
    loss = loss_fn(attribute_token_attn_prob, attribute_seg_maps,)
    loss = loss * attribute_token_idx_mask.view(b, 1, MAX_REFERENCE_ATTRIBUTE_NUM,)
    attribute_token_cnt = attribute_token_idx_mask.sum(dim=1).view(b, 1) + 1e-5
    loss = (loss.sum(dim=2) / attribute_token_cnt).mean()
    
    return loss


def get_attribute_localization_loss(reference_attention,attribute_seg_maps,img_token_idx,img_token_idx_mask,loss_fn,):
    num_layers = len(reference_attention)
    loss = 0
    for k, v in reference_attention.items():
        layer_loss = get_attribute_localization_loss_for_one_layer(v, attribute_seg_maps, img_token_idx, img_token_idx_mask, loss_fn,)
        loss += layer_loss
    
    return loss / num_layers


@torch.no_grad()
def inference_for_diverse_person(self,description: Union[str, List[str]] = None,height: Optional[int] = None,width: Optional[int] = None,num_inference_steps: int = 50,guidance_scale: float = 7.5,negative_description: Optional[Union[str, List[str]]] = None,num_images_per_description: Optional[int] = 1,eta: float = 0.0,generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,latents: Optional[torch.FloatTensor] = None,description_embeddings: Optional[torch.FloatTensor] = None,description_embeddings_text_only: Optional[torch.FloatTensor] = None,negative_description_embeddings: Optional[torch.FloatTensor] = None,output_type: Optional[str] = "pil",loss_dict: bool = True,callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,callback_steps: int = 1,cross_attention_kwargs: Optional[Dict[str, Any]] = None,start_merge_step=0,):
    height = height or self.unet.config.sample_size * self.vae_scale_factor
    width = width or self.unet.config.sample_size * self.vae_scale_factor
    self.check_inputs(description,height,width,callback_steps,negative_description,description_embeddings,negative_description_embeddings,)
    
    if description is not None and isinstance(description, str):
        batch_size = 1
    elif description is not None and isinstance(description, list):
        batch_size = len(description)
    else:
        batch_size = description_embeddings.shape[0]
    device = self._execution_device
    do_classifier_free_guidance = guidance_scale > 1.0
    
    assert do_classifier_free_guidance
    
    description_embeddings = self._encode_description(description,device,num_images_per_description,do_classifier_free_guidance,negative_description,description_embeddings=description_embeddings,negative_description_embeddings=negative_description_embeddings,)
    description_embeddings = torch.cat([description_embeddings, description_embeddings_text_only], dim=0,)
    
    self.scheduler.set_timesteps(num_inference_steps, device=device,)
    time_steps = self.scheduler.time_steps
    num_channels_latents = self.unet.in_channels
    
    latents = self.prepare_latents(batch_size * num_images_per_description,num_channels_latents,height,width,description_embeddings,generator=generator,latents=latents,device=device,eta=eta,)
    
    
    
    img_tokens = self.init_img_tokens(batch_size * num_images_per_description, height, width, latents, device=device,)
    latents_and_img_tokens = torch.cat([latents, img_tokens], dim=-1,)
    
    texts = self.get_texts(description, negative_description, batch_size)
    texts = self.repeat_text(texts,num_images_per_description,)
    latent_tokens = self.unet.condition_tokens(latents_and_img_tokens)
    latent_tokens = self.unet.latent_positional_encoder(latent_tokens)
    
    
    token_attention_mask = self.get_attention_mask(latents_and_img_tokens,)
    
    out = self.unet(
        token_latent=latent_tokens,
        time_steps=time_steps,
        encoder_hidden_states=texts["hidden_states"] if self.unet.has_encoder else None,
        attention_mask=token_attention_mask,
        cross_attention_kwargs=cross_attention_kwargs,
    )
    
    output = StableDiffusionPipelineOutput(
        predition=out.sample if isinstance(out, torch.distributions.Distribution) else out,
        sequence_latents=latents,
        sequence_img_tokens=img_tokens,
        sequence_texts=texts,
    )
    
    return output


def prepare_latents(
    self,
    batch_size: int,
    num_channels_latents: int,
    height: int,
    width: int,
    text_embeddings: torch.FloatTensor,
    generator: Optional[torch.Generator] = None,
    latents: Optional[torch.FloatTensor] = None,
    device: Optional[torch.device] = None,
    eta: float = 0.0,
) -> torch.FloatTensor:
    batch_size = batch_size
    
    if latents is None:
        latents = self.vae.sample_prior(batch_size, num_channels_latents, height, width, generator=generator, device=device, eta=eta,)
    else:
        assert latents.shape[0] == batch_size
        
        
    return latents


def init_img_tokens(
    self,
    batch_size: int,
    height: int,
    width: int,
    latents: torch.FloatTensor,
    device: Optional[torch.device] = None,
) -> torch.FloatTensor:
    
    img_tokens = self.unet.init_img_tokens(batch_size, height, width, latents, device=device,)
    
    return img_tokens


def get_texts(self,description: Union[str, List[str]], negative_description: Union[str, List[str]], batch_size: int) -> Dict[str, torch.Tensor]:
    texts = self.unet.get_texts(description, negative_description, batch_size)
    
    return texts


def repeat_text(self,texts: Dict[str, torch.Tensor], num_repeats: int) -> Dict[str, torch.Tensor]:
    texts["input_ids"] = texts["input_ids"].repeat(1, num_repeats)
    texts["attention_mask"] = texts["attention_mask"].repeat(1, num_repeats)
    texts["img_token_mask"] = texts["img_token_mask"].repeat(1, num_repeats)
    texts["reference_attribute_num"] = texts["reference_attribute_num"].repeat(1, num_repeats)
    
    return texts


def get_attention_mask(
    self,
    sequence_latents: torch.FloatTensor,
) -> Optional[torch.Tensor]:
    
    return sequence_latents.new_ones(sequence_latents.shape[:2], dtype=torch.long) * (-1e5)


setattr(StableDiffusionPipeline, "inference_for_diverse_person", inference_for_diverse_person)
setattr(StableDiffusionPipeline, "prepare_latents", prepare_latents)
setattr(StableDiffusionPipeline, "init_img_tokens", init_img_tokens)
setattr(StableDiffusionPipeline, "get_texts", get_texts)
setattr(StableDiffusionPipeline, "repeat_text", repeat_text)
setattr(StableDiffusionPipeline, "get_attention_mask", get_attention_mask)