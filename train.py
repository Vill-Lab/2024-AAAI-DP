import torch
import torch.utils.checkpoint
import math
import os
import shutil
from accelerate import Accelerator
from accelerate.utils import set_seed
from torchvision import transforms as T
from diffusers import DDPMScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils.import_utils import is_xformers_available
from collections import OrderedDict
from tqdm.auto import tqdm
from transformers import CLIPTokenizer
from pathlib import Path
from model.utils import parse_args
from model.diffusers_model import DiversePersonModel
from model.data_processor import Transform4TrainWithSegmap, Processor4SegmentRandom
from model.data_processor import DiversePersonDataset

def collate_fn(examples):
    pix_v = torch.stack([example["pix_v"] for example in examples])
    input_ids = torch.cat([example["input_ids"] for example in examples])
    img_ids = torch.stack([example["img_ids"] for example in examples])
    img_token_mask = torch.cat([example["img_token_mask"] for example in examples])
    img_token_idx = torch.cat([example["img_token_idx"] for example in examples])
    img_token_idx_mask = torch.cat([example["img_token_idx_mask"] for example in examples])
    attribute_v = torch.stack([example["attribute_v"] for example in examples])
    attribute_seg_maps = torch.stack([example["attribute_seg_maps"] for example in examples])
    reference_attribute_num = torch.stack([example["reference_attribute_num"] for example in examples])
    return {"pix_v": pix_v, "input_ids": input_ids, "img_token_mask": img_token_mask, "img_token_idx": img_token_idx, "img_token_idx_mask": img_token_idx_mask, "attribute_v": attribute_v, "attribute_seg_maps": attribute_seg_maps, "reference_attribute_num": reference_attribute_num, "img_ids": img_ids}

if __name__ == "__main__":
    args = parse_args()
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, mixed_precision=args.mixed_precision, log_with=args.report_to, logging_dir=args.logging_dir)

    if accelerator.is_main_process:
        for dir_path in [args.output_dir, args.logging_dir]:
            if dir_path is not None:
                os.makedirs(dir_path, exist_ok=True)
    accelerator.wait_for_everyone()

    if args.seed is not None:
        set_seed(args.seed)

    noise_sched = DDPMScheduler.from_pretrained(args.pretrained_model_hf, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_hf, subfolder="tokenizer", revision=args.revision)

    model = DiversePersonModel.load_pretrained_model_hf(args)
    weight_dtype = torch.float16 if accelerator.mixed_precision == "fp16" else torch.bfloat16 if accelerator.mixed_precision == "bf16" else torch.float32

    for param in model.parameters():
        param.requires_grad = False
        param.data = param.data.to(weight_dtype)

    model.unet.requires_grad_(True).to(torch.float32)

    if args.text_image_linking == "control" and not args.activate_controller:
        model.controller.requires_grad_(True).to(torch.float32)

    if args.train_text_encoder:
        model.text_encoder.requires_grad_(True).to(torch.float32)

    if args.train_image_encoder:
        if args.image_encoder_trainable_layers > 0:
            for idx in range(args.image_encoder_trainable_layers):
                model.image_encoder.vision_model.encoder.layers[-1 - idx].requires_grad_(True).to(torch.float32)
        else:
            model.image_encoder.requires_grad_(True).to(torch.float32)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            model.unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if args.gradient_checkpointing and args.train_text_encoder:
        model.text_encoder.gradient_checkpointing_enable()

    if args.scale_lr:
        args.lr *= args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes

    optimizer_cls = torch.optim.AdamW
    unet_params = [p for p in model.unet.parameters() if p.requires_grad]
    other_params = [p for n, p in model.named_parameters() if p.requires_grad and "unet" not in n]
    parameters = unet_params + other_params

    optimizer = optimizer_cls([
        {"params": unet_params, "lr": args.lr * args.unet_lr_scale},
        {"params": other_params, "lr": args.lr},
    ], betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay, eps=args.adam_epsilon)

    train_transforms = Transform4TrainWithSegmap(args)
    attribute_transforms = torch.nn.Sequential(OrderedDict([
        ("resize", T.Resize((args.resolution, args.resolution), interpolation=T.InterpolationMode.BILINEAR, antialias=True)),
        ("convert_to_float", T.ConvertImageDtype(torch.float32))
    ]))
    attribute_processor = Processor4SegmentRandom()
    reference_attribute_types = args.reference_attribute_types.split("_") if args.reference_attribute_types and args.reference_attribute_types != "all" else None

    train_dataset = DiversePersonDataset(args.dataset_name, tokenizer, train_transforms, attribute_transforms, attribute_processor, device=accelerator.device, MAX_REFERENCE_ATTRIBUTE_NUM=args.MAX_REFERENCE_ATTRIBUTE_NUM, img_token_num=args.img_token_num, reference_attribute_types=reference_attribute_types, split="train")
    
    train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=True, collate_fn=collate_fn, batch_size=args.train_batch_size, num_workers=0)

    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(args.lr_scheduler, optimizer=optimizer, num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps, num_training_steps=args.max_train_steps * args.gradient_accumulation_steps)

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(model, optimizer, train_dataloader, lr_scheduler)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        accelerator.init_trackers("DiversePerson", config=vars(args))

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    global_step = 0
    first_epoch = 0

    if args.resume_from_checkpoint:
        path = os.path.basename(args.resume_from_checkpoint) if args.resume_from_checkpoint != "latest" else sorted([d for d in os.listdir(args.output_dir) if d.startswith("checkpoint")], key=lambda x: int(x.split("-")[1]))[-1] if os.path.exists(args.output_dir) else None
        if path is None:
            accelerator.print(f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run.")
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])
            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)
            model.to(accelerator.device)

    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)

    for epoch in range(first_epoch, args.num_train_epochs):
        model.train()
        train_loss = 0.0
        denoise_loss = 0.0
        localization_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            progress_bar.set_description(f"Global step: {global_step}")
            with accelerator.accumulate(model), torch.backends.cuda.sdp_kernel(enable_flash=not args.disable_flashattention):
                loss_dict = model(batch, noise_sched)
                loss = loss_dict["loss"]
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps
                avg_denoise_loss = accelerator.gather(loss_dict["denoise_loss"].repeat(args.train_batch_size)).mean()
                denoise_loss += avg_denoise_loss.item() / args.gradient_accumulation_steps
                if "localization_loss" in loss_dict:
                    avg_localization_loss = accelerator.gather(loss_dict["localization_loss"].repeat(args.train_batch_size)).mean()
                    localization_loss += avg_localization_loss.item() / args.gradient_accumulation_steps
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(parameters, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss, "denoise_loss": denoise_loss, "localization_loss": localization_loss}, step=global_step)
                train_loss = 0.0
                denoise_loss = 0.0
                localization_loss = 0.0
                if global_step % args.checkpointing_steps == 0 and accelerator.is_local_main_process:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    if args.keep_only_last_checkpoint:
                        for file in os.listdir(args.output_dir):
                            if file.startswith("checkpoint") and file != os.path.basename(save_path):
                                ckpt_num = int(file.split("-")[1])
                                if args.keep_interval is None or ckpt_num % args.keep_interval != 0:
                                    shutil.rmtree(os.path.join(args.output_dir, file))
            logs = {"l_noise": loss_dict["denoise_loss"].detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            if "localization_loss" in loss_dict:
                logs["l_loc"] = loss_dict["localization_loss"].detach().item()
            progress_bar.set_postfix(**logs)
            if global_step >= args.max_train_steps:
                break
            
    accelerator.wait_for_everyone()
    
    if accelerator.is_main_process:
        model = accelerator.unwrap_model(model)
        pipeline = model.convert_model()
        pipeline.save_pretrained(args.output_dir)
        
    accelerator.end_training()
