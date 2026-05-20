import os.path as osp
from collections import OrderedDict
import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

import os
import torchattacks
from autoattack import AutoAttack
from trades import trades_loss
from tqdm import tqdm
from torch.utils.data import TensorDataset
from torchvision import transforms

_tokenizer = _Tokenizer()

MOE_STATE_NAMES = ("expert_prompts", "gate", "tau", "alpha", "base_prompt")


def is_moe_state_parameter(name):
    return any(key in name for key in MOE_STATE_NAMES)


def is_trainable_moe_parameter(name, train_tau=True, train_base_prompt=True):
    if "tau" in name and not train_tau:
        return False
    if "base_prompt" in name and not train_base_prompt:
        return False
    return is_moe_state_parameter(name)


def unwrap_model(model):
    return model.module if isinstance(model, nn.DataParallel) else model


def is_moe_gate_parameter(name):
    return "gate.weight" in name or "gate.bias" in name


def is_moe_scale_parameter(name):
    return ("alpha" in name) or ("tau" in name)


def build_moe_param_groups(model, base_lr, weight_decay, lr_mult_gate, lr_mult_scale):
    prompt_params = []
    gate_params = []
    scale_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if is_moe_scale_parameter(name):
            scale_params.append(param)
        elif is_moe_gate_parameter(name):
            gate_params.append(param)
        else:
            prompt_params.append(param)

    param_groups = []
    if prompt_params:
        param_groups.append({"params": prompt_params, "lr": base_lr, "weight_decay": weight_decay})
    if gate_params:
        param_groups.append({"params": gate_params, "lr": base_lr * lr_mult_gate, "weight_decay": weight_decay})
    if scale_params:
        param_groups.append({"params": scale_params, "lr": base_lr * lr_mult_scale, "weight_decay": 0.0})

    return param_groups


def load_clip_to_cpu(cfg):
    moe_cfg = cfg.TRAINER.MoEAdvIVLP
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    design_details = {
        "trainer": "IVLP_MoE",
        "vision_depth": moe_cfg.PROMPT_DEPTH_VISION,
        "language_depth": moe_cfg.PROMPT_DEPTH_TEXT,
        "vision_ctx": moe_cfg.N_CTX_VISION,
        "language_ctx": moe_cfg.N_CTX_TEXT,
        "num_experts": moe_cfg.NUM_EXPERTS,
        "delta_scale_init": moe_cfg.DELTA_SCALE_INIT,
        "gate_mode": moe_cfg.GATE_MODE,
        "gate_hybrid_lambda": moe_cfg.GATE_HYBRID_LAMBDA,
        "alpha_min": moe_cfg.ALPHA_MIN,
        "alpha_max": moe_cfg.ALPHA_MAX,
        "tau_min": moe_cfg.TAU_MIN,
        "tau_max": moe_cfg.TAU_MAX,
    }
    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


class VLPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        moe_cfg = cfg.TRAINER.MoEAdvIVLP
        n_cls = len(classnames)
        assert moe_cfg.PROMPT_DEPTH_TEXT >= 1, "In Independent VL prompting, Language prompt depth should be >=1\nPlease use VPT trainer if you want to learn only vision branch"
        n_ctx = moe_cfg.N_CTX_TEXT
        ctx_init = moe_cfg.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init and n_ctx <= 4:
            ctx_init = ctx_init.replace("_", " ")
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1:1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
        print("Adversarial Independent V-L design")
        print(f'Initial text context: "{prompt_prefix}"')
        print(f"Number of context words (tokens) for Language prompting: {n_ctx}")
        print(f"Number of context words (tokens) for Vision prompting: {moe_cfg.N_CTX_VISION}")
        self.ctx = nn.Parameter(ctx_vectors)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts
        self.name_lens = name_lens

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat([
            prefix,
            ctx,
            suffix,
        ], dim=1)
        return prompts

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = self.construct_prompts(ctx, prefix, suffix)
        return prompts


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.moe_cfg = cfg.TRAINER.MoEAdvIVLP
        self.prompt_learner = VLPromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.moe_aux_loss_scale = 1.0
        self.normalize = transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        )

    def compute_moe_aux_components(self):
        total_balance = self.logit_scale.new_tensor(0.0)
        total_diversity = self.logit_scale.new_tensor(0.0)
        for transformer in (self.image_encoder.transformer, self.text_encoder.transformer):
            if hasattr(transformer, "moe_aux_losses"):
                balance, diversity = transformer.moe_aux_losses()
                total_balance = total_balance + balance
                total_diversity = total_diversity + diversity
        return total_balance, total_diversity

    def compute_moe_aux_loss(self):
        total_balance, total_diversity = self.compute_moe_aux_components()
        balance_w = getattr(self.moe_cfg, "AUX_BALANCE_W_TARGET", self.moe_cfg.AUX_BALANCE_W)
        diversity_w = getattr(self.moe_cfg, "AUX_DIVERSITY_W_TARGET", self.moe_cfg.AUX_DIVERSITY_W)
        return self.moe_aux_loss_scale * (balance_w * total_balance + diversity_w * total_diversity)

    def set_moe_aux_loss_scale(self, scale):
        self.moe_aux_loss_scale = float(scale)

    def compute_moe_diagnostics(self):
        zero = self.logit_scale.new_tensor(0.0).float()
        alpha_values = []
        tau_values = []
        gate_entropies = []

        for module in self.modules():
            if hasattr(module, "alpha") and hasattr(module, "_bounded_scale"):
                alpha = module._bounded_scale(
                    module.alpha,
                    getattr(module, "alpha_min", None),
                    getattr(module, "alpha_max", None),
                )
                alpha_values.append(alpha.detach().float().mean())

            if hasattr(module, "tau") and hasattr(module, "_bounded_scale"):
                tau = module._bounded_scale(
                    module.tau,
                    getattr(module, "tau_min", None),
                    getattr(module, "tau_max", None),
                )
                tau_values.append(tau.detach().float().mean())

            gate_weights = getattr(module, "_last_gate_weights", None)
            if gate_weights is None or gate_weights.numel() == 0:
                continue

            probs = gate_weights.detach().float().clamp_min(1e-12)
            entropy = -(probs * probs.log()).sum(dim=-1)
            if probs.shape[-1] > 1:
                entropy = entropy / math.log(probs.shape[-1])
            gate_entropies.append(entropy.mean())

        total_balance, total_diversity = self.compute_moe_aux_components()

        def mean_or_zero(values):
            return torch.stack(values).mean() if values else zero

        return {
            "alpha_mean": mean_or_zero(alpha_values),
            "tau_mean": mean_or_zero(tau_values),
            "gate_entropy": mean_or_zero(gate_entropies),
            "aux_balance": total_balance.detach().float(),
            "aux_diversity": total_diversity.detach().float(),
        }

    def forward(self, image, label=None):
        image = self.normalize(image)

        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        prompts = self.prompt_learner()
        text_features = self.text_encoder(prompts, tokenized_prompts)
        if image.shape[0] == 0:
            return text_features.new_empty((0, text_features.shape[0]))
        image_features = self.image_encoder(image.type(self.dtype))

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logits = logit_scale * image_features @ text_features.t()

        if self.prompt_learner.training:
            return F.cross_entropy(logits, label) + self.compute_moe_aux_loss()

        return logits


@TRAINER_REGISTRY.register()
class MoEAdvIVLP(TrainerX):
    def before_adv_test(self, attack='pgd', eps=1/255, alpha=1/255, steps=100):
        r"""
        Arguments:
            eps (float): maximum perturbation. (Default: 4/255)
            alpha (float): step size. (Default: 1/255)
            steps (int): number of steps. (Default: 10)
            random_start (bool): using random initialization of delta. (Default: True)
        """
        dataset_save_path = os.path.join(self.cfg.OUTPUT_DIR, self.cfg.DATASET.NAME + "_adv_dataset.pkl")
        print(attack)

        if os.path.isfile(dataset_save_path):
            self.adv_test_pkl, _ = torch.load(dataset_save_path).tensors
            return

        if attack == 'auto':
            attacker = AutoAttack(self.model, norm='Linf', eps=eps, version='standard')
        elif attack == 'pgd':
            attacker = torchattacks.PGD(self.model, eps=eps, alpha=alpha, steps=steps, random_start=True)
        elif attack == 'di':
            attacker = torchattacks.DIFGSM(self.model, eps=eps, alpha=alpha, steps=steps)
        elif attack == 'ti':
            attacker = torchattacks.TIFGSM(self.model, eps=eps, alpha=alpha, steps=steps)
        elif attack == 'cw':
            attacker = torchattacks.CW(self.model)
        else:
            raise ValueError(f"Unknown attack: {attack}")

        self.adv_test_pkl = torch.empty(size=[len(self.test_loader.dataset), 3, 224, 224])
        all_labels = torch.empty(size=[len(self.test_loader.dataset)])

        for batch_idx, batch in enumerate(tqdm(self.test_loader)):
            input, label = self.parse_batch_test(batch)
            if attack == 'auto':
                adv_input = attacker.run_standard_evaluation(input, label)
            else:
                adv_input = attacker(input, label)
            with torch.no_grad():
                start_idx = batch_idx * self.test_loader.batch_size
                end_idx = start_idx + input.size(0)
                self.adv_test_pkl[start_idx:end_idx] = adv_input.detach().cpu()
                all_labels[start_idx:end_idx] = label.detach().cpu()

        adv_test_dataset = TensorDataset(self.adv_test_pkl, all_labels)
        torch.save(adv_test_dataset, dataset_save_path)
        print(f"Saving to: {dataset_save_path}")

    @torch.no_grad()
    def test_adv(self, split=None):
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"
            data_loader = self.test_loader

        array_to_pkl = self.adv_test_pkl
        print(f"Evaluate on the *{split}* set")

        for batch_idx, batch in enumerate(tqdm(data_loader)):
            _, label = self.parse_batch_test(batch)
            adv_input = array_to_pkl[batch_idx * data_loader.batch_size:(batch_idx + 1) * data_loader.batch_size]
            adv_output = self.model_inference(adv_input.to(label.device))
            self.evaluator.process(adv_output, label)

        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0]

    def check_cfg(self, cfg):
        assert cfg.TRAINER.MoEAdvIVLP.PREC in ["fp16", "fp32", "amp"]

    def get_moe_aux_loss_scale(self):
        moe_cfg = self.cfg.TRAINER.MoEAdvIVLP
        warmup_epochs = max(int(getattr(moe_cfg, "AUX_WARMUP_EPOCHS", 1)), 1)
        current_epoch = getattr(self, "epoch", 0) + 1
        return min(1.0, current_epoch / warmup_epochs)

    def build_moe_optimizer(self, base_lr):
        moe_cfg = self.cfg.TRAINER.MoEAdvIVLP
        param_groups = build_moe_param_groups(
            self.model,
            base_lr=base_lr,
            weight_decay=self.cfg.OPTIM.WEIGHT_DECAY,
            lr_mult_gate=moe_cfg.LR_MULT_GATE,
            lr_mult_scale=moe_cfg.LR_MULT_SCALE,
        )
        return build_optimizer(self.model, self.cfg.OPTIM, param_groups=param_groups)

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.MoEAdvIVLP.PREC in ["fp32", "amp"]:
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        train_tau = cfg.TRAINER.MoEAdvIVLP.TRAIN_TAU
        train_base_prompt = cfg.TRAINER.MoEAdvIVLP.TRAIN_BASE_PROMPT
        for name, param in self.model.named_parameters():
            if "prompt_learner" in name or "VPT" in name or is_trainable_moe_parameter(name, train_tau, train_base_prompt):
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)

        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        self.optim = self.build_moe_optimizer(cfg.OPTIM.LR)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("VLPromptLearner", self.model, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.MoEAdvIVLP.PREC == "amp" else None

        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        model = self.model
        optim = self.optim
        scaler = self.scaler
        unwrap_model(model).set_moe_aux_loss_scale(self.get_moe_aux_loss_scale())

        eps = self.cfg.AT.TRAIN.EPS / 255.0
        alpha = self.cfg.AT.TRAIN.ALPHA / 255.0
        steps = self.cfg.AT.TRAIN.STEPS
        loss_type = self.cfg.AT.TRAIN.AT_LOSS_TYPE

        attacker = torchattacks.PGD(self.model, eps=eps, alpha=alpha, steps=steps, random_start=True)

        N = image.size(0)
        if loss_type == "clean":
            com_images = image
        elif loss_type == "adv_full":
            adv_image = attacker(image, label)
            com_images = adv_image
        elif loss_type == "adv_half":
            adv_image = attacker(image[N//2:], label[N//2:])
            com_images = torch.cat([image[:N//2], adv_image], dim=0)
        else:
            raise ValueError(f"Invalid loss type: {loss_type}")

        prec = self.cfg.TRAINER.MoEAdvIVLP.PREC
        if prec == "amp":
            with autocast():
                loss = model(com_images, label)
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            loss = model(com_images, label)
            optim.zero_grad()
            loss.backward()
            optim.step()

        diagnostics = unwrap_model(model).compute_moe_diagnostics()
        loss_summary = {"loss": loss.item()}
        loss_summary.update({name: value.item() for name, value in diagnostics.items()})

        if (self.batch_idx + 1) == self.num_batches:
            for name, value in diagnostics.items():
                self.write_scalar(f"train/{name}", value.item(), self.epoch)
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            if "prompt_learner.token_prefix" in state_dict:
                del state_dict["prompt_learner.token_prefix"]

            if "prompt_learner.token_suffix" in state_dict:
                del state_dict["prompt_learner.token_suffix"]

            print("Loading weights to {} ".format(name) + 'from "{}" (epoch = {})'.format(model_path, epoch))
            self._models[name].load_state_dict(state_dict, strict=False)
