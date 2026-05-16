import argparse
import contextlib
import gc
import os
import pickle
import time
import warnings
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from qwen_vl_utils import process_vision_info
from sklearn.metrics import accuracy_score, f1_score
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

DEFAULT_MODEL_PATH = r"C:\Users\rog72\.cache\modelscope\hub\models\Qwen\Qwen2.5-VL-3B-Instruct"
DEFAULT_PROCESSED_DIR = r"C:\Users\rog72\kaggle\Review_model\processed_data"
DEFAULT_CHECKPOINT_DIR = r"C:\Users\rog72\kaggle\Review_model\checkpoints"

IMAGE_PROMPT = "\u8bf7\u5206\u6790\u8fd9\u5f20\u56fe\u7247\u7684\u5185\u5bb9\u7c7b\u578b\u3002"
TEXT_PROMPT_PREFIX = "\u8bf7\u5224\u65ad\u4ee5\u4e0b\u6587\u672c\u662f\u5426\u8fdd\u89c4\uff1a\n"
EMPTY_CONTENT = "\u65e0\u5185\u5bb9"
FALLBACK_TEXT = "\u5185\u5bb9\u5ba1\u6838"


@dataclass
class TrainConfig:
    model_path: str
    processed_dir: str
    checkpoint_dir: str
    epochs: int = 5
    grad_accum_steps: int = 8
    eval_every_steps: int = 200
    save_every_steps: int = 500
    max_grad_norm: float = 1.0
    max_length: int = 384
    num_classes: int = 5
    hidden_size: int = 2048
    lora_lr: float = 2e-5
    head_lr: float = 1e-3
    weight_decay: float = 0.01
    eval_max_batches: int = 200


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train the Qwen2.5-VL moderation classifier.")
    parser.add_argument("--model-path", default=os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH))
    parser.add_argument("--processed-dir", default=os.environ.get("PROCESSED_DIR", DEFAULT_PROCESSED_DIR))
    parser.add_argument("--checkpoint-dir", default=os.environ.get("CHECKPOINT_DIR", DEFAULT_CHECKPOINT_DIR))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--eval-every-steps", type=int, default=200)
    parser.add_argument("--save-every-steps", type=int, default=500)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--eval-max-batches", type=int, default=200)
    args = parser.parse_args()
    return TrainConfig(
        model_path=args.model_path,
        processed_dir=args.processed_dir,
        checkpoint_dir=args.checkpoint_dir,
        epochs=args.epochs,
        grad_accum_steps=args.grad_accum_steps,
        eval_every_steps=args.eval_every_steps,
        save_every_steps=args.save_every_steps,
        max_grad_norm=args.max_grad_norm,
        max_length=args.max_length,
        eval_max_batches=args.eval_max_batches,
    )


def normalize_path(path: str) -> str:
    path = os.path.expandvars(os.path.expanduser(path))
    if os.name != "nt" and len(path) >= 3 and path[1] == ":" and path[2] in ("\\", "/"):
        drive = path[0].lower()
        rest = path[3:].replace("\\", "/")
        return f"/mnt/{drive}/{rest}"
    return path


def model_device(backbone) -> torch.device:
    device = getattr(backbone, "device", None)
    if device is not None:
        return device
    return next(backbone.parameters()).device


def autocast_context():
    if torch.cuda.is_available():
        return autocast("cuda", dtype=torch.float16)
    return contextlib.nullcontext()


class Qwen2VLClassifier(nn.Module):
    def __init__(self, backbone, num_classes: int = 5, hidden_size: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.backbone = backbone
        self.hidden_size = hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 512), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512, 128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        ).float().to(model_device(backbone))
        self._init_classifier()

    def _init_classifier(self) -> None:
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, input_ids, attention_mask, pixel_values=None, image_grid_thw=None, labels=None, **kwargs):
        with autocast_context():
            outputs = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                output_hidden_states=True,
                return_dict=True,
            )

        hidden = outputs.hidden_states[-1]
        batch_size = hidden.size(0)
        seq_lengths = attention_mask.sum(dim=1).sub(1).clamp(min=0)
        pooled = hidden[torch.arange(batch_size, device=hidden.device), seq_lengths]
        logits = self.classifier(pooled.float())
        loss = F.cross_entropy(logits, labels) if labels is not None else None
        return loss, logits


class ModerationDataset(Dataset):
    def __init__(self, samples, processor, max_length: int = 384):
        self.samples = samples
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_path = sample.get("image")
        text = sample.get("text") or ""
        label = sample["label"]
        has_image = bool(image_path and os.path.exists(image_path))

        content = []
        prompt = IMAGE_PROMPT if has_image else ""
        if has_image:
            content.append({"type": "image", "image": f"file://{image_path}"})
        if text:
            prompt += f"{TEXT_PROMPT_PREFIX}{text[:200]}"
        content.append({"type": "text", "text": prompt or EMPTY_CONTENT})

        messages = [{"role": "user", "content": content}]
        try:
            text_input = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            if has_image:
                image_inputs, _ = process_vision_info(messages)
                inputs = self.processor(
                    text=[text_input],
                    images=image_inputs,
                    padding="max_length",
                    max_length=self.max_length,
                    truncation=True,
                    return_tensors="pt",
                )
            else:
                inputs = self.processor(
                    text=[text_input],
                    padding="max_length",
                    max_length=self.max_length,
                    truncation=True,
                    return_tensors="pt",
                )
        except Exception:
            inputs = self.processor(
                text=[FALLBACK_TEXT],
                padding="max_length",
                max_length=self.max_length,
                truncation=True,
                return_tensors="pt",
            )

        result = {key: value.squeeze(0) for key, value in inputs.items()}
        result["labels"] = torch.tensor(label, dtype=torch.long)
        return result


def build_collate_fn(pad_token_id: int):
    def collate_fn(batch):
        max_len = max(item["input_ids"].size(0) for item in batch)
        input_ids_list = []
        attention_mask_list = []

        for item in batch:
            input_ids = item["input_ids"]
            attention_mask = item["attention_mask"]
            pad_len = max_len - input_ids.size(0)
            if pad_len > 0:
                input_ids = F.pad(input_ids, (0, pad_len), value=pad_token_id)
                attention_mask = F.pad(attention_mask, (0, pad_len), value=0)
            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)

        result = {
            "input_ids": torch.stack(input_ids_list),
            "attention_mask": torch.stack(attention_mask_list),
            "labels": torch.stack([item["labels"] for item in batch]),
        }

        image_items = [item for item in batch if "pixel_values" in item]
        if image_items:
            result["pixel_values"] = torch.cat([item["pixel_values"] for item in image_items], dim=0)
            grids = []
            for item in image_items:
                grid = item["image_grid_thw"]
                grids.append(grid.unsqueeze(0) if grid.dim() == 1 else grid)
            result["image_grid_thw"] = torch.cat(grids, dim=0)

        return result

    return collate_fn


def move_batch(batch, device):
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if hasattr(value, "to") else value
    return moved


def load_samples(processed_dir: str):
    with open(os.path.join(processed_dir, "train_samples.pkl"), "rb") as handle:
        train_samples = pickle.load(handle)
    with open(os.path.join(processed_dir, "val_samples.pkl"), "rb") as handle:
        val_samples = pickle.load(handle)
    return train_samples, val_samples


@torch.no_grad()
def evaluate(model, val_loader, device, max_batches: int):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    count = 0

    for batch_idx, batch in enumerate(val_loader):
        if max_batches and batch_idx >= max_batches:
            break

        batch = move_batch(batch, device)
        loss, logits = model(**batch)
        total_loss += float(loss.item())
        all_preds.extend(logits.argmax(dim=-1).detach().cpu().numpy())
        all_labels.extend(batch["labels"].detach().cpu().numpy())
        count += 1

    if not all_labels:
        model.train()
        return 0.0, 0.0, 0.0

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    model.train()
    return total_loss / max(count, 1), acc, f1


def save_checkpoint(model, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    model.backbone.save_pretrained(output_dir)
    torch.save(model.classifier.state_dict(), os.path.join(output_dir, "classifier_head.pt"))


def build_model(config: TrainConfig):
    processor = AutoProcessor.from_pretrained(
        config.model_path,
        min_pixels=4 * 28 * 28,
        max_pixels=128 * 28 * 28,
    )
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        config.model_path,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    )
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="FEATURE_EXTRACTION",
    )
    peft_model = get_peft_model(base_model, lora_config)
    model = Qwen2VLClassifier(peft_model, num_classes=config.num_classes, hidden_size=config.hidden_size)
    return model, processor


def train(config: TrainConfig) -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    config.model_path = normalize_path(config.model_path)
    config.processed_dir = normalize_path(config.processed_dir)
    config.checkpoint_dir = normalize_path(config.checkpoint_dir)
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    print("loading model and processor")
    model, processor = build_model(config)
    device = model_device(model.backbone)

    print("loading processed samples")
    train_samples, val_samples = load_samples(config.processed_dir)
    pad_token_id = getattr(processor.tokenizer, "pad_token_id", None) or 0
    collate_fn = build_collate_fn(pad_token_id)

    train_dataset = ModerationDataset(train_samples, processor, max_length=config.max_length)
    val_dataset = ModerationDataset(val_samples, processor, max_length=config.max_length)
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    param_groups = [
        {"params": param, "lr": config.lora_lr, "name": name}
        for name, param in model.backbone.named_parameters()
        if param.requires_grad
    ]
    param_groups.append({"params": model.classifier.parameters(), "lr": config.head_lr, "name": "classifier"})

    optimizer = AdamW(param_groups, weight_decay=config.weight_decay)
    total_steps = max(1, (len(train_loader) * config.epochs + config.grad_accum_steps - 1) // config.grad_accum_steps)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)
    try:
        scaler = GradScaler("cuda", enabled=torch.cuda.is_available())
    except TypeError:
        scaler = GradScaler(enabled=torch.cuda.is_available())

    print(f"train samples={len(train_dataset)}, val samples={len(val_dataset)}, steps={total_steps}")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    best_f1 = 0.0
    global_step = 0

    for epoch in range(config.epochs):
        epoch_start = time.time()
        accum_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.epochs}")

        for batch_idx, batch in enumerate(progress):
            try:
                batch = move_batch(batch, device)
                loss, _ = model(**batch)
                loss = loss / config.grad_accum_steps
                scaler.scale(loss).backward()
                accum_loss += float(loss.item())

                should_step = (batch_idx + 1) % config.grad_accum_steps == 0 or (batch_idx + 1) == len(train_loader)
                if not should_step:
                    continue

                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], config.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

                global_step += 1
                progress.set_postfix({"loss": f"{accum_loss:.4f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"})
                accum_loss = 0.0

                if config.eval_every_steps and global_step % config.eval_every_steps == 0:
                    val_loss, val_acc, val_f1 = evaluate(model, val_loader, device, config.eval_max_batches)
                    print(f"step={global_step} val_loss={val_loss:.4f} acc={val_acc:.4f} f1={val_f1:.4f}")
                    if val_f1 > best_f1:
                        best_f1 = val_f1
                        save_checkpoint(model, os.path.join(config.checkpoint_dir, "best_model"))
                        print(f"saved best checkpoint f1={best_f1:.4f}")

                if config.save_every_steps and global_step % config.save_every_steps == 0:
                    save_checkpoint(model, os.path.join(config.checkpoint_dir, f"step_{global_step}"))

            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    print(f"out of memory at batch={batch_idx}; batch skipped")
                    optimizer.zero_grad(set_to_none=True)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                    continue
                raise

        val_loss, val_acc, val_f1 = evaluate(model, val_loader, device, max_batches=len(val_loader))
        elapsed = (time.time() - epoch_start) / 60
        print(f"epoch={epoch + 1} minutes={elapsed:.1f} val_loss={val_loss:.4f} acc={val_acc:.4f} f1={val_f1:.4f}")
        if val_f1 > best_f1:
            best_f1 = val_f1
            save_checkpoint(model, os.path.join(config.checkpoint_dir, "best_model"))
            print(f"saved best checkpoint f1={best_f1:.4f}")

    print(f"training complete best_f1={best_f1:.4f}")


if __name__ == "__main__":
    train(parse_args())
