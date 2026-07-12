import gc
import os
import sys

import torch
from dotenv import load_dotenv
from huggingface_hub import HfApi, login, snapshot_download, upload_folder, list_repo_files
from peft import PeftModel
from transformers import EarlyStoppingCallback, TrainingArguments

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

sys.path.insert(0, os.path.dirname(__file__))

from config import (
    HUB_REPO, EPOCHS, BATCH_SIZE_TRAIN, BATCH_SIZE_EVAL,
    GRAD_ACCUM_STEPS, LEARNING_RATE, LR_SCHEDULER, WARMUP_RATIO,
    WEIGHT_DECAY, LABEL_SMOOTHING, EARLY_STOPPING_PAT, METRIC_FOR_BEST,
    OUTPUT_DIR, FINAL_MODEL_DIR,
    LORA_R, LORA_ALPHA, LORA_DROPOUT, HEAD_DROPOUT,
)
from dataset import (
    load_splits, HAMDataset, eval_collate_fn, build_sampler_and_weights
)
from trainer import HAMTrainer, SoftTargetCrossEntropy
from metrics import compute_metrics

hf_token = os.environ.get("HF_TOKEN", "")
if not hf_token:
    raise EnvironmentError("HF_TOKEN is not set. Add it to the .env file at the project root.")
login(hf_token)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
print(f"Device={DEVICE}")

train_raw, val_raw, test_raw = load_splits()
train_ds = HAMDataset(train_raw, augment=True)
val_ds   = HAMDataset(val_raw,   augment=False)
test_ds  = HAMDataset(test_raw,  augment=False)
print(f"train={len(train_ds):,}  val={len(val_ds):,}  test={len(test_ds):,}")

sampler, class_weights = build_sampler_and_weights(train_raw)
class_weights = class_weights.to(DEVICE)

loss_fn = SoftTargetCrossEntropy(weight=class_weights, label_smoothing=LABEL_SMOOTHING)

from model.architecture import build_model

gc.collect(); torch.cuda.empty_cache()
print(f"GPU free: {torch.cuda.mem_get_info()[0] / 1e9:.2f} GB")

HfApi().create_repo(HUB_REPO, token=hf_token, private=False, exist_ok=True)

resume_from = None
model       = None

try:
    files       = list(list_repo_files(HUB_REPO, token=hf_token))
    hub_folders = {f.split("/")[0] for f in files}

    resume_folder = (
        "last-checkpoint" if "last-checkpoint" in hub_folders
        else (
            sorted(
                {f for f in hub_folders if f.startswith("checkpoint-")},
                key=lambda p: int(p.split("-")[-1]),
            )[-1]
            if any(f.startswith("checkpoint-") for f in hub_folders)
            else None
        )
    )

    if resume_folder:
        print(f"Checkpoint found → {resume_folder}. Downloading...")
        _resume_dir = os.path.join(os.path.dirname(__file__), "..", "resume")
        local       = snapshot_download(HUB_REPO, token=hf_token,
                                        local_dir=_resume_dir)
        resume_from = os.path.join(local, resume_folder)

        model    = build_model(lora_r=LORA_R, lora_alpha=LORA_ALPHA,
                                lora_dropout=LORA_DROPOUT, head_dropout=HEAD_DROPOUT)
        lora_dir  = os.path.join(resume_from, "dinov2_lora")
        head_path = os.path.join(resume_from, "head.pt")
        if os.path.isdir(lora_dir):
            model.dinov2 = PeftModel.from_pretrained(
                model.dinov2.base_model.model, lora_dir, is_trainable=True
            )
        if os.path.isfile(head_path):
            model.head.load_state_dict(torch.load(head_path, map_location="cpu"))
    else:
        print("No checkpoint found — starting fresh.")

except Exception as e:
    print(f"Hub lookup failed ({e}) — starting fresh.")

if model is None:
    model = build_model(lora_r=LORA_R, lora_alpha=LORA_ALPHA,
                        lora_dropout=LORA_DROPOUT, head_dropout=HEAD_DROPOUT)

model.to(DEVICE)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE_TRAIN,
    per_device_eval_batch_size=BATCH_SIZE_EVAL,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    learning_rate=LEARNING_RATE,
    lr_scheduler_type=LR_SCHEDULER,
    warmup_ratio=WARMUP_RATIO,
    weight_decay=WEIGHT_DECAY,
    fp16=True,
    optim="adamw_torch_fused",
    logging_steps=50,
    save_strategy="epoch",
    eval_strategy="epoch",
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model=METRIC_FOR_BEST,
    greater_is_better=True,
    push_to_hub=True,
    hub_model_id=HUB_REPO,
    hub_strategy="checkpoint",
    hub_token=hf_token,
    report_to="none",
    dataloader_num_workers=0,
    remove_unused_columns=False,
)

trainer = HAMTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=eval_collate_fn,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PAT)],
    sampler=sampler,
    loss_fn=loss_fn,
)

print(f"{'Resuming' if resume_from else 'Starting fresh'} training...")
trainer.train(resume_from_checkpoint=resume_from)

os.makedirs(f"{FINAL_MODEL_DIR}/dinov2_lora", exist_ok=True)
model.dinov2.save_pretrained(f"{FINAL_MODEL_DIR}/dinov2_lora")
torch.save(model.head.state_dict(), f"{FINAL_MODEL_DIR}/head.pt")
print("Final model saved locally.")

upload_folder(
    repo_id=HUB_REPO,
    folder_path=FINAL_MODEL_DIR,
    commit_message="HAM10000 DINOv2-LoRA final",
    token=hf_token,
)
print("Done, model pushed to Hub.")
