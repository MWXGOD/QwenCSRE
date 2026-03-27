import os
import math
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import json, time, torch, argparse, swanlab
from tool import Hyperargs, read_data
from tqdm import tqdm
from datetime import datetime
from lightning.pytorch import seed_everything
from CSRTE_model import QwenAudioRTEModel
from loguru import logger
from accelerate import Accelerator
from torch.optim import AdamW
from CSRTE_data_module import SRTEDataModule
from transformers import get_cosine_schedule_with_warmup
from bitsandbytes.optim import PagedAdamW32bit


parser = argparse.ArgumentParser()
parser.add_argument('--args_path', type=str, default='csrte_args/CoNLL04_Qwen_LoRA.json')
# parser.add_argument('--args_path', type=str, default='argsfile/aishell_ner_args_4_whisper_medium.json')
shell_args = parser.parse_args()
args_dict = read_data(shell_args.args_path)
hyperargs = Hyperargs(**args_dict)
seed_everything(hyperargs.seed, workers=True)

time_str = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
swanlab.init(
    project=hyperargs.swanlab_project_name,
    config=hyperargs.__dict__,
    experiment_name=f"MWX-Qwen2Audio-Demo-{time_str}"
)

# 数据
data_module = SRTEDataModule(**hyperargs.__dict__)
data_module.setup(stage = "train")
train_dataloader = data_module.train_dataloader()
data_module.setup(stage = "dev")
dev_dataloader = data_module.dev_dataloader()

# 模型
model = QwenAudioRTEModel(**hyperargs.__dict__)
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"trainable: {trainable}, total: {total}")
# for name, _ in model.named_modules():
#     print(name)


# 优化器
trainable_params = [p for p in model.parameters() if p.requires_grad]
optimizer = PagedAdamW32bit(trainable_params, lr=hyperargs.learning_rate, weight_decay = hyperargs.weight_decay)

# 学习率调度器
num_update_steps_per_epoch = math.ceil(len(train_dataloader) / hyperargs.gradient_accumulation_steps)
num_training_steps = hyperargs.epochs_num * num_update_steps_per_epoch
num_warmup_steps = int(hyperargs.warmup_rate * num_training_steps)
scheduler = get_cosine_schedule_with_warmup(
    optimizer = optimizer, 
    num_warmup_steps = num_warmup_steps, 
    num_training_steps=num_training_steps
)

# 混合精度
accelerator = Accelerator(
    mixed_precision=hyperargs.mixed_precision, 
    gradient_accumulation_steps=hyperargs.gradient_accumulation_steps
)
model, optimizer, train_dataloader, dev_dataloader, scheduler = accelerator.prepare(
    model, optimizer, train_dataloader, dev_dataloader, scheduler
)

max_f1 = 0
for epoch in range(hyperargs.epochs_num):
    train_start = time.time()
    train_bar = model.on_train_batch_start(epoch, train_dataloader)
    train_loss_per_epoch = 0
    for batch in train_bar:
        loss = model.training_step(batch, optimizer, scheduler, accelerator)
        train_loss_per_epoch += loss.item()
        train_bar.set_postfix({"loss": loss.item()})
    swanlab.log({"train_loss_per_step": train_loss_per_epoch/len(train_dataloader)})
    train_end = time.time()
    logger.info(f"训练结束，第{epoch+1}轮总时长：{(train_end-train_start)/60:.2f}分钟")

    dev_start = time.time()
    dev_bar = model.on_validation_batch_start(epoch, dev_dataloader)
    dev_loss_per_epoch = 0
    gen_text_per_epoch = []
    lab_text_per_epoch = []
    with torch.no_grad():
        for batch in dev_bar:
            gen_text_batch, lab_text_batch, val_loss = model.validation_step(batch, accelerator=accelerator)
            dev_loss_per_epoch += val_loss.item()
            dev_bar.set_postfix({"loss": val_loss.item()})
            gen_text_per_epoch += gen_text_batch
            lab_text_per_epoch += lab_text_batch
    swanlab.log({"dev_loss_per_step": dev_loss_per_epoch/len(dev_dataloader)})
    P_NER, R_NER, F1_NER, P_RE, R_RE, F1_RE, P_RTE, R_RTE, F1_RTE = model.on_validation_batch_end()
    final_f1 = F1_RTE
    swanlab.log({"final_f1": final_f1})
    swanlab.log({
        "P_NER": P_NER,
        "R_NER": R_NER,
        "F1_NER": F1_NER,
        "P_RE": P_RE,
        "R_RE": R_RE,
        "F1_RE": F1_RE,
        "P_RTE": P_RTE,
        "R_RTE": R_RTE,
        "F1_RTE": F1_RTE
    })
    os.makedirs(hyperargs.output_result_path, exist_ok=True)
    with open(f"{hyperargs.output_result_path}/gen_text_batch_{epoch}.json", 'w', encoding='utf-8') as f:
        json.dump({"pred_label": gen_text_per_epoch}, f, indent=4, ensure_ascii=False)
    with open(f"{hyperargs.output_result_path}/lab_text_batch_{epoch}.json", 'w', encoding='utf-8') as f:
        json.dump({"gold_label": lab_text_per_epoch}, f, indent=4, ensure_ascii=False)
    
    if max_f1<final_f1:
        max_f1 = final_f1
        os.makedirs(hyperargs.output_model_path, exist_ok=True)
        save_path = os.path.join(hyperargs.output_model_path, f"{hyperargs.output_model_path.split('/')[-1]}.bin")
        torch.save(
            {
                "state_dict": model.state_dict(),
                "hparams": dict(model.hparams),
            }, 
            save_path
        )
        os.makedirs(hyperargs.output_result_path, exist_ok=True)
        with open(f"{hyperargs.output_result_path}/best_gen_text_batch.json", 'w', encoding='utf-8') as f:
            json.dump({"pred_label": gen_text_per_epoch}, f, indent=4, ensure_ascii=False)
        with open(f"{hyperargs.output_result_path}/best_lab_text_batch.json", 'w', encoding='utf-8') as f:
            json.dump({"gold_label": lab_text_per_epoch}, f, indent=4, ensure_ascii=False)
        logger.info("模型已保存")
    logger.info(f"评价指标F: {final_f1:.2f}")
    dev_end = time.time()
    logger.info(f"验证结束，第{epoch+1}轮总时长：{(dev_end-dev_start)/60:.2f}分钟")









