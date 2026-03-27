import os
import json
import time
import torch
import argparse
import swanlab

from tool import Hyperargs, read_data
from datetime import datetime
from lightning.pytorch import seed_everything
from CSRTE_model import QwenAudioRTEModel
from loguru import logger
from accelerate import Accelerator
from CSRTE_data_module import SRTEDataModule


os.environ["CUDA_VISIBLE_DEVICES"] = "1"


parser = argparse.ArgumentParser()
parser.add_argument('--args_path', type=str, default='csrte_args/CoNLL04_Qwen_LoRA.json')
parser.add_argument(
    '--ckpt_path',
    type=str,
    default='output/CoNLL04/Qwen2Audio-7B_checkpoint/Qwen2Audio-7B_checkpoint.bin'
)
shell_args = parser.parse_args()

# 读取参数
args_dict = read_data(shell_args.args_path)
hyperargs = Hyperargs(**args_dict)
seed_everything(hyperargs.seed, workers=True)

time_str = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
swanlab.init(
    project=hyperargs.swanlab_project_name,
    config=hyperargs.__dict__,
    experiment_name=f"MWX-QwenAudio-Test-{time_str}"
)

# 数据
data_module = SRTEDataModule(**hyperargs.__dict__)
data_module.setup(stage="test")
test_dataloader = data_module.test_dataloader()

# 模型
model = QwenAudioRTEModel(**hyperargs.__dict__)

total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"trainable: {trainable}, total: {total}")

# 加载权重
logger.info(f"加载模型权重: {shell_args.ckpt_path}")
ckpt = torch.load(shell_args.ckpt_path, map_location="cpu")

state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

logger.info(f"missing_keys: {missing_keys}")
logger.info(f"unexpected_keys: {unexpected_keys}")

# 混合精度
accelerator = Accelerator(
    mixed_precision=hyperargs.mixed_precision
)

model, test_dataloader = accelerator.prepare(model, test_dataloader)

# 开始测试
test_start = time.time()
model.eval()

# 沿用 dev 的验证逻辑
test_bar = model.on_test_epoch_start(test_dataloader)
test_loss_per_epoch = 0
gen_text_per_epoch = []
lab_text_per_epoch = []

with torch.no_grad():
    for batch in test_bar:
        gen_text_batch, lab_text_batch = model.test_step(
            batch,
            accelerator=accelerator
        )

        gen_text_per_epoch += gen_text_batch
        lab_text_per_epoch += lab_text_batch

avg_test_loss = test_loss_per_epoch / len(test_dataloader)
swanlab.log({"test_loss": avg_test_loss})

# 计算指标
P_NER, R_NER, F1_NER, P_RE, R_RE, F1_RE, P_RTE, R_RTE, F1_RTE = model.on_validation_batch_end()

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

logger.info(f"Test Loss: {avg_test_loss:.6f}")
logger.info(f"NER  -> P: {P_NER:.4f}, R: {R_NER:.4f}, F1: {F1_NER:.4f}")
logger.info(f"RE   -> P: {P_RE:.4f}, R: {R_RE:.4f}, F1: {F1_RE:.4f}")
logger.info(f"RTE  -> P: {P_RTE:.4f}, R: {R_RTE:.4f}, F1: {F1_RTE:.4f}")

# 保存结果
os.makedirs(hyperargs.output_result_path, exist_ok=True)

with open(f"{hyperargs.output_result_path}/test_gen_text.json", "w", encoding="utf-8") as f:
    json.dump({"pred_label": gen_text_per_epoch}, f, indent=4, ensure_ascii=False)

with open(f"{hyperargs.output_result_path}/test_lab_text.json", "w", encoding="utf-8") as f:
    json.dump({"gold_label": lab_text_per_epoch}, f, indent=4, ensure_ascii=False)

with open(f"{hyperargs.output_result_path}/test_metrics.json", "w", encoding="utf-8") as f:
    json.dump({
        "test_loss": avg_test_loss,
        "P_NER": P_NER,
        "R_NER": R_NER,
        "F1_NER": F1_NER,
        "P_RE": P_RE,
        "R_RE": R_RE,
        "F1_RE": F1_RE,
        "P_RTE": P_RTE,
        "R_RTE": R_RTE,
        "F1_RTE": F1_RTE
    }, f, indent=4, ensure_ascii=False)

test_end = time.time()
logger.info(f"测试结束，总时长：{(test_end - test_start)/60:.2f}分钟")
