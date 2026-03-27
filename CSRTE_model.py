import lightning as L
import torch
from tool import *
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

class QwenAudioRTEModel(L.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()  # 记录超参更稳, 通过self.hparams.XXX调用

        # bnb_config = BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_compute_dtype=torch.bfloat16,
        #     bnb_4bit_use_double_quant=True,
        #     bnb_4bit_quant_type="nf4"
        # )

        self.qwenaudio = Qwen2AudioForConditionalGeneration.from_pretrained(
            self.hparams.model_name_or_path,
            torch_dtype=torch.bfloat16
            # quantization_config=bnb_config,
            # device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(self.hparams.model_name_or_path)
        
        # 开启 gradient checkpointing
        if self.hparams.gradient_checkpointing:
            self.qwenaudio.gradient_checkpointing_enable()
            self.qwenaudio.config.use_cache = False

        # self.qwenaudio = prepare_model_for_kbit_training(self.qwenaudio)

        # LoRA配置
        lora_config = LoraConfig(
            r=self.hparams.lora_rank,
            lora_alpha=self.hparams.lora_alpha,
            lora_dropout=self.hparams.lora_dropout,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            exclude_modules=["audio_tower"],
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        # 冻结所有参数
        for p in self.qwenaudio.parameters():
            p.requires_grad = False
        # 应用LoRA，仅用于语言模型，因为lora不需要插入音频塔
        self.qwenaudio.language_model = get_peft_model(
            self.qwenaudio.language_model,
            lora_config
        )

        # 打印可训练参数
        # self.qwenaudio.language_model.print_trainable_parameters()
        trainable = 0
        total = 0
        for n, p in self.qwenaudio.named_parameters():
            total += p.numel()
            if p.requires_grad:
                trainable += p.numel()

        print("trainable =", trainable, "total =", total, "trainable%", "%.4f" % (trainable / total * 100))
        # print("gradient_checkpointing:", self.qwenaudio.is_gradient_checkpointing)
        # print("use_cache:", getattr(self.qwenaudio.config, "use_cache", None))
        # print("attn_impl:", getattr(self.qwenaudio.config, "_attn_implementation", None))
        # print("dtype:", next(self.qwenaudio.parameters()).dtype)


        # 定义PRF的变量
        self.P_NER = 0.0
        self.R_NER = 0.0
        self.C_NER = 0.0
        self.P_RE = 0.0
        self.R_RE = 0.0
        self.C_RE = 0.0
        self.P_RTE = 0.0
        self.R_RTE = 0.0
        self.C_RTE = 0.0

        # 定义f1
        self.max_ner_f1 = 0.0
        self.max_re_f1 = 0.0
        self.max_rte_f1 = 0.0

    def forward(self, **input):
        outputs = self.qwenaudio(**input)
        return outputs

    def clear_PRC(self):
        self.P_NER = 0.0
        self.R_NER = 0.0
        self.C_NER = 0.0
        self.P_RE = 0.0
        self.R_RE = 0.0
        self.C_RE = 0.0
        self.P_RTE = 0.0
        self.R_RTE = 0.0
        self.C_RTE = 0.0
        self.max_ner_f1 = 0.0
        self.max_re_f1 = 0.0
        self.max_rte_f1 = 0.0

    def on_train_batch_start(self, epoch, train_dataloader):
        self.train()
        train_bar = tqdm(
            train_dataloader,
            desc=f"Epoch [{epoch+1}/{self.hparams.epochs_num}] Training",
            leave=False
        )
        return train_bar
    
    def training_step(self, batch, optimizer=None, scheduler=None, accelerator=None):

        with accelerator.accumulate(self):
            train_loss = self.qwenaudio(
                **batch["train_input"],
                labels=batch["labels"],
                return_dict=False,
                use_cache=False,
            )[0]

            accelerator.backward(train_loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(self.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        return train_loss.detach()

    def on_validation_batch_start(self, epoch, dev_dataloader):
        self.eval()
        self.clear_PRC()
        dev_bar = tqdm(
            dev_dataloader,
            desc=f"Epoch [{epoch+1}/{self.hparams.epochs_num}] Validation",
            leave=False
        )
        return dev_bar

    def validation_step(self, batch, accelerator=None):

        val_loss = None
        with accelerator.autocast():
            val_loss = self.qwenaudio(
                **batch["train_input"],
                labels=batch["labels"],
                return_dict=False,
                use_cache=False,
            )[0]

        with torch.no_grad():
            with accelerator.autocast():
                generated_ids = self.qwenaudio.generate(
                    **batch["dev_input"], 
                    max_new_tokens=128,
                    do_sample=False,
                    temperature=0.0,
                    top_p=1.0,
                    use_cache=True
                )
        generated_ids = generated_ids[:, batch["dev_input"]["input_ids"].shape[1]:]

        labels = batch["labels"].clone()
        labels[labels == -100] = self.processor.tokenizer.pad_token_id
        gen_text_batch = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        lab_text_batch = self.processor.batch_decode(labels, skip_special_tokens=True)

        batch_rte_lab = self.batch_text2rte(lab_text_batch)
        batch_rte_gen = self.batch_text2rte(gen_text_batch)

        self.compute_metric_step_update_4_rte(batch_rte_lab, batch_rte_gen)

        return gen_text_batch, lab_text_batch, val_loss

    def on_validation_batch_end(self):
        P_NER = self.C_NER / self.P_NER if self.P_NER > 0.0 else 0.0
        R_NER = self.C_NER / self.R_NER if self.R_NER > 0.0 else 0.0
        F1_NER = 2 * P_NER * R_NER / (P_NER + R_NER) if (P_NER + R_NER) > 0.0 else 0.0
        if self.max_ner_f1 < F1_NER:
            self.max_ner_f1 = F1_NER

        P_RE = self.C_RE / self.P_RE if self.P_RE > 0.0 else 0.0
        R_RE = self.C_RE / self.R_RE if self.R_RE > 0.0 else 0.0
        F1_RE = 2 * P_RE * R_RE / (P_RE + R_RE) if (P_RE + R_RE) > 0.0 else 0.0
        if self.max_re_f1 < F1_RE:
            self.max_re_f1 = F1_RE

        P_RTE = self.C_RTE / self.P_RTE if self.P_RTE > 0.0 else 0.0
        R_RTE = self.C_RTE / self.R_RTE if self.R_RTE > 0.0 else 0.0
        F1_RTE = 2 * P_RTE * R_RTE / (P_RTE + R_RTE) if (P_RTE + R_RTE) > 0.0 else 0.0
        if self.max_rte_f1 < F1_RTE:
            self.max_rte_f1 = F1_RTE

        return P_NER, R_NER, F1_NER, P_RE, R_RE, F1_RE, P_RTE, R_RTE, F1_RTE

    def on_test_epoch_start(self, test_dataloader):
        self.eval()
        self.clear_PRC()
        test_bar = tqdm(
            test_dataloader,
            desc=f"Testing",
            leave=False
        )
        return test_bar

    def test_step(self, batch, accelerator=None): 

        with torch.no_grad():
            with accelerator.autocast():
                generated_ids = self.qwenaudio.generate(
                    **batch["dev_input"], 
                    max_new_tokens=128,
                    do_sample=False,
                    temperature=0.0,
                    top_p=1.0,
                    use_cache=True
                )
        generated_ids = generated_ids[:, batch["dev_input"]["input_ids"].shape[1]:]

        labels = batch["labels"].clone()
        labels[labels == -100] = self.processor.tokenizer.pad_token_id
        gen_text_batch = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        lab_text_batch = self.processor.batch_decode(labels, skip_special_tokens=True)

        batch_rte_lab, batch_ner_lab, batch_re_lab = self.batch_text2rte(lab_text_batch)
        batch_rte_gen, batch_ner_gen, batch_re_gen = self.batch_text2rte(gen_text_batch)

        self.compute_metric_step_update_4_rte(batch_rte_lab, batch_rte_gen)
        self.compute_metric_step_update_4_ner(batch_ner_lab, batch_ner_gen)
        self.compute_metric_step_update_4_re(batch_re_lab, batch_re_gen)

        return gen_text_batch, lab_text_batch

    def on_test_batch_end(self):
        P_NER = self.C_NER / self.P_NER if self.P_NER > 0.0 else 0.0
        R_NER = self.C_NER / self.R_NER if self.R_NER > 0.0 else 0.0
        F1_NER = 2 * P_NER * R_NER / (P_NER + R_NER) if (P_NER + R_NER) > 0.0 else 0.0
        if self.max_ner_f1 < F1_NER:
            self.max_ner_f1 = F1_NER

        P_RE = self.C_RE / self.P_RE if self.P_RE > 0.0 else 0.0
        R_RE = self.C_RE / self.R_RE if self.R_RE > 0.0 else 0.0
        F1_RE = 2 * P_RE * R_RE / (P_RE + R_RE) if (P_RE + R_RE) > 0.0 else 0.0
        if self.max_re_f1 < F1_RE:
            self.max_re_f1 = F1_RE

        P_RTE = self.C_RTE / self.P_RTE if self.P_RTE > 0.0 else 0.0
        R_RTE = self.C_RTE / self.R_RTE if self.R_RTE > 0.0 else 0.0
        F1_RTE = 2 * P_RTE * R_RTE / (P_RTE + R_RTE) if (P_RTE + R_RTE) > 0.0 else 0.0
        if self.max_rte_f1 < F1_RTE:
            self.max_rte_f1 = F1_RTE

        return P_NER, R_NER, F1_NER, P_RE, R_RE, F1_RE, P_RTE, R_RTE, F1_RTE


    def text2rte(self, text):
        # text_re = "Douglas Flint##person title##chairman$$Douglas Flint##person title##Chief Financial Officer"
        final_ner = []
        final_re = []
        final_rte = []
        for re_item in text.split("$$"):
            re_item = re_item.strip()
            final_rte.append(re_item)
            if re_item:
                re_item_element = re_item.split("##")
                if len(re_item_element) == 3:
                    final_ner.append(re_item_element[0])
                    final_re.append(re_item_element[1])
                    final_ner.append(re_item_element[2])
        return final_rte, final_ner, final_re

    def batch_text2rte(self, batch_text):
        batch_rte = []
        batch_ner = []
        batch_re = []
        for text_item in batch_text:
            final_rte, final_ner, final_re = self.text2rte(text_item)
            batch_rte.append(final_rte)
            batch_ner.append(final_ner)
            batch_re.append(final_re)
        return batch_rte, batch_ner, batch_re

    def compute_metric_step_update_4_rte(self, batch_rte_lab, batch_rte_gen):
        for brl, brp in zip(batch_rte_lab, batch_rte_gen):
            brl = [l.lower() for l in brl if l != "None"]
            brp = [p.lower() for p in brp if p != "None"]
            self.P_RTE += len(set(brp))
            self.R_RTE += len(set(brl))
            self.C_RTE += len(set(brp) & set(brl))

    def compute_metric_step_update_4_ner(self, batch_ner_lab, batch_ner_gen):
        for brl, brp in zip(batch_ner_lab, batch_ner_gen):
            brl = [l.lower() for l in brl if l != "None"]
            brp = [p.lower() for p in brp if p != "None"]
            self.P_NER += len(set(brp))
            self.R_NER += len(set(brl))
            self.C_NER += len(set(brp) & set(brl))

    def compute_metric_step_update_4_re(self, batch_re_lab, batch_re_gen):
        for brl, brp in zip(batch_re_lab, batch_re_gen):
            brl = [l.lower() for l in brl if l != "None"]
            brp = [p.lower() for p in brp if p != "None"]
            self.P_RE += len(set(brp))
            self.R_RE += len(set(brl))
            self.C_RE += len(set(brp) & set(brl))



    def test_func(self):
        pass