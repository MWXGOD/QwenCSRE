import os
import torch
import librosa
import numpy as np
import lightning as L
import numpy as np
from tool import *
from typing import Any
from transformers import AutoProcessor
from torch.utils.data import Dataset, DataLoader



class SRTEDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = read_data(self.data_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_item = self.data[index]
        # if "entities" in data_item.keys() and "relations" in data_item.keys():
        #     return data_item["audio_path"], data_item["target_text"], data_item["entities"], data_item["relations"]
        return data_item["audio_path"], '$$'.join(data_item["triplets_list"])



class SRTEDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_path,
        model_name_or_path,
        batch_size=8,
        num_workers=4,
        max_length=2048,
        sample_rate=16000,
        **kwargs
    ):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_length = max_length
        self.sample_rate = sample_rate
        self.processor = AutoProcessor.from_pretrained(model_name_or_path)
        self.save_hyperparameters()  # 记录超参更稳, 通过self.hparams.XXX调用
        print(self.hparams)
        self.audio_bos_token = self.processor.audio_bos_token
        self.audio_token = self.processor.audio_token
        self.audio_eos_token = self.processor.audio_eos_token

        self.relations = ",".join(read_data(os.path.join(self.data_path, "relation.json")))
        self.ins_begin_token = f"{self.audio_bos_token}{self.audio_token}{self.audio_eos_token}"
        # self.ins_instruction_text = f"Please transcribe this voice:"
        self.ins_instruction_text = f"请从语音中抽取关系三元组，其中关系类型包括:{self.relations}。"
        self.instruction = f"{self.ins_begin_token}\n\n### Instruction:\n{self.ins_instruction_text}\n\n### Response:\n"
        self.instruction_ids = self.processor.tokenizer(
            self.instruction,
            add_special_tokens=False
        )["input_ids"]
        print("instruction:", self.instruction)

        self.stage = "train"


    def setup(self, stage=None):
        self.stage = stage
        if stage in (None, "train"):
            # full_train = SRTEDataset(
            #     os.path.join(self.data_path, "train_target.json")
            # )

            # self.train_dataset = Subset(full_train, range(32))
            self.train_dataset = SRTEDataset(os.path.join(self.data_path, "train.json"))
        if stage in (None, "dev"):
            self.dev_dataset = SRTEDataset(os.path.join(self.data_path, "dev.json"))
        if stage in (None, "test"):
            self.test_dataset  = SRTEDataset(os.path.join(self.data_path, "test.json"))



    def collate_fn(self, batch):
        audio_list = []
    
        infer_input_texts = []
        # infer_output_texts = []
        train_input_texts = []
        # infer_output_ids = []

        labels = []
        labels_ids = []

        for x in batch:

            audio_path, srte_answer = x
            # 语音
            ext = os.path.splitext(audio_path)[1].lower()
            if ext == ".npy":
                wav = np.load(audio_path)
            elif ext == ".wav":
                wav, _ = librosa.load(audio_path, sr=self.sample_rate, mono=True)
            else:
                raise ValueError(f"Unsupported audio file format: {audio_path}")

            audio_list.append(wav)
            
            infer_input_text = self.instruction
            infer_output_text = srte_answer + self.processor.tokenizer.pad_token
            train_input_text = infer_input_text + infer_output_text
            infer_output_ids = self.processor.tokenizer(infer_output_text, add_special_tokens=False)["input_ids"]

            infer_input_texts.append(infer_input_text)
            # infer_output_texts.append(infer_output_text)
            train_input_texts.append(train_input_text)
            labels_ids.append(infer_output_ids)

        # if self.stage == "train" or self.stage == "dev":
        train_proc = self.processor(
            text=train_input_texts,          # 每条样本的prompt，里面包含audio占位
            audio=audio_list,
            sampling_rate=self.sample_rate,
            text_kwargs={
                "truncation": True,
                "max_length": self.max_length,   # 只限制文本 token 长度
                "padding": True,
                "return_tensors": "pt",   # 放这里
            }
        )
        labels = torch.full_like(train_proc["input_ids"], -100)
        for i, ids in enumerate(labels_ids):
            labels[i, -len(ids):] = torch.tensor(ids, dtype=torch.long)
        # train_proc["labels"] = labels
        
        if self.stage == "dev" or self.stage == "test":
            dev_proc = self.processor(
            text=infer_input_texts,          # 每条样本的prompt，里面包含audio占位
            audio=audio_list,
            sampling_rate=self.sample_rate,
            text_kwargs={
                "truncation": True,
                "max_length": self.max_length,   # 只限制文本 token 长度
                "padding": True,
                "return_tensors": "pt",   # 放这里
            }
            )
            # dev_proc["ref_labels"] = labels
            if self.stage == "test":
                return {"dev_input": dev_proc, "labels": labels}
            else:
                return {"dev_input": dev_proc, "train_input": train_proc, "labels": labels}
        return {"train_input": train_proc, "labels": labels}

    def train_dataloader(self):
        return DataLoader[Any](
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )

    def dev_dataloader(self):
        return DataLoader(
            self.dev_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )


if __name__ == "__main__":
    
    from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

    # processor = AutoProcessor.from_pretrained(
    #     "cache/Qwen2-Audio-7B"
    # )

    data_path = "data/CONLL04"
    model_name_or_path = "cache/Qwen2-Audio-7B"
    batch_size = 8
    num_workers = 4
    max_length = 128
    sample_rate = 16000
    # processor = AutoProcessor.from_pretrained(processor_name_or_path, language="zh")
    model = Qwen2AudioForConditionalGeneration.from_pretrained(model_name_or_path)

    dm = SRTEDataModule(
        data_path,
        model_name_or_path,
        batch_size,
        num_workers,
        max_length,
        sample_rate,
        is_train=False,
    )
    dm.setup(stage="test")
    test_loader = dm.test_dataloader()
    for batch in test_loader:
        print(batch["dev_input"]["input_ids"].shape)
        print(batch["dev_input"]["attention_mask"].shape)
        print(batch["dev_input"]["input_features"].shape)
        print(batch["dev_input"]["feature_attention_mask"].shape)
        print(batch["labels"].shape)
        # print(batch["generate_output_ids"].shape)
        generated_ids = model.generate(**batch["dev_input"], max_new_tokens=128)
        # generated_ids = generated_ids[:, batch["dev_input"]["input_ids"].shape[1]:]
        
        transcription = dm.processor.batch_decode(generated_ids, skip_special_tokens=True)

        labels = batch["labels"].clone()
        labels[labels == -100] = dm.processor.tokenizer.pad_token_id
        labels = dm.processor.batch_decode(labels, skip_special_tokens=True)