import os
import torch
import librosa
import lightning as L
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
        return data_item["audio_path"], data_item["target_text"]



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

        self.audio_bos_token = self.processor.audio_bos_token
        self.audio_token = self.processor.audio_token
        self.audio_eos_token = self.processor.audio_eos_token

        self.relations = ",".join(read_data(os.path.join(self.data_path, "relation.json")))
        self.ins_begin_token = f"{self.audio_bos_token}{self.audio_token}{self.audio_eos_token}"
        self.ins_instruction_text = f"Please transcribe this voice:"
        # self.ins_instruction_text = f"{self.ins_begin_token}请从语音中抽取关系三元组，其中关系类型包括:{self.relations}。"
        self.instruction = f"{self.ins_begin_token}\n\n### Instruction:\n{self.ins_instruction_text}\n\n### Response:\n"
        self.instruction_ids = self.processor.tokenizer(
            self.instruction,
            add_special_tokens=False
        )["input_ids"]

        self.stage = "train"


    def setup(self, stage=None):
        self.stage = stage
        if stage in (None, "train"):
            # full_train = SRTEDataset(
            #     os.path.join(self.data_path, "train_target.json")
            # )

            # self.train_dataset = Subset(full_train, range(32))
            self.train_dataset = SRTEDataset(os.path.join(self.data_path, "train_target_RTE.json"))
        if stage in (None, "dev"):
            self.dev_dataset = SRTEDataset(os.path.join(self.data_path, "dev_target.json"))
        if stage in (None, "test"):
            self.test_dataset  = SRTEDataset(os.path.join(self.data_path, "test_target.json"))



    def collate_fn(self, batch):
        audio_list = []

        input_ids = []
        attention_mask = []
        labels = []

        generate_input_ids = []
        generate_attention_mask = []
        generate_labels = []

        pad_token_id = self.processor.tokenizer.pad_token_id
        input_dict = {}

        for x in batch:

            audio_path, srte_answer = x
            # 语音
            wav, _ = librosa.load(audio_path, sr=self.sample_rate, mono=True)
            audio_list.append(wav)
            
            # if self.stage == "train" or self.stage == "dev":
            #     # 构建训练时需要的input_ids和labels
            instruction_ids = self.instruction_ids.copy()
            response_ids = self.processor.tokenizer(srte_answer, add_special_tokens=False)["input_ids"]
            response_ids = response_ids + [self.processor.tokenizer.pad_token_id]

            cur_input_ids = instruction_ids + response_ids
            cur_labels = [-100] * len(instruction_ids) + response_ids
            cur_attention_mask = [1] * len(cur_input_ids)

            input_ids.append(cur_input_ids)
            labels.append(cur_labels)
            attention_mask.append(cur_attention_mask)
            
            if self.stage == "dev" or self.stage == "test":
                generate_input_ids.append(instruction_ids)
                generate_labels.append(response_ids)
                generate_attention_mask.append([1] * len(instruction_ids))

        # ====== 对 train/dev 的 teacher-forcing 输入做 batch 内动态 padding ======
        if len(input_ids) > 0:
            max_len = max(len(x) for x in input_ids)

            padded_input_ids = []
            padded_labels = []
            padded_attention_mask = []

            for ids, lbs, mask in zip(input_ids, labels, attention_mask):
                pad_len = max_len - len(ids)

                padded_input_ids.append([pad_token_id] * pad_len + ids)
                padded_labels.append([-100] * pad_len + lbs)
                padded_attention_mask.append([0] * pad_len + mask)

            input_ids = torch.tensor(padded_input_ids, dtype=torch.long)
            labels = torch.tensor(padded_labels, dtype=torch.long)
            attention_mask = torch.tensor(padded_attention_mask, dtype=torch.long)

        # ====== 对 dev/test 的 generate 输入做 batch 内动态 padding ======
        if len(generate_input_ids) > 0:
            max_gen_input_len = max(len(x) for x in generate_input_ids)
            max_gen_label_len = max(len(x) for x in generate_labels)

            padded_generate_input_ids = []
            padded_generate_attention_mask = []
            padded_generate_labels = []

            for ids, mask, lbs in zip(generate_input_ids, generate_attention_mask, generate_labels):
                input_pad_len = max_gen_input_len - len(ids)
                label_pad_len = max_gen_label_len - len(lbs)

                padded_generate_input_ids.append([pad_token_id] * input_pad_len + ids)
                padded_generate_attention_mask.append([0] * input_pad_len + mask)
                padded_generate_labels.append(lbs + [-100] * label_pad_len)

            generate_input_ids = torch.tensor(padded_generate_input_ids, dtype=torch.long)
            generate_attention_mask = torch.tensor(padded_generate_attention_mask, dtype=torch.long)
            generate_labels = torch.tensor(padded_generate_labels, dtype=torch.long)

        audio_inputs = self.processor.feature_extractor(
            audio_list, 
            sampling_rate=self.sample_rate, 
            return_attention_mask=True, 
            return_tensors='pt'
        )

        if self.stage == "train" or self.stage == "dev":
            input_dict["input_ids"] = input_ids
            input_dict["attention_mask"] = attention_mask
            input_dict["labels"] = labels
        if self.stage == "test" or self.stage == "dev":
            input_dict["generate_input_ids"] = generate_input_ids
            input_dict["generate_attention_mask"] = generate_attention_mask
            input_dict["generate_labels"] = generate_labels
        
        input_dict["input_features"] = audio_inputs["input_features"]
        input_dict["feature_attention_mask"] = audio_inputs["attention_mask"]

        return input_dict



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
        print(batch["generate_input_ids"].shape)
        print(batch["generate_attention_mask"].shape)
        print(batch["input_features"].shape)
        print(batch["feature_attention_mask"].shape)
        # print(batch["labels"].shape)
        # print(batch["generate_output_ids"].shape)
        generated_ids = model.generate( 
            input_ids=batch["generate_input_ids"],
            attention_mask=batch["generate_attention_mask"],
            input_features=batch["input_features"],
            feature_attention_mask=batch["feature_attention_mask"],
            max_new_tokens=128)
        generated_ids = generated_ids[:, batch["generate_input_ids"].shape[1]:]
        
        transcription = dm.processor.batch_decode(generated_ids, skip_special_tokens=True)

        labels = batch["generate_labels"].clone()
        labels[labels == -100] = dm.processor.tokenizer.pad_token_id
        labels = dm.processor.batch_decode(labels, skip_special_tokens=True)

        print(transcription)
        print(labels)   
        break





