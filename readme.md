
---

## 1. 数据格式

训练数据中，`instruction` 和 `answer` 是通过一个 `\n` 拼接的：

```text
你是一个关系抽取助手。
<audio>请从语音中抽取关系三元组。
Khieu Samphan##Work_For##Khmer Rouge
```

对应代码：

```python
total_content = '\n'.join(total_content)
```

本质上是将 prompt 和 target 放在同一个序列中，采用 **Causal LM 的方式训练**。

---

## 2. 文本与音频处理方式

### 文本

```python
tokenizer(
    context,
    return_attention_mask=False,
    add_special_tokens=False,
    **kwargs
)
```

### 音频

```python
audio_inputs = self.processor.feature_extractor(
    audios,
    sampling_rate=self.sampling_rate,
    return_attention_mask=True,
    return_tensors='pt'
)
```

典型输入形状：

```python
input_ids.shape              # [1, 109]
attention_mask.shape         # [1, 109]
input_features.shape         # [1, 128, 3000]
feature_attention_mask.shape # [1, 3000]
```

其中：

* `feature_attention_mask[:1827] = 1`（有效音频）
* `feature_attention_mask[1827:] = 0`（padding）

---

## 3. ⚠️ Qwen2-Audio 的特殊对齐机制（非常重要）

Qwen2-Audio 的输入中包含特殊 token：

* `<|audio_bos|>`
* 多个 `<|AUDIO|>`
* `<|audio_eos|>`

关键点：

👉 `<|AUDIO|>` 的数量必须与音频特征长度严格对齐
👉 这个对齐逻辑 **必须由 processor 自动完成**

如果你手动构造 `input_ids`，很容易出现：

* shape 不匹配
* forward 直接报错
* 生成异常

✅ **强烈建议：始终使用 `Qwen2AudioProcessor` 处理多模态输入**

---

## 4. Label 构造方式

训练时只对 **答案部分计算 loss**，其余部分全部 mask：

```python
labels = [-100, ..., -100, token_ids_of_answer]
```

示例：

```python
'labels' = [
    -100, -100, ..., -100,
    46788, 25173, 8224, 9943, ...
]
```

规则：

* prompt / system / user → `-100`
* answer → 正常 token id

👉 这是标准 Causal LM 训练方式，但在多模态场景中更容易出错，一定要检查。

---

## 5. LoRA 作用层选择

LoRA 需要精确匹配模块名称。

推荐匹配模式：

```python
'.*\\.(q_proj|k_proj|down_proj|v_proj|gate_proj|up_proj|o_proj)'
```

完整配置示例：

```python
lora_kwargs = {
    'r': 8,
    'target_modules': '^(language_model(?=\\.).*\\.(q_proj|k_proj|down_proj|v_proj|gate_proj|up_proj|o_proj))$',
    'lora_alpha': 32,
    'lora_dropout': 0.05,
    'bias': 'none',
    'modules_to_save': [],
    'use_rslora': False,
    'use_dora': False,
    'lorap_lr_ratio': None,
    'init_lora_weights': True
}
```

👉 注意这里 **只作用在 language_model 上**

---

## 6. 参数冻结策略（关键坑点）

⚠️ 一定要注意：

* ❌ 不要错误冻结 `audio_tower`
* ✅ 需要明确控制 Qwen language model 的训练方式

如果冻结策略错误，可能导致：

* loss 不下降
* 参数没有被训练
* 多模态能力失效

---

## 7. 可训练参数规模（用于 sanity check）

```text
总参数：8.4B
可训练参数：约 20M
占比：0.2375%
```

```text
trainable params: 19,988,480
all params: 8,417,083,392
```

👉 如果你的数值差很多，优先检查：

* LoRA target_modules
* 是否正确注入 LoRA
* freeze 是否正确

---

## 8. generate 行为说明

`qwen.generate()` 有一个容易误解的行为：

👉 以 **batch 为单位 padding**

表现为：

```text
<|endoftext|> <|endoftext|> <|endoftext|> ...
```

解释：

* 第一个 `<|endoftext|>` → 真正的结束符
* 后面的 → padding

👉 不要把多个 eos 当成模型输出

---

## 9. ⚠️ 训练时必须关闭 logits（显存优化关键）

音频序列很长，如果返回 logits，会极大占用显存。

👉 推荐只返回 loss：

```python
train_loss = self.qwenaudio(
    **batch["train_input"],
    labels=batch["labels"],
    return_dict=False,
    use_cache=False,
)[0]
```

完整训练步骤：

```python
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
```

✅ 效果：

* 显著降低显存占用
* 提高训练稳定性

---

## 10. 总结（最容易踩坑的点）

在自定义实现 Qwen2-Audio + LoRA 时，最容易出问题的地方：

1. **音频 token 对齐（最关键）**
2. **label mask 是否正确**
3. **LoRA 注入层是否匹配**
4. **freeze 策略是否正确**
5. **是否关闭 logits（显存问题）**

---

## 一句话总结

> Qwen2-Audio 的微调不仅是 LoRA 的问题，更关键的是 **多模态对齐 + 数据构造 + 训练细节控制**。

---
