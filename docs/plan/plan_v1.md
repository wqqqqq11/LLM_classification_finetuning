V1 总体定位

V1 采用：

Gemma 2 9B IT 4bit + LoRA + 三分类分类头


任务是根据：

prompt
response_a
response_b


预测三类概率：

winner_model_a
winner_model_b
winner_tie


提交文件必须包含 id,winner_model_a,winner_model_b,winner_tie 四列，三类概率和为 1。


一、输入字段

输入只保留：

prompt
response_a
response_b


二、V1 输入模板

V1 使用简洁模板：

You are judging which assistant response a human user would prefer.
The order of responses is arbitrary. Judge only the content.

[User Prompt]
{prompt}

[Response A]
{response_a}

[Response B]
{response_b}

Predict the human preference:
A = Response A is better
B = Response B is better
Tie = both responses are similarly preferred

Preference:


模型读完整段文本后，在最后位置做三分类：

0 -> winner_model_a
1 -> winner_model_b
2 -> winner_tie


这套模板的目的有四个：

明确 prompt、A、B 的边界
保证 A/B 结构对称
告诉模型这是用户偏好预测，不是标准答案评分
给分类头一个稳定落点：Preference:


三、V1 截断策略


1. 最大长度

V1 默认：

max_length = 3072 tokens

如果资源允许，后续再尝试：

max_length = 4096

2. 基础预算

以 3072 tokens 为例：

模板预留：128 ~ 160
prompt：最多 512
response_a：最多 1200
response_b：最多 1200

核心原则：

response_a 和 response_b 的初始预算必须相等，否则模型会受到长度不公平影响。

3. 动态补偿

如果 prompt 很短，没有用完 512 tokens，剩余 token 平均分给 A/B。

如果 A 很短、B 很长，A 没用完的预算可以给 B；反过来也一样。

但是基础设计仍然保持 A/B 对称。

4. 头尾保留

长文本不只保留开头，也不只保留结尾。

V1 规则：

prompt：head 40% + tail 60%
response：head 65% + tail 35%


四、V1 数据增强

V1 训练阶段加入 A/B swap augmentation。

原始样本：

prompt, response_a, response_b -> label = A


交换后：

prompt, response_b, response_a -> label = B


标签映射：

A赢 -> B赢
B赢 -> A赢
Tie -> Tie


注意顺序：

先划分训练集 / 验证集
再只对训练集做 swap augmentation


不能先增强再划分，否则原样本和交换样本可能分别进入训练集和验证集，导致验证分数虚高。

五、V1 验证方案

V1 使用：

Stratified split


或者资源允许时：

Stratified K-Fold


第一版建议先单折验证，跑通流程。

标签分布比较均衡：

winner_model_a: 34.91%
winner_model_b: 34.19%
winner_tie: 30.90%


所以 V1 暂时不加 class weight。

验证指标只盯：

log loss

accuracy 可以看，但不是主要指标。

六、V1 模型方案

模型：

gemma-2-9b-it-4bit


训练方式：

LoRA 微调
三分类 classification head


项目里已经有对应的模型目录和模块结构，gemma_classifier.py 适合承载 Gemma + LoRA 分类模型，trainer.py 负责训练，evaluation.py 负责 log loss。


V1 不做复杂生成式预测。

不让模型输出：

A / B / Tie


而是直接输出三类 logits，再 softmax 成概率。

七、V1 推理方案

推理时也做双向预测。

第一遍：

prompt, response_a, response_b


得到：

p_a, p_b, p_tie


第二遍交换 A/B：

prompt, response_b, response_a


得到：

p'_a, p'_b, p'_tie


然后映射回来：

swapped_p_a = p'_b
swapped_p_b = p'_a
swapped_p_tie = p'_tie


最终平均：

final_p_a = (p_a + swapped_p_a) / 2
final_p_b = (p_b + swapped_p_b) / 2
final_p_tie = (p_tie + swapped_p_tie) / 2


这个策略主要是降低 A/B 位置偏差。

八、V1 提交文件

最终生成：

submission.csv

格式：

id,winner_model_a,winner_model_b,winner_tie


每一行：

id, P(A赢), P(B赢), P(Tie)


三类概率必须和为 1。

测试集示例只有 3 条，但正式评分时平台会替换完整隐藏测试集，所以 V1 不能依赖当前 test.csv 的分布。
