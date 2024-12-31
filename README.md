# RecLLM
基于大语言模型的推荐系统（工程实践与科技创新4G大作业）

## 环境配置与安装
- 将解压出来的 `ml-1m` 数据集文件夹放在主目录中
- 根据 `requirements.txt` 安装合适的 packages

## 如何运行？
- 使用 `python main.py --help` 查看命令行参数的帮助说明
- 修改 `run.sh` 中的命令行参数
  - 修改 `--exp-name="NCF_test1"` 即可指定本次实验的 `log` 日志文件和模型参数保存的路径
- 运行 `bash run.sh` 
  - 从终端中查看模型训练的进度
  - 从日志文件中查看完整的训练和测试结果

## 如何查看实验记录？
- 在 `$exp_dir` (默认为 `./exp`) 中查找相应的实验文件夹
- 实验文件夹中保存以下文件
  - `args`: 记录实验环境和命令行参
  - `log`: ：记录训练和测试的实验结果
  - `ckpt.pth`: 在 evaluation dataset 上评估得到的最好的模型参数

## 改进总结
- 修正 backward 逻辑，使模型有效训练
- 修正 early stop 逻辑
- 引入 metadata
- 测试时会访问到不存在的 item ids，在获取 metadata 时出现 KeyError，使用 mask 来避免这一点
- 添加 LLM
- LLM 需要添加 padding，输出时通过 attention mask 来避免计算 padding token 的 hidden state
- 添加 LoRA 微调机制
- 测试时提前计算 item embedding 以避免重复计算，节省测试时间
- 测试时分批次计算 item embedding 以避免显存溢出
- 修正 predict 函数，测试时模型的输出为预测的分数而非 item embedding 和 user embedding 的 cosine 相似度
- 添加多头注意力层，用于更好地融合 user embedding 和 item embedding