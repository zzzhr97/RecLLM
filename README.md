# RecLLM
基于大语言模型的推荐系统（工程实践与科技创新4G大作业）

## 环境配置与安装
- 将解压出来的 `ml-1m` 数据集文件夹放在主目录中
- 根据 `requirements.txt` 安装合适的 packages

## 运行
- 使用 `python main.py --help` 查看帮助
- 修改 `run.sh` 中的参数
- 运行 `bash run.sh`

## 实验记录
- 在 `$exp_dir` (默认为 `./exp`) 中查找相应的实验文件夹
- 实验文件夹中保存以下文件
  - `args`: 保存命令行参数
  - `log`: 实验记录
  - `ckpt.pth`: 在 evaluation dataset 上评估得到的最好的模型参数