# Chinese Hubert Soft

> 本仓库代码不怎么优雅，仅供参考

本仓库基于 
- https://github.com/TencentGameMate/chinese_speech_pretrain
- https://github.com/bshall/hubert

基本训练流程:

```bash
python get_hubert_dsicrete_features.py
python convert_features_to_kmeans.py
python train.py
python clean_checkpoint.py
```
