# LGSC-for-FAS pytorch
## WORK IN PROGRESS

This repository is reimplementation of ["LGSC-for-FAS"](https://github.com/VIS-VAR/LGSC-for-FAS) repository, which is implementation
of ["Learning Generalized Spoof Cues for FaceAnti-spoofing (LGSC)"](https://arxiv.org/abs/2005.03922)

Code of this repository uses ["pytorch"](https://github.com/pytorch/pytorch) and ["pytorch_lightning"](https://github.com/PyTorchLightning/pytorch-lightning)

### Train
There are 2 ways:
 1. Configure your own dataset, redefine `val_dataloader` & `train_dataloader` functions in `pl_model.py`
 2. Specify in `configs/train_config.yml` next variables: `train_df` - path to csv with train info. 
 Each object should have `target` 
 (1 for live, 0 for spoof) and `path` 
 (relative path, see `src/dataset` to understand, how path is build), 
 Also specify `val_df` (like `train_df`),
 `path_root` (absolute path to your data) and 
 `default_root_dir` (path to store `pytorch_lightning` logs)
 
 **Note** You can crop faces online with [facenet-pytorch](https://github.com/timesler/facenet-pytorch),
 just specify parameters in `configs/train_config.yml`

Then run `python src/train.py -c configs/train_config`

### Infer
1. Specify parameters in `configs/infer_configs.yml`
2. Run `src/infer.py`

## Citation
```
@inproceedings{
    feng2020deep,
    title={Learning Generalized Spoof Cues for Face Anti-spoofing},
    author={Haocheng Feng and Zhibin Hong and Haixiao Yue and Yang Chen and Keyao Wang and 
    Junyu Han and Jingtuo Liu and Errui Ding},
    year={2020},
    eprint={2005.03922},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```