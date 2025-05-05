# GAPrompt: Geometry-Aware Point Cloud Prompt for 3D Vision Model
<div align="center">
    Zixiang Ai<sup>1</sup>&emsp; Zichen Liu<sup>1</sup>&emsp; Yuanhang Lei<sup>2</sup>&emsp; Zhenyu Cui<sup>1</sup>&emsp; Xu Zou<sup>3</sup>&emsp; Jiahuan Zhou<sup>†</sup><sup>1</sup> <br>
    <small>
    <sup>1 Wangxuan Institute of Computer Technology, Peking University&emsp; <br></sup>
    <sup>2 State Key Laboratory of CAD&CG, Zhejiang University&emsp; <br></sup>
    <sup>3 School of Artificial Intelligence and Automation, Huazhong University of Science and Technology</sup>
    </small>
</div>

<p align="center">
  <a href="https://github.com/zhoujiahuan1991/ICML2025-GAPrompt"><img src="https://hitscounter.dev/api/hit?url=https%3A%2F%2Fgithub.com%2Fzhoujiahuan1991%2FICML2025-GAPrompt&label=GAPrompt&icon=github&color=%230a58ca"></a>
</p>

<div align="center">
Official implementation of 'GAPrompt: Geometry-Aware Point Cloud Prompt for 3D Vision Model'.


The paper has been accepted by **ICML 2025**.
</div>




<p align="center"><img src="files/pipeline.png" width="60%"></p>


Pre-trained 3D vision models have gained significant attention for their promising performance on point cloud data. However, fully fine-tuning these models for downstream tasks is computationally expensive and storage-intensive. Existing parameter-efficient fine-tuning (PEFT) approaches, which focus primarily on input token prompting, struggle to achieve competitive performance due to their limited ability to capture the geometric information inherent in point clouds. 

To address this challenge, we propose a novel Geometry-Aware Point Cloud Prompt (GAPrompt) that leverages geometric cues to enhance the adaptability of 3D vision models. First, we introduce a Point Prompt that serves as an auxiliary input alongside the original point cloud, explicitly guiding the model to capture fine-grained geometric details. Additionally, we present a Point Shift Prompter designed to extract global shape information from the point cloud, enabling instance-specific geometric adjustments at the input level. Moreover, our proposed Prompt Propagation mechanism incorporates the shape information into the model's feature extraction process, further strengthening its ability to capture essential geometric characteristics. Extensive experiments demonstrate that GAPrompt significantly outperforms state-of-the-art PEFT methods and achieves competitive results compared to full fine-tuning on various benchmarks, while utilizing only 2.19\% of trainable parameters.

## Main Results
Classification on three variants of the ScanObjectNN and the ModelNet40, including the number of trainable parameters (Param) and overall accuracy (Acc). We report ScanObjectNN and ModelNet40 results without voting.

<p align="center"><img src="files/results.png" width="60%"></p>



## Checkpoint Release
The backbone checkpoints used in our paper are provided below.
| Backbones | Reference | Checkpoints |
| :-----: |:-----:| :-----:|
| Point-MAE | ECCV 22 | [mae_base.pth](https://github.com/Pang-Yatian/Point-MAE/releases/download/main/pretrain.pth) |
| ReCon | ICML 23 | [recon_base.pth](https://drive.google.com/file/d/1L-TlZUi7umBCDpZW-1F0Gf4X-9Wvf_Zo/view?usp=share_link) |
| Point-GPT | NeurIPS 23 | [pointgpt-post-pretrained-L.pth](https://drive.google.com/file/d/1Kh6f6gFR12Y86FAeBtMU9NbNpB5vZnpu/view?usp=sharing) |
| Point-FEMAE | AAAI 24 | [mae_base.pth](https://drive.google.com/drive/folders/1q0A-yXC1fmKKg38fbaqIxM79lvXpj4AO?usp=drive_link) |


## Environment
Create a conda environment and install basic dependencies:
```bash
git clone git@github.com:zhoujiahuan1991/ICML2025-GAPrompt.git
cd ICML2025-GAPrompt

# Not necessary
conda create -n gaprompt python=3.9
conda activate gaprompt

# Install the corresponding versions of Torch and TorchVision; other compatible versions are also acceptable.
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116

pip install -r requirements.txt
```
Install GPU-related packages:
```bash
# Chamfer Distance and EMD
cd ./extensions/chamfer_dist
pip install .

cd ../emd
pip install .

# PointNet++
pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"

# GPU kNN
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
```

## Dataset
For pre-training and fine-tuning, please follow [DATASET.md](https://github.com/lulutang0608/Point-BERT/blob/master/DATASET.md) to install ModelNet40, ScanObjectNN, and ShapeNetPart datasets, referring to Point-BERT. Specially Put the unzip folder under `data/`.

The final directory structure should be:
```
│ICML2025-GAPrompt/
├──cfgs/
├──datasets/
├──data/
│   ├──ModelNet/
│   ├──ScanObjectNN/
├──...
```

## Parameter-efficient Fine-tuning

We provide commands to parameter-efficiently fine-tune the pre-trained backbones, taking Point-MAE checkpoints for examples.
```
# GAPrompt based on Point-MAE in ModelNet40
python main.py  --config  cfgs/gaprompt_modelnet.yaml  --ckpts  pretrained_bases/mae_base.pth --peft

# GAPrompt based on Point-MAE in scan_hardest
python main.py  --config  cfgs/gaprompt_scan_hardest.yaml  --ckpts  pretrained_bases/mae_base.pth --peft

# GAPrompt based on Point-MAE in scan_objbg
python main.py  --config  cfgs/gaprompt_scan_objbg.yaml  --ckpts  pretrained_bases/mae_base.pth --peft

# GAPrompt based on Point-MAE in scan_objonly
python main.py  --config  cfgs/gaprompt_scan_objonly.yaml  --ckpts  pretrained_bases/mae_base.pth --peft
```


## Citation
If you find our paper and code useful in your research, please consider giving a star and citation.
To do.
<!-- ```bash
@inproceedings{tang2024point,
  title={Point-PEFT: Parameter-efficient fine-tuning for 3D pre-trained models},
  author={Tang, Yiwen and Zhang, Ray and Guo, Zoey and Ma, Xianzheng and Zhao, Bin and Wang, Zhigang and Wang, Dong and Li, Xuelong},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={6},
  pages={5171--5179},
  year={2024}
}
``` -->

## Acknowledgement
This repo benefits from [Point-BERT](https://github.com/lulutang0608/Point-BERT), [Point-MAE](https://github.com/Pang-Yatian/Point-MAE), [ReCon](https://github.com/qizekun/ReCon), [PointGPT](https://github.com/CGuangyan-BIT/PointGPT), [PointFEMAE](https://github.com/zyh16143998882/AAAI24-PointFEMAE), [IDPT](https://github.com/zyh16143998882/ICCV23-IDPT), [DAPT](https://github.com/LMD0311/DAPT), and [Point-PEFT](https://github.com/Ivan-Tang-3D/Point-PEFT). Thanks for their wonderful works.

