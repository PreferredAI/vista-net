# VistaNet

This is the code for the paper:

**[VistaNet: Visual Aspect Attention Network for Multimodal Sentiment Analysis](https://drive.google.com/file/d/12d8SZiNeKFgIGmO5VHSrZV2jkgwYZpNp)**
<br>
[Quoc-Tuan Truong](http://www.qttruong.com/) and [Hady W. Lauw](http://www.hadylauw.com/)
<br>
Presented at [AAAI 2019](https://aaai.org/Conferences/AAAI-19/)

We provide:

- Code to train and evaluate the model
- [Data](https://smu-my.sharepoint.com/:f:/g/personal/hadywlauw_smu_edu_sg/ErrrZOQqAEhPomyExrxMGbUBqpmvmlaj1pM7xnAed5BCNQ) used for the experiments

If you find the code and data useful in your research, please cite:

```
@inproceedings{truong2019vistanet,
  title={Vistanet: Visual aspect attention network for multimodal sentiment analysis},
  author={Truong, Quoc-Tuan and Lauw, Hady W},
  booktitle={Proceedings of the AAAI conference on artificial intelligence},
  volume={33},
  number={01},
  pages={305--312},
  year={2019}
}
```

## Requirements

- Python 3
- Tensorflow >=1.12,<2.0
- Tqdm
- [GloVe](https://nlp.stanford.edu/projects/glove/) word embeddings

## How to run

1. Make sure the data is ready. Run script to pre-process the data:
```bash
python data_preprocess.py
```

2. Train `VistaNet`:
```bash
python train.py --hidden_dim 50 --att_dim 100 --num_images 3 --batch_size 32 --learning_rate 0.001 --num_epochs 20
```

## Contact
Questions and discussion are welcome: www.qttruong.com
