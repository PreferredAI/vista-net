# VistaNet

This is the code for the paper:

**[VistaNet: Visual Aspect Attention Network for Multimodal Sentiment Analysis](https://drive.google.com/file/d/12d8SZiNeKFgIGmO5VHSrZV2jkgwYZpNp)**
<br>
[Quoc-Tuan Truong](http://www.qttruong.info/) and [Hady W. Lauw](http://www.hadylauw.com/)
<br>
Presented at [AAAI 2019](https://aaai.org/Conferences/AAAI-19/)

We provide:

- Code to train and evaluate the model
- [Data](https://goo.gl/jgESp4) used for the experiments

If you find the code and data useful in your research, please cite:

```
@inproceedings{truong2019vistanet,
  title={VistaNet: Visual Aspect Attention Network for Multimodal Sentiment Analysis},
  author={Truong, Quoc-Tuan and Lauw, Hady W},
  publisher={AAAI Press},
  year={2019},
}
```

## Requirements

- Python 3
- Tensorflow >=1.12,<2.0
- Tqdm
- [GloVe](https://nlp.stanford.edu/projects/glove/) word embeddings

## How to run

1. Make sure [data](https://goo.gl/jgESp4) is ready. Run script to pre-process the data:
```bash
python data_preprocess.py
```

2. Train `VistaNet`:
```bash
python train.py --hidden_dim 50 --att_dim 100 --num_images 3 --batch_size 32 --learning_rate 0.001 --num_epochs 20
```

## Contact
Questions and discussion are welcome: www.qttruong.info