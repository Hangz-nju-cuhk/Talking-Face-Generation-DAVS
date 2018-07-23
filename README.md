# Talking Face Generation by Adversarially Disentangled Audio-Visual Representation

We propose `Disentangled Audio-Visual System (DAVS)` to address arbitrary-subject talking face generation in this work, which aims to synthesize a sequence of face images
that correspond to given speech semantics, conditioning on either an unconstrained speech audio or video.

[[Project]](https://liuziwei7.github.io/projects/TalkingFace) [[Paper]](https://arxiv.org/abs/1807.07860) [[Demo]](https://www.youtube.com/watch?v=-J2zANwdjcQ)

<img src='./misc/teaser.png' width=880>

## Requirements
* [PyTorch](https://pytorch.org/)

## Getting started
* Run the testing script:
``` bash
python test_all.py
```

## Sample Results
* Talking Effect on Human Characters
<img src='./misc/demo_human.gif' width=640>

* Talking Effect on Non-human Characters (Trained on Human Faces Only)
<img src='./misc/demo_nonhuman.gif' width=640>

## License and Citation
The use of this software is RESTRICTED to **non-commercial research and educational purposes**.

```
@article{zhou2018talking,
  title={Talking Face Generation by Adversarially Disentangled Audio-Visual Representation},
  author={Hang Zhou, Yu Liu, Ziwei Liu, Ping Luo, Xiaogang Wang},
  journal={arXiv preprint arXiv:1807.07860},
  year={2018}
}
```
