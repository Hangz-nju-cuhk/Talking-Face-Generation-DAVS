# Talking Face Generation by Adversarially Disentangled Audio-Visual Representation

We propose `Disentangled Audio-Visual System (DAVS)` to address arbitrary-subject talking face generation in this work, which aims to synthesize a sequence of face images
that correspond to given speech semantics, conditioning on either an unconstrained speech audio or video.

[[Project]](https://liuziwei7.github.io/projects/TalkingFace) [[Paper]](https://arxiv.org/abs/1807.07860) [[Demo]](https://www.youtube.com/watch?v=-J2zANwdjcQ)

<img src='./misc/teaser.png' width=880>

## Requirements
* [PyTorch](https://pytorch.org/) 0.2.0
* [Opencv2](https://pytorch.org/)

## Getting started
* Download the pre-trained model [checkpoint](https://drive.google.com/file/d/1WltJlIWhG0xT-HSAFUh19F5yEkIfEW5m/view?usp=sharing)
``` bash
Create the default folder "checkpoints" and put the checkpoint in it or get the CHECKPOINT_PATH
``` 

* Samples for testing can be found in this [folder](https://drive.google.com/open?id=1ykjOZwwFfyP2V1vdUVsm2v4r1QSM-uxa).This is a pre-processed sample from the [Voxceleb](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/) Dataset. 

* Run the testing script for generation from video:

``` bash
python test_all.py  --test_root './0572_0019_0003/video' --test_type 'video' --test_audio_video_length 99 --test_resume_path CHECKPOINT_PATH 
```
* Run the testing script for generation from audio:
``` bash
python test_all.py  --test_root './0572_0019_0003/audio' --test_type 'audio' --test_audio_video_length 99 --test_resume_path CHECKPOINT_PATH 
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
