# Talking Face Generation by Adversarially Disentangled Audio-Visual Representation

We propose `Disentangled Audio-Visual System (DAVS)` to address arbitrary-subject talking face generation in this work, which aims to synthesize a sequence of face images
that correspond to given speech semantics, conditioning on either an unconstrained speech audio or video.

[[Project]](https://liuziwei7.github.io/projects/TalkingFace) [[Paper]](https://arxiv.org/abs/1807.07860) [[Demo]](https://www.youtube.com/watch?v=-J2zANwdjcQ)

<img src='./misc/teaser.png' width=880>

## Requirements
* [PyTorch](https://pytorch.org/) < 0.4.0
* [opencv2](https://opencv.org/releases.html)

## Getting started
* Download the pre-trained model [checkpoint](https://drive.google.com/file/d/1WltJlIWhG0xT-HSAFUh19F5yEkIfEW5m/view?usp=sharing)
``` bash
Create the default folder "checkpoints" and put the checkpoint in it or get the CHECKPOINT_PATH
``` 

* Samples for testing can be found in this folder named [0572_0019_0003](https://drive.google.com/open?id=1ykjOZwwFfyP2V1vdUVsm2v4r1QSM-uxa). This is a pre-processed sample from the [Voxceleb](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/) Dataset. 

* Run the testing script to generate videos from video:

``` bash
python test_all.py  --test_root './0572_0019_0003/video' --test_type 'video' --test_audio_video_length 99 --test_resume_path CHECKPOINT_PATH 
```
* Run the testing script to generate videos from audio:
``` bash
python test_all.py  --test_root './0572_0019_0003/audio' --test_type 'audio' --test_audio_video_length 99 --test_resume_path CHECKPOINT_PATH 
```

## Sample Results
* Talking Effect on Human Characters
<img src='./misc/demo_human.gif' width=640>

* Talking Effect on Non-human Characters (Trained on Human Faces Only)
<img src='./misc/demo_nonhuman.gif' width=640>

## Create more samples

The face detection tool used in the demo videos can be found at [RSA](https://github.com/sciencefans/RSA-for-object-detection). It will return a Matfile with 5 key point locations in a row for each image. The key points for face alignement we used are the two for the eyes and the average point of the corners of the mouth. 

With each image's PATH and the face POINTS, you can find our way of face alignment at `preprocess/face_align.py`.

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

## Acknowledgement
The structure of this code is borrowed from [pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
