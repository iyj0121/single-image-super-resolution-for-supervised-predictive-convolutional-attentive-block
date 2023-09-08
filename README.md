**About PyTorch 1.2.0**
  * Now the master branch supports PyTorch 1.2.0 by default.

# EDSR-sspcab

**About PyTorch 1.1.0**
  * There have been minor changes with the 1.1.0 update. Now we support PyTorch 1.1.0 by default, and please use the legacy branch if you prefer older version.

<img width="712" alt="스크린샷 2023-04-09 오후 1 01 52" src="https://user-images.githubusercontent.com/90498398/236676450-5e7d3073-e2b0-47da-bc13-8187581af0e2.png">
<img width="298" alt="스크린샷 2023-05-03 오후 8 07 38" src="https://user-images.githubusercontent.com/90498398/236676461-02a1bc78-14c7-4144-852f-4886d3ffae60.png">
https://github.com/yulunzhang/RCAN/raw/master/Figs/RCAN.PNG

This study aims to restore resolution to improve image quality. The entire framework consists of two models, vdsr and mask-attention. If SR-reconstruction works well using two models, the loss value is low, and if SR-reconstruction does not work well, the loss value is high, so it is a model that focuses on hard samples. This allows us to focus on samples that are difficult to rebuild, and is a kind of ensemble (mining technique that focuses on hard samples) as an opportunity to make more use of local details.
In addition, this study placed restrictions on convolution using masks when learning models. Through this, when testing the model, it can be effectively applied to reconstruct the image.

Model 을 2 개 사용하여 SR-reconstruction 이 잘 되면 loss 값이 낮고 SR-reconstruction 이 잘 안되면 loss 값이 높으므로 하드 샘플에 집중하는 모델이다. 이를 통하여 재구축이 힘든 샘플에 집중할 수 있게 되고,localdetail을 좀 더 살릴 수 있는 계기로(하드 샘플에 집중하는 마이닝 기법) 앙상블의 일종이다.
또한 이 연구는 모델 학습을 할 때, 컨볼루션에 마스크를 사용하여 재한을 두었다. 이를 통하여 모델을 테스트 할 때, 이미지 재구성이 하는데 효과적으로 적용할 수 있다.

[1] Bee Lim, Sanghyun Son, Heewon Kim, Seungjun Nah, and Kyoung Mu Lee, **"Enhanced Deep Residual Networks for Single Image Super-Resolution,"** <i>2nd NTIRE: New Trends in Image Restoration and Enhancement workshop and challenge on image super-resolution in conjunction with **CVPR 2017**. </i> [[PDF](http://openaccess.thecvf.com/content_cvpr_2017_workshops/w12/papers/Lim_Enhanced_Deep_Residual_CVPR_2017_paper.pdf)] [[arXiv](https://arxiv.org/abs/1707.02921)] [[Slide](https://cv.snu.ac.kr/research/EDSR/Presentation_v3(release).pptx)]
```
@InProceedings{Lim_2017_CVPR_Workshops,
  author = {Lim, Bee and Son, Sanghyun and Kim, Heewon and Nah, Seungjun and Lee, Kyoung Mu},
  title = {Enhanced Deep Residual Networks for Single Image Super-Resolution},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
  month = {July},
  year = {2017}
}
@inproceedings{Ristea-CVPR-2022,
  title={Self-Supervised Predictive Convolutional Attentive Block for Anomaly Detection},
  author={Ristea, Nicolae-Catalin and Madan, Neelu and Ionescu, Radu Tudor and Nasrollahi, Kamal and Khan, Fahad Shahbaz and Moeslund, Thomas B and Shah, Mubarak},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2022}
}
```

## Dependencies
* Python 3.6
* PyTorch >= 1.0.0
* numpy
* skimage
* **imageio**
* matplotlib
* tqdm
* cv2 >= 3.xx (Only if you want to use video input/output)

## Code
Clone this repository into any place you want.
```bash
git@git.ajou.ac.kr:iyj0121/single-image-super-resolution-for-supervised-predictive-convolutional-attentive-block.git
cd single-image-super-resolution-for-supervised-predictive-convolutional-attentive-block
```
You can evaluate your models with widely-used benchmark datasets:

[Set5 - Bevilacqua et al. BMVC 2012](http://people.rennes.inria.fr/Aline.Roumy/results/SR_BMVC12.html),

[Set14 - Zeyde et al. LNCS 2010](https://sites.google.com/site/romanzeyde/research-interests),

[B100 - Martin et al. ICCV 2001](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/),

[Urban100 - Huang et al. CVPR 2015](https://sites.google.com/site/jbhuang0604/publications/struct_sr).
