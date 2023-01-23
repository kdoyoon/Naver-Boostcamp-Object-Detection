![header](https://capsule-render.vercel.app/api?type=rect&color=gradient&text=재활용%20품목%20분류를%20위한%20Object%20Detection&fontSize=30)
<div align="left">
	<img src="https://img.shields.io/badge/Python-3776AB?style=flat&logo=Python&logoColor=white" />
	<img src="https://img.shields.io/badge/Pytorch-EE4C2C?style=flat&logo=Pytorch&logoColor=white" />
	<img src="https://img.shields.io/badge/OpenMMLab-181717?style=flat&logo=Github&logoColor=white" />
</div>


# Members
- **김도윤**  : 2 stage model(detector, feature extractor  위주의  실험  진행_faster, cascade, htc, etc.), ensemble(WBF),   
Hyperparameter tuning 
- **김윤호**  : Augmentation(Auto-augmentation, Mosaic, Multi-Scale)  실험, 2stage model(ATSS-Dyhead, cascade rcnn), StratifiedGroupKfold 구현
- **김종해**  : 1 stage model (RetinaNet, yolov7)  실험, StratifiedGroupKfold  구현, WBF실험
- **조재효**  : Augmentation(TTA, albumentation), 1stage model (yolov7), 2 stage model(cascade_swin_b),    
hyperparameter tuning, k-fold, ensemble(WBF)
- **허진녕**  : EDA, 1 stage model (yolov3, yolof, yolox)  실험, hyperparameter tuning(atss_dyhead), k-fold, ensemble(WBF)

&nbsp;

# 프로젝트 개요
> 잘 분리배출 된 쓰레기는 자원으로서 가치를 인정받아 재활용되지만, 잘못 분리배출 되면 그대로 폐기물로 분류되어 매립 또는 소각됩니다. 따라서 우리는 사진에서 쓰레기를 Detection하는 모델을 만들어 이러한 문제점을 해결해보고자 합니다.   
빠르고 정확한 분리수거를 통해 직접적으로 환경을 개선할 뿐만 아니라 아이들의 분리수거 교육 등에 활용되어 많은 사람들이 분리수거를 올바르게 행할 수 있도록 영향력을 줄 것입니다. 🌎 <sup>[[1]](#footnote_1)</sup>

&nbsp;

# 데이터셋 구조
```
├─ input
│	├─ code
│	│	└─ submision
│	└─ data
│		├─ train
│		│	├─ images
│		│	│		
│		│	└─ mask
│		│			
│		├─ valid
│		│	├─ images
│		│	│		
│		│	└─ mask
│		│			
│		└─ test
│			└─ images
│					
│
└─ mmdetection
	├─ configs
	  ├─ _base_
	  ├─ CV03
	  │	├─ ATSS/DyHead
	  │	├─ swin-base
	  │	└─ YOLOV7
	  └─ utils
```	


&nbsp;

# 프로젝트 수행 절차
<h3> 1. 데이터 EDA  </h3>
<h3> 2. 모델 및 Augmentation 기법 탐색  </h3>
<h3> 3. Baseline 모델 선정 및 최적 Augmentation 기법 선정  </h3>
<h3> 4. Hyperparameter Tuning  </h3>
<h3> 5. Model Ensemble(WBF)  </h3>

&nbsp;

# 문제정의
<h3> 1. 데이터의 불균형   </h3>  
<img width="850" alt="image" src="https://user-images.githubusercontent.com/93971443/213986454-de1e2d96-6e74-40ed-95ab-a7a333b06dc3.png">

- EDA 결과 심한 data 불균형 관찰됨
- 적은 label을 제대로 detect 하지 못할 경우 key problem 설정 후 이를 정복하는 방식으로 접근
<h3> 2. 균일한 Valid Dataset 구성   </h3>

- 한 이미지 내에 여러 클래스의 annotation이 존재하여 일반적인 data split으로 접근하면 안됨
- Stratified Group K-fold를 활용하여 균일한 validation set 구성


&nbsp;

# Advanced Techniques

<h3> 1. Model Ensemble   </h3>  

- WBF(Weighted Box Fusion)
	- bbox가 너무 많이 그려져 결과를 분석하는데 어려움을 겪음
	- soft-nms 사용하여 성능 향상을 이끌어내었으나 nms 방식은 bbox를 일부 제거하므로 정보 손실이 불가피하였음
	- 여러 개의 bbox를 앙상블하는 WBF를 사용하여 정보 손실을 최소화하였고 큰 성능 향상을 이끌어 냄


# Reference
<a name="footnote_1">[1]</a>  : AIstage
