---
layout: single
title:  "DETR & Deformable DETR"
categories: deeplearning
tag: ['od', 'vision', 'vit']
toc: true
use_math: true
typora-root-url: ../ 
---



 

 ## x. 코드리뷰



> Model은 `backbone`, `deformable_detr`, `deformable_transformer`, `matcher`, `position_encoding`, `segmention`으로 구성

### backbone.py

- torchvision에서 제공해주는 `IntermediateLayerGetter` 클래스 : 모델의 중간 레이어의 출력을 가져오는데 도움을 주는 것이 주요 목적
  - 기존에는 nn.Sequential을 통해서 코드를 수정해줬었는데, 이 클래스를 이용해주면 가지고 올 레이어만 정의해주고 인스턴스를 생성하면 되는 것 같다.

- `FrozenBatchNorm2d` 클래스
  - 2D batch norm을 수정하여 w와 b를 고정하여, 훈련 시 파라미터가 업데이트 되지 않도록. Pretrained feature에서 feature를 뽑을 때 사용함.
  - 어짜피 고정된 w와 b를 사용하는거면 `num_batches_tracked` 파라미터를 제거하여 충돌을 방지.

- `Backbone` 클래스
  - `backbone.py`에서 BackboneBase 클래스를 상속 받은 클래스
  - 어느 레이어까지를 중간 layer로 설정할지, backbone을 추가적으로 훈련 시킬지를 결정
  - <font color = 'green'>NestedTensor가 모델의 feature map과 해당 위치 정보를 통합하는 방법이라는데 잘 모르겠음</font>

#### deformable_transformer.py

- 사용할 transformer를 만드는 모듈
- 

#### segmentation.py

- `DETRsegm` 클래스
  - `__init__` 메서드
    - `detr` : backbone + transformer
    - `bbox_attention` : bbox attention을 수행
    - `mask_head` : segmentation을 수행
  - `forward` 메서드
    - 

#### dformable_detr.py

- `DeformableTransformer` 클래스. 

  - `dec_n_points`와 `enc_n_points`가 있음. 왜 encoder에도 있지? detection할 포인트가 아닌가? 아 deformable이라서 이를 반영하여 선택할 feature들인가보구나.
  - `two_stage_num_proposals` : 여기에서 detection할 bbox 기본 설정을 하는 듯. 기본적으로 300개(DETR에서는 100개 였음)
  - `_reset_parameters` 함수: 모델 파라미터의 초기화를 수행한다.
  - `get_proposal_pos_embed` 함수 : 주어진 proposals에 대한 위치 임베딩(position embedding)을 계산
    - <font color = 'green'>저기서 proposal 인자가 feature인지 제안 영역인지</font>
    - 
  - `gen_encoder_output_proposals` 함수 : 인코더(encoder)의 출력 `memory`, `memory_padding_mask`, 및 `spatial_shapes`를 기반으로 제안된 물체 영역(proposals)에 대한 정보인 `output_memory`와 `output_proposals`를 생성
    - <font color = 'green'>일단 나중에.. 막 중요한 코드 부분 아님</font>

  



이미지의 feature들 간의 유사도를 계산해서 차이를 최소화 시켜야돼. 보통 mse loss를 쓰잖아.  근데 ce loss가 괜찮을 경우가 있어? 왜냐하면 성능이 잘 나왔거든. 이미지에서 차이를 최소화 시킬 부분이 배경 영역은 걍 0 픽셀 값인데 너무 많을 경우에 이게 의미가 있을까? 

ChatGPT

ChatGPT

이미지의 특징들 간의 유사도를 계산하고 차이를 최소화하기 위해 일반적으로 Mean Squared Error (MSE) 손실 함수를 사용하는 것은 맞습니다. 그런데 Cross-Entropy (CE) 손실 함수가 더 나은 성능을 보인 경우에 대해서 설명해보겠습니다.

1. **MSE Loss의 기본 원리**: MSE는 예측값과 실제값 간의 평균 제곱 오차를 계산합니다. 이는 연속적인 값에 대한 회귀 문제에서 널리 사용됩니다. 예를 들어, 이미지 복원이나 초해상도에서 이 손실 함수가 많이 사용됩니다.
2. **CE Loss의 기본 원리**: CE 손실은 분류 문제에 주로 사용됩니다. 각 클래스에 대한 예측 확률을 실제 레이블과 비교하여 손실을 계산합니다. 이는 분류 문제에서 각 클래스에 대한 예측의 '불확실성'을 잘 측정합니다.
3. **CE Loss 사용 시점**: CE 손실이 유용한 경우는 주로 이미지의 특정 부분(예: 객체)을 분류하거나, 픽셀 단위의 분류 문제(예: 이미지 분할)에 있습니다. 이 경우, CE 손실은 각 픽셀이 특정 클래스(예: 객체, 배경)에 속하는 확률을 효과적으로 모델링할 수 있습니다.
4. **배경 영역의 처리**: 당신이 언급한 배경 영역이 0 픽셀 값으로 많은 부분을 차지하는 경우, MSE는 이 영역에 대해 매우 낮은 오차를 계산할 것입니다. 이는 모델이 배경에 대해 '과도하게' 학습되는 것을 의미할 수 있습니다. 반면, CE 손실은 배경과 객체를 더 명확하게 구분하는 데 도움이 될 수 있습니다, 특히 각 픽셀이 특정 클래스에 속하는 확률을 계산할 때 유용합니다.
5. **성능 향상의 이유**: CE 손실 함수가 더 나은 성능을 보인 이유는, 이 함수가 문제의 특성(분류, 객체 인식, 분할 등)과 더 잘 맞았을 수 있기 때문입니다. CE는 클래스 간의 불균형이 큰 경우에도 잘 작동하며, 모델이 더 중요한 특징에 집중하도록 유도할 수 있습니다.

요약하자면, CE 손실 함수가 MSE 대비 더 나은 성능을 보인 경우는 이미지의 특정 부분(예: 객체와 배경)을 더 명확하게 구분하거나, 클래스 간 불균형을 다루는 문제에서 발생할 수 있습니다. 배경이 0 픽셀 값으로 큰 부분을 차지하는 경우, CE는 이를 효과적으로 처리하고 모델이 중요한 특징에 더 집중하도록 할 수 있습니다.





![User](https://lh3.googleusercontent.com/a/ALm5wu1ZHa5Ut9IYrjZVWamWdpB1mtvH0PTk2n8oRwEQ=s96-c)

You

위를 수식 기반으로 설명해줄 수 있어?

ChatGPT

ChatGPT

물론입니다. 수식을 기반으로 하여 Mean Squared Error (MSE) 손실과 Cross-Entropy (CE) 손실을 설명하겠습니다.

### Mean Squared Error (MSE) 손실

MSE 손실은 다음과 같이 계산됩니다:

MSE=1�∑�=1�(��−�^�)2MSE=*N*1∑*i*=1*N*(*y**i*−*y*^*i*)2

여기서:

- �*N*은 데이터 포인트의 수입니다.
- ��*y**i*는 �*i*번째 데이터 포인트의 실제 값입니다.
- �^�*y*^*i*는 �*i*번째 데이터 포인트의 예측 값입니다.

MSE는 예측 값과 실제 값 간의 차이의 제곱을 평균내어 계산합니다. 이는 연속적인 값을 가진 데이터(예: 이미지 복원, 초해상도)에 주로 사용됩니다.

### Cross-Entropy (CE) 손실

CE 손실은 주로 분류 문제에 사용되며, 다음과 같이 계산됩니다:

CE=−∑�=1���log⁡(�^�)CE=−∑*i*=1*N**y**i*log(*y*^*i*)

여기서:

- �*N*은 클래스의 수입니다.
- ��*y**i*는 실제 레이블을 원-핫 인코딩한 벡터입니다.
- �^�*y*^*i*는 모델이 예측한 확률 분포입니다.

CE 손실은 실제 레이블에 해당하는 클래스의 예측 확률의 로그를 취한 값의 음수를 취하여 계산합니다. 이는 모델이 올바른 클래스에 대해 높은 확률을 예측하도록 유도합니다.

### 이미지 처리에서의 응용

이미지 처리에서 MSE 손실은 각 픽셀의 실제 값과 예측 값의 차이를 최소화하는 데 초점을 맞춥니다. 예를 들어, 이미지 복원에서는 실제 이미지와 복원된 이미지 간의 픽셀 단위 차이를 최소화하는 데 사용됩니다.

반면, CE 손실은 이미지 분류나 픽셀 단위의 분류(예: 이미지 분할)에서 유용합니다. 여기서는 각 픽셀이 특정 클래스(예: 객체, 배경)에 속할 확률을 예측하며, CE 손실은 이러한 확률을 최적화하는 데 도움을 줍니다.

### 결론

따라서, 이미지의 특정 부분(예: 객체 대 배경)을 분류하거나, 클래스 간의 불균형이 있는 경우 CE 손실이 MSE 손실보다 더 나은 성능을 보일 수 있습니다. CE 손실은 특정 클래스(예: 중요한 객체)에 대한 예측의 정확도를 높이는 데 중점을 둡니다.

 $x_t = f(x_{t-1})$