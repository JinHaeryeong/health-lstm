# health-lstm
만들고 있는 프로젝트에서 사용할 위험, 주의, 정상 라벨링을 위한 LSTM
----
LOSS
*pytorch로* 학습시킨 것의 Loss가 keras로 학습시켰을 때보다 높은 이유

둘이 loss를 계산하는 방식이 다르기 때문

## pytorch의 경우
<img width="1028" height="107" alt="image" src="https://github.com/user-attachments/assets/df9a677f-4b5b-421a-92b1-99be3408fc2e" />

$$\text{PyTorch Loss} = \frac{1}{N} \sum_{i=1}^{N} (\text{Sample Loss}_i \times \text{Class Weight}_i)$$

## keras의 경우
<img width="1695" height="121" alt="image" src="https://github.com/user-attachments/assets/86e37cf9-e73c-4511-bba5-29db32624996" />

$$\text{Keras Loss} \approx \frac{1}{N} \sum_{i=1}^{N} (\text{Sample Loss}_i)$$

keras는 단순히 sample loss(패널티가 1점이면 1점만 부여) 하는 방식이지만 pytorch는 틀렸을 때 sample loss * class weight(PyTorch Class Weights: tensor([15.8032,  0.7776,  0.6058]) 니까 위험이면 15.8배가 곱해지고 정상일땐 0.77 주의일땐 0.6058이 곱해지는거임

