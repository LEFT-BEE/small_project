---
title: "mnist_csv_img_recog.ipynb 해설"
---

### 간단하게 kaggle에서 mnist 손글씨 csv파일을 받아 이를 분류하는 프로젝트이다.

```
import pandas as pd
import numpy as np
from keras import layers
```

머신러닝 데이터처리에 필요한 라이브러리이다 pd와 np는 둘다 데이터처리를 하는데 있어 공통점이 있으나 numpy는 더욱 고차원 텐서의 연산 즉 계산하는데 이점이 있다면
pandas는 데이터 프레임을 처리하는데에 있어 중점이된다(온전히 나의 생각이다 틀릴 수도 있다.) 마지막으로 딥러닝 층을 만들기위해 keras의 layers를 가져온다.

```
test_data = pd.read_csv("/content/drive/MyDrive/Kaggle project/recognition_digit/test.csv")
train_data = pd.read_csv("/content/drive/MyDrive/Kaggle project/recognition_digit/train.csv")
print(train_data.shape)#42000개의 데이터를 가진 csv파일 train에는 label 이 있어 axis=1의 크기가 785이다
print(test_data.shape)


실행결과
-----
(42000, 785)
(28000, 784)
-----
```
google drive에 있는 csv파일을 읽기 위해 pd.read_csv 메소드를 이용하였다 이때 test와trani의 컬럼수가 다른데 이는 train에 label이 들어가 있기 때문이다 따라서

```
label = train_data.label
train_data = train_data.drop("label" , axis =1);
```
pd데이터는 이렇게나 쉽게 특정 컬럼을 다룰 수 있다 label을 따로 추출하여 train_data를 재정의 해준다

```
model = keras.models.Sequential([
    
    layers.Dense(input_dim = 784 , units = 512, activation="relu"),
    layers.Dense(256,activation="relu"),
    layers.Dense(10, activation="softmax")
])
```
모델을 만들었다 input값은 28 * 28 인 784이고 이후 fully connective network구조로 모델을 만들었다.

```
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
```
만든 모델을 compile해주는데 optimizer는 adam을 loss는 분류모델이기에 sparse_categorical_crossentropy를 사용하였다.

```
val_train  , val_label = train_data[40000:42000] , label[40000:42000]
model.fit(train_data , label , epochs = 10)#이거 왜 데이터가 1313개만 학습이 될까

-----

실행결과


Epoch 1/10
1313/1313 [==============================] - 8s 6ms/step - loss: 0.1303 - accuracy: 0.9656
Epoch 2/10
1313/1313 [==============================] - 8s 6ms/step - loss: 0.1226 - accuracy: 0.9680
Epoch 3/10
1313/1313 [==============================] - 8s 6ms/step - loss: 0.1162 - accuracy: 0.9705
Epoch 4/10
1313/1313 [==============================] - 8s 6ms/step - loss: 0.1035 - accuracy: 0.9737
Epoch 5/10
1313/1313 [==============================] - 8s 6ms/step - loss: 0.0988 - accuracy: 0.9743
Epoch 6/10
1313/1313 [==============================] - 8s 6ms/step - loss: 0.0883 - accuracy: 0.9777
Epoch 7/10
1313/1313 [==============================] - 8s 6ms/step - loss: 0.0783 - accuracy: 0.9797
Epoch 8/10
1313/1313 [==============================] - 8s 6ms/step - loss: 0.0887 - accuracy: 0.9796
Epoch 9/10
1313/1313 [==============================] - 8s 6ms/step - loss: 0.0778 - accuracy: 0.9817
Epoch 10/10
1313/1313 [==============================] - 8s 6ms/step - loss: 0.0687 - accuracy: 0.9839
<tensorflow.python.keras.callbacks.History at 0x7f3eed236c90>
-------
```
나중에 모델을 평가하기 위해 2000개의 데이터를 따로 빼주어 검증 데이터세트를 만들어준다. 그리고 fit을 이용해
10epochs만큼 반복 학습한다. 이상하게도 데이터가 1313개 밖에 학습이 되질않는다.....? 
이러면 overfitting의 문제점이 발생하겠지만 일단은 넘어가도록 하자. loss는 0.06이며 accuracy는 98%정도가 나온다 

```
loss, accuracy = [], []
for i in range(10):
    model.fit(val_train, val_label, epochs=1)
    loss.append((model.evaluate(val_train), val_label)[0])
    accuracy.append(model.evaluate(val_train, val_label)[1])
    
from matplotlib import pyplot as plt
y_value = accuracy
loss_value = loss
x_value = [1,2,3,4,5,6,7,8,9,10]

plt.plot(x_value , y_value , label = "accuracy")
plt.title("accuracy")
plt.show()
```
마지막으로 아까 빼놓았던 검증데이터셋을 통해 accuracy를 시각화 하였다.


![다운로드](https://user-images.githubusercontent.com/65720894/110230848-26840280-7f57-11eb-8506-68268177c8db.png)








