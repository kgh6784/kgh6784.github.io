## RNN

CNN이 이미지 구역 별로 같은 weight를 공유한다면, RNN은 `시간` 별로 같은 weight를 공유한다.

#### First-order System

case 1) 현재 시간의 상태가 이전 시간의 상태와 관련이 있다고 가정 (only 이전 상태)

- x_t = f(x_{t-1})

![image-20231226132355695](C:\Users\User\AppData\Roaming\Typora\typora-user-images\image-20231226132355695.png)

초기 조건만 주어지면 외부 입력 없이 잘 돌아가는 autonomous system이다.



case 2) 현재 시간의 상태가 이전 시간의 상태와 현재의 입력에 관계가 있는 경우

- x_t = f(x_{t-1}, u_t)

autonomous system은 아니다. 함수는 input을 두 개 필요로 한다.



#### State-Space Model

모든 시간 t에서 모든 상태 x_t를 관측할 수 없는 경우도 있다. 일부만 관측가능할 수 있다. 관측가능한 변수들의 모음을 따로 만들어 주어야 한다. (y_t)

- 관측 가능한 상태의 모음 : 출력 y_t = h(x_t)

- 어떤 시스템을 해석하기 위한 3요소 : 입력(u), 상태(x), 출력(y)

![image-20231226134034601](C:\Users\User\AppData\Roaming\Typora\typora-user-images\image-20231226134034601.png)

- x_t는 이전까지의 상태와 이전까지의 입력을 대표하여 최대한 상세하게 표현할 수 있어야 한다.



원래 풀고 싶었던 문제는 이전의 모든 input을 이용하여 함수를 풀어야 하는데, 이전의 상태만(`First-order Markov Model`) 고려해서 문제를 푼다. 이전의 정보들은 hidden state를 거쳐서 대신해서 정보를 얻는다. 



#### RNN Problem types

`Many-to-many`,  `Many-to-one` , `One-to-many`



## GRU / LSTM

기존의 RNN의 문제점 : exploding or vanishing gradient

- RNN 구조에서는 W_xx가 계속 곱해지게 된다. Gradient clipping을 통해 이를 해결할 수 있지만 근본적인 해결책이 아니다.

### LSTM(Long short-term memory)

Graident flow를 제어할 수 있는 벨브 역할을 한다고 보면 된다. 

적당히 이용할 정보와 버릴 정보를 종합한다. 



![Long Short-Term Memory (LSTM) 이해하기](https://t1.daumcdn.net/cfile/tistory/9982923F5ACB86A10E)

![Long Short-Term Memory (LSTM) 이해하기](https://t1.daumcdn.net/cfile/tistory/995E00425ACB86A018)

GRU는 lstm의 간소화 버전