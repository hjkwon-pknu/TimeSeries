### TimeSeries
### GA_GRU, GA_LSTM은 GA과정이 오래걸리는 점을 감안해서 따로 분리한 코드입니다.
### 위 2개의 코드를 실행하시면 GA 값이 추출되고 그 값은 json으로 저장됩니다.
### 다음으로, main.py를 실행하신다면 json파일을 가져와서 실행됩니다.
### 이러한 과정을 넣은 이유는 로컬 모델이 아닌 글로벌 모델로 발전할 시에 보다 쉽게 비교하기 위해서 설정했습니다.
### 추가로 pyESN 파일은 import 과정에서 오류가 생길 떄가 많아서 명시했습니다.
### 감사합니다.
