## 목적

WIDER 데이터 셋의 2차원 영상으로부터 가상의 깊이 정보를 생성  
<br/>


## 파일 설명
- generator.py: 깊이 정보 생성을 위한 파이썬 코드
- generator.cfg: generator.py 코드를 실행시키는 데 필요한 설정 값
<br/>

## Configuration File(generator.cfg) 설명

- source_path: WIDER 데이터셋 디렉터리
- dest_path: 생성된 깊이 정보를 저장할 디렉터리
<br/>
  
## 필요 라이브러리
```
PyYAML >= 6.0
tqdm >= 4.62.3
torch >= 1.7.0
opencv-python >= 4.5.1.48
skimage >= 0.18.3
matplotlib >= 3.4.1
numpy >= 1.21.3
timm >= 0.4.12
```


## 프로그램 실행
```
python generator.py
```
