train.py를 실행하면 checkpoint 유무에 따라 학습이 진행되거나 시를 생성합니다.

또는 test.py를 실행할 수도 있습니다. models/model.h5가 있거나 checkpoints 디렉토리 안에 체크포인트가 있어야 합니다.

현재 h5 형태의 파일을 읽어올 때 버그가 있어서 test.py는 checkpoint로 대체했습니다.