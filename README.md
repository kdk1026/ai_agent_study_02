# Python AI Agent Study
#### [강의] (https://youtu.be/O93x9JvDQd0?si=Df8uZxTuQC9BWpV2)

## 라이브러리
```
pip install -qU langchain_community langgraph langchain-google-genai neo4j-driver python-dotenv
```

<br />

## 가상환경 생성
```
python -m venv venv
```

<br />

## 가상환경 활성화
```
(Windows)
venv\Scripts\activate

(Linux/Mac)
source venv/bin/activate
```

<br />

## VS Code 인터프리터 설정 (자동으로 가상환경 켜기)
1. `Ctrl + Shift + P` -> `Python: Select Interpreter` 선택
2. 리스트에서 `./venv/Scripts/activate` 선택
3. 터미널 새로 열기 `Ctrl + Shift + ~`

<br />

## 설치된 라이브러리 목록 추출
#### - Git에 올리는 경우
```
pip freeze > requirements.txt
```

<br />

## 라이브러리 한 번에 설치
#### - Git에서 내려받는 경우
```
pip install -r requirements.txt
```

<br />

## Editable 모드로 설치
```
pip install -e .
```