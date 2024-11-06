# 바이낸스 비트코인 백테스트

## 개요
@collect_data.md 를 바탕으로 생성한 데이터를 활용하여 백테스트를 수행하고, 딥러닝을 통해 전략을 최적화합니다.

## 설치 및 실행

### 1. 필요 패키지 설치
```bash
pip install -r requirements.txt
```

### 2. 데이터베이스 준비
- @collect_data.md 의 데이터베이스 설정 및 데이터 수집이 완료되어 있어야 함
- market_data 테이블에 기술적 지표가 계산되어 있어야 함

### 3. 실행 방법
```bash
# 기본 백테스트 실행
python src/run_backtest.py

# 딥러닝 최적화 포함 실행
python src/run_backtest.py --optimize
```

## 시스템 구조

### 1. 전략 최적화 (src/optimizer.py)
- 딥러닝 기반 파라미터 최적화
- 전체 데이터 활용 학습
- 가중치 자동 저장 및 로드
- 멀티 타임프레임 분석
- 지표별 가중치 최적화
- 거래 빈도 및 안정성 고려

### 2. 전략 정의 (src/strategy.py)
- 6개 기술적 지표 통합 (MA, RSI, BB, MACD, Stochastic, ADX)
- 멀티 타임프레임 신호 생성 (1H, 4H, 1D)
- 지표별 가중치 기반 신호 생성
- ATR 기반 포지션 사이징
- 지표/타임프레임 On/Off 스위치

### 3. 백테스트 엔진 (src/backtest.py)
- 레버리지 3배 적용
- 롱/숏 포지션 지원
- 수수료 0.005% 반영
- 상세한 성과 분석
- 월별 수익 분석

## 주요 기능

### 1. 딥러닝 최적화
- 파라미터 자동 최적화
- 1000회 학습 진행
- 배치 사이즈 64
- 학습률 자동 조정
- 최적 파라미터 저장

### 2. 거래 전략
- 지표별 가중치 적용
- 타임프레임별 가중치 적용
- 동적 임계값 설정
- 리스크 관리
- 동적 포지션 사이징

### 3. 보상 함수
- 거래 빈도 중시 (월 20회 목표)
- 수익률 안정성 고려
- 승률 가중치 부여
- 최대 손실폭 제한
- 연속 거래 보너스

## 파일 구조
```
src/
├── optimizer.py    # 딥러닝 최적화
├── strategy.py     # 거래 전략
├── backtest.py     # 백테스트 엔진
└── run_backtest.py # 실행 스크립트

models/             # 학습 모델 저장
├── strategy_weights.pth
└── best_parameters.json
```

## 주요 파라미터

### 1. 전략 파라미터
- RSI: 매수(10-50), 매도(50-90)
- BB: 임계값(0.001-0.1)
- ADX: 임계값(10-50)
- 손절: 2-10%
- 익절: 3-20%
- 레버리지: 3배 고정

### 2. 학습 파라미터
- 에피소드: 1000회
- 배치 크기: 64
- 학습률: 0.0005
- 드롭아웃: 0.2
- 은닉층: 128 노드

### 3. 보상 가중치
- 거래 빈도: 35%
- 승률: 20%
- 수익률 안정성: 20%
- 총 수익률: 15%
- 최대 손실폭: 10%

## 주의사항
- 최소 50회 이상의 거래 필요
- 월 20회 이상 거래 권장
- 최대 손실폭 20% 제한
- 승률 55% 이상 목표
- 평균 수익률 0.5-2.0% 목표

## 향후 개선사항
- 시장 상황별 전략 분리
- 멀티 GPU 지원
- 앙상블 전략 추가
- 실시간 학습 구현
- 백테스트 시각화