# 바이낸스 선물 데이터 수집

## 데이터 수집 목표

- 바이낸스 선물 데이터 수집
  - 과거 데이터 수집 (3년치)
  - 타임프레임별 데이터 집계
    - 1분봉
    - 3분봉
    - 5분봉
    - 15분봉
    - 30분봉
    - 1시간봉
    - 4시간봉
    - 일봉

## 데이터 수집 방법

- 바이낸스 REST API 사용
  - 과거 데이터: /fapi/v1/klines 엔드포인트
  - 각 API 호출당 최대 1000개 데이터
  - API 호출 제한 고려 (0.5초 간격)
  - 벌크 데이터 처리 (10,000개 단위)

## 데이터 정형화

- 데이터 정형화 방법
  - REST API 응답 데이터 파싱
  - 시계열 데이터 포맷 통일 (UTC 기준)
  - 각 타임프레임별 데이터 집계
  - 기술지표 계산
  - 데이터 타입 변환 (Decimal to Float)
  - NaN/Inf 값 처리

## 데이터베이스 구조

### market_data 테이블 (통합 테이블)
- timestamp (TIMESTAMPTZ) - NOT NULL
- symbol (VARCHAR(20)) - NOT NULL
- timeframe (VARCHAR(10)) - NOT NULL
- open (DECIMAL) - NOT NULL
- high (DECIMAL) - NOT NULL
- low (DECIMAL) - NOT NULL
- close (DECIMAL) - NOT NULL
- volume (DECIMAL) - NOT NULL
- 기술지표 컬럼들:
  - ma_5, ma_10, ma_20, ma_60, ma_120, ma_200 (DECIMAL)
  - ema_5, ema_10, ema_20, ema_60, ema_120, ema_200 (DECIMAL)
  - bb_upper, bb_middle, bb_lower (DECIMAL)
  - rsi (DECIMAL)
  - macd, macd_signal, macd_hist (DECIMAL)
  - stoch_k, stoch_d (DECIMAL)
  - atr (DECIMAL)
  - obv (DECIMAL)
  - dmi_plus, dmi_minus, adx (DECIMAL)
- created_at (TIMESTAMPTZ) - DEFAULT CURRENT_TIMESTAMP
- UNIQUE (timestamp, symbol, timeframe)

### 인덱스
- idx_market_data_timestamp (timestamp)
- idx_market_data_symbol (symbol)
- idx_market_data_timeframe (timeframe)
- idx_market_data_symbol_timeframe (symbol, timeframe)

## 데이터 수집 프로세스

### 1. 초기 데이터 수집
- 각 타임프레임별로 순차적 처리
- 각 심볼에 대해:
  1. 마지막 저장 데이터 확인
  2. 데이터가 없는 경우:
     - 3년치 과거 데이터 수집
     - 1000개 단위로 API 호출
     - 10,000개 단위로 벌크 저장
  3. 데이터가 있는 경우:
     - 마지막 저장 시점부터 현재까지 데이터 수집
     - 누락 데이터 벌크 저장

### 2. 실시간 데이터 갱신
- 1분 간격으로 최신 데이터 확인
- 각 심볼/타임프레임별 마지막 데이터 이후의 신규 데이터만 수집
- 중복 방지를 위한 UNIQUE 제약조건 활용
- 신규 데이터 벌크 저장

### 3. 기술지표 계산
- OHLCV 데이터 저장 후 자동 계산
- 계산 대상 지표:
  - 이동평균선 (MA, EMA)
  - 볼린저 밴드
  - RSI
  - MACD
  - Stochastic
  - ATR
  - OBV
  - DMI/ADX
- 최근 10개 데이터에 대해서만 기술지표 업데이트
- 데이터 타입 변환 및 예외 처리
  - Decimal to Float 변환
  - 무한값(inf) 처리
  - NULL 값 처리

## 설치 및 실행

### 1. 필수 패키지 설치
```bash
pip install requests pandas numpy psycopg2-binary
```

### 2. 데이터베이스 설정
- PostgreSQL 설치
- 데이터베이스 생성 및 스키마 적용
  - 사용자: postgres
  - 비밀번호: 1234
  - 데이터베이스명: binance_futures
  - 호스트: localhost
  - 포트: 5432
  - 인코딩: SQL_ASCII

### 3. 프로그램 실행
```bash
python src/main.py
```

## 에러 처리

- API 호출 실패 시 재시도
- 데이터베이스 연결 오류 처리
- 기술지표 계산 오류 처리
- 인코딩 관련 오류 처리 (Windows 환경 고려)
- 데이터 타입 변환 오류 처리
- 트랜잭션 롤백 처리

## 모니터링

- 데이터 수집 진행 상황 로깅
- 타임프레임별 수집 완료 상태 표시
- 에러 발생 시 상세 로그 기록
- 벌크 저장 상태 모니터링
- 기술지표 계산 진행 상황 표시