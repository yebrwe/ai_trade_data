# 실시간 거래 모니터링 시스템

## 개요
이 문서는 실시간 거래 모니터링 시스템의 구현 사항을 설명합니다. 시스템은 데이터베이스에서 시장 데이터를 주기적으로 조회하여 거래 신호를 모니터링하고 표시합니다.

## 주요 기능
1. 전체 거래 내역 표시 (첫 실행 시)
2. 새로운 거래 발생 시 실시간 알림
3. 타임프레임별 최신 데이터 상태 모니터링
4. 수익/손실 컬러 표시

## 데이터베이스 설정 
```python
db_config = {
'dbname': 'binance_futures',
'user': 'postgres',
'password': '1234',
'host': 'localhost',
'port': '5432'
}
```
## 데이터 조회 쿼리
```sql
SELECT
m.timestamp,
m.open, m.high, m.low, m.close, m.volume,
m.ma_5, m.ma_10, m.ma_20, m.ma_60, m.ma_120, m.ma_200,
m.ema_5, m.ema_10, m.ema_20, m.ema_60, m.ema_120, m.ema_200,
m.bb_upper, m.bb_middle, m.bb_lower,
m.rsi,
m.macd, m.macd_signal, m.macd_hist,
m.stoch_k, m.stoch_d,
m.atr,
m.obv,
m.dmi_plus, m.dmi_minus, m.adx
FROM market_data m
WHERE m.symbol = 'BTCUSDT'
AND m.timeframe = %s
AND CASE
WHEN %s IS NOT NULL THEN m.timestamp > %s
ELSE TRUE
END
ORDER BY m.timestamp
```

## 구현 세부사항

### 1. 시간 처리
- UTC 시간을 한국 시간(KST)으로 변환: `+ timedelta(hours=9)`
- 타임스탬프 출력 형식: `%Y-%m-%d %H:%M:%S`

### 2. 데이터 조회 최적화
- 첫 실행: 전체 데이터 조회
- 이후 실행: 마지막 타임스탬프 이후의 데이터만 조회
- 각 타임프레임('1h', '4h', '1d')별 마지막 타임스탬프 관리

### 3. 출력 형식
- 진입: {timestamp} - {type} @ ${price:,.2f}
- 청산: {timestamp} - PnL: ${pnl:,.2f} ({return_pct:.2f}%)
- 수익은 초록색(Fore.GREEN), 손실은 빨간색(Fore.RED)으로 표시

### 4. 로깅 설정
```python
logging.basicConfig(
level=logging.INFO,
format='%(asctime)s - %(levelname)s - %(message)s',
encoding='cp949'
)
```

### 5. 업데이트 주기
- 기본 업데이트 간격: 60초
- 오류 발생 시 대기 시간: 5초

## 필요한 패키지
pandas
numpy
psycopg2-binary
torch
scikit-learn
matplotlib
sqlalchemy>=1.4.0
colorama>=0.4.6

## 주의사항
1. 데이터베이스 연결이 끊어질 경우 자동 재연결 시도
2. 타임스탬프는 KST 기준으로 표시
3. 불필요한 경고 메시지 출력 제거
4. 컬러 출력 후 반드시 Style.RESET_ALL 적용

## 실행 방법
```bash
python src/run_realtime_monitor.py --interval 60
```

## 종료
- Ctrl+C로 안전하게 종료 가능
- 종료 시 "모니터링을 종료합니다." 메시지 출력
