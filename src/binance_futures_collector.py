import requests
import json
import time
from datetime import datetime, timezone, timedelta
import psycopg2
from typing import Dict, List
import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass

@dataclass
class MarketData:
    timestamp: int
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float

class BinanceFuturesCollector:
    def __init__(self, db_config: Dict):
        self.base_url = "https://fapi.binance.com"
        try:
            self.db_conn = psycopg2.connect(**db_config)
        except Exception as e:
            raise Exception(f"데이터베이스 연결 실패: {e}")
        
        self.setup_logging()
        self.timeframes = ['1m', '3m', '5m', '15m', '30m', '1h', '4h', '1d']
        
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            encoding='cp949'
        )
        self.logger = logging.getLogger(__name__)

    def get_klines(self, symbol: str, interval: str, start_time: int = None, end_time: int = None, limit: int = 1000) -> List[MarketData]:
        """바이낸스 API에서 캔들스틱 데이터 조회"""
        endpoint = f"{self.base_url}/fapi/v1/klines"
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }
        
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time
        
        try:
            response = requests.get(endpoint, params=params)
            response.raise_for_status()
            data = response.json()
            
            return [
                MarketData(
                    timestamp=candle[0],
                    symbol=symbol,
                    open=float(candle[1]),
                    high=float(candle[2]),
                    low=float(candle[3]),
                    close=float(candle[4]),
                    volume=float(candle[5])
                )
                for candle in data
            ]
        except Exception as e:
            self.logger.error(f"데이터 조회 중 오류 발생: {e}")
            return []

    def save_market_data_bulk(self, market_data_list: List[MarketData], timeframe: str):
        """시장 데이터 벌크 저장"""
        if not market_data_list:
            return
            
        try:
            with self.db_conn.cursor() as cur:
                # 벌크 INSERT 쿼리 생성
                args = [(
                    datetime.fromtimestamp(data.timestamp / 1000),
                    data.symbol,
                    timeframe,
                    data.open,
                    data.high,
                    data.low,
                    data.close,
                    data.volume
                ) for data in market_data_list]
                
                # executemany 대신 더 효율적인 execute_values 사용
                from psycopg2.extras import execute_values
                
                execute_values(
                    cur,
                    """
                    INSERT INTO market_data (timestamp, symbol, timeframe, open, high, low, close, volume)
                    VALUES %s
                    ON CONFLICT (timestamp, symbol, timeframe) DO UPDATE 
                    SET open = EXCLUDED.open,
                        high = EXCLUDED.high,
                        low = EXCLUDED.low,
                        close = EXCLUDED.close,
                        volume = EXCLUDED.volume
                    """,
                    args,
                    page_size=10000  # 한 번에 처리할 레코드 수
                )
                self.db_conn.commit()
                
                # 기술지표 일괄 계산
                self.calculate_technical_indicators(market_data_list[0].symbol, timeframe)
                
        except Exception as e:
            self.logger.error(f"벌크 데이터 저장 중 오류 발생: {e}")
            self.db_conn.rollback()

    def collect_historical_data(self, symbols: List[str], days: int = 1095, timeframe: str = None):
        """과거 데이터 수집"""
        end_time = int(datetime.now().timestamp() * 1000)
        start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
        
        for symbol in symbols:
            timeframes_to_collect = [timeframe] if timeframe else self.timeframes
            
            for tf in timeframes_to_collect:
                self.logger.info(f"{symbol} {tf} 과거 {days}일 데이터 수집 시작...")
                current_start = start_time
                
                while current_start < end_time:
                    self.logger.info(f"{symbol} {tf} 데이터 수집 중... ({datetime.fromtimestamp(current_start/1000)})")
                    
                    candles = self.get_klines(
                        symbol=symbol,
                        interval=tf,
                        start_time=current_start,
                        limit=1000
                    )
                    
                    if not candles:
                        break
                    
                    # 벌크 저장 사용
                    self.save_market_data_bulk(candles, tf)
                    
                    # 다음 구간 설정
                    current_start = candles[-1].timestamp + 1
                    
                    # API 호출 제한 방지
                    time.sleep(0.5)
                
                self.logger.info(f"{symbol} {tf} 과거 데이터 수집 완료")

    def collect_recent_data(self, symbols: List[str]):
        """최신 데이터 수집 및 갱신"""
        while True:
            try:
                for symbol in symbols:
                    for timeframe in self.timeframes:
                        last_timestamp = self.get_last_timestamp(symbol, timeframe)
                        
                        if last_timestamp:
                            start_time = int((last_timestamp - timedelta(minutes=10)).timestamp() * 1000)
                            self.logger.info(f"{symbol} {timeframe} 최신 데이터 수집 중... (마지막 데이터: {last_timestamp})")
                            
                            candles = self.get_klines(
                                symbol=symbol,
                                interval=timeframe,
                                start_time=start_time
                            )
                            
                            if candles:
                                # 마지막 저장 시점 이후의 데이터만 필터링
                                new_candles = [
                                    candle for candle in candles 
                                    if candle.timestamp > int(last_timestamp.timestamp() * 1000)
                                ]
                                
                                if new_candles:
                                    self.save_market_data_bulk(new_candles, timeframe)
                                    self.logger.info(f"{symbol} {timeframe} {len(new_candles)}개 새로운 데이터 저장 완료")
                    
                    time.sleep(0.5)
                
                self.logger.info("다음 갱신까지 대기 중...")
                time.sleep(60)
                
            except Exception as e:
                self.logger.error(f"데이터 수집 중 오류 발생: {e}")
                time.sleep(5)

    def get_last_timestamp(self, symbol: str, timeframe: str) -> datetime:
        """마지막으로 저장된 데이터의 timestamp 조회"""
        try:
            with self.db_conn.cursor() as cur:
                cur.execute("""
                    SELECT timestamp 
                    FROM market_data 
                    WHERE symbol = %s AND timeframe = %s 
                    ORDER BY timestamp DESC 
                    LIMIT 1
                """, (symbol, timeframe))
                
                result = cur.fetchone()
                return result[0] if result else None
                
        except Exception as e:
            self.logger.error(f"마지막 timestamp 조회 중 오류 발생: {e}")
            return None

    def collect_data(self, symbols: List[str]):
        """전체 데이터 수집 프로세스"""
        try:
            # 각 타임프레임별로 모든 심볼 처리
            for timeframe in self.timeframes:
                self.logger.info(f"========== {timeframe} 데이터 수집 시작 ==========")
                
                for symbol in symbols:
                    # 1. 데이터 존재 여부 확인
                    last_timestamp = self.get_last_timestamp(symbol, timeframe)
                    
                    if last_timestamp is None:
                        # 2-1. 데이터가 없는 경우: 3년치 과거 데이터부터 수집
                        self.logger.info(f"{symbol} {timeframe} 과거 데이터 수집 시작...")
                        self.collect_historical_data([symbol], timeframe=timeframe)
                    else:
                        # 2-2. 데이터가 있는 경우: 마지막 저장 시점부터 현재까지 수집
                        self.logger.info(f"{symbol} {timeframe} 기존 데이터 발견. 마지막 데이터: {last_timestamp}")
                        current_time = datetime.now(timezone.utc)
                        
                        # 마지막 저장 시점부터 현재까지의 데이터 수집
                        start_time = int(last_timestamp.timestamp() * 1000)
                        end_time = int(current_time.timestamp() * 1000)
                        
                        self.logger.info(f"{symbol} {timeframe} 누락 데이터 수집 시작... ({last_timestamp} ~ {current_time})")
                        
                        current_start = start_time
                        while current_start < end_time:
                            candles = self.get_klines(
                                symbol=symbol,
                                interval=timeframe,
                                start_time=current_start,
                                limit=1000
                            )
                            
                            if not candles:
                                break
                                
                            for candle in candles:
                                self.save_market_data(candle, timeframe)
                            
                            current_start = candles[-1].timestamp + 1
                            time.sleep(0.5)  # API 호출 제한 방지
                        
                        self.logger.info(f"{symbol} {timeframe} 누락 데이터 수집 완료")
                
                self.logger.info(f"========== {timeframe} 데이터 수집 완료 ==========")
                
            # 3. 모든 과거/누락 데이터 수집 완료 후 실시간 데이터 수집 시작
            self.logger.info("모든 과거 데이터 수집 완료. 실시간 데이터 수집 시작...")
            self.collect_recent_data(symbols)
            
        except Exception as e:
            self.logger.error(f"데이터 수집 중 오류 발생: {e}")
            raise

    def save_market_data(self, market_data: MarketData, timeframe: str):
        """시장 데이터를 데이터베이스에 저장하고 기술지표 업데이트"""
        try:
            with self.db_conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO market_data (timestamp, symbol, timeframe, open, high, low, close, volume)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (timestamp, symbol, timeframe) DO UPDATE 
                    SET open = EXCLUDED.open,
                        high = EXCLUDED.high,
                        low = EXCLUDED.low,
                        close = EXCLUDED.close,
                        volume = EXCLUDED.volume
                """, (
                    datetime.fromtimestamp(market_data.timestamp / 1000),
                    market_data.symbol,
                    timeframe,
                    market_data.open,
                    market_data.high,
                    market_data.low,
                    market_data.close,
                    market_data.volume
                ))
                self.db_conn.commit()
                
                # OHLCV 데이터 저장 후 기술지표 계산 및 업데이트
                self.calculate_technical_indicators(market_data.symbol, timeframe)
                
        except Exception as e:
            self.logger.error(f"데이터 저장 중 오류 발생: {e}")
            self.db_conn.rollback()

    def calculate_technical_indicators(self, symbol: str, timeframe: str):
        """기술지표 계산 및 업데이트"""
        try:
            # 충분한 데이터 조회 (200일 이동평균을 위해 최소 200개 이상)
            with self.db_conn.cursor() as cur:
                cur.execute("""
                    SELECT timestamp, open, high, low, close, volume 
                    FROM market_data 
                    WHERE symbol = %s AND timeframe = %s 
                    ORDER BY timestamp DESC 
                    LIMIT 300
                """, (symbol, timeframe))
                
                rows = cur.fetchall()
                if not rows:
                    return
                    
                # DataFrame 생성 및 데이터 타입 변환
                df = pd.DataFrame(rows, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df = df.sort_values('timestamp')  # 시간순 정렬
                
                # Decimal을 float로 변환
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = df[col].astype(float)
                
                # 이동평균선 (MA)
                for period in [5, 10, 20, 60, 120, 200]:
                    df[f'ma_{period}'] = df['close'].rolling(window=period).mean()
                    df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
                
                # 볼린저 밴드 (20일 기준)
                df['bb_middle'] = df['close'].rolling(window=20).mean()
                std = df['close'].rolling(window=20).std()
                df['bb_upper'] = df['bb_middle'] + (std * 2)
                df['bb_lower'] = df['bb_middle'] - (std * 2)
                
                # RSI (14일 기준)
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                df['rsi'] = 100 - (100 / (1 + rs))
                
                # MACD (12, 26, 9)
                exp1 = df['close'].ewm(span=12, adjust=False).mean()
                exp2 = df['close'].ewm(span=26, adjust=False).mean()
                df['macd'] = exp1 - exp2
                df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
                df['macd_hist'] = df['macd'] - df['macd_signal']
                
                # Stochastic (14, 3, 3)
                low_14 = df['low'].rolling(window=14).min()
                high_14 = df['high'].rolling(window=14).max()
                df['stoch_k'] = ((df['close'] - low_14) / (high_14 - low_14)) * 100
                df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
                
                # ATR (14일 기준)
                tr1 = df['high'] - df['low']
                tr2 = abs(df['high'] - df['close'].shift())
                tr3 = abs(df['low'] - df['close'].shift())
                tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                df['atr'] = tr.rolling(window=14).mean()
                
                # OBV
                df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
                
                # DMI/ADX
                plus_dm = df['high'].diff()
                minus_dm = df['low'].diff()
                plus_dm = plus_dm.where(plus_dm > minus_dm, 0)
                minus_dm = minus_dm.where(minus_dm > plus_dm, 0)
                tr = tr.fillna(0)
                
                df['dmi_plus'] = (plus_dm.rolling(window=14).mean() / tr.rolling(window=14).mean()) * 100
                df['dmi_minus'] = (minus_dm.rolling(window=14).mean() / tr.rolling(window=14).mean()) * 100
                df['adx'] = abs(df['dmi_plus'] - df['dmi_minus']) / (df['dmi_plus'] + df['dmi_minus']) * 100
                df['adx'] = df['adx'].rolling(window=14).mean()
                
                # NaN 값을 None으로 변환
                df = df.replace([np.inf, -np.inf], np.nan)
                df = df.where(pd.notnull(df), None)
                
                # 기술지표 업데이트
                for _, row in df.iloc[-10:].iterrows():  # 최근 10개 데이터만 업데이트
                    self.update_market_data_indicators(symbol, timeframe, row)
                    
        except Exception as e:
            self.logger.error(f"기술지표 계산 중 오류 발생: {e}")

    def update_market_data_indicators(self, symbol: str, timeframe: str, row: pd.Series):
        """기술지표 값 업데이트"""
        try:
            with self.db_conn.cursor() as cur:
                cur.execute("""
                    UPDATE market_data 
                    SET ma_5 = %s, ma_10 = %s, ma_20 = %s, ma_60 = %s, ma_120 = %s, ma_200 = %s,
                        ema_5 = %s, ema_10 = %s, ema_20 = %s, ema_60 = %s, ema_120 = %s, ema_200 = %s,
                        bb_upper = %s, bb_middle = %s, bb_lower = %s,
                        rsi = %s,
                        macd = %s, macd_signal = %s, macd_hist = %s,
                        stoch_k = %s, stoch_d = %s,
                        atr = %s,
                        obv = %s,
                        dmi_plus = %s, dmi_minus = %s, adx = %s
                    WHERE symbol = %s AND timeframe = %s AND timestamp = %s
                """, (
                    row.get('ma_5'), row.get('ma_10'), row.get('ma_20'),
                    row.get('ma_60'), row.get('ma_120'), row.get('ma_200'),
                    row.get('ema_5'), row.get('ema_10'), row.get('ema_20'),
                    row.get('ema_60'), row.get('ema_120'), row.get('ema_200'),
                    row.get('bb_upper'), row.get('bb_middle'), row.get('bb_lower'),
                    row.get('rsi'),
                    row.get('macd'), row.get('macd_signal'), row.get('macd_hist'),
                    row.get('stoch_k'), row.get('stoch_d'),
                    row.get('atr'),
                    row.get('obv'),
                    row.get('dmi_plus'), row.get('dmi_minus'), row.get('adx'),
                    symbol, timeframe, 
                    # timestamp 처리 수정
                    row['timestamp'] if isinstance(row['timestamp'], datetime) else datetime.fromtimestamp(row['timestamp'])
                ))
                self.db_conn.commit()
        except Exception as e:
            self.logger.error(f"기술지표 업데이트 중 오류 발생: {e}")
            self.db_conn.rollback()