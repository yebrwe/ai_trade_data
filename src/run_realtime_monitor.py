import pandas as pd
import logging
import sys
from pathlib import Path
import json
from typing import List, Dict
import time
from datetime import datetime, timedelta, timezone
from sqlalchemy import create_engine
from colorama import init, Fore, Style

# 현재 스크립트의 디렉토리를 파이썬 경로에 추가
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

from strategy import Strategy, TradingParameters
from backtest import Backtest

# 데이터베이스 설정
db_config = {
    'dbname': 'binance_futures',
    'user': 'postgres',
    'password': '1234',
    'host': 'localhost',
    'port': '5432'
}

# SQLAlchemy 엔진 생성을 위한 URL 생성
db_url = f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['dbname']}"

# 실시간 데이터 조회 쿼리 수정 - 시작 시간 파라미터 추가
realtime_query = """
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
"""

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        encoding='cp949'
    )
    return logging.getLogger(__name__)

def get_realtime_data(timeframe: str, start_time: datetime) -> pd.DataFrame:
    """실시간 데이터를 가져옵니다"""
    logger = logging.getLogger(__name__)
    
    try:
        engine = create_engine(db_url, pool_pre_ping=True)
        
        with engine.connect() as conn:
            df = pd.read_sql_query(
                realtime_query,
                conn,
                params=(timeframe, start_time, start_time)
            )
            
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp']) + timedelta(hours=9)  # UTC to KST
                df.set_index('timestamp', inplace=True)
                df = df.sort_index()
                
                logger.info(f"{timeframe} 데이터 조회: {len(df)}개 (최신: {df.index[-1].strftime('%Y-%m-%d %H:%M:%S')})")
                return df.dropna()
            return pd.DataFrame()
        
    except Exception as e:
        logger.error(f"실시간 데이터 로드 중 오류: {str(e)}")
        return pd.DataFrame()
    finally:
        engine.dispose()

def monitor_realtime_trades(update_interval: int = 60):
    """실시간 거래 모니터링"""
    logger = setup_logging()
    
    try:
        # 파라미터 로드
        try:
            with open('models/best_parameters.json', 'r') as f:
                best_params = json.load(f)
                logger.info("\n=== 거래 파라미터 로드 ===")
                for param_name, value in best_params.items():
                    logger.info(f"{param_name}: {value}")
                params = TradingParameters(**best_params)
        except FileNotFoundError:
            logger.info("파라미터 파일이 없습니다. 기본값으로 시작합니다.")
            params = TradingParameters()

        strategy = Strategy(params)
        backtest = Backtest(initial_capital=10000.0, leverage=3.0, fee_rate=0.00005)
        
        # 첫 실행 여부 플래그와 마지막 타임스탬프 저장
        is_first_run = True
        last_trade_count = 0
        last_timestamps = {}  # 각 타임프레임별 마지막 타임스탬프 저장
        
        init()  # colorama 초기화
        
        while True:
            try:
                timeframes = ['1h', '4h', '1d']
                dfs = {}
                
                for tf in timeframes:
                    # 첫 실행이면 전체 데이터, 아니면 마지막 타임스탬프 이후 데이터만
                    last_ts = None if is_first_run else last_timestamps.get(tf)
                    df = get_realtime_data(tf, last_ts)
                    
                    if not df.empty:
                        dfs[tf] = df
                        # 마지막 타임스탬프 업데이트
                        last_timestamps[tf] = df.index[-1]
                        logger.info(f"{tf} - 최신: {df.index[-1].strftime('%Y-%m-%d %H:%M:%S')} (총 {len(df)}개 데이터)")
                    # else 부분 제거 - 업데이트가 없는 경우 아무 메시지도 출력하지 않음
                
                if all(tf in dfs for tf in timeframes):
                    results = backtest.run(dfs, strategy)
                    
                    current_trade_count = len(backtest.trades)
                    
                    # 첫 실행 시 전체 거래 내역 표시
                    if is_first_run and current_trade_count > 0:
                        logger.info("\n=== 전체 거래 내역 ===")
                        for trade in backtest.trades:
                            trade_time = pd.to_datetime(trade['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
                            if trade['type'].startswith('enter'):
                                logger.info(f"진입: {trade_time} - {trade['type']} @ ${trade['price']:,.2f}")
                            else:
                                pnl = trade['pnl']
                                color = Fore.GREEN if pnl > 0 else Fore.RED
                                logger.info(f"청산: {trade_time} - PnL: {color}${pnl:,.2f} ({trade.get('return_pct', 0):.2f}%){Style.RESET_ALL}")
                        is_first_run = False
                        last_trade_count = current_trade_count
                    # 이후 새로운 거래만 표시
                    elif current_trade_count > last_trade_count:
                        logger.info("\n=== 새로운 거래 발생 ===")
                        for trade in backtest.trades[last_trade_count:]:
                            trade_time = pd.to_datetime(trade['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
                            if trade['type'].startswith('enter'):
                                logger.info(f"진입: {trade_time} - {trade['type']} @ ${trade['price']:,.2f}")
                            else:
                                pnl = trade['pnl']
                                color = Fore.GREEN if pnl > 0 else Fore.RED
                                logger.info(f"청산: {trade_time} - PnL: {color}${pnl:,.2f} ({trade.get('return_pct', 0):.2f}%){Style.RESET_ALL}")
                        last_trade_count = current_trade_count
                
                time.sleep(update_interval)
                
            except Exception as e:
                logger.error(f"모니터링 중 오류 발생: {str(e)}")
                time.sleep(5)
                
    except KeyboardInterrupt:
        logger.info("\n모니터링을 종료합니다.")
    except Exception as e:
        logger.error(f"프로그램 실행 중 오류 발생: {str(e)}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--interval', type=int, default=60, help='데이터 업데이트 간격(초)')
    args = parser.parse_args()
    
    monitor_realtime_trades(update_interval=args.interval) 