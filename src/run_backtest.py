import pandas as pd
import logging
import sys
from pathlib import Path
from dataclasses import asdict
import json

# 현재 스크립트의 디렉토리를 파이썬 경로에 추가
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

from strategy import Strategy, TradingParameters
from backtest import Backtest
import psycopg2
from datetime import datetime, timedelta
from optimizer import StrategyOptimizer

# 데이터베이스 설정 추가
db_config = {
    'dbname': 'binance_futures',
    'user': 'postgres',
    'password': '1234',
    'host': 'localhost',
    'port': '5432',
    'client_encoding': 'SQL_ASCII'
}

# 쿼리 정의 추가
query = """
SELECT 
    timestamp, 
    open, high, low, close, volume,
    ma_5, ma_10, ma_20, ma_60, ma_120, ma_200,
    ema_5, ema_10, ema_20, ema_60, ema_120, ema_200,
    bb_upper, bb_middle, bb_lower,
    rsi,
    macd, macd_signal, macd_hist,
    stoch_k, stoch_d,
    atr,
    obv,
    dmi_plus, dmi_minus, adx
FROM market_data
WHERE symbol = 'BTCUSDT' 
AND timeframe = %s
ORDER BY timestamp
"""

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        encoding='cp949'
    )
    return logging.getLogger(__name__)

def get_data_from_db(timeframe: str) -> pd.DataFrame:
    """특정 타임프레임의 데이터를 가져옵니다"""
    logger = logging.getLogger(__name__)
    
    try:
        conn = psycopg2.connect(**db_config)
        df = pd.read_sql_query(query, conn, params=(timeframe,))
        conn.close()
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # 데이터 검증 로깅
        logger.info(f"\n{timeframe} 데이터 검증:")
        logger.info(f"데이터 행 수: {len(df)}")
        logger.info(f"컬럼: {df.columns.tolist()}")
        logger.info(f"결측치:\n{df.isnull().sum()}")
        
        # 기술적 지표 값 범위 확인
        for col in ['rsi', 'macd', 'adx', 'stoch_k']:
            if col in df.columns:
                logger.info(f"{col} 범위: {df[col].min():.2f} ~ {df[col].max():.2f}")
        
        df.set_index('timestamp', inplace=True)
        return df.dropna()
        
    except Exception as e:
        logger.error(f"데이터 로드 중 상세 오류: {str(e)}")
        raise

def run_backtest_analysis(optimize: bool = True):
    logger = setup_logging()
    
    try:
        # 여러 타임프레임의 데이터 로드
        timeframes = ['1h', '4h', '1d']
        dfs = {}
        
        for tf in timeframes:
            logger.info(f"{tf} 타임프레임 데이터 로드 중...")
            df = get_data_from_db(tf)
            if not df.empty:
                dfs[tf] = df
                logger.info(f"{tf} 데이터 기간: {df.index[0]} ~ {df.index[-1]}")
            else:
                logger.error(f"{tf} 데이터가 비어있습니다.")
                return
        
        # 데이터 정합성 체크
        if not all(tf in dfs for tf in timeframes):
            logger.error("일부 타임프레임의 데이터가 누락되었습니다.")
            return
        
        # 파라미터 설정
        try:
            # 기존 최적 파라미터 로드 시도
            with open('models/best_parameters.json', 'r') as f:
                best_params = json.load(f)
                logger.info("\n=== 기존 최적 파라미터 로드 ===")
                for param_name, value in best_params.items():
                    logger.info(f"{param_name}: {value}")
                params = TradingParameters(**best_params)
        except FileNotFoundError:
            logger.info("기존 최적 파라미터가 없습니다. 기본값으로 시작합니다.")
            params = TradingParameters()
        
        # 최적화 모드인 경우 기존 파라미터를 기반으로 추가 최적화 수행
        if optimize:
            logger.info("\n=== 전략 최적화 시작 ===")
            optimizer = StrategyOptimizer(dfs)
            optimized_params = optimizer.optimize(episodes=500)
            
            # 최적화 결과가 더 좋은 경우에만 파라미터 업데이트
            if optimized_params is not None:
                params = optimized_params
                logger.info("\n=== 새로운 최적 파라미터 ===")
                for param_name, value in asdict(optimized_params).items():
                    logger.info(f"{param_name}: {value}")
        
        # 백테스트 실행
        logger.info("\n=== 백테스트 실행 시작 ===")
        strategy = Strategy(params)
        backtest = Backtest(initial_capital=10000.0, leverage=3.0, fee_rate=0.00005)
        
        # 멀티 타임프레임 데이터로 백테스트 실행
        results = backtest.run(dfs, strategy)
        
        # 거래 내역 출력
        if len(backtest.trades) > 0:
            logger.info("\n=== 거래 내역 샘플 ===")
            for trade in backtest.trades[:5]:
                if trade['type'].startswith('enter'):
                    logger.info(f"진입: {trade['timestamp']} - {trade['type']} @ ${trade['price']:,.2f}")
                else:
                    logger.info(f"청산: {trade['timestamp']} - PnL: ${trade['pnl']:,.2f} ({trade.get('return_pct', 0):.2f}%)")
            
            # 월별 수익률 계산
            trades_df = pd.DataFrame(backtest.trades)
            trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
            trades_df['month'] = trades_df['timestamp'].dt.strftime('%Y-%m')
            
            monthly_pnl = trades_df[trades_df['type'] == 'exit'].groupby('month')['pnl'].sum()
            logger.info("\n=== 월별 손익 ===")
            for month, pnl in monthly_pnl.items():
                logger.info(f"{month}: ${pnl:,.2f}")
        
        # 최종 결과 출력
        logger.info("\n=== 백테스트 최종 결과 ===")
        logger.info(f"백테스트 기간: {dfs['1h'].index[0]} ~ {dfs['1h'].index[-1]}")
        logger.info(f"초기 자본: ${results['initial_capital']:,.2f}")
        logger.info(f"최종 자본: ${results['final_capital']:,.2f}")
        logger.info(f"총 수익률: {results['total_return']:.2f}%")
        logger.info(f"승률: {results['win_rate']:.2f}%")
        logger.info(f"총 거래 횟수: {results['num_trades']} (롱: {results['num_long_trades']}, 숏: {results['num_short_trades']})")
        logger.info(f"평균 수익률: {results['avg_return_per_trade']:.2f}%")
        logger.info(f"최대 손실폭: {results['max_drawdown']:.2f}%")
        logger.info(f"총 손익: ${results['total_pnl']:,.2f}")
        logger.info(f"총 수수료: ${results['total_fees']:,.2f}")
        
        # 연간 수익률 계산
        days = (dfs['1h'].index[-1] - dfs['1h'].index[0]).days
        if days > 0:
            years = days / 365.0
            annual_return = ((1 + results['total_return']/100) ** (1/years) - 1) * 100
            logger.info(f"연간 수익률: {annual_return:.2f}%")
        
    except Exception as e:
        logger.error(f"백테스트 실행 중 오류 발생: {str(e)}")
        raise

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--optimize', action='store_true', help='전략 최적화 실행')
    args = parser.parse_args()
    
    run_backtest_analysis(optimize=args.optimize) 