from binance_futures_collector import BinanceFuturesCollector
import sys
import locale
import logging

def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        encoding='cp949'
    )
    logger = logging.getLogger(__name__)
    
    if sys.platform.startswith('win'):
        locale.setlocale(locale.LC_ALL, 'C')
    
    db_config = {
        'dbname': 'binance_futures',
        'user': 'postgres',
        'password': '1234',
        'host': 'localhost',
        'port': '5432',
        'client_encoding': 'SQL_ASCII'
    }
    
    symbols = ['BTCUSDT']
    
    try:
        logger.info("바이낸스 선물 데이터 수집 시작...")
        collector = BinanceFuturesCollector(db_config)
        logger.info(f"수집 대상 심볼: {', '.join(symbols)}")
        collector.collect_data(symbols)
    except Exception as e:
        logger.error(f"오류 발생: {str(e).encode('cp949', 'ignore').decode('cp949')}")
        sys.exit(1)

if __name__ == "__main__":
    main() 