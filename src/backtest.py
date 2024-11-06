import pandas as pd
import numpy as np
from typing import Dict, List
from strategy import Strategy, TradingParameters
import logging

class Backtest:
    def __init__(self, initial_capital: float = 10000.0, leverage: float = 3.0, fee_rate: float = 0.00005):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.leverage = leverage
        self.fee_rate = fee_rate
        self.position = 0  # 양수: 롱, 음수: 숏
        self.trades: List[Dict] = []
        self.strategy = None
        
    def run(self, dfs: Dict[str, pd.DataFrame], strategy: Strategy) -> Dict:
        """백테스트를 실행하고 결과를 반환합니다"""
        self.strategy = strategy
        signals = strategy.generate_signals(dfs)
        df = dfs['1h']
        
        logger = logging.getLogger(__name__)
        
        for i in range(1, len(df)):
            current_price = df['close'].iloc[i]
            signal = signals.iloc[i]
            
            try:
                # 포지션이 없을 때
                if self.position == 0:
                    if signal != 0:  # 신규 진입
                        position_size = strategy.calculate_position_size(df, i, self.capital)
                        if signal == 1:  # 롱 진입
                            self.enter_position(current_price, df.index[i], position_size, 'long')
                        else:  # 숏 진입
                            self.enter_position(current_price, df.index[i], position_size, 'short')
                
                # 포지션이 있을 때
                else:
                    # 청산 조건 확인
                    should_exit = False
                    
                    # 1. 반대 신호 발생
                    if (self.position > 0 and signal == -1) or (self.position < 0 and signal == 1):
                        should_exit = True
                    
                    # 2. 손절/익절 확인
                    elif self._check_stop_loss_take_profit(current_price):
                        should_exit = True
                    
                    # 청산 실행
                    if should_exit:
                        self.exit_position(current_price, df.index[i])
            
            except Exception as e:
                logger.error(f"거래 처리 중 오류: {str(e)}")
                continue
        
        # 마지막 포지션 청산
        if self.position != 0:
            self.exit_position(df['close'].iloc[-1], df.index[-1])
        
        return self._calculate_statistics()
    
    def enter_position(self, price: float, timestamp, position_size: float, position_type: str):
        """포지션 진입"""
        # 레버리지 적용
        leveraged_position = position_size * self.leverage
        
        # 수수료 계산 및 차감
        entry_fee = abs(leveraged_position * price * self.fee_rate)
        self.capital -= entry_fee
        
        self.position = leveraged_position if position_type == 'long' else -leveraged_position
        self.entry_price = price
        
        self.trades.append({
            'timestamp': timestamp,
            'type': f'enter_{position_type}',
            'price': price,
            'size': abs(self.position),
            'leverage': self.leverage,
            'fee': entry_fee,
            'capital': self.capital
        })
    
    def exit_position(self, price: float, timestamp):
        """포지션 청산"""
        # 수수료 계산
        exit_fee = abs(self.position * price * self.fee_rate)
        
        # PnL 계산 (레버리지 효과 포함)
        if self.position > 0:  # 롱 포지션
            pnl = (price - self.entry_price) * self.position
        else:  # 숏 포지션
            pnl = (self.entry_price - price) * abs(self.position)
        
        # 수수료 차감
        self.capital = self.capital + pnl - exit_fee
        
        self.trades.append({
            'timestamp': timestamp,
            'type': 'exit',
            'price': price,
            'pnl': pnl,
            'fee': exit_fee,
            'capital': self.capital,
            'return_pct': (pnl / (self.entry_price * abs(self.position))) * 100
        })
        self.position = 0
    
    def _check_stop_loss_take_profit(self, current_price: float) -> bool:
        """스탑로스/익절 체크"""
        if self.position == 0:
            return False
        
        if self.position > 0:  # 롱 포지션
            return_pct = (current_price - self.entry_price) / self.entry_price
        else:  # 숏 포지션
            return_pct = (self.entry_price - current_price) / self.entry_price
            
        return (return_pct <= -self.strategy.params.stop_loss or 
                return_pct >= self.strategy.params.take_profit)
    
    def _calculate_statistics(self) -> Dict:
        """백테스트 결과 통계 계산"""
        if not self.trades:
            return self._get_empty_stats()
        
        exit_trades = [t for t in self.trades if t['type'] == 'exit']
        if not exit_trades:
            return self._get_empty_stats()
        
        trades_df = pd.DataFrame(exit_trades)
        
        # 수수료 총액 계산
        total_fees = sum(t['fee'] for t in self.trades)
        
        # 순손익 (수수료 차감)
        net_pnl = trades_df['pnl'].sum() - total_fees
        
        num_trades = len(trades_df)
        win_trades = len(trades_df[trades_df['pnl'] > 0])
        
        # 롱/숏 포지션별 성과 계산
        long_trades = [t for t in self.trades if t['type'] == 'enter_long']
        short_trades = [t for t in self.trades if t['type'] == 'enter_short']
        
        stats = {
            'initial_capital': self.initial_capital,
            'final_capital': self.capital,
            'total_return': (self.capital - self.initial_capital) / self.initial_capital * 100,
            'num_trades': num_trades,
            'num_long_trades': len(long_trades),
            'num_short_trades': len(short_trades),
            'win_rate': (win_trades / num_trades * 100) if num_trades > 0 else 0,
            'total_pnl': net_pnl,
            'total_fees': total_fees,
            'avg_return_per_trade': trades_df['return_pct'].mean() if not trades_df.empty else 0,
            'max_drawdown': self._calculate_max_drawdown(trades_df) if not trades_df.empty else 0,
            'leverage_used': self.leverage
        }
        
        return stats
    
    def _get_empty_stats(self) -> Dict:
        """빈 통계 결과 반환"""
        return {
            'initial_capital': self.initial_capital,
            'final_capital': self.capital,
            'total_return': 0.0,
            'num_trades': 0,
            'num_long_trades': 0,
            'num_short_trades': 0,
            'win_rate': 0.0,
            'total_pnl': 0.0,
            'total_fees': 0.0,
            'avg_return_per_trade': 0.0,
            'max_drawdown': 0.0,
            'leverage_used': self.leverage
        }
    
    def _calculate_max_drawdown(self, trades_df: pd.DataFrame) -> float:
        """최대 손실폭 계산"""
        try:
            capital_history = trades_df['capital']
            rolling_max = capital_history.expanding().max()
            drawdowns = (capital_history - rolling_max) / rolling_max * 100
            return abs(drawdowns.min()) if not drawdowns.empty else 0
        except Exception as e:
            logging.error(f"최대 손실폭 계산 중 오류 발생: {str(e)}")
            return 0.0