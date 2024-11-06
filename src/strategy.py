from dataclasses import dataclass
from typing import Dict, List
import pandas as pd
import numpy as np

@dataclass
class TradingParameters:
    # 이동평균선 설정
    ma_short: str = 'ma_20'
    ma_mid: str = 'ma_60'
    ma_long: str = 'ma_200'
    
    # RSI 설정
    rsi_buy_threshold: float = 35.0
    rsi_sell_threshold: float = 75.0
    
    # 볼린저 밴드 설정
    bb_buy_threshold: float = 0.02
    bb_sell_threshold: float = 0.02
    
    # ADX 설정
    adx_threshold: float = 25.0
    
    # 리스크 관리
    stop_loss: float = 0.05
    take_profit: float = 0.15
    max_position_size: float = 0.3
    
    # 지표 사용 여부 (비트마스크)
    use_ma: bool = True
    use_rsi: bool = True
    use_bb: bool = True
    use_macd: bool = True
    use_stoch: bool = True
    use_adx: bool = True
    
    # 타임프레임 사용 여부 (비트마스크)
    use_1h: bool = True
    use_4h: bool = True
    use_1d: bool = True
    
    # 신호 발생 임계값
    signal_threshold: float = 0.6
    
    # 지표별 가중치 (optimizer.py와 일치하도록 추가)
    weight_ma: float = 0.2
    weight_rsi: float = 0.15
    weight_bb: float = 0.15
    weight_macd: float = 0.2
    weight_stoch: float = 0.15
    weight_adx: float = 0.15

class Strategy:
    def __init__(self, params: TradingParameters = None):
        self.params = params or TradingParameters()
    
    def analyze_timeframe(self, df: pd.DataFrame, timeframe: str) -> pd.Series:
        """각 타임프레임별 분석을 수행합니다"""
        signals = pd.Series(0, index=df.index)
        
        try:
            # 각 지표별 점수 계산 (가중치 적용)
            buy_score = pd.Series(0.0, index=df.index)
            sell_score = pd.Series(0.0, index=df.index)
            total_weight = 0
            
            # 이동평균선
            if self.params.use_ma:
                ma_buy = (df[self.params.ma_short] > df[self.params.ma_mid]) & (df[self.params.ma_mid] > df[self.params.ma_long])
                ma_sell = (df[self.params.ma_short] < df[self.params.ma_mid]) & (df[self.params.ma_mid] < df[self.params.ma_long])
                buy_score += ma_buy.astype(float) * self.params.weight_ma
                sell_score += ma_sell.astype(float) * self.params.weight_ma
                total_weight += self.params.weight_ma
            
            # RSI
            if self.params.use_rsi:
                rsi_buy = df['rsi'] < self.params.rsi_buy_threshold
                rsi_sell = df['rsi'] > self.params.rsi_sell_threshold
                buy_score += rsi_buy.astype(float) * self.params.weight_rsi
                sell_score += rsi_sell.astype(float) * self.params.weight_rsi
                total_weight += self.params.weight_rsi
            
            # 볼린저 밴드
            if self.params.use_bb:
                bb_buy = df['close'] <= df['bb_lower'] * (1 + self.params.bb_buy_threshold)
                bb_sell = df['close'] >= df['bb_upper'] * (1 - self.params.bb_sell_threshold)
                buy_score += bb_buy.astype(float) * self.params.weight_bb
                sell_score += bb_sell.astype(float) * self.params.weight_bb
                total_weight += self.params.weight_bb
            
            # MACD
            if self.params.use_macd:
                macd_buy = (df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1))
                macd_sell = (df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1))
                buy_score += macd_buy.astype(float) * self.params.weight_macd
                sell_score += macd_sell.astype(float) * self.params.weight_macd
                total_weight += self.params.weight_macd
            
            # 스토캐스틱
            if self.params.use_stoch:
                stoch_buy = df['stoch_k'] < 30
                stoch_sell = df['stoch_k'] > 70
                buy_score += stoch_buy.astype(float) * self.params.weight_stoch
                sell_score += stoch_sell.astype(float) * self.params.weight_stoch
                total_weight += self.params.weight_stoch
            
            # ADX/DMI
            if self.params.use_adx:
                adx_buy = (df['adx'] > self.params.adx_threshold) & (df['dmi_plus'] > df['dmi_minus'])
                adx_sell = (df['adx'] > self.params.adx_threshold) & (df['dmi_plus'] < df['dmi_minus'])
                buy_score += adx_buy.astype(float) * self.params.weight_adx
                sell_score += adx_sell.astype(float) * self.params.weight_adx
                total_weight += self.params.weight_adx
            
            if total_weight > 0:
                # 가중치 합으로 정규화
                buy_score = buy_score / total_weight
                sell_score = sell_score / total_weight
                
                # 임계값을 낮춰서 더 많은 신호 생성 (0.3으로 낮춤)
                signals[buy_score >= 0.3] = 1
                signals[sell_score >= 0.3] = -1
            
            return signals
            
        except Exception:
            return pd.Series(index=df.index, data=0)
    
    def generate_signals(self, dfs: Dict[str, pd.DataFrame]) -> pd.Series:
        """멀티 타임프레임 분석을 통한 매매 신호 생성"""
        try:
            base_index = dfs['1h'].index
            signal_sum = pd.Series(0.0, index=base_index)
            active_count = 0
            
            # 각 타임프레임별 신호를 합산
            if self.params.use_1h:
                signals_1h = self.analyze_timeframe(dfs['1h'], '1h')
                signal_sum += signals_1h
                active_count += 1
            
            if self.params.use_4h:
                signals_4h = self.analyze_timeframe(dfs['4h'], '4h')
                signals_4h = signals_4h.reindex(base_index, method='ffill')
                signal_sum += signals_4h
                active_count += 1
            
            if self.params.use_1d:
                signals_1d = self.analyze_timeframe(dfs['1d'], '1d')
                signals_1d = signals_1d.reindex(base_index, method='ffill')
                signal_sum += signals_1d
                active_count += 1
            
            # 최종 신호 생성
            final_signals = pd.Series(0, index=base_index)
            if active_count > 0:
                threshold = active_count * 0.5  # 50% 이상의 타임프레임이 동의할 때
                final_signals[signal_sum >= threshold] = 1
                final_signals[signal_sum <= -threshold] = -1
            
            return final_signals
            
        except Exception:
            return pd.Series(index=dfs['1h'].index, data=0)
    
    def calculate_position_size(self, df: pd.DataFrame, index: int, capital: float) -> float:
        """ATR을 기반으로 포지션 크기를 조절합니다"""
        atr = df['atr'].iloc[index]
        close_price = df['close'].iloc[index]
        
        risk_per_trade = capital * self.params.stop_loss
        position_size = risk_per_trade / (3 * atr)
        max_position = (capital * self.params.max_position_size) / close_price
        
        return min(position_size, max_position)