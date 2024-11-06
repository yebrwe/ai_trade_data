import numpy as np
import pandas as pd
from typing import Dict, List
import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import asdict
import json
from pathlib import Path
import logging
from strategy import Strategy, TradingParameters
from backtest import Backtest

class StrategyNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 128):
        super().__init__()
        
        # 파라미터 범위 수정 (지표/타임프레임 스위치 추가)
        self.param_ranges = {
            # 기술적 지표 사용 여부 (0 또는 1)
            'use_ma': (0, 1),      # 이동평균선 사용 여부
            'use_rsi': (0, 1),     # RSI 사용 여부
            'use_bb': (0, 1),      # 볼린저밴드 사용 여부
            'use_macd': (0, 1),    # MACD 사용 여부
            'use_stoch': (0, 1),   # 스토캐스틱 사용 여부
            'use_adx': (0, 1),     # ADX 사용 여부
            
            # 타임프레임 사용 여부 (0 또는 1)
            'use_1h': (0, 1),      # 1시간봉 사용 여부
            'use_4h': (0, 1),      # 4시간 용 여부
            'use_1d': (0, 1),      # 일봉 사용 여부
            
            # 기존 파라미터들...
            'rsi_buy_threshold': (10, 50),
            'rsi_sell_threshold': (50, 90),
            'bb_buy_threshold': (0.001, 0.1),
            'bb_sell_threshold': (0.001, 0.1),
            'adx_threshold': (10, 50),
            'stop_loss': (0.02, 0.10),
            'take_profit': (0.03, 0.20),
            'max_position_size': (0.2, 0.8),
            'signal_threshold': (0.1, 0.5)
        }
        
        # 네트워크 레이어 정의
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, len(self.param_ranges))
        )
    
    def forward(self, x):
        raw_output = self.network(x)
        scaled_params = {}
        
        # 각 파라미터를 적절한 범위로 스케일링
        for i, (param_name, param_range) in enumerate(self.param_ranges.items()):
            if param_name.startswith('use_'):  # 스위치 파라미터
                # 시그모이드 출력을 이진값으로 변환 (0.5를 임계값으로 사용)
                scaled_params[param_name] = int(torch.sigmoid(raw_output[:, i]).mean() > 0.5)
            else:  # 연속값 파라미터
                scaled_params[param_name] = torch.sigmoid(raw_output[:, i]).mean() * (param_range[1] - param_range[0]) + param_range[0]
        
        # MA 컬럼은 고정값 사용
        scaled_params['ma_short'] = 'ma_20'
        scaled_params['ma_mid'] = 'ma_60'
        scaled_params['ma_long'] = 'ma_200'
        
        return scaled_params

class StrategyOptimizer:
    def __init__(self, dfs: Dict[str, pd.DataFrame], save_path: str = 'models/strategy_weights.pth'):
        self.logger = logging.getLogger(__name__)
        self.dfs = dfs
        self.save_path = Path(save_path)
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 네트워크 초기화
        self.network = StrategyNetwork(input_size=self._calculate_input_size())
        self.optimizer = optim.Adam(self.network.parameters(), lr=0.0005)
        
        # param_ranges를 인스턴스 변수로 저장
        self.param_ranges = self.network.param_ranges
        
        # 이전 가중치 로드 시도
        try:
            if self.save_path.exists():
                # 이전 모델 구조와 현재 모델 구조 다른 경우 처리
                old_state_dict = torch.load(self.save_path)
                if len(old_state_dict['network.6.bias']) != len(self.param_ranges):
                    self.logger.warning("모델 구조가 변경되어 새로 시작합니다.")
                    # 이전 가중치 파일 백업
                    backup_path = self.save_path.with_suffix('.pth.bak')
                    self.save_path.rename(backup_path)
                    # best_parameters.json도 백업
                    params_path = Path('models/best_parameters.json')
                    if params_path.exists():
                        params_path.rename(params_path.with_suffix('.json.bak'))
                else:
                    self.network.load_state_dict(old_state_dict)
                    self.logger.info("기존 가중치를 로드했습니다.")
        except Exception as e:
            self.logger.warning(f"가중치 로드 실패, 새로 시작합니다: {str(e)}")
    
    def _create_trading_parameters(self, params: Dict) -> TradingParameters:
        try:
            # 기본 파라미터 변환
            param_dict = {
                'ma_short': 'ma_20',
                'ma_mid': 'ma_60',
                'ma_long': 'ma_200',
                'rsi_buy_threshold': float(params['rsi_buy_threshold'].item() if hasattr(params['rsi_buy_threshold'], 'item') else params['rsi_buy_threshold']),
                'rsi_sell_threshold': float(params['rsi_sell_threshold'].item() if hasattr(params['rsi_sell_threshold'], 'item') else params['rsi_sell_threshold']),
                'bb_buy_threshold': float(params['bb_buy_threshold'].item() if hasattr(params['bb_buy_threshold'], 'item') else params['bb_buy_threshold']),
                'bb_sell_threshold': float(params['bb_sell_threshold'].item() if hasattr(params['bb_sell_threshold'], 'item') else params['bb_sell_threshold']),
                'adx_threshold': float(params['adx_threshold'].item() if hasattr(params['adx_threshold'], 'item') else params['adx_threshold']),
                'stop_loss': float(params['stop_loss'].item() if hasattr(params['stop_loss'], 'item') else params['stop_loss']),
                'take_profit': float(params['take_profit'].item() if hasattr(params['take_profit'], 'item') else params['take_profit']),
                'max_position_size': float(params['max_position_size'].item() if hasattr(params['max_position_size'], 'item') else params['max_position_size']),
                'signal_threshold': float(params['signal_threshold'].item() if hasattr(params['signal_threshold'], 'item') else params['signal_threshold']),
            }
            
            # 지표와 타임프레임 설정을 랜덤하게 결정
            indicators = ['use_ma', 'use_rsi', 'use_bb', 'use_macd', 'use_stoch', 'use_adx']
            num_active = np.random.randint(2, len(indicators) + 1)
            active_indicators = np.random.choice(indicators, size=num_active, replace=False)
            
            for indicator in indicators:
                param_dict[indicator] = indicator in active_indicators
            
            timeframes = ['use_1h', 'use_4h', 'use_1d']
            num_active_tf = np.random.randint(1, len(timeframes) + 1)
            active_timeframes = np.random.choice(timeframes, size=num_active_tf, replace=False)
            
            for timeframe in timeframes:
                param_dict[timeframe] = timeframe in active_timeframes
            
            return TradingParameters(**param_dict)
            
        except Exception as e:
            self.logger.error(f"파라미터 변환 중 오류: {str(e)}")
            self.logger.debug(f"params: {params}")
            raise

    def _prepare_state(self, df: pd.DataFrame) -> torch.Tensor:
        """현재 시장 상태를 네트워크 입력으로 변환"""
        try:
            # 기술적 지표들의 상태 계산
            features = {
                'trend': (df['close'] / df['close'].shift(20) - 1) * 100,  # 20일 추세
                'volatility': df['atr'] / df['close'] * 100,  # 변동성
                'rsi': df['rsi'],  # RSI
                'macd_hist': df['macd_hist'],  # MACD 히스토그램
                'bb_position': (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower']),  # BB 내 위치
                'stoch_k': df['stoch_k'],  # 스토캐스틱
                'adx': df['adx'],  # ADX
                'dmi_trend': df['dmi_plus'] - df['dmi_minus'],  # DMI 트렌드
                'ma_trend_short': (df['ma_20'] / df['close'] - 1) * 100,
                'ma_trend_mid': (df['ma_60'] / df['close'] - 1) * 100,
                'ma_trend_long': (df['ma_200'] / df['close'] - 1) * 100,
                'volume_trend': (df['volume'] / df['volume'].rolling(20).mean() - 1) * 100,  # 거래량 추세
                'price_momentum': (df['close'] / df['close'].shift(1) - 1) * 100  # 가격 모멘텀
            }
            
            # 데이터프레임으로 변환
            features_df = pd.DataFrame(features)
            features_df = features_df.fillna(0)
            
            # 정규화
            features_mean = features_df.mean()
            features_std = features_df.std() + 1e-8
            normalized_features = (features_df - features_mean) / features_std
            
            return torch.FloatTensor(normalized_features.values)
            
        except Exception as e:
            self.logger.error(f"상태 준비 중 오류 발생: {str(e)}")
            raise

    def _calculate_reward(self, results: Dict) -> float:
        """백테스트 결과를 보상으로 변환"""
        # 거래가 너무 적은 경우 큰 페널티
        if results['num_trades'] < 50:  # 최소 50회 이상 거래 요구 (이전 30회에서 상향)
            return -20 * (1 - results['num_trades'] / 50)  # 거래 횟수에 비례한 페널티
        
        # 기본 지표들
        returns = results['total_return']
        win_rate = results['win_rate']
        max_drawdown = abs(results['max_drawdown'])
        num_trades = results['num_trades']
        avg_return = results['avg_return_per_trade']
        
        # 거래 빈도 점수 (월 평균 20회 이상을 목표로 상향)
        days = (results.get('end_date') - results.get('start_date')).days
        months = max(1, days / 30)
        target_trades = months * 20  # 월 20회로 상향
        trade_frequency_score = min(2.0, num_trades / target_trades)
        
        # 평균 수익률의 안정성 (변동성 고려)
        if 'trades' in results:
            returns_std = np.std([t.get('return_pct', 0) for t in results['trades']])
            stability_score = 1.0 / (1.0 + returns_std)  # 변동성이 낮을수록 높은 점수
        else:
            stability_score = 0
        
        # 기본 보상 계산 (가중치 조정)
        reward = (
            returns * 0.15 +                    # 총 수익률 (15% 비중 감소)
            win_rate * 0.20 +                   # 승률 (20%)
            trade_frequency_score * 0.35 +      # 거래 빈도 (35%로 증가)
            stability_score * 0.20 +            # 수익률 안정성 (20%)
            (-max_drawdown * 0.10)             # 최대 손실폭 (10%)
        )
        
        # 추가 보너스/페널티
        if num_trades >= target_trades * 1.5:  # 목표 거래 횟수의 150% 이상
            reward *= 1.3  # 더 큰 보너스
        elif num_trades >= target_trades:
            reward *= 1.2  # 기본 보너스
        
        if win_rate >= 55:
            reward *= 1.1  # 높은 승률 보너스
        
        if max_drawdown > 20:
            reward *= 0.8  # 큰 손실 페널티
        
        # 평균 수익률 안정성 보너스
        if 0.5 <= abs(avg_return) <= 2.0:  # 적정 수준의 평균 수익률
            reward *= 1.2
        elif abs(avg_return) > 5:  # 지나치게 큰 평균 수익률은 페널티
            reward *= 0.8
        
        # 연속 거래 보너스 (거래가 고르게 분포되어 있는지)
        if 'trades' in results:
            trade_gaps = []
            for i in range(1, len(results['trades'])):
                current_trade = pd.to_datetime(results['trades'][i]['timestamp'])
                prev_trade = pd.to_datetime(results['trades'][i-1]['timestamp'])
                gap_hours = (current_trade - prev_trade).total_seconds() / 3600
                trade_gaps.append(gap_hours)
            
            if trade_gaps:
                gap_std = np.std(trade_gaps)
                if gap_std < 24:  # 거래 간격이 24시간 이내로 고른 경우
                    reward *= 1.2
        
        return reward
    
    def _calculate_max_consecutive_losses(self) -> int:
        """최대 연속 손실 횟수 계산"""
        consecutive_losses = 0
        max_consecutive_losses = 0
        
        for trade in self.trades:
            if trade.get('pnl', 0) < 0:
                consecutive_losses += 1
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
            else:
                consecutive_losses = 0
        
        return max_consecutive_losses
    
    def _save_weights(self):
        """현재 가중치 저장"""
        try:
            torch.save(self.network.state_dict(), self.save_path)
            self.logger.info("새로운 가중치를 저장했습니다.")
        except Exception as e:
            self.logger.error(f"가중치 저장 중 오류 발생: {str(e)}")
    
    def optimize(self, episodes: int = None):
        try:
            episodes = 100000
            train_dfs = self.dfs
            best_reward = float('-inf')
            best_params = None
            best_annual_return = -100.0
            best_results = None  # 최고 성능의 전체 결과 저장
            
            print("\n최적화 진행 중...")
            for episode in range(episodes):
                try:
                    # 진행률 표시
                    progress = (episode + 1) / episodes * 100
                    print(f"\r진행률: [{('=' * int(progress/2)).ljust(50)}] {progress:.1f}%", end='')
                    
                    # 미니배치 학습
                    batch_size = 64
                    batch_rewards = []
                    
                    # 전체 데이터에서 랜덤 윈도우로 학습
                    for _ in range(batch_size):
                        window_size = np.random.randint(2400, 4800)  # 100~200일
                        max_start = max(0, len(train_dfs['1h']) - window_size)
                        start_idx = np.random.randint(0, max_start) if max_start > 0 else 0
                        
                        window_dfs = {
                            tf: df.iloc[start_idx:start_idx+window_size].copy() 
                            for tf, df in train_dfs.items()
                        }
                        
                        # 학습 단계
                        self.network.train()
                        self.optimizer.zero_grad()
                        
                        state = self._prepare_state(window_dfs['1h'])
                        params_dict = self.network(state)
                        trading_params = self._create_trading_parameters(params_dict)
                        strategy = Strategy(trading_params)
                        backtest = Backtest(initial_capital=10000.0, leverage=3.0, fee_rate=0.00005)
                        
                        results = backtest.run(window_dfs, strategy)
                        reward = self._calculate_reward(results)
                        batch_rewards.append(reward)
                        
                        # 역전파
                        loss = -torch.tensor(reward, dtype=torch.float32, requires_grad=True)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
                        self.optimizer.step()
                    
                    # 배치 학습 후 전체 데이터로 검증
                    if len(batch_rewards) > 0:
                        self.network.eval()
                        with torch.no_grad():
                            val_params_dict = self.network(self._prepare_state(train_dfs['1h']))
                            val_trading_params = self._create_trading_parameters(val_params_dict)
                            val_strategy = Strategy(val_trading_params)
                            val_backtest = Backtest(initial_capital=10000.0, leverage=3.0, fee_rate=0.00005)
                            val_results = val_backtest.run(train_dfs, val_strategy)
                            
                            # 연간 수익률 계산
                            days = (train_dfs['1h'].index[-1] - train_dfs['1h'].index[0]).days
                            annual_return = ((1 + val_results['total_return']/100) ** (365/days) - 1) * 100
                            
                            # 종합 성능 점수 계산
                            performance_score = (
                                annual_return * 0.3 +              # 연간 수익률 (30%)
                                val_results['win_rate'] * 0.2 +    # 승률 (20%)
                                min(val_results['num_trades'], 100) * 0.3 +  # 거래 횟수 (30%, 최대 100회까지)
                                (-abs(val_results['max_drawdown'])) * 0.2    # 최대 손실폭 (20%)
                            )
                            
                            # 성능 출력 (10 에피소드마다)
                            if episode % 10 == 0:
                                print(f"\n에피소드 {episode + 1}")
                                print(f"배치 평균 보상: {np.mean(batch_rewards):.2f}")
                                print(f"거래 횟수: {val_results['num_trades']}")
                                print(f"승률: {val_results['win_rate']:.2f}%")
                                print(f"연간 수익률: {annual_return:.2f}%")
                                print(f"최대 손실폭: {val_results['max_drawdown']:.2f}%")
                                print(f"종합 점수: {performance_score:.2f}")
                            
                            # 성능 향상 시에만 저장
                            if performance_score > best_reward:
                                best_reward = performance_score
                                best_params = val_trading_params
                                best_annual_return = annual_return
                                best_results = val_results
                                self._save_weights()
                                
                                # 성능 향상 시 상세 결과 출력
                                print("\n=== 새로운 최고 성능 ===")
                                print(f"종합 점수: {performance_score:.2f}")
                                print(f"연간 수익률: {annual_return:.2f}%")
                                print(f"총 수익률: {val_results['total_return']:.2f}%")
                                print(f"승률: {val_results['win_rate']:.2f}%")
                                print(f"거래 횟수: {val_results['num_trades']}")
                                print(f"최대 손실폭: {val_results['max_drawdown']:.2f}%")
                                print(f"평균 수익률: {val_results['avg_return_per_trade']:.2f}%")
                                print(f"총 수수료: ${val_results['total_fees']:.2f}")
                                
                                # 파라미터 저장
                                with open('models/best_parameters.json', 'w') as f:
                                    json.dump(asdict(val_trading_params), f, indent=4)
                
                except Exception as e:
                    print(f"\n에피소드 {episode + 1} 오류: {str(e)}")
                    continue
            
            print("\n최적화 완료!")
            if best_results:
                print("\n=== 최종 최고 성능 ===")
                print(f"종합 점수: {best_reward:.2f}")
                print(f"연간 수익률: {best_annual_return:.2f}%")
                print(f"총 수익률: {best_results['total_return']:.2f}%")
                print(f"승률: {best_results['win_rate']:.2f}%")
                print(f"거래 횟수: {best_results['num_trades']}")
                print(f"최대 손실폭: {best_results['max_drawdown']:.2f}%")
                print(f"평균 수익률: {best_results['avg_return_per_trade']:.2f}%")
                print(f"총 수수료: ${best_results['total_fees']:.2f}")
            
            return best_params
            
        except Exception as e:
            print(f"\n최적화 중 오류 발생: {str(e)}")
            raise
        
        return self._load_best_parameters()
    
    def _load_best_parameters(self) -> TradingParameters:
        """저장된 최적 파라미터 로드"""
        try:
            with open('models/best_parameters.json', 'r') as f:
                params = json.load(f)
            return TradingParameters(**params)
        except Exception as e:
            self.logger.error(f"적 파라미터 로드 실패: {str(e)}")
            return TradingParameters() 

    def _calculate_input_size(self) -> int:
        """시장 상태를 나타내는 입력 특성의 수를 계산"""
        try:
            # 기술적 지표들의 상태 계산에 사용되는 특성 수
            features = [
                'trend',              # 20일 추세
                'volatility',         # 변동성
                'rsi',               # RSI
                'macd_hist',         # MACD 히스토그램
                'bb_position',        # BB 내 위치
                'stoch_k',           # 스토캐스틱
                'adx',               # ADX
                'dmi_trend',         # DMI 트렌드
                'ma_trend_short',     # 단기 이평선 트렌드
                'ma_trend_mid',       # 중기 이평선 트렌드
                'ma_trend_long',      # 장기 이평선 트렌드
                'volume_trend',       # 거래량 추세
                'price_momentum'      # 가격 모멘텀
            ]
            
            self.logger.info(f"입력 특성 수: {len(features)}")
            self.logger.info("사용되는 특성:")
            for feature in features:
                self.logger.info(f"- {feature}")
            
            return len(features)
            
        except Exception as e:
            self.logger.error(f"입력 크기 계산 중 오류 발생: {str(e)}")
            raise