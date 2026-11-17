"""
记录器模块：负责记录模拟过程中的所有数据
"""
import json
import os
import time
from datetime import datetime
from typing import Dict, Any, List
from collections import defaultdict


class Recorder:
    """数据记录器类"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.record_path = config['data_path']['record']

        self.output_dir = os.path.dirname(self.record_path)
        if self.output_dir and not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.records = {
            'META': [],
            'TRAJ': [],
            'GROUP_STATS': [],
            'SUMMARY': [],
            'EMOTION': [],
            'BEHAVIOR': [],
            'INTERVENTION': [],
            'NETWORK': []
        }

        self.stats = {
            'total_records': 0,
            'start_time': datetime.now().isoformat(),
            'record_counts': defaultdict(int)
        }
        
    def add_record(self, record_type: str, record_data: Any):
        """添加记录"""
        if record_type not in self.records:
            print(f"警告：未知的记录类型 {record_type}")
            record_type = 'META'

        if isinstance(record_data, dict):
            record_data['_timestamp'] = time.time()
            record_data['_datetime'] = datetime.now().isoformat()
            
        self.records[record_type].append(record_data)
        self.stats['total_records'] += 1
        self.stats['record_counts'][record_type] += 1
        
    def add_trajectory(self, agent_id: int, trajectory_data: Dict):
        """添加代理轨迹"""
        self.add_record('TRAJ', {
            'agent_id': agent_id,
            'trajectory': trajectory_data
        })
        
    def add_emotion_event(self, event_data: Dict):
        """添加情绪事件"""
        self.add_record('EMOTION', event_data)
        
    def add_behavior_pattern(self, pattern_data: Dict):
        """添加行为模式"""
        self.add_record('BEHAVIOR', pattern_data)
        
    def add_intervention_event(self, intervention_data: Dict):
        """添加干预事件"""
        self.add_record('INTERVENTION', intervention_data)
        
    def write_prompt_level(self):
        """写入所有记录到文件"""
        self.stats['end_time'] = datetime.now().isoformat()

        output_data = {
            'metadata': {
                'version': '1.0',
                'simulation_type': 'enhanced_trendsim',
                'config': self.config,
                'stats': self.stats
            },
            'records': self.records
        }

        try:
            with open(self.record_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2, default=str)
            print(f"记录已保存到: {self.record_path}")

            self._write_separate_records()
            
        except Exception as e:
            print(f"保存记录时出错: {e}")
            backup_path = self.record_path.replace('.json', '_backup.json')
            try:
                with open(backup_path, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, ensure_ascii=False, indent=2, default=str)
                print(f"记录已保存到备份文件: {backup_path}")
            except Exception as e2:
                print(f"保存备份记录也失败: {e2}")
                
    def _write_separate_records(self):
        """分别保存不同类型的记录"""
        if self.records['TRAJ']:
            traj_path = os.path.join(self.output_dir, 'trajectories.json')
            self._write_json(traj_path, self.records['TRAJ'])

        if self.records['GROUP_STATS']:
            stats_path = os.path.join(self.output_dir, 'group_stats.json')
            self._write_json(stats_path, self.records['GROUP_STATS'])

            self._write_emotion_timeline_csv()

        if self.records['EMOTION']:
            emotion_path = os.path.join(self.output_dir, 'emotion_events.json')
            self._write_json(emotion_path, self.records['EMOTION'])

        if self.records['INTERVENTION']:
            intervention_path = os.path.join(self.output_dir, 'interventions.json')
            self._write_json(intervention_path, self.records['INTERVENTION'])
            
    def _write_json(self, filepath: str, data: Any):
        """写入JSON文件"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        except Exception as e:
            print(f"写入文件 {filepath} 失败: {e}")
            
    def _write_emotion_timeline_csv(self):
        """生成情绪时间线CSV文件"""
        csv_path = os.path.join(self.output_dir, 'emotion_timeline.csv')
        
        try:
            import csv
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)

                writer.writerow([
                    'timestamp', 'datetime', 'phase',
                    'anger_avg', 'anxiety_avg', 'disappointment_avg', 
                    'trust_avg', 'positivity_avg',
                    'high_anger_count', 'high_anxiety_count', 
                    'low_trust_count', 'negative_ratio'
                ])

                for stat in self.records['GROUP_STATS']:
                    if 'emotion_averages' in stat:
                        emotions = stat['emotion_averages']
                        high_counts = stat.get('high_emotion_counts', {})
                        media_state = stat.get('social_media_state', {})
                        
                        writer.writerow([
                            stat.get('timestamp', ''),
                            stat.get('_datetime', ''),
                            stat.get('phase', ''),
                            emotions.get('anger', 0),
                            emotions.get('anxiety', 0),
                            emotions.get('disappointment', 0),
                            emotions.get('trust', 0.5),
                            emotions.get('positivity', 0.5),
                            high_counts.get('anger', 0),
                            high_counts.get('anxiety', 0),
                            high_counts.get('low_trust', 0),
                            media_state.get('negative_comment_ratio', 0)
                        ])
                        
            print(f"情绪时间线CSV已保存到: {csv_path}")
            
        except Exception as e:
            print(f"生成CSV文件失败: {e}")
            
    def get_summary_statistics(self) -> Dict[str, Any]:
        """获取摘要统计信息"""
        summary = {
            'total_records': self.stats['total_records'],
            'record_types': dict(self.stats['record_counts']),
            'trajectory_count': len(self.records['TRAJ']),
            'emotion_events': len(self.records['EMOTION']),
            'intervention_events': len(self.records['INTERVENTION']),
            'time_range': {
                'start': self.stats['start_time'],
                'end': self.stats.get('end_time', 'ongoing')
            }
        }

        if self.records['GROUP_STATS']:
            max_anger = max((s.get('emotion_averages', {}).get('anger', 0) 
                           for s in self.records['GROUP_STATS']), default=0)
            max_anxiety = max((s.get('emotion_averages', {}).get('anxiety', 0) 
                            for s in self.records['GROUP_STATS']), default=0)
            
            summary['emotion_peaks'] = {
                'max_anger': max_anger,
                'max_anxiety': max_anxiety
            }

            if self.records['GROUP_STATS']:
                final_stats = self.records['GROUP_STATS'][-1]
                summary['final_state'] = {
                    'phase': final_stats.get('phase', 'unknown'),
                    'emotion_averages': final_stats.get('emotion_averages', {}),
                    'negative_ratio': final_stats.get('social_media_state', {}).get('negative_comment_ratio', 0)
                }
                
        return summary
        
    def export_for_analysis(self, export_path: str = None):
        """导出数据供分析使用"""
        if export_path is None:
            export_path = os.path.join(self.output_dir, 'analysis_data.json')
            
        analysis_data = {
            'group_stats_history': self.records['GROUP_STATS'],
            'emotion_contagion_events': self.records['EMOTION'],
            'behavior_patterns': self._extract_behavior_patterns(),
            'intervention_events': self.records['INTERVENTION'],
            'summary': self.get_summary_statistics()
        }
        
        self._write_json(export_path, analysis_data)
        return analysis_data
        
    def _extract_behavior_patterns(self) -> Dict[str, int]:
        """从轨迹中提取行为模式"""
        patterns = defaultdict(int)
        
        for traj_record in self.records['TRAJ']:
            if 'trajectory' not in traj_record:
                continue
                
            trajectory = traj_record['trajectory']
            for step in trajectory:
                action = step.get('action', '')
                state = step.get('State', '')

                pattern_key = f"{state}_{action}"
                patterns[pattern_key] += 1

                if action == 'Comment':
                    action_detail = step.get('action_detail', {})
                    emotion_state = action_detail.get('emotion_state', {})
                    if emotion_state.get('anger', 0) > 0.6:
                        patterns['angry_comment'] += 1
                    elif emotion_state.get('anxiety', 0) > 0.6:
                        patterns['anxious_comment'] += 1
                        
        return dict(patterns)