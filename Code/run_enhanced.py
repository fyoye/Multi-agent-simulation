import os
os.environ['TZ'] = 'Asia/Shanghai'
import argparse
import time
import json
import os
from datetime import datetime

import random

from config_enhanced import CONFIG
random.seed(CONFIG.get('seed', 42))

from llm import create_LLM
from recorder import Recorder
from simulator_enhanced import Simulator
from utils_enhanced import CheckPoint, update_config
from analysis_module import SimulationAnalyzer


def run(load_checkpoint=False, analyze_results=True):
    """
    运行增强版模拟

    Args:
        load_checkpoint: 是否从检查点恢复
        analyze_results: 是否在模拟后自动分析结果
    """

    print("=== 启动增强版TrendSim模拟系统 ===")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"配置: {CONFIG['simulation_config']['degree']} 攻击强度, "
          f"{CONFIG['simulation_config']['baseline']} 基线")

    output_dirs = [
        CONFIG['data_path']['analysis_output'],
        os.path.dirname(CONFIG['data_path']['record']),
        os.path.dirname(CONFIG['data_path']['save_checkpoint'])
    ]
    for dir_path in output_dirs:
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path)

    if not load_checkpoint:
        print("\n初始化新的模拟...")

        recorder = Recorder(CONFIG)
        recorder.add_record('META', {
            'configuration': CONFIG,
            'start_time': datetime.now().isoformat(),
            'version': 'enhanced_v1.0'
        })

        st = time.time()

        print("初始化语言模型...")
        llm = create_LLM(CONFIG)

        checkpoint = CheckPoint(CONFIG, st)

        print("创建模拟环境...")
        simulator = Simulator(CONFIG, llm, recorder, checkpoint)

        print(f"\n模拟参数:")
        print(f"- 代理总数: {len(simulator.agents)}")
        print(f"- 攻击者数量: {len([a for a in simulator.agents if a.type != 'normal'])}")
        print(f"- 干预代理数量: {len(simulator.intervention_agents)}")
        print(f"- 模拟时长: {CONFIG['time_config']['start_time']} 至 {CONFIG['time_config']['end_time']}")

        print("\n开始模拟...")
        simulator.run()

        ed = time.time()
        total_time = ed - st

        print(f"\n模拟完成！总用时: {total_time:.2f} 秒")

        recorder.add_record('META', {
            'simulation_total_time': total_time,
            'end_time': datetime.now().isoformat()
        })

        recorder.write_prompt_level()

        analysis_data = simulator.get_analysis_data()
        analysis_data['summary'] = simulator.generate_summary_report()

    else:
        print("\n从检查点恢复模拟...")
        
        checkpoint = CheckPoint(CONFIG, None)
        if not checkpoint.load_checkpoint():
            print("检查点加载失败，将从头开始运行")
            recorder = Recorder(CONFIG)
            recorder.add_record('META', {
                'configuration': CONFIG,
                'start_time': datetime.now().isoformat(),
                'version': 'enhanced_v1.0',
                'note': '检查点加载失败，重新初始化'
            })
            st = time.time()
            llm = create_LLM(CONFIG)
            checkpoint = CheckPoint(CONFIG, st)
            simulator = Simulator(CONFIG, llm, recorder, checkpoint)
        else:
            simulator = checkpoint.simulator
            if simulator is None:
                raise ValueError("检查点中未找到有效的simulator对象")
            recorder = simulator.recorder

        print("继续运行模拟...")
        simulator.run()

        recorder.add_record('META', {
            'simulation_total_time': '[CHECKPOINT]',
            'checkpoint_resume': datetime.now().isoformat()
        })
        recorder.write_prompt_level()

        analysis_data = simulator.get_analysis_data()
        analysis_data['summary'] = simulator.generate_summary_report()

    if analyze_results:
        print("\n=== 开始分析模拟结果 ===")
        analyzer = SimulationAnalyzer(CONFIG)

        try:
            report = analyzer.analyze_simulation_results(analysis_data)

            print("\n=== 关键发现 ===")
            print(f"最高愤怒水平: {report['summary']['emotion_peaks']['max_anger']:.2%}")
            print(f"最高焦虑水平: {report['summary']['emotion_peaks']['max_anxiety']:.2%}")
            print(f"平均信任度: {report['summary']['emotion_peaks']['avg_trust']:.2%}")
            print(f"情绪传染事件: {report['summary']['contagion']['total_events']} 次")

            if 'intervention' in report['summary'] and report['summary']['intervention']:
                print(f"\n干预效果:")
                print(f"- 愤怒降低(干预后窗口): {report['summary']['intervention'].get('anger_reduction_percentage', 0):.1f}%")
                print(f"- 干预阶段即时降低: {report['summary']['intervention'].get('intervention_phase_anger_reduction_percentage', 0):.1f}%")
                print(f"- 干预评论数: {report['summary']['intervention'].get('intervention_comments', 0)}")
                print(f"- 过滤评论数: {report['summary']['intervention'].get('filtered_comments', 0)}")

            print(f"\n分析报告已保存至: {CONFIG['data_path']['analysis_output']}")

        except Exception as e:
            print(f"分析过程中出现错误: {e}")
            import traceback
            traceback.print_exc()

    print("\n=== 模拟和分析完成 ===")


def batch_run(experiment_configs):
    """
    批量运行多个实验配置（用于参数对比、变量控制等实验设计）

    Args:
        experiment_configs: 实验配置列表，每个配置为一组参数组合
    """
    results = []

    for i, exp_config in enumerate(experiment_configs):
        print(f"\n\n{'=' * 50}")
        print(f"实验 {i + 1}/{len(experiment_configs)}")
        print(f"{'=' * 50}")

        global CONFIG
        for key, value in exp_config.items():
            if '.' in key:
                parts = key.split('.')
                current = CONFIG
                for part in parts[:-1]:
                    current = current[part]
                current[parts[-1]] = value
            else:
                CONFIG[key] = value

        exp_name = f"exp_{i + 1}_{exp_config.get('name', 'unnamed')}"
        CONFIG['data_path']['record'] = f'output/{exp_name}/record.json'
        CONFIG['data_path']['analysis_output'] = f'output/{exp_name}/analysis/'

        try:
            run(load_checkpoint=False, analyze_results=True)
            results.append({
                'experiment': exp_name,
                'config': exp_config,
                'status': 'success'
            })
        except Exception as e:
            print(f"实验 {exp_name} 失败: {e}")
            results.append({
                'experiment': exp_name,
                'config': exp_config,
                'status': 'failed',
                'error': str(e)
            })

    with open('output/batch_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n\n批量实验完成！成功: {sum(1 for r in results if r['status'] == 'success')}/{len(results)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='运行增强版TrendSim模拟')

    parser.add_argument('--checkpoint', action='store_true',
                        help='从检查点恢复运行')
    parser.add_argument('--no-analysis', action='store_true',
                        help='跳过自动分析')

    parser.add_argument('--degree', choices=['mild', 'medium', 'severe'],
                        help='攻击强度')
    parser.add_argument('--baseline', choices=['clean', 'full'],
                        help='基线配置')
    parser.add_argument('--intervention', choices=['none', 'rumor', 'emotion', 'rational', 'all'],
                        help='干预类型')
    parser.add_argument('--defense-time', type=float,
                        help='防御触发时间（0-1）')

    parser.add_argument('--batch', action='store_true',
                        help='运行批量实验')

    args, extra_args = parser.parse_known_args()

    if args.degree:
        CONFIG['simulation_config']['degree'] = args.degree
    if args.baseline:
        CONFIG['simulation_config']['baseline'] = args.baseline

    if args.intervention:
        if args.intervention == 'none':
            CONFIG['intervention']['rumor_refutation']['enabled'] = False
            CONFIG['intervention']['emotion_soothing']['enabled'] = False
            CONFIG['intervention']['rational_guidance']['enabled'] = False
        elif args.intervention == 'rumor':
            CONFIG['intervention']['rumor_refutation']['enabled'] = True
            CONFIG['intervention']['emotion_soothing']['enabled'] = False
            CONFIG['intervention']['rational_guidance']['enabled'] = False
        elif args.intervention == 'emotion':
            CONFIG['intervention']['rumor_refutation']['enabled'] = False
            CONFIG['intervention']['emotion_soothing']['enabled'] = True
            CONFIG['intervention']['rational_guidance']['enabled'] = False
        elif args.intervention == 'rational':
            CONFIG['intervention']['rumor_refutation']['enabled'] = False
            CONFIG['intervention']['emotion_soothing']['enabled'] = False
            CONFIG['intervention']['rational_guidance']['enabled'] = True
        elif args.intervention == 'all':
            CONFIG['intervention']['rumor_refutation']['enabled'] = True
            CONFIG['intervention']['emotion_soothing']['enabled'] = True
            CONFIG['intervention']['rational_guidance']['enabled'] = True

    if args.defense_time is not None:
        CONFIG['intervention']['comment_poisoning_detect']['split_time'] = args.defense_time

    CONFIG = update_config(CONFIG, extra_args)
    if args.batch:
        experiments = [
            {'name': 'baseline_mild', 'simulation_config.degree': 'mild',
             'intervention.rumor_refutation.enabled': False},
            {'name': 'baseline_medium', 'simulation_config.degree': 'medium',
             'intervention.rumor_refutation.enabled': False},
            {'name': 'baseline_severe', 'simulation_config.degree': 'severe',
             'intervention.rumor_refutation.enabled': False},
            {'name': 'intervention_mild', 'simulation_config.degree': 'mild',
             'intervention.rumor_refutation.enabled': True},
            {'name': 'intervention_medium', 'simulation_config.degree': 'medium',
             'intervention.rumor_refutation.enabled': True},
            {'name': 'intervention_severe', 'simulation_config.degree': 'severe',
             'intervention.rumor_refutation.enabled': True},
        ]
        batch_run(experiments)
    else:
        run(load_checkpoint=args.checkpoint,
            analyze_results=not args.no_analysis)
