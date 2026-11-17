# -*- coding: utf-8 -*-
CONFIG = {
    'seed': 42,
    

    'LLM_config': {
        'model_name': 'glm-4-air',
        'API_KEY': '',
        'use_mock': False
    },

    # Configuration for Agents (Initialize and Hyper-parameters).
    'agent_config': {
        'default_score': {
            'memory': '',
            'emotion': 0.5,
            'social_confidence': 0.5,
            'opinion': ''
        },
        'execute_gap_params': {
            'alpha': 1.0,
            'beta': 14400,
            'T': 5.0
        },
        'lightweight_mode': {
            'enabled': True,
            'comment_threshold': 0.62,
            'reply_threshold': 0.58,
            'detail_threshold': 0.42,
            'leave_threshold': 0.18,
            'neutral_comment_probability': 0.14,
            'min_comment_probability': 0.05
        },
        'emotion_config': {
            'dimensions': ['anger', 'anxiety', 'disappointment', 'trust', 'positivity'],
            'negative_dimensions': ['anger', 'anxiety', 'disappointment'],
            'susceptibility_range': {
                'min': 0.5,
                'max': 1.5
            },
            'decay_rate': 0.02,
            'contagion_strength': 0.4,
            'contagion_detection_threshold': 0.02
        }
    },

    # Configuration for file paths.
    'data_path': {
        'normal_user': 'data/user_400.csv',
        'tweets': 'data/tweets.csv',
        'record': 'output/record.json',
        'load_checkpoint': 'checkpoint/check.pickle',
        'save_checkpoint': 'checkpoint/check.pickle',
        'analysis_output': 'output/analysis/',
        'emotion_timeline': 'output/emotion_timeline.csv',
        'network_graph': 'output/network_graph.json'
    },

    # Configuration of Simulation.
    'simulation_config': {
        'max_step': 5,
        'degree': 'medium',
        'baseline': 'full',
        'checkpointing': {
            'enabled': True,
            'save_every_n_events': 1000,
            'save_on_phase_transition': True
        },
        # 实验阶段配置
        'experiment_phases': {
            "baseline_duration": 0.15,
            "intervention_duration": 0.75,
            'allow_preemptive_intervention': False,
            'enable_auto_transition': True,

            'intervention_trigger': {
                'emotion_soothing': {
                    'anger_threshold': 0.3
                },
                'rumor_refutation': {
                    'negative_ratio_threshold': 0.4
                },
                'rational_guidance': {
                    'low_trust_ratio_threshold': 0.2
                }
            }
        },
        # 群体统计间隔
        'stats_interval_ratio': 0.05
    },

    # Configuration of Social Media.
    'media_config': {
        'tweet_index': 1,
        'comment_pagesize': 4,
        'reply_pagesize': 4,
        'content_weight': {
            'like_factor': 1.0,
            'visibility_factor': 1.0,
            'recency_factor': 0.1
        }
    },

    # Configuration for Intervention
    'intervention': {
        'runtime': {
            'check_interval_divisor': 30,
            'burst_interval_divisor': 60,
            'lightweight_observation': True
        },
        'comment_poisoning_detect': {
            'type': 'LLM',
            'split_time': 0.3,
            'detection_threshold': 0.5
        },

        'content_filtering': {
            'enabled': False,
            'negative_threshold': 0.5,
            'visibility_reduction': 0.2,
            'severe_threshold': 0.7,
            'filter_dimensions': {
                'anger': 0.6,
                'anxiety': 0.6,
                'negativity': 0.5,
                'misinformation': 0.4
            }
        },

        'rumor_refutation': {
            'enabled': True,
            'agent_count': 8,
            'activation_delay': 0.15,
            'cooldown_minutes': 2,
            'max_interventions': 10,
            'intervention_probability': 0.8
        },

        'emotion_soothing': {
            'enabled': True,
            'agent_count': 8,
            'activation_delay': 0.2,
            'cooldown_minutes': 2,
            'max_interventions': 8,
            'intervention_probability': 0.85
        },

        'rational_guidance': {
            'enabled': True,
            'agent_count': 8,
            'activation_delay': 0.25,
            'cooldown_minutes': 3,
            'max_interventions': 6,
            'intervention_probability': 0.7
        },

        'positive_content': {
            'enabled': False,
            'boost_factor': 2.0,
            'initial_likes': 5
        }
    },

    'analysis_config': {
        'emotion_heatmap': True,
        'network_visualization': True,
        'behavior_pattern_analysis': True,
        'intervention_effectiveness': True,
        'export_formats': ['json', 'csv', 'png'],
        # 分析参数
        'contagion_chain_min_length': 2,
        'emotion_threshold': {
            'high': 0.6,
            'medium': 0.4,
            'low': 0.2
        }
    },

    'time_config': {
        'start_time': '2025-08-10 06:00:00',
        'end_time': '2025-08-10 10:00:00',
        'epoch_num': 2
    }
}
