import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import json
from matplotlib.lines import Line2D
'''
We used an LLM to generate this code.
'''

def plot_learning_curves_from_json(json_path, title='PPO Learning Curves Across Seeds', save_path=None):
    with open(json_path, 'r') as f:
        results_data = json.load(f)
    return plot_learning_curves(results_data, title, save_path)


def plot_learning_curves(results_data, title='PPO Learning Curves Across Seeds', save_path=None):
    plot_data = []
    for result in results_data:
        seed = result['seed']
        learning_curve = result['learning_curve']
        for point in learning_curve:
            plot_data.append({
                'timesteps': point['timesteps'],
                'avg_reward': point['avg_reward'],
                'seed': seed
            })

    df = pd.DataFrame(plot_data)
    plt.figure(figsize=(12, 8))

    sns.lineplot(data=df, x='timesteps', y='avg_reward', units='seed',
                 estimator=None, alpha=0.3, color='gray', linewidth=1)
    sns.lineplot(data=df, x='timesteps', y='avg_reward',
                 estimator='mean', ci=95, linewidth=2.5, color='blue')

    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Timesteps', fontsize=14)
    plt.ylabel('Average Episode Reward', fontsize=14)
    plt.grid(True, alpha=0.3)

    legend_elements = [
        Line2D([0], [0], color='gray', alpha=0.3, linewidth=1, label='Individual Seeds'),
        Line2D([0], [0], color='blue', linewidth=2.5, label='Mean')
    ]
    plt.legend(handles=legend_elements, loc='lower right')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Plot saved to {save_path}')
    plt.show()

    final_rewards = df.groupby('seed')['avg_reward'].last()
    print(f'\nFinal Performance Summary:')
    print(f'Mean final reward: {final_rewards.mean():.4f} ± {final_rewards.std():.4f}')
    print(f'Best seed: {final_rewards.idxmax()} (reward: {final_rewards.max():.4f})')
    print(f'Worst seed: {final_rewards.idxmin()} (reward: {final_rewards.min():.4f})')


def plot_episode_length_curves(results_data, title='Average Episode Length Across Seeds', save_path=None):
    plot_data = []

    for result in results_data:
        seed = result['seed']
        for point in result['episode_length_curve']:
            plot_data.append({
                'timesteps': point['timesteps'],
                'avg_episode_length': point['avg_episode_length'],
                'seed': seed
            })

    df = pd.DataFrame(plot_data)
    plt.figure(figsize=(12, 8))

    sns.lineplot(data=df, x='timesteps', y='avg_episode_length', units='seed',
                 estimator=None, alpha=0.3, color='gray', linewidth=1)
    sns.lineplot(data=df, x='timesteps', y='avg_episode_length',
                 estimator='mean', ci=95, linewidth=2.5, color='green')

    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Timesteps', fontsize=14)
    plt.ylabel('Average Episode Length', fontsize=14)
    plt.grid(True, alpha=0.3)

    legend_elements = [
        Line2D([0], [0], color='gray', alpha=0.3, linewidth=1, label='Individual Seeds'),
        Line2D([0], [0], color='green', linewidth=2.5, label='Mean')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Plot saved to {save_path}')
    plt.show()

def plot_success_rate_curves(results_data, title='Success Rate Across Seeds', save_path=None):
    '''Plot success rate curves across multiple seeds'''
    plot_data = []
    
    for result in results_data:
        seed = result['seed']
        for point in result['success_rate_curve']:
            plot_data.append({
                'timesteps': point['timesteps'],
                'success_rate': point['success_rate'],
                'seed': seed
            })

    df = pd.DataFrame(plot_data)
    plt.figure(figsize=(12, 8))

    sns.lineplot(data=df, x='timesteps', y='success_rate', units='seed',
                 estimator=None, alpha=0.3, color='gray', linewidth=1)
    
    sns.lineplot(data=df, x='timesteps', y='success_rate',
                 estimator='mean', ci=95, linewidth=2.5, color='orange')

    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Timesteps', fontsize=14)
    plt.ylabel('Success Rate', fontsize=14)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)

    legend_elements = [
        Line2D([0], [0], color='gray', alpha=0.3, linewidth=1, label='Individual Seeds'),
        Line2D([0], [0], color='orange', linewidth=2.5, label='Mean')
    ]
    plt.legend(handles=legend_elements, loc='lower right')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Plot saved to {save_path}')
    plt.show()

    final_success_rates = df.groupby('seed')['success_rate'].last()
    print(f'\nFinal Success Rate Summary:')
    print(f'Mean success rate: {final_success_rates.mean():.4f} ± {final_success_rates.std():.4f}')
    print(f'Best seed: {final_success_rates.idxmax()} (success rate: {final_success_rates.max():.4f})')
    print(f'Worst seed: {final_success_rates.idxmin()} (success rate: {final_success_rates.min():.4f})')


def plot_policy_behavior_curves(results_data, metric, title=None, save_path=None):
    assert metric in ['policy_loss', 'total_loss', 'entropy_loss'], \
        f'Metric {metric} not used.'

    plot_data = []
    for result in results_data:
        seed = result['seed']
        for point in result['policy_behavior_curve']:
            plot_data.append({
                'timesteps': point['timesteps'],
                metric: point[metric],
                'seed': seed
            })

    df = pd.DataFrame(plot_data)
    plt.figure(figsize=(12, 8))

    sns.lineplot(data=df, x='timesteps', y=metric, units='seed',
                 estimator=None, alpha=0.3, color='gray', linewidth=1)
    sns.lineplot(data=df, x='timesteps', y=metric,
                 estimator='mean', ci=95, linewidth=2.5, color='red')

    if not title:
        title = f"{metric.replace('_', ' ').title()} Over Timesteps"

    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Timesteps', fontsize=14)
    plt.ylabel(metric.replace('_', ' ').title(), fontsize=14)
    plt.grid(True, alpha=0.3)

    legend_elements = [
        Line2D([0], [0], color='gray', alpha=0.3, linewidth=1, label='Individual Seeds'),
        Line2D([0], [0], color='red', linewidth=2.5, label='Mean')
    ]
    plt.legend(handles=legend_elements, loc='best')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Plot saved to {save_path}')
    plt.show()


# Example usage
if __name__ == '__main__':
    json_path = 'results/MiniGrid-DoorKey-8x8-v0_SIL_results.json'
    
    with open(json_path, 'r') as f:
        results = json.load(f)

    # Plot reward curve
    plot_learning_curves(results, title='PPO Learning Curve')

    # Plot episode length
    plot_episode_length_curves(results, title='Episode Length Curve')
    # Plot Sucess Rate
    plot_success_rate_curves(results, title='PPO Success Rate')

    # Plot policy behavior metrics
    plot_policy_behavior_curves(results, metric='policy_loss')
    plot_policy_behavior_curves(results, metric='total_loss')
    plot_policy_behavior_curves(results, metric='entropy_loss')
