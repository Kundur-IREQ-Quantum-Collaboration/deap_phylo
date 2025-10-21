import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
from collections import defaultdict
from deap import gp


class EvolutionaryCategoryObserver:
    def __init__(self, primitive_categories: Dict[str, str]):
        self.primitive_categories = primitive_categories
        self.unique_categories = list(set(primitive_categories.values()))
        self.generation_history = []
        self.individual_history = []
        
    def analyze_individual(self, individual) -> Dict[str, float]:
        category_counts = defaultdict(int)
        tracked_nodes = 0
        
        # Track nodes that are NOT operations (i.e., actual kernels/functions we care about)
        for node in individual:
            primitive_name = getattr(node, 'name', str(node))
            if primitive_name in self.primitive_categories:
                category = self.primitive_categories[primitive_name]
                # Only track if it's not an operation
                if category != 'operation':
                    category_counts[category] += 1
                    tracked_nodes += 1
        
        return {cat: category_counts[cat] / max(tracked_nodes, 1) for cat in self.unique_categories}
    
    def track_population(self, population: List, generation: int, fitnesses: Optional[List[float]] = None):
        gen_data = []
        for i, individual in enumerate(population):
            percentages = self.analyze_individual(individual)
            percentages.update({'generation': generation, 'fitness': fitnesses[i] if fitnesses else 0})
            gen_data.append(percentages)
            self.individual_history.append(percentages)
        
        stats = {f'{cat}_mean': np.mean([p[cat] for p in gen_data]) for cat in self.unique_categories}
        stats['generation'] = generation
        self.generation_history.append(stats)
        
    def get_evolution_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.generation_history)
    
    def get_fitness_correlation(self, category: str) -> float:
        df = pd.DataFrame(self.individual_history)
        return df[category].corr(df['fitness']) if 'fitness' in df.columns else 0.0
    
    def get_summary(self) -> Dict[str, any]:
        """Get comprehensive summary of category evolution"""
        if not self.generation_history:
            return {}
        
        df = pd.DataFrame(self.individual_history)
        
        # Initial and final distributions
        initial_dist = {cat: self.generation_history[0][f'{cat}_mean'] for cat in self.unique_categories}
        final_dist = {cat: self.generation_history[-1][f'{cat}_mean'] for cat in self.unique_categories}
        
        # Changes
        change = {cat: final_dist[cat] - initial_dist[cat] for cat in self.unique_categories}
        
        # Fitness correlations
        fitness_correlations = {cat: self.get_fitness_correlation(cat) for cat in self.unique_categories}
        
        return {
            'categories': self.unique_categories,
            'initial_distribution': initial_dist,
            'final_distribution': final_dist,
            'change': change,
            'fitness_correlations': fitness_correlations
        }
    
    def plot_category_evolution(self, figsize=(10, 6)):
        """Line plot showing category evolution over generations"""
        df = self.get_evolution_dataframe()
        if df.empty:
            return None
        
        # Filter out 'operation' category
        categories = [cat for cat in self.unique_categories if cat != 'operation']
        
        fig, ax = plt.subplots(figsize=figsize)
        colors = plt.cm.Set3(np.linspace(0, 1, len(categories)))
        
        for i, category in enumerate(categories):
            mean_col = f'{category}_mean'
            if mean_col in df.columns:
                ax.plot(df['generation'], df[mean_col] * 100, 
                        label=category, color=colors[i], linewidth=2)
        
        ax.set_xlabel('Generation')
        ax.set_ylabel('Usage Percentage (%)')
        ax.set_title('Category Evolution Over Generations')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_category_distribution(self, figsize=(10, 8)):
        """Pie chart showing final category distribution with improved layout"""
        summary = self.get_summary()
        if not summary:
            return None
            
        final_dist = summary['final_distribution']
        
        # Filter out 'operation' category
        categories = [cat for cat in final_dist.keys() if cat != 'operation']
        values = [final_dist[cat] * 100 for cat in categories]
        colors = plt.cm.Set3(np.linspace(0, 1, len(categories)))
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create pie chart with better spacing for labels
        wedges, texts, autotexts = ax.pie(
            values, 
            labels=categories, 
            colors=colors, 
            autopct='%1.1f%%',
            startangle=90,
            pctdistance=0.85,
            labeldistance=1.15  # Push labels further out
        )
        
        # Adjust text properties for better readability
        for text in texts:
            text.set_fontsize(10)
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(9)
            autotext.set_weight('bold')
        
        ax.set_title('Final Category Distribution', fontsize=12, pad=20)
        
        # Add a legend if there are many categories
        if len(categories) > 6:
            ax.legend(wedges, categories, loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
        
        plt.tight_layout()
        return fig
    
    def plot_fitness_correlations(self, figsize=(8, 6)):
        """Bar chart showing category-fitness correlations"""
        summary = self.get_summary()
        if not summary:
            return None
            
        correlations = summary['fitness_correlations']
        
        # Filter out 'operation' category
        categories = [cat for cat in correlations.keys() if cat != 'operation']
        corr_values = [correlations[cat] for cat in categories]
        
        fig, ax = plt.subplots(figsize=figsize)
        colors = ['green' if c > 0 else 'red' for c in corr_values]
        
        ax.barh(categories, corr_values, color=colors, alpha=0.7)
        ax.set_xlabel('Correlation with Fitness')
        ax.set_title('Category-Fitness Correlations')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_xlim([-1, 1])
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        return fig

