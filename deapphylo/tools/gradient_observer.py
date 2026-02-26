import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Dict, List, Optional
from util.symbolic_metric import *
from util.neuronal_inputs import *
from util.activations_base import Activation, ActivationRegistry
from util.register_activations import *

from autofd import Grid

class EvolutionaryCategoryGradientObserver:
    def __init__(self, registry: ActivationRegistry, data: torch.Tensor):
        self.nn_map = {}
        self.nn_inputs = data

        self.generation_history = []
        self.individual_history = []

        self.registry = registry
        self.terminals = registry.terminals()
        self.operations = registry.operations()

        self.family_map = registry.activation_names_by_category()
        self.unique_categories = list(self.family_map.keys())
        self.func_map = registry.sympy_func_map()

        self.torch_activation_types = tuple(
            cls for _, cls in inspect.getmembers(nn.modules.activation, inspect.isclass)
            if issubclass(cls, nn.Module)
            and cls is not nn.Module
            and cls.__module__ == "torch.nn.modules.activation"
        )
        self.activation_classes = self.torch_activation_types + (Activation,)
        self.nan_count = 0

    def analyze_individual(self, individual) -> Dict[str, float]:
        print("INDIVIDUAL", str(individual))
        primitive_gradients = {}
        inputs, _, acts, exprs = get_activation_inputs_and_exprs_from_model(self.nn_map[str(individual)], self.nn_inputs, activation_classes=(self.activation_classes))
        for act_name, _ in acts.items():
            expr = exprs[act_name]
            for primitive in self.terminals.keys():
                primitive_measure = compute_gateaux_measure(expr, self.func_map, sp.Function(primitive), inputs[act_name])
                if primitive_measure is None or primitive_measure != primitive_measure:
                    print(f"Activation {act_name} is not well-defined on input domain.")
                    return None
                primitive_gradients[primitive] = primitive_gradients.get(primitive, 0) + primitive_measure
        
        gradients_sum = sum(primitive_gradients.values())
        normalization_factor = 1.0 / gradients_sum if gradients_sum != 0 else 0.0
        for k in primitive_gradients.keys():
            primitive_gradients[k] = primitive_gradients[k] * normalization_factor
        print("Normalized gradients:", primitive_gradients)
        print("")
        return primitive_gradients
    
    def analyze_individual_autofd(self, individual) -> Dict[str, float]:

        primitive_gradients = {}
        for primitive in self.terminals.keys():
            primitive_gradients[primitive] = 0.0

        inputs, _, acts, _ = get_activation_inputs_and_exprs_from_model(self.nn_map[str(individual)], self.nn_inputs, activation_classes=(self.activation_classes))
        for act_name, act in acts.items():
            input = inputs[act_name]
            xs, nan_frac = self.process_inputs(input)
            if nan_frac > 0:
                print(f"Warning: {nan_frac*100:.2f}% of inputs for {act_name} are NaN")
                return None
            weights = jnp.ones_like(xs) / xs.size

            fs, fns, ids = act.get_jax_fn()
            for fn in fns:
                fn.grid = Grid(nodes=(xs,), weights=(weights,))
            def F(*args):
                if len(args) != len(fns):
                    raise TypeError(f"Expected exactly {len(fns)} positional arguments, got {len(args)}")
                return o.integrate(fs(*args))
        
            dF_s = jax.grad(F, argnums=range(len(ids)))(*fns)         
            gradients = [jnp.sqrt(jnp.sum(jnp.square(jax.vmap(df)(xs)))).item() for df in dF_s]

            for i, id in enumerate(ids):
                primitive_gradients[id] += gradients[i]
        
        gradients_sum = sum(primitive_gradients.values())
        normalization_factor = 1.0 / gradients_sum if gradients_sum != 0 else 0.0
        for k in primitive_gradients.keys():
            primitive_gradients[k] = primitive_gradients[k] * normalization_factor

        return primitive_gradients
    
    def analyze_individual_sympy(self, individual) -> Dict[str, float]:

        primitive_gradients = {}
        for primitive in self.terminals.keys():
            primitive_gradients[primitive] = 0.0

        inputs, _, acts, exprs = get_activation_inputs_and_exprs_from_model(self.nn_map[str(individual)], self.nn_inputs, activation_classes=(self.activation_classes))
        for act_name, _ in acts.items():
            input = inputs[act_name]
            expr = exprs[act_name]

            for primitive in self.terminals.keys():
                dF = functional_derivative(sp.simplify(expr), primitive_to_sp(primitive))
                simplified_dF = sp.simplify(dF)
                callable_dF = sympy_expr_to_torch_callable(simplified_dF, self.func_map)
                primitive_measure = compute_empirical_measure_torch(input, callable_dF)

                if primitive_measure != primitive_measure:  # Check for NaN
                    return None
                primitive_gradients[primitive] = primitive_gradients.get(primitive, 0) + primitive_measure
        
        gradients_sum = sum(primitive_gradients.values())
        normalization_factor = 1.0 / gradients_sum if gradients_sum != 0 else 0.0
        for k in primitive_gradients.keys():
            primitive_gradients[k] = primitive_gradients[k] * normalization_factor

        return primitive_gradients
    
    def track_population(self, population: List, generation: int, fitnesses: Optional[List[float]] = None):
        gen_data = []
        for i, individual in enumerate(population):
            percentages = self.analyze_individual(individual)
            if percentages is None:
                self.nan_count += 1
                continue
            percentages.update({
                'generation': generation,
                'fitness': fitnesses[i] if fitnesses is not None else np.nan
            })
            gen_data.append(percentages)
            self.individual_history.append(percentages)
        
        stats = {
            'mean': {
                cat: np.mean([
                    np.sum([p.get(k,0) for k in self.terminals.keys() if self.terminals[k].category == cat]) 
                    for p in gen_data
                ]) 
                for cat in self.unique_categories 
            }
        }
        stats['generation'] = generation
        structural_analysis = self.analyze_structure_generation(population, generation)

        stats['terminal_structural'] = structural_analysis['terminals']['category_structural']
        stats['terminal_presence'] = structural_analysis['terminals']['category_presence']

        stats['operations_structural'] = structural_analysis['operations']['structural_ratio']
        stats['operations_presence'] = structural_analysis['operations']['presence_ratio']

        self.generation_history.append(stats)
        
    def get_evolution_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.generation_history)
    
    def get_fitness_correlation(self, category: str) -> float:
        df = pd.DataFrame(self.individual_history)
        # Compute mean per family
        cols = self.family_map[category]
        valid_cols = [c for c in cols if c in df.columns]
        if valid_cols:
            df[category] = df[valid_cols].mean(axis=1)

        return df[category].corr(df['fitness']) if 'fitness' in df.columns else 0.0
    
    def get_summary(self) -> Dict[str, any]:
        """Get comprehensive summary of category evolution"""
        if not self.generation_history:
            return {}
        
        initial_dist = self.generation_history[0][f'mean'] 
        final_dist = self.generation_history[-1]['mean']

        change = {cat: final_dist[cat] - initial_dist[cat] for cat in self.unique_categories}
        fitness_correlations = {cat: self.get_fitness_correlation(cat) for cat in self.unique_categories}

        initial_structural_analysis_term = self.generation_history[0]['terminal_structural']
        initial_structural_analysis_op = self.generation_history[0]['operations_structural']
        final_structural_analysis_term = self.generation_history[-1]['terminal_structural']
        final_structural_analysis_op = self.generation_history[-1]['operations_structural']
        
        return {
            'categories': self.unique_categories,
            'initial_distribution': initial_dist,
            'final_distribution': final_dist,
            'change': change,
            'fitness_correlations': fitness_correlations,
            'initial_structural_term': initial_structural_analysis_term,
            'initial_structural_op': initial_structural_analysis_op,
            'final_structural_term': final_structural_analysis_term,
            'final_structural_op': final_structural_analysis_op,
        }
    
    def plot_category_evolution(self, figsize=(10, 6)):
        df = self.get_evolution_dataframe()
        if df.empty:
            return None
        
        fig, ax = plt.subplots(figsize=figsize)
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.unique_categories)))
        
        for i, category in enumerate(self.unique_categories):
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
    
    def freeze(self):
        self.nn_map = {}
        if hasattr(self.nn_inputs, "cpu"):
            self.nn_inputs = self.nn_inputs.cpu()

    def export_history(self):
        out = {
            "individual_history": self.individual_history,
            "generation_history": self.generation_history,
        }
        return out
    
    def process_inputs(self, input_tensor: torch.Tensor):
        x_flat = input_tensor.detach().cpu().view(-1)
        valid_mask = ~torch.isnan(x_flat)  
        
        n_total = x_flat.numel()
        n_nan = (~valid_mask).sum().item()
        fraction_nan = n_nan / n_total
        
        x_clean = x_flat[valid_mask]
        xs = jnp.array(x_clean)
        
        return xs, fraction_nan
    
    def count_individual_nodes(self, individual, key_dict):
        counts = {k: 0 for k in key_dict.keys()}

        for node in individual:
            name = getattr(node, "name", None)
            if name in counts:
                counts[name] += 1
            # total_nodes += 1

        return counts

    def analyze_counts(self, population, key_dict, category_map=None):

        indiv_presence = {k: 0 for k in key_dict.keys()}
        total_counts = {k: 0 for k in key_dict.keys()}

        if category_map:
            cat_presence = {cat: 0 for cat in self.unique_categories}

        for individual in population:
            counts = self.count_individual_nodes(individual, key_dict)

            # primitive presence
            for k, v in counts.items():
                if v > 0:
                    indiv_presence[k] += 1
                total_counts[k] += v

            # category presence (union within individual)
            if category_map:
                seen_cats = set()
                for k, v in counts.items():
                    if v > 0:
                        seen_cats.add(category_map[k])
                for cat in seen_cats:
                    cat_presence[cat] += 1

        n_individuals = len(population)
        total_nodes_all = sum(total_counts.values())

        presence_ratio = {
            k: indiv_presence[k] / n_individuals if n_individuals else 0
            for k in indiv_presence
        }

        structural_ratio = {
            k: total_counts[k] / total_nodes_all if total_nodes_all else 0
            for k in total_counts
        }

        if category_map:
            category_presence_ratio = {
                cat: cat_presence[cat] / n_individuals if n_individuals else 0
                for cat in cat_presence
            }
            return presence_ratio, structural_ratio, category_presence_ratio
        
        return presence_ratio, structural_ratio

    def aggregate_terminal_categories(self, ratio_dict):
        category_totals = {cat: 0.0 for cat in self.unique_categories}

        for term_name, value in ratio_dict.items():
            cat = self.terminals[term_name].category
            category_totals[cat] += value

        return category_totals

    def analyze_structure_generation(self, population, generation):

        term_cat_map = self.registry.terminal_category_lookup()

        term_presence, term_struct, term_cat_presence = self.analyze_counts(
            population, self.terminals, term_cat_map
        )

        op_presence, op_struct = self.analyze_counts(
            population, self.operations
        )

        term_cat_struct = self.aggregate_terminal_categories(term_struct)

        return {
            "generation": generation,
            "terminals": {
                "presence_ratio": term_presence,
                "structural_ratio": term_struct,
                "category_presence": term_cat_presence,
                "category_structural": term_cat_struct,
            },
            "operations": {
                "presence_ratio": op_presence,
                "structural_ratio": op_struct,
            }
        }

    def plot_entropy_evolution(self, keys, labels, figsize=(8, 5), title="Entropy"):
        fig, ax = plt.subplots(figsize=figsize)
        for i, key in enumerate(keys):
            generations = []
            entropies = []

            for g in self.generation_history:
                generations.append(g['generation'])
                entropies.append(
                    self.compute_entropy(g[key])
                )

            ax.plot(generations, entropies, label=labels[i])

        ax.set_title(title)
        ax.set_xlabel("Generation")
        ax.set_ylabel("Entropy")
        ax.grid(True, alpha=0.3)
        ax.legend()

        return fig
    
    def compute_entropy(self, distribution_dict):
        probs = np.array(list(distribution_dict.values()))
        probs = probs[probs > 0]
        return -np.sum(probs * np.log(probs))

    def plot_distribution_divergence(
        self, keys, labels,
        functional_suffix="_mean",
        figsize=(8, 5),
        title="Structural vs Functional Divergence"
    ):
        fig, ax = plt.subplots(figsize=figsize)

        for i, key in enumerate(keys):
            generations = []
            divergences = []

            all_categories = set()
            for g in self.generation_history:
                all_categories.update(g[key].keys())

            all_categories = sorted(all_categories)

            for g in self.generation_history:
                generations.append(g['generation'])

                structural = g[key]

                functional = {
                    cat: g.get(f"{cat}{functional_suffix}", 0)
                    for cat in all_categories
                }

                s = np.array([structural.get(cat, 0) for cat in all_categories])
                f = np.array([functional.get(cat, 0) for cat in all_categories])

                divergences.append(np.linalg.norm(s - f))

            ax.plot(generations, divergences, label=labels[i])

        ax.set_xlabel("Generation")
        ax.set_ylabel("L2 Distance")
        ax.legend()
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

        return fig

    def plot_initial_vs_final(self, figsize=(10, 6), key='operations_structural', title="Operations Structural", init_index=0, final_index=-1):
        if not self.generation_history:
            return None

        initial = self.generation_history[init_index][key]
        final = self.generation_history[final_index][key]

        keys = sorted(set(initial.keys()) | set(final.keys()))

        initial_vals = [initial[key] * 100 for key in keys]
        final_vals = [final[key] * 100 for key in keys]

        x = np.arange(len(keys))
        width = 0.35

        fig, ax = plt.subplots(figsize=figsize)
        ax.bar(x - width/2, initial_vals, width, label="Initial")
        ax.bar(x + width/2, final_vals, width, label="Final")

        ax.set_xticks(x)
        ax.set_xticklabels(keys, rotation=45, ha='right')
        ax.set_ylabel("Structural Usage (%)")
        ax.set_title(f"{title} Distribution: Initial vs Final")
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)

        plt.tight_layout()
        return fig
    
    def plot_initial_vs_final_pie(self, figsize=(10, 6), key='mean', title="Activation Category", init_index=0, final_index=-1):
        if not self.generation_history:
            return None

        initial = self.generation_history[init_index][key]
        final = self.generation_history[final_index][key]

        keys = sorted(set(initial.keys()) | set(final.keys()))

        initial_vals = [initial[key] * 100 for key in keys]
        final_vals = [final[key] * 100 for key in keys]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        colors = plt.cm.Set3(np.linspace(0, 1, len(keys)))

        ax1.pie(initial_vals, labels=keys, autopct='%1.0f%%', colors=colors)
        ax1.set_title(f'{title} Initial Distribution')

        ax2.pie(final_vals, labels=keys, autopct='%1.0f%%', colors=colors)
        ax1.set_title(f'{title} Final Distribution')

        plt.tight_layout()
        return fig

    def plot_distribution_evolution(
        self, subset, key,
        figsize=(10, 6),
        title="Distribution",
        ylabel="Usage (%)",
        percent=True
    ):
        if not self.generation_history:
            return None

        fig, ax = plt.subplots(figsize=figsize)

        generations = [g['generation'] for g in self.generation_history]
        multiplier = 100 if percent else 1

        for item in subset:
            values = [
                g.get(key, {}).get(item, 0) * multiplier
                for g in self.generation_history
            ]
            ax.plot(generations, values, label=item, linewidth=2)

        ax.set_xlabel("Generation")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Selected {title} Evolution")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig
