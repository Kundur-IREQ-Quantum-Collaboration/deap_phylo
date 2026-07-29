import numpy as np
import pandas as pd
import logging

from typing import Dict, List, Optional, Any
from util.symbolic_metric import *
from util.neuronal_inputs import *
from util.activations_base import Activation
from util.register_activations import *

from autofd import Grid

@dataclass
class CountAnalysis:
    presence_ratio: dict
    structural_ratio: dict
    category_presence_ratio: dict | None = None

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
        
        self.activation_classes = (Activation,)
        self.nan_count = 0

        self.logger = logging.getLogger(__name__)

    def _analyze_individual(self, individual, measure_fn) -> Optional[Dict[str, float]]:
        """
        Shared skeleton for all three analysis strategies. measure_fn(act_name, act, expr, input_)
        must return a dict {primitive_name: measure} for ONE activation site, or None to signal
        the activation is not well-defined (e.g. NaN) -- which aborts the whole individual.
        """
        primitive_gradients = {k: 0.0 for k in self.terminals.keys()}

        inputs, _, acts, exprs = get_activation_inputs_and_exprs_from_model(
            self.nn_map[str(individual)], self.nn_inputs, activation_classes=self.activation_classes
        )

        for act_name, act in acts.items():
            expr = exprs.get(act_name)
            input_ = inputs[act_name]

            contributions = measure_fn(act_name=act_name, act=act, expr=expr, input_=input_)
            if contributions is None:
                self.logger.info(f"Activation {act_name} is not well-defined on input domain.")
                return None

            for primitive, measure in contributions.items():
                primitive_gradients[primitive] += measure

        return self._normalize(primitive_gradients)

    def _normalize(self, gradients: Dict[str, float]) -> Dict[str, float]:
        total = sum(gradients.values())
        if total == 0:
            self.logger.info("Warning: all gradients zero for individual")
            factor = 1.0   # leave raw (zero) measures as-is
        else:
            factor = 1.0 / total
        return {k: v * factor for k, v in gradients.items()}
    
    def _gateaux_measure(self, act_name, act, expr, input_):
        contributions = {}
        for primitive in self.terminals.keys():
            m = compute_gateaux_measure(expr, self.func_map, sp.Function(primitive), input_)
            if m is None or m != m:  # NaN check
                return None
            contributions[primitive] = m
        return contributions

    def _sympy_measure(self, act_name, act, expr, input_):
        contributions = {}
        for primitive in self.terminals.keys():
            dF = functional_derivative(sp.simplify(expr), primitive_to_sp(primitive))
            simplified_dF = sp.simplify(dF)
            callable_dF = sympy_expr_to_torch_callable(simplified_dF, self.func_map)
            m = compute_empirical_measure_torch(input_, callable_dF)
            if m != m:  # NaN check
                return None
            contributions[primitive] = m
        return contributions

    def _autofd_measure(self, act_name, act, expr, input_):
        xs, nan_frac = self.process_inputs(input_)
        if nan_frac > 0:
            self.logger.info(f"Warning: {nan_frac*100:.2f}% of inputs for {act_name} are NaN")
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

        return dict(zip(ids, gradients))
    
    def analyze_individual(self, individual) -> Optional[Dict[str, float]]:
        self.logger.info(f"INDIVIDUAL {str(individual)}")
        return self._analyze_individual(individual, self._gateaux_measure)

    def analyze_individual_autofd(self, individual) -> Optional[Dict[str, float]]:
        return self._analyze_individual(individual, self._autofd_measure)

    def analyze_individual_sympy(self, individual) -> Optional[Dict[str, float]]:
        return self._analyze_individual(individual, self._sympy_measure)
    
    def track_population(self, population: List, generation: int, fitnesses: Optional[List[float]] = None):
        gen_data = []
        self.logger.info(f'POPULATION SIZE: {len(population)}')
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

        cols = self.family_map[category]
        valid_cols = [c for c in cols if c in df.columns]
        if valid_cols:
            df[category] = df[valid_cols].mean(axis=1)

        return df[category].corr(df['fitness']) if 'fitness' in df.columns else 0.0
    
    def get_summary(self) -> Dict[str, Any]:
        if not self.generation_history:
            return {}
        
        initial_dist = self.generation_history[0]['mean'] 
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

            for k, v in counts.items():
                if v > 0:
                    indiv_presence[k] += 1
                total_counts[k] += v

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
            return CountAnalysis(presence_ratio=presence_ratio, structural_ratio=structural_ratio, category_presence_ratio=category_presence_ratio)
        return CountAnalysis(presence_ratio=presence_ratio, structural_ratio=structural_ratio)

    def aggregate_terminal_categories(self, ratio_dict):
        category_totals = {cat: 0.0 for cat in self.unique_categories}
        for term_name, value in ratio_dict.items():
            cat = self.terminals[term_name].category
            category_totals[cat] += value
        return category_totals

    def analyze_structure_generation(self, population, generation):

        term_cat_map = self.registry.terminal_category_lookup()
        term_count_analysis = self.analyze_counts(
            population, self.terminals, term_cat_map
        )
        op_count_anaysis = self.analyze_counts(
            population, self.operations
        )
        term_cat_struct = self.aggregate_terminal_categories(term_count_analysis.structural_ratio)

        return {
            "generation": generation,
            "terminals": {
                "presence_ratio": term_count_analysis.presence_ratio,
                "structural_ratio": term_count_analysis.structural_ratio,
                "category_presence": term_count_analysis.category_presence_ratio,
                "category_structural": term_cat_struct,
            },
            "operations": {
                "presence_ratio": op_count_anaysis.presence_ratio,
                "structural_ratio": op_count_anaysis.structural_ratio,
            }
        }
