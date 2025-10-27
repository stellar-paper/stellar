from opensbt.model_ga.population import PopulationExtended as Population

import logging as log
import os
from pathlib import Path
from typing import Optional

import dill
import numpy as np
import weave
import wandb
from pymoo.core.result import Result

from llm import llm_output
from opensbt import config
from opensbt.config import EXPERIMENTAL_MODE, RESULTS_FOLDER, WRITE_ALL_INDIVIDUALS
from opensbt.evaluation.fitness import Fitness
from opensbt.experiment.search_configuration import SearchConfiguration
from opensbt.utils.sorting import get_nondominated_population
from opensbt.visualization import output_metric, visualizer, visualizer_llm


class SimulationResult(Result):
    """
    This class extends pymoo's Result class to output simulation results
    and extract information from the test data.
    """

    def __init__(self) -> None:
        super().__init__()
        self._additional_data = dict()

    @classmethod
    def from_result(cls, result: Result) -> "SimulationResult":
        result.__class__ = cls
        return result

    def obtain_history_design(self):
        hist = self.history

        if hist is not None:
            n_evals = []  # corresponding number of function evaluations
            hist_X = []  # the objective space values in each
            pop = Population()
            for algo in hist:
                n_evals.append(
                    algo.evaluator.n_eval
                )  # store the number of function evaluations
                pop = Population.merge(pop, algo.pop)
                feas = np.where(pop.get("feasible"))[
                    0
                ]  # filter out only the feasible and append and objective space values
                hist_X.append(pop.get("X")[feas])
        else:
            n_evals = None
            hist_X = None
        return n_evals, hist_X

    def get_first_critical(self):
        """Identifies the iteration number when the first critical solutions was found"""

        hist = self.history
        archive = self.obtain_archive()
        res = Population()
        if hist is not None and archive is not None:
            for index, algo in enumerate(hist):
                # n_evals.append(algo.evaluator.n_eval)  # store the number of function evaluations
                inds = archive[: algo.evaluator.n_eval]
                crit = np.where((inds.get("CB")))[0]
                feas = np.where((inds.get("feasible")))[0]
                feas = list(set(crit) & set(feas))
                res = inds[feas]
                if len(res) == 0:
                    continue
                else:
                    return index, res
        return 0, res

    def obtain_history(self, critical=False):
        """Returns the set of test inputs over all genreation based on feasibility and criticality
        according to number of function evaluations"""
        hist = self.history
        if hist is not None:
            n_evals = []  # corresponding number of function evaluations
            hist_F = []  # the objective space values in each generation
            for algo in hist:
                n_evals.append(
                    algo.evaluator.n_eval
                )  # store the number of function evaluations
                opt = algo.opt  # retrieve the optimum from the algorithm
                if critical:
                    crit = np.where((opt.get("CB")))[0]
                    feas = np.where((opt.get("feasible")))[0]
                    feas = list(set(crit) & set(feas))
                else:
                    feas = np.where(
                        opt.get("feasible")
                    )[
                        0
                    ]  # filter out only the feasible and append and objective space values
                hist_F.append(opt.get("F")[feas])
        else:
            n_evals = None
            hist_F = None
        return n_evals, hist_F

    def obtain_all_population(self):
        """Returns all test inputs over all generations"""
        all_population = Population()
        hist = self.history
        if hist is not None:
            for generation in hist:
                all_population = Population.merge(all_population, generation.pop)
        return all_population

    def obtain_archive(self):
        """Returns all archived individuals."""
        return self.archive

    def obtain_history_archive(self, critical=False):
        """Returns all archived test inputs over all generations"""
        hist = self.history
        archive = self.obtain_archive()
        if hist is not None:
            n_evals = []  # corresponding number of function evaluations
            hist_F = []  # the objective space values in each generation
            n_eval_last = 0
            for i, algo in enumerate(hist):
                n_eval = (
                    algo.evaluator.n_eval - n_eval_last
                )  # get the number of evals for the current iteration
                n_evals.append(n_eval)  # store the number of function evaluations
                inds = archive[n_eval_last : algo.evaluator.n_eval]
                if critical:
                    crit = np.where((inds.get("CB")))[0]
                    feas = np.where((inds.get("feasible")))[0]
                    feas = list(set(crit) & set(feas))
                else:
                    feas = np.where(
                        inds.get("feasible")
                    )[
                        0
                    ]  # filter out only the feasible and append and objective space values
                hist_F.append(inds.get("F")[feas])
                # update for next calculation
                n_eval_last = algo.evaluator.n_eval
        else:
            n_evals = None
            hist_F = None
        return n_evals, hist_F

    def obtain_history_hitherto_archive(self, critical=False, optimal=True, var="F"):
        """Returns the set of test inputs over all generations based on feasibility and criticality
        according to number of function evaluations considering all evaluated test inputs (aggregated)"""
        hist = self.history
        n_evals = []  # corresponding number of function evaluations
        hist_F = []  # the objective space values in each generation
        archive = self.obtain_archive()
        all = Population()
        for i, algo in enumerate(hist):
            n_eval = algo.evaluator.n_eval
            n_evals.append(n_eval)
            all = archive[:n_eval]
            if optimal:
                all = get_nondominated_population(all)

            if critical:
                crit = np.where((all.get("CB")))[0]
                feas = np.where((all.get("feasible")))[0]
                feas = list(set(crit) & set(feas))
            else:
                feas = np.where(all.get("feasible"))[
                    0
                ]  # filter out only the feasible and append and objective space values
            hist_F.append(all.get(var)[feas])
        return n_evals, hist_F

    def obtain_history_hitherto(self, critical=False, optimal=True, var="F"):
        """Returns the set of test inputs over all generations based on feasibility and criticality
        according to number of function evaluations (aggregated)"""

        hist = self.history
        n_evals = []  # corresponding number of function evaluations
        hist_F = []  # the objective space values in each generation

        all = Population()
        for algo in hist:
            n_evals.append(algo.evaluator.n_eval)
            all = Population.merge(all, algo.pop)
            if optimal:
                all = get_nondominated_population(all)

            if critical:
                crit = np.where((all.get("CB")))[0]
                feas = np.where((all.get("feasible")))[0]
                feas = list(set(crit) & set(feas))
            else:
                feas = np.where(all.get("feasible"))[
                    0
                ]  # filter out only the feasible and append and objective space values
            hist_F.append(all.get(var)[feas])
        return n_evals, hist_F

    """ Write down the result object by pickling """
    def persist(self, save_folder):
        Path(save_folder).mkdir(parents=True, exist_ok=True)
        with open(save_folder + os.sep + "result", "wb") as f:
            dill.dump(self, f)

    """ Load the result object which was pickled before """
    @staticmethod
    def load(save_folder, name="result"):
        with open(save_folder + os.sep + name, "rb") as f:
            return dill.load(f)

    @property
    def additional_data(self):
        return self._additional_data

    @property
    def ref_point(self):
        fitness: Fitness = self.problem.fitness_function
        point = []
        for direction in fitness.min_or_max:
            point.append(int(direction == "min"))
        return np.array(point)

    """ Write the results artefacts for the current experiment"""

    @weave.op()
    def write_results(
        self,
        results_folder=RESULTS_FOLDER,
        params=None,
        is_experimental=EXPERIMENTAL_MODE,
        search_config: Optional[SearchConfiguration] = None,
    ):
        algorithm = self.algorithm

        print("[write_results] results_folder is: ", results_folder)
        # WHen algorithm is developed without subclassing pymoos Algorithm,
        # we need to use the explicit algorithm name passed via params

        # if type(algorithm) is Algorithm:
        #     algorithm_name = params["algorithm_name"]
        # else:
        
        if params is not None and "algorithm_name" in params:
            algorithm_name = params["algorithm_name"]
        else:
            algorithm_name = algorithm.__class__.__name__

        log.info(f"=====[{algorithm_name}] Writing results to: ")

        save_folder = visualizer.create_save_folder(
            self.problem,
            results_folder,
            algorithm_name,
            is_experimental=is_experimental,
        )
        log.info(save_folder)

        # Mostly for algorithm evaluation relevant

        try:
            # output_metric.gd_analysis(self, save_folder)
            output_metric.hypervolume_analysis(
                self, save_folder, ref_point_hv=self.ref_point
            )
            # output_metric.spread_analysis(self, save_folder)
        except Exception:
            print("Hypervolume analysis not possible.")
            pass
        if search_config is not None:
            visualizer.write_search_config(self, save_folder, search_config)

        visualizer.write_generations(self, save_folder)
        visualizer.write_calculation_properties(
            self, save_folder, algorithm_name, algorithm_parameters=params
        )

        # visualizer_llm.design_space(self, save_folder)
        llm_output.show_critical_heatmap_features(self, save_folder)

        visualizer.objective_space(self, save_folder)
        visualizer.optimal_individuals(self, save_folder)
        visualizer.all_critical_individuals(self, save_folder)
        visualizer.write_summary_results(self, save_folder, params=params)
        llm_output.write_failures_over_time(self, save_folder)
        llm_output.calculate_diversity(self, save_folder)

        # visualizer.write_simulation_output(self,save_folder,
        #                                    mode= config.MODE_WRITE_SIMOUT,
        #                                    write_max=config.NUM_SIMOUT_MAX)
        # visualizer.plot_timeseries_basic(self,
        #    save_folder,
        #    mode= config.MODE_PLOT_TIME_TRACES,
        #     write_max = config.NUM_PLOT_TIME_TRACES)

        llm_output.write_tests_to_json(self, save_folder)
        llm_output.write_tests_to_json(
            self, save_folder, critical_only=True, filename="all_critical_utterances"
        )

        llm_output.copy_config(save_folder)
        llm_output.copy_prompts(save_folder)

        # visualizer.simulations(self,
        # save_folder,
        # mode = config.MODE_WRITE_GIF,
        # write_max = config.NUM_GIF_MAX)
        if WRITE_ALL_INDIVIDUALS:
            visualizer.all_individuals(self, save_folder)
            visualizer_llm.all_individuals_llm(self, save_folder)

        artifact = wandb.Artifact("results_folder", "output")
        artifact.add_dir(save_folder)

        wandb.log_artifact(artifact)

        return save_folder
