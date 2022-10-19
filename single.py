import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from pymoo.algorithms.soo.nonconvex.brkga import BRKGA
from pymoo.algorithms.soo.nonconvex.cmaes import CMAES
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.algorithms.soo.nonconvex.direct import DIRECT
from pymoo.algorithms.soo.nonconvex.es import ES
from pymoo.algorithms.soo.nonconvex.g3pcx import G3PCX
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.soo.nonconvex.ga_niching import NicheGA
from pymoo.algorithms.soo.nonconvex.isres import ISRES
from pymoo.algorithms.soo.nonconvex.nelder import NelderMead
from pymoo.algorithms.soo.nonconvex.pattern import PatternSearch
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.algorithms.soo.nonconvex.pso_ep import EPPSO
from pymoo.algorithms.soo.nonconvex.random_search import RandomSearch
from pymoo.algorithms.soo.nonconvex.sres import SRES

from pymoo.config import Config
from pymoo.termination.default import DefaultSingleObjectiveTermination

from problems import SINGLE_OBJECTIVE_PROBLEMS

Config.warnings['not_compiled'] = False
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

B = 8
G = 1.164
VERBOSE = False
SEED = 1410

# Prepare results dir
RESULTS = os.path.realpath("_soo")
os.makedirs(RESULTS, exist_ok=True)

OPTIMIZERS = (
    ("BRKGA", BRKGA, {}),
    # ("CMAES", CMAES, {}),
    ("DE", DE, {}),
    ("DIRECT", DIRECT, {}),
    ("ES", ES, {}),
    ("G3PCX", G3PCX, {}),
    ("GA", GA, {}),
    ("NicheGA", NicheGA, {}),
    ("ISRES", ISRES, {}),
    ("NelderMead", NelderMead, {}),
    ("PatternSearch", PatternSearch, {}),
    ("PSO", PSO, {}),
    ("EPPSO", EPPSO, {}),
    ("RandomSearch", RandomSearch, {}),
    ("SRES", SRES, {}),
)

SOLVER_TERMINATION_ARGS = {
    'n_max_gen': 1000,
}

problems_bar = tqdm(SINGLE_OBJECTIVE_PROBLEMS, leave=False)

for problem_name, problem_cb, problem_args in problems_bar:
    try:
        problem = problem_cb(**problem_args)
    except Exception as e:
        logger.info(f"[P] {problem_name}: {e}")
        continue

    problems_bar.set_description(problem_name)

    PROBLEM_RESULTS = os.path.join(RESULTS, problem_name)
    os.makedirs(PROBLEM_RESULTS, exist_ok=True)

    for opt_name, optimizer_cb, optimizer_args in OPTIMIZERS:
        progress = tqdm(desc=opt_name, leave=False)
    
        try:
            optimizer = optimizer_cb(**optimizer_args)

            optimizer.setup(
                problem, termination=DefaultSingleObjectiveTermination(**SOLVER_TERMINATION_ARGS),
                seed=SEED, verbose=VERBOSE, save_history=True
            )

            while optimizer.has_next():
                optimizer.next()
                progress.update()

        except Exception as e:
            logger.info(f"[O] {opt_name}: {e}")
            continue

        progress.close()

        results = [[chap.F[0] for chap in epoch.pop] for epoch in optimizer.history]

        min_r = [np.min(r) for r in results]
        med_r = [np.median(r) for r in results]
        max_r = [np.max(r) for r in results]

        fig, axs = plt.subplots(1, 1, figsize=(B, B * G))

        fig.suptitle(f"{problem_name}: {opt_name}")

        ax = axs
        ax.plot(med_r, color='crimson')
        ax.plot(max_r, color='crimson', ls='--', lw=0.8)
        ax.plot(min_r, color='crimson', ls='--', lw=0.8)
        ax.grid(True, ls=':')

        plt.tight_layout()
        plt.savefig("foo.png")
        plt.savefig(os.path.join(PROBLEM_RESULTS, f"{opt_name}.png"))
        plt.close()
        plt.clf()
