import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from pymoo.algorithms.moo.age import AGEMOEA
from pymoo.algorithms.moo.age2 import AGEMOEA2
from pymoo.algorithms.moo.ctaea import CTAEA
from pymoo.algorithms.moo.dnsga2 import DNSGA2
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.rnsga2 import RNSGA2
from pymoo.algorithms.moo.rnsga3 import RNSGA3
from pymoo.algorithms.moo.rvea import RVEA
from pymoo.algorithms.moo.sms import SMSEMOA
from pymoo.algorithms.moo.spea2 import SPEA2
from pymoo.algorithms.moo.unsga3 import UNSGA3

from pymoo.config import Config
from pymoo.termination.default import DefaultMultiObjectiveTermination
from pymoo.util.ref_dirs import get_reference_directions

from problems import MULTI_OBJECTIVE_PROBLEMS

Config.warnings['not_compiled'] = False
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

B = 8
G = 1.164
VERBOSE = False
SEED = 1410

# Prepare results dir
RESULTS = os.path.realpath("_moo")
os.makedirs(RESULTS, exist_ok=True)

REF_DIRS = get_reference_directions("uniform", 2, n_points=12)

OPTIMIZERS = (
    ("AGEMOEA", AGEMOEA, {}),
    ("AGEMOEA2", AGEMOEA2, {}),
    ("CTAEA", CTAEA, {"ref_dirs": REF_DIRS}),
    ("DNSGA2", DNSGA2, {}),
    ("MOEAD", MOEAD, {"ref_dirs": REF_DIRS}),
    ("NSGA2", NSGA2, {}),
    ("NSGA3", NSGA3, {"ref_dirs": REF_DIRS}),
    ("RNSGA2", RNSGA2, {}),
    ("RNSGA3", RNSGA3, {}),
    ("RVEA", RVEA, {"ref_dirs": REF_DIRS}),
    ("SMSEMOA", SMSEMOA, {}),
    ("SPEA2", SPEA2, {}),
    ("UNSGA3", UNSGA3, {"ref_dirs": REF_DIRS}),
)

SOLVER_TERMINATION_ARGS = {
    'n_max_gen': 1000,
}

problems_bar = tqdm(MULTI_OBJECTIVE_PROBLEMS, leave=False)

for problem_name, problem_cb, problem_args in problems_bar:
    try:
        problem = problem_cb(**problem_args)
    except Exception as e:
        logger.info(f"[P] {problem_name}: {e}")
        continue

    if problem.n_obj > 2:
        # Plotting is prepared for 2 objectives
        continue

    problems_bar.set_description(problem_name)

    PROBLEM_RESULTS = os.path.join(RESULTS, problem_name)
    os.makedirs(PROBLEM_RESULTS, exist_ok=True)

    for opt_name, optimizer_cb, optimizer_args in OPTIMIZERS:
        progress = tqdm(desc=opt_name, leave=False)
    
        try:
            optimizer = optimizer_cb(**optimizer_args)

            optimizer.setup(
                problem, termination=DefaultMultiObjectiveTermination(**SOLVER_TERMINATION_ARGS),
                seed=SEED, verbose=VERBOSE, save_history=True
            )

            while optimizer.has_next():
                optimizer.next()
                progress.update()

        except Exception as e:
            logger.info(f"[O] {opt_name}: {e}")
            continue

        progress.close()

        results = [[(chap.F[0] + chap.F[1]) / 2 for chap in epoch.pop] for epoch in optimizer.history]

        min_r = [np.min(r) for r in results]
        med_r = [np.median(r) for r in results]
        max_r = [np.max(r) for r in results]

        fig, axs = plt.subplots(1, 2, figsize=(B * 2, B))

        fig.suptitle(f"{problem_name}: {opt_name}")

        ax = axs[0]
        ax.plot(med_r, color='black', lw=0.8, label="mean")
        ax.plot(max_r, color='black', ls='--', lw=0.5)
        ax.plot(min_r, color='black', ls='--', lw=0.5)

        results = [[chap.F[0] for chap in epoch.pop] for epoch in optimizer.history]

        min_r = [np.min(r) for r in results]
        med_r = [np.median(r) for r in results]
        max_r = [np.max(r) for r in results]

        ax.plot(med_r, color='blue', label="f1")
        ax.plot(max_r, color='blue', ls='--', lw=0.8)
        ax.plot(min_r, color='blue', ls='--', lw=0.8)

        results = [[chap.F[1] for chap in epoch.pop] for epoch in optimizer.history]

        min_r = [np.min(r) for r in results]
        med_r = [np.median(r) for r in results]
        max_r = [np.max(r) for r in results]

        ax.plot(med_r, color='crimson', label='f2')
        ax.plot(max_r, color='crimson', ls='--', lw=0.8)
        ax.plot(min_r, color='crimson', ls='--', lw=0.8)

        ax.legend()

        ax.grid(True, ls=':')

        ax = axs[1]
        cols = plt.cm.YlGn(np.linspace(0.4, 1, len(optimizer.history)))
        for col, epoch in zip(cols, optimizer.history):
            X = np.array([[chap.F[0], chap.F[1]] for chap in epoch.opt])
            ax.scatter(*X.T, color=col, alpha=1.0, s=10)

        X = np.array([[chap.F[0], chap.F[1]] for chap in optimizer.history[-1].opt])
        ax.scatter(*X.T, alpha=1.0, s=50, linewidths=0.8, marker='D', facecolors='none', edgecolors='black')

        ax.grid(True, ls=':')

        plt.tight_layout()
        plt.savefig("foo.png")
        plt.savefig(os.path.join(PROBLEM_RESULTS, f"{opt_name}.png"))
        plt.close()
        plt.clf()
