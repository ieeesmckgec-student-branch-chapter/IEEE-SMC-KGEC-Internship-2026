import random

from optimizers.ga import ga
from simulation.fitness import fitness

# --------------------------------------------------
# Available Servers
# --------------------------------------------------

SERVERS = [
    "EDGE1",
    "EDGE2",
    "EDGE3",
    "CLOUD1",
    "CLOUD2"
]


# --------------------------------------------------
# Differential Evolution Refinement
# --------------------------------------------------

def de_refinement(pride, tasks, F=0.5, CR=0.8):

    new_pride = []

    for i in range(len(pride)):

        candidates = list(range(len(pride)))
        candidates.remove(i)

        a, b, c = random.sample(candidates, 3)

        trial = pride[i].copy()

        for j in range(len(trial)):

            if random.random() < CR:

                if random.random() < F:

                    trial[j] = pride[a][j]

                else:

                    trial[j] = random.choice(
                        [pride[b][j], pride[c][j]]
                    )

        if fitness(tasks, trial) < fitness(tasks, pride[i]):

            new_pride.append(trial)

        else:

            new_pride.append(pride[i])

    return new_pride


def adaptive_mutation_rate(
    generation,
    iterations,
    mr_max=0.30,
    mr_min=0.05
):

    return mr_max - (
        (mr_max - mr_min)
        * (generation / iterations)
    )


def hybrid_ga_lion(
    tasks,
    lions=50,
    iterations=50,
    mating_probability=0.30,
    roam_probability=0.25
):

    # --------------------------------------------------
    # Phase 1 : Global Exploration using GA
    # --------------------------------------------------

    best_solution = ga(
        tasks,
        population_size=50,
        generations=50
    )

    best_score = fitness(tasks, best_solution)

    # --------------------------------------------------
    # Initialize Lion Pride
    # --------------------------------------------------

    pride = []

    for _ in range(lions):

        lion = best_solution.copy()

        for i in range(len(lion)):

            if random.random() < 0.30:

                lion[i] = random.choice(SERVERS)

        pride.append(lion)

    # --------------------------------------------------
    # Lion Optimization
    # --------------------------------------------------

    elite_size = 2

    for generation in range(iterations):

        pride.sort(
            key=lambda x: fitness(tasks, x)
        )

        elites = [

            lion.copy()

            for lion in pride[:elite_size]

        ]

        leader_count = max(5, lions // 10)
        leaders = pride[:leader_count]

        # --------------------------------------------
        # Hunting
        # --------------------------------------------

        for lion in pride:

            leader = random.choice(leaders)

            for i in range(len(tasks)):

                if random.random() < 0.50:

                    lion[i] = leader[i]

        # --------------------------------------------
        # Roaming
        # --------------------------------------------

        for lion in pride:

            for i in range(len(tasks)):

                if random.random() < roam_probability:

                    lion[i] = random.choice(SERVERS)

        # --------------------------------------------
        # Mating
        # --------------------------------------------

        children = []

        for _ in range(len(pride) // 4):

            father = random.choice(leaders)

            mother = random.choice(pride)

            if random.random() < mating_probability:

                point = random.randint(
                    1,
                    len(tasks) - 1
                )

                child = (

                    father[:point]

                    +

                    mother[point:]

                )

                children.append(child)

        # --------------------------------------------
        # Adaptive Mutation
        # --------------------------------------------

        adaptive_mutation = adaptive_mutation_rate(
            generation,
            iterations
        )

        for child in children:

            for i in range(len(tasks)):

                if random.random() < adaptive_mutation:

                    child[i] = random.choice(SERVERS)

        pride.extend(children)

        # --------------------------------------------
        # DE Refinement
        # Every 5 Iterations
        # --------------------------------------------

        if generation % 10 == 0:

            pride = de_refinement(
                pride,
                tasks
            )

        # --------------------------------------------
        # Elitism
        # --------------------------------------------

        pride.extend(elites)

        # --------------------------------------------
        # Survival
        # --------------------------------------------

        pride.sort(
            key=lambda x: fitness(tasks, x)
        )

        pride = pride[:lions]

        # --------------------------------------------
        # Update Best Solution
        # --------------------------------------------

        current_best = pride[0]

        current_score = fitness(
            tasks,
            current_best
        )

        if current_score < best_score:

            best_score = current_score

            best_solution = current_best.copy()

    return best_solution


if __name__ == "__main__":

    from simulation.task_generator import generate_tasks

    tasks = generate_tasks(20)

    result = hybrid_ga_lion(tasks)

    print(result)