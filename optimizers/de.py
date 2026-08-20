from asyncio import tasks
import random

from simulation.fitness import fitness

# ---------------------------------------------------
# Available Servers
# ---------------------------------------------------

SERVERS = [
    "EDGE1",
    "EDGE2",
    "EDGE3",
    "CLOUD1",
    "CLOUD2"
]


def de(
    tasks,
    population_size=40,
    generations=100,
    F=0.8,
    CR=0.9,
    mutation_rate=0.05
):

    # ---------------------------------------------------
    # Initialize Population
    # ---------------------------------------------------

    population = []

    for _ in range(population_size):

        solution = [

            random.choice(SERVERS)

            for _ in tasks

        ]

        population.append(solution)

    # ---------------------------------------------------
    # Evolution
    # ---------------------------------------------------

    for generation in range(generations):

        new_population = []

        # Adaptive Mutation Factor
        adaptive_F = F - (0.3 * generation / generations)

        adaptive_CR = CR - (0.2 * generation / generations)

        # Elite Preservation

        elite = min(

            population,

            key=lambda x: fitness(tasks, x)

        ).copy()

        for i in range(population_size):

            candidates = list(range(population_size))

            candidates.remove(i)

            a, b, c = random.sample(candidates, 3)

            target = population[i]

            A = population[a]

            B = population[b]

            C = population[c]

            trial = []

            for j in range(len(tasks)):

                rand = random.random()

                # ---------------------------------
                # Crossover
                # ---------------------------------

                if rand < adaptive_CR:

                    if B[j] != C[j]:

                        gene = A[j]

                    else:

                        gene = random.choice(SERVERS)

                else:

                    gene = target[j]

                # ---------------------------------
                # Mutation
                # ---------------------------------

                if random.random() < mutation_rate:

                    gene = random.choice(SERVERS)

                trial.append(gene)

            # ---------------------------------
            # Greedy Selection
            # ---------------------------------

            trial_score = fitness(tasks, trial)
            target_score = fitness(tasks, target)

            if trial_score < target_score:
                new_population.append(trial)
            else:
                new_population.append(target)

        # ---------------------------------
        # Elitism
        # ---------------------------------

        scores = [fitness(tasks, sol) for sol in new_population]

        worst = scores.index(max(scores))

        new_population[worst] = elite

    # ---------------------------------------------------
    # Best Solution
    # ---------------------------------------------------

    best_solution = min(

        population,

        key=lambda x: fitness(tasks, x)

    )

    return best_solution