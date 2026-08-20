import random

from simulation.fitness import fitness

# -------------------------------------------------
# Available Servers
# -------------------------------------------------

SERVERS = [
    "EDGE1",
    "EDGE2",
    "EDGE3",
    "CLOUD1",
    "CLOUD2"
]


def tournament_selection(population, tasks, tournament_size=4):

    competitors = random.sample(population, tournament_size)

    return min(
        competitors,
        key=lambda chromosome: fitness(tasks, chromosome)
    )


def crossover(parent1, parent2):

    size = len(parent1)

    point1 = random.randint(1, size - 2)
    point2 = random.randint(point1 + 1, size - 1)

    child = (
        parent1[:point1] +
        parent2[point1:point2] +
        parent1[point2:]
    )

    return child


def mutate(chromosome, mutation_rate):

    for i in range(len(chromosome)):

        if random.random() < mutation_rate:

            chromosome[i] = random.choice(SERVERS)

    return chromosome


def adaptive_mutation_rate(
    generation,
    generations,
    mr_max=0.30,
    mr_min=0.05
):

    return mr_max - (
        (mr_max - mr_min) *
        (generation / generations)
    )


def ga(
    tasks,
    population_size=50,
    generations=100,
    crossover_rate=0.90,
):

    # -----------------------------------------
    # Initialize Population
    # -----------------------------------------

    population = []

    for _ in range(population_size):

        chromosome = [

            random.choice(SERVERS)

            for _ in tasks

        ]

        population.append(chromosome)

    # -----------------------------------------
    # Evolution
    # -----------------------------------------

    elite_size = 2

    for generation in range(generations):

        population.sort(

            key=lambda chromosome:

            fitness(tasks, chromosome)

        )

        adaptive_mutation = adaptive_mutation_rate(
            generation,
            generations
        )

        new_population = [

            chromosome.copy()

            for chromosome in population[:elite_size]

        ]

        while len(new_population) < population_size:

            parent1 = tournament_selection(
                population[:25],
                tasks
            )

            parent2 = tournament_selection(
                population[:25],
                tasks
            )

            if random.random() < crossover_rate:

                child = crossover(
                    parent1,
                    parent2
                )

            else:

                child = parent1.copy()

            child = mutate(
                child,
                adaptive_mutation
            )

            new_population.append(child)

        population = new_population

    # -----------------------------------------
    # Best Solution
    # -----------------------------------------

    best_solution = min(

        population,

        key=lambda chromosome:

        fitness(tasks, chromosome)

    )

    return best_solution


if __name__ == "__main__":

    from simulation.task_generator import generate_tasks

    tasks = generate_tasks(20)

    result = ga(tasks)

    print(result)