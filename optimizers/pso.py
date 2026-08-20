import random
import math

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


def adaptive_mutation_rate(
    iteration,
    iterations,
    mr_max=0.30,
    mr_min=0.05
):

    return mr_max - (
        (mr_max - mr_min)
        * (iteration / iterations)
    )


def pso(
    tasks,
    particles=50,
    iterations=100,
    inertia=0.7,
    cognitive=1.8,
    social=1.8
):

    population = []
    velocities = []

    # -------------------------------------------------
    # Initialize Population
    # -------------------------------------------------

    for _ in range(particles):

        solution = [
            random.choice(SERVERS)
            for _ in tasks
        ]

        velocity = [
            random.uniform(-1, 1)
            for _ in tasks
        ]

        population.append(solution)
        velocities.append(velocity)

    # -------------------------------------------------
    # Personal Best
    # -------------------------------------------------

    personal_best = [

        particle.copy()

        for particle in population

    ]

    personal_best_score = [

        fitness(tasks, particle)

        for particle in population

    ]

    best_index = personal_best_score.index(

        min(personal_best_score)

    )

    global_best = personal_best[best_index].copy()

    global_best_score = personal_best_score[best_index]

    elite_size = 2

    # -------------------------------------------------
    # Main PSO Loop
    # -------------------------------------------------

    for iteration in range(iterations):

        adaptive_mutation = adaptive_mutation_rate(
            iteration,
            iterations
        )

        ranked = sorted(

            zip(population, personal_best_score),

            key=lambda x: x[1]

        )

        elites = [

            particle.copy()

            for particle, _ in ranked[:elite_size]

        ]

        # -----------------------------------------
        # Evaluate Particles
        # -----------------------------------------
        particle_scores = []

        for i in range(particles):

            score = fitness(tasks, population[i])

            particle_scores.append(score)

            if score < personal_best_score[i]:

                personal_best_score[i] = score

                personal_best[i] = population[i].copy()

            if score < global_best_score:

                global_best_score = score

                global_best = population[i].copy()

        # -----------------------------------------
        # Velocity & Position Update
        # -----------------------------------------

        for i in range(particles):

            for j in range(len(tasks)):

                r1 = random.random()

                r2 = random.random()

                velocities[i][j] = (

                    inertia * velocities[i][j]

                    + cognitive * r1

                    + social * r2

                )

                probability = 1 / (

                    1 + math.exp(-velocities[i][j])

                )

                rand = random.random()

                if rand < probability * 0.45:

                    population[i][j] = global_best[j]

                elif rand < probability * 0.85:

                    population[i][j] = personal_best[i][j]

                else:

                    population[i][j] = random.choice(SERVERS)

                if random.random() < adaptive_mutation:

                    population[i][j] = random.choice(SERVERS)

        # -----------------------------------------
        # Elitism
        # -----------------------------------------

        ranked_population = [

        particle

        for particle, _ in sorted(

            zip(population, particle_scores),

            key=lambda x: x[1]

    )

]

        ranked_population[-elite_size:] = elites

        population = ranked_population

    return global_best


if __name__ == "__main__":

    from simulation.task_generator import generate_tasks

    tasks = generate_tasks(20)

    result = pso(tasks)

    print(result)