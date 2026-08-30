import random

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


def lion(
    tasks,
    lions=50,
    iterations=100,
    roam_probability=0.25,
    mating_probability=0.30
):

    # --------------------------------------------------
    # Initialize Pride
    # --------------------------------------------------

    pride = []

    for _ in range(lions):

        solution = [

            random.choice(SERVERS)

            for _ in tasks

        ]

        pride.append(solution)

    pride.sort(

        key=lambda x: fitness(tasks, x)

    )

    best_solution = pride[0].copy()

    best_score = fitness(tasks, best_solution)

    elite_size = 2

    # --------------------------------------------------
    # Main Optimization
    # --------------------------------------------------

    for iteration in range(iterations):

        pride.sort(

            key=lambda x: fitness(tasks, x)

        )

        elites = [

            lion.copy()

            for lion in pride[:elite_size]

        ]

        leader_count = max(5, lions // 10)

        leaders = pride[:leader_count]

        # ---------------------------------------------
        # Hunting
        # ---------------------------------------------

        for lion_solution in pride:

            leader = random.choice(leaders)

            for j in range(len(tasks)):

                if random.random() < 0.50:

                    lion_solution[j] = leader[j]

        # ---------------------------------------------
        # Roaming
        # ---------------------------------------------

        for lion_solution in pride:

            for j in range(len(tasks)):

                if random.random() < roam_probability:

                    lion_solution[j] = random.choice(SERVERS)

        # ---------------------------------------------
        # Mating
        # ---------------------------------------------

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

        # ---------------------------------------------
        # Adaptive Mutation
        # ---------------------------------------------

        adaptive_mutation = adaptive_mutation_rate(
            iteration,
            iterations
        )

        for child in children:

            for j in range(len(tasks)):

                if random.random() < adaptive_mutation:

                    child[j] = random.choice(SERVERS)

        pride.extend(children)

        # ---------------------------------------------
        # Elitism
        # ---------------------------------------------

        pride.extend(elites)

        # ---------------------------------------------
        # Survival
        # ---------------------------------------------

        pride.sort(

            key=lambda x: fitness(tasks, x)

        )

        pride = pride[:lions]

        # ---------------------------------------------
        # Update Best Solution
        # ---------------------------------------------

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

    result = lion(tasks)

    print(result)