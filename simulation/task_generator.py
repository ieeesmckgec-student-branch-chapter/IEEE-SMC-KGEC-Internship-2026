import random


def generate_tasks(num_tasks):

    tasks = []

    for i in range(num_tasks):

        size = random.randint(1000, 10000)

        cpu = random.randint(20, 95)

        memory = random.randint(256, 4096)

        bandwidth = random.randint(10, 100)

        priority = random.randint(1, 5)

        # Higher priority → smaller deadline
        if priority == 5:
            deadline = random.randint(30, 80)
        elif priority == 4:
            deadline = random.randint(60, 120)
        elif priority == 3:
            deadline = random.randint(100, 200)
        elif priority == 2:
            deadline = random.randint(180, 300)
        else:
            deadline = random.randint(250, 500)

        arrival = random.randint(0, num_tasks)

        # Energy depends on task size
        energy = round(size / 2500 + cpu / 120, 2)

        # Estimated execution time
        execution_time = round(size / 150, 2)

        task = {

            "id": i,

            "size": size,

            "cpu": cpu,

            "memory": memory,

            "bandwidth": bandwidth,

            "priority": priority,

            "deadline": deadline,

            "arrival": arrival,

            "energy": energy,

            "execution_time": execution_time

        }

        tasks.append(task)

    return tasks


if __name__ == "__main__":

    tasks = generate_tasks(10)

    for task in tasks:
        print(task)