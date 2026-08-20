from simulation.servers import all_servers


def baseline(tasks):
    """
    Classical Min-Min Scheduling Algorithm
    """

    # Reset all servers
    for server in all_servers:
        server.reset()

    allocation = [None] * len(tasks)

    remaining_tasks = list(range(len(tasks)))

    while remaining_tasks:

        best_task = None
        best_server = None
        global_min_completion = float("inf")

        # -----------------------------------
        # Evaluate every remaining task
        # -----------------------------------

        for task_index in remaining_tasks:

            task = tasks[task_index]

            task_best_server = None
            task_best_completion = float("inf")

            for server in all_servers:

                if not server.available():
                    continue

                execution_time = (
                    task["size"] /
                    server.processing_speed
                )

                queue_delay = server.queue_length * 0.5

                completion_time = (
                    execution_time +
                    server.network_delay +
                    queue_delay
                )

                if completion_time < task_best_completion:

                    task_best_completion = completion_time
                    task_best_server = server

            # -----------------------------------
            # Global Minimum
            # -----------------------------------

            if task_best_completion < global_min_completion:

                global_min_completion = task_best_completion
                best_task = task_index
                best_server = task_best_server

        # -----------------------------------
        # Assign Selected Task
        # -----------------------------------

        best_server.assign_task(tasks[best_task]["cpu"])

        allocation[best_task] = best_server.name

        remaining_tasks.remove(best_task)

    return allocation


if __name__ == "__main__":

    from simulation.task_generator import generate_tasks

    tasks = generate_tasks(10)

    allocation = baseline(tasks)

    print("\nMIN-MIN SCHEDULING\n")

    for task, server in zip(tasks, allocation):

        print(f"Task {task['id']:02d} ---> {server}")