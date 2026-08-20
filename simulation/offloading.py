from simulation.servers import edge_servers, cloud_servers


def calculate_score(task, server):

    score = 0

    # ----------------------------
    # CPU Requirement
    # ----------------------------
    cpu_ratio = task["cpu"] / server.cpu_capacity

    if cpu_ratio <= 0.30:
        score += 4
    elif cpu_ratio <= 0.60:
        score += 2
    else:
        score -= 2

    # ----------------------------
    # Memory Requirement
    # ----------------------------
    memory_ratio = task["memory"] / server.memory

    if memory_ratio <= 0.30:
        score += 3
    elif memory_ratio <= 0.60:
        score += 2
    else:
        score -= 2

    # ----------------------------
    # Deadline
    # ----------------------------
    if task["deadline"] <= 100:

        if server.server_type == "EDGE":
            score += 4
        else:
            score += 1

    else:

        if server.server_type == "CLOUD":
            score += 3

    # ----------------------------
    # Priority
    # ----------------------------
    score += task["priority"]

    # ----------------------------
    # Queue Length
    # ----------------------------
    score -= server.queue_length * 0.5

    # ----------------------------
    # CPU Utilization
    # ----------------------------
    score -= server.utilization() * 0.05

    # ----------------------------
    # Network Delay
    # ----------------------------
    score -= server.network_delay * 0.3

    # ----------------------------
    # Cost
    # ----------------------------
    score -= server.cost_rate * 1000

    # ----------------------------
    # Availability
    # ----------------------------
    if not server.available():
        score = -999999

    return score


def offload(tasks):

    allocation = []

    servers = edge_servers + cloud_servers

    # Reset every server before scheduling
    for server in servers:
        server.reset()

    for task in tasks:

        best_server = None
        best_score = float("-inf")

        for server in servers:

            score = calculate_score(task, server)

            if score > best_score:

                best_score = score
                best_server = server

        best_server.assign_task(task["cpu"])

        allocation.append(best_server.name)

    return allocation


if __name__ == "__main__":

    from simulation.task_generator import generate_tasks

    tasks = generate_tasks(10)

    allocation = offload(tasks)

    for task, server in zip(tasks, allocation):

        print(
            f"Task {task['id']:02d}  --->  {server}"
        )