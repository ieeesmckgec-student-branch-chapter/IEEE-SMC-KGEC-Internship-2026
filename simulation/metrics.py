from simulation.servers import all_servers


def get_server(server_name):

    for server in all_servers:

        if server.name == server_name:
            return server

    return None


def metrics(tasks, allocation):

    latency = 0.0
    energy = 0.0
    cost = 0.0
    response_time = 0.0
    throughput = 0.0
    sla_violation = 0

    server_usage = {}

    edge_tasks = 0
    cloud_tasks = 0

    # -----------------------------------------
    # Initialize Server Usage
    # -----------------------------------------

    for server in all_servers:

        server_usage[server.name] = 0

    # -----------------------------------------
    # Process Each Task
    # -----------------------------------------

    for task, server_name in zip(tasks, allocation):

        server = get_server(server_name)

        if server is None:
            continue

        server_usage[server.name] += 1

        execution_time = task["size"] / server.processing_speed

        transmission_delay = server.network_delay

        queue_delay = server.queue_delay()

        total_delay = execution_time + transmission_delay + queue_delay

        latency += total_delay

        response_time += total_delay

        energy += task["energy"] * server.energy_rate

        cost += task["size"] * server.cost_rate

        throughput += task["size"] / max(total_delay, 1)

        if total_delay > task["deadline"]:

            sla_violation += 1

        if server.server_type == "EDGE":

            edge_tasks += 1

        else:

            cloud_tasks += 1

    # -----------------------------------------
    # Load Balance
    # -----------------------------------------

    loads = list(server_usage.values())

    if len(loads) > 0:

        average_load = sum(loads) / len(loads)

        load_balance = (

            sum(abs(load - average_load) for load in loads)

            / len(loads)

        )

    else:

        load_balance = 0.0

    # -----------------------------------------
    # Percentages
    # -----------------------------------------

    total_tasks = max(len(tasks), 1)

    edge_percent = (edge_tasks / total_tasks) * 100

    cloud_percent = (cloud_tasks / total_tasks) * 100

    offload_percent = (cloud_tasks / total_tasks) * 100

    # -----------------------------------------
    # Return Metrics
    # -----------------------------------------

    return {

        "latency": round(latency, 2),

        "energy": round(energy, 2),

        "cost": round(cost, 2),

        "response_time": round(response_time, 2),

        "throughput": round(throughput, 2),

        "sla_violation": sla_violation,

        "load_balance": round(load_balance, 2),

        "edge_percent": round(edge_percent, 2),

        "cloud_percent": round(cloud_percent, 2),

        "offload_percent": round(offload_percent, 2),

        "server_usage": server_usage

    }


if __name__ == "__main__":

    from simulation.task_generator import generate_tasks
    from simulation.offloading import offload

    tasks = generate_tasks(20)

    allocation = offload(tasks)

    result = metrics(tasks, allocation)

    for key, value in result.items():

        print(f"{key:20} : {value}")