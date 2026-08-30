class Server:
    """
    Generic Server Class for Edge and Cloud Computing
    """

    def __init__(
        self,
        name,
        server_type,
        cpu_capacity,
        memory,
        bandwidth,
        processing_speed,
        energy_rate,
        cost_rate,
        network_delay,
        max_tasks
    ):

        self.name = name
        self.server_type = server_type

        # Hardware Resources
        self.cpu_capacity = cpu_capacity
        self.memory = memory              # MB
        self.bandwidth = bandwidth        # Mbps

        # Performance
        self.processing_speed = processing_speed

        # Cost & Energy
        self.energy_rate = energy_rate
        self.cost_rate = cost_rate

        # Network
        self.network_delay = network_delay

        # Task Management
        self.max_tasks = max_tasks
        self.current_tasks = 0
        self.queue_length = 0
        self.cpu_used = 0

    # ----------------------------
    # Check Server Availability
    # ----------------------------

    def available(self):
        return self.current_tasks < self.max_tasks

    # ----------------------------
    # Assign Task
    # ----------------------------

    def assign_task(self, cpu_required):

        if self.available():

            self.current_tasks += 1
            self.queue_length += 1
            self.cpu_used += cpu_required

            return True

        return False

    # ----------------------------
    # Release Task
    # ----------------------------

    def release_task(self, cpu_required):

        if self.current_tasks > 0:

            self.current_tasks -= 1
            self.queue_length -= 1

            self.cpu_used = max(0, self.cpu_used - cpu_required)

    # ----------------------------
    # CPU Utilization (%)
    # ----------------------------

    def utilization(self):

        return round(
            (self.cpu_used / self.cpu_capacity) * 100,
            2
        )

    # ----------------------------
    # Queue Delay
    # ----------------------------

    def queue_delay(self):

        return self.queue_length * 0.5

    # ----------------------------
    # Reset Server
    # ----------------------------

    def reset(self):

        self.current_tasks = 0
        self.queue_length = 0
        self.cpu_used = 0

    # ----------------------------
    # Display Information
    # ----------------------------

    def info(self):

        return {
            "Name": self.name,
            "Type": self.server_type,
            "CPU Capacity": self.cpu_capacity,
            "Memory": self.memory,
            "Bandwidth": self.bandwidth,
            "Processing Speed": self.processing_speed,
            "Energy Rate": self.energy_rate,
            "Cost Rate": self.cost_rate,
            "Network Delay": self.network_delay,
            "CPU Used": self.cpu_used,
            "CPU Utilization": self.utilization(),
            "Queue Length": self.queue_length,
            "Current Tasks": self.current_tasks
        }


# =====================================================
# Create Edge Servers
# =====================================================

edge_servers = [

    Server(
        name="EDGE1",
        server_type="EDGE",
        cpu_capacity=120,
        memory=4096,
        bandwidth=1000,
        processing_speed=120,
        energy_rate=0.50,
        cost_rate=0.0005,
        network_delay=2,
        max_tasks=150
    ),

    Server(
        name="EDGE2",
        server_type="EDGE",
        cpu_capacity=110,
        memory=4096,
        bandwidth=900,
        processing_speed=115,
        energy_rate=0.55,
        cost_rate=0.00055,
        network_delay=3,
        max_tasks=150
    ),

    Server(
        name="EDGE3",
        server_type="EDGE",
        cpu_capacity=100,
        memory=4096,
        bandwidth=850,
        processing_speed=110,
        energy_rate=0.60,
        cost_rate=0.00060,
        network_delay=4,
        max_tasks=150
    )

]

# =====================================================
# Create Cloud Servers
# =====================================================

cloud_servers = [

    Server(
        name="CLOUD1",
        server_type="CLOUD",
        cpu_capacity=500,
        memory=32768,
        bandwidth=10000,
        processing_speed=250,
        energy_rate=1.20,
        cost_rate=0.0020,
        network_delay=10,
        max_tasks=1000
    ),

    Server(
        name="CLOUD2",
        server_type="CLOUD",
        cpu_capacity=600,
        memory=65536,
        bandwidth=12000,
        processing_speed=280,
        energy_rate=1.30,
        cost_rate=0.0025,
        network_delay=12,
        max_tasks=1000
    )

]

# =====================================================
# All Servers
# =====================================================

all_servers = edge_servers + cloud_servers

# =====================================================
# Testing
# =====================================================

if __name__ == "__main__":

    print("=" * 70)
    print("SERVER CONFIGURATION")
    print("=" * 70)

    for server in all_servers:

        print(server.info())
        print("-" * 70)