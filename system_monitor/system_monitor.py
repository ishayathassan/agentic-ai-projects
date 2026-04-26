import sys
import time
from pprint import pprint
import psutil

from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain.tools import tool

from langgraph.checkpoint.memory import InMemorySaver


@tool
def cpu_usage(per_cpu_usage: bool) -> list[float] | float:
    """
    Retrieves the current CPU utilization as a percentage.

    Args:
        per_cpu_usage (bool): If True, returns a list of percentages for each 
            individual CPU core. If False, returns the aggregate CPU usage as 
            a single float.

    Returns:
        list[float] | float: The CPU usage percentage(s).
    """
    return psutil.cpu_percent(interval=0.5, percpu=per_cpu_usage)

@tool
def ram_usage() -> dict[str, float]:
    """
    Retrieves the system's current virtual memory statistics.

    The returned dictionary contains memory metrics in Megabytes (MB) and the
    overall usage percentage.

    Returns:
        dict[str, int | float]: A dictionary with the following keys:
            - 'total': Total physical memory in MB.
            - 'available': The memory that can be given immediately to processes in MB.
            - 'used': Memory used in MB.
            - 'percent': The percentage usage calculated as (total - available) / total * 100.
    """
    mem = psutil.virtual_memory()
    factor = 1024 ** 2
    return {
        "total": mem.total / factor,
        "available": mem.available / factor,
        "used": mem.used / factor,
        "percent": mem.percent
    }

@tool
def disk_usage() -> dict[str, float]:
    """
    Retrieves the disk usage statistics for the root directory ('/').

    Returns:
        dict[str, float]: A dictionary containing disk usage metrics in 
        Megabytes (MB) and the usage percentage.
            - 'total': Total disk space in MB.
            - 'available': Total free disk space in MB.
            - 'used': Used disk space in MB.
            - 'percent': The percentage of disk space currently in use.
    """
    mem = psutil.disk_usage("/")
    factor = 1024 ** 2
    return {
        "total": mem.total / factor,
        "available": mem.free / factor,
        "used": mem.used / factor,
        "percent": mem.percent
    }
    
    
@tool
def get_top_processes(metric: str = 'cpu', count: int = 3) -> list[dict]:
    """
    Identifies and returns the top running processes based on a specified resource metric.

    This function performs a two-pass sampling approach to accurately calculate 
    CPU usage, as a single, immediate check would return 0.0 without a 
    reference time window. Memory usage is calculated based on Resident Set 
    Size (RSS).

    Args:
        metric (str): The resource metric to sort by. Accepts 'cpu' 
            (percentage) or 'memory' (Resident Set Size in MB). Defaults to 'cpu'.
        count (int): The number of top processes to retrieve. Defaults to 3.

    Returns:
        list[dict]: A list of dictionaries, where each dictionary contains:
            - 'pid' (int): The process ID.
            - 'name' (str): The name of the process.
            - 'value' (float): The CPU percentage or memory usage in MB.

    Note:
        This function blocks execution for 0.5 seconds to accurately measure 
        CPU utilization. Processes that terminate or deny access during 
        the iteration are silently skipped.
    """
    processes = []
    
    # First pass: Initialize cpu_percent for all processes
    procs = []
    for proc in psutil.process_iter(['pid', 'name', 'memory_info']):
        try:
            # We don't need cpu_percent in process_iter here
            proc.cpu_percent(interval=None) 
            procs.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    
    # Wait a short interval to capture usage
    time.sleep(0.5)
    
    # Second pass: Calculate actual usage
    for proc in procs:
        try:
            if metric == 'cpu':
                value = proc.cpu_percent(interval=None)
            else:
                value = proc.memory_info().rss / (1024 * 1024)
            
            processes.append({
                'pid': proc.pid,
                'name': proc.name(),
                'value': value
            })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
            
    return sorted(processes, key=lambda x: x['value'], reverse=True)[:count]


system_prompt = """

You are the Internal System Administrator. Your sole purpose is to monitor and report on the technical health and performance of the host machine.

Operational Rules:
1. DATA-ONLY OUTPUT: Provide metrics, status updates, and actionable insights. Do not include introductory phrases, filler, or AI-assistant pleasantries (e.g., "I can help with that," "Here is the information," or "Hope this helps").
2. THRESHOLD AWARENESS: If any metric is in a warning state (e.g., CPU > 85%, Memory > 90%, or Disk > 90% full), highlight this specifically and provide an urgent status note.
3. CONCISENESS: When reporting on multiple metrics, use bullet points for readability. If asked about "top processes," provide only the top 3 unless otherwise specified.
4. ACTIONABLE INSIGHTS: If performance is poor, briefly suggest the most likely cause based on the metrics you retrieve.
5. NO HALLUCINATIONS: If a tool provides data, report it accurately. Do not invent details or provide opinions on software beyond its resource impact.

Start your response directly with the requested data.

"""

model = ChatOllama(
    model="gemma4:e2b"
)

agent = create_agent(
    model=model,
    tools=[cpu_usage, ram_usage, disk_usage, get_top_processes],
    system_prompt=system_prompt,
    checkpointer=InMemorySaver()
)

config = {"configurable": {"thread_id": "1"}}


# --- Interactive Script Loop ---
print("="*50)
print(" SYSTEM ADMINISTRATOR AGENT INITIALIZED")
print("="*50)
print("Type your inquiry below.")
print("Type 'q', 'quit', or 'exit' to terminate the session.\n")

while True:
    try:
        user_input = input("System Inquiry: ").strip()
        
        # Break condition
        if user_input.lower() in ['q', 'quit', 'exit']:
            print("\nTerminating session. Goodbye.")
            break
            
        # Prevent empty queries from breaking the agent
        if not user_input:
            continue
            
        question = HumanMessage(content=user_input)

        print("Analyzing...", end="", flush=True)

        # Reset for every new question
        prefix_printed = False

        for event in agent.stream({"messages": [question]}, stream_mode="updates", config=config):
            for node_name, node_data in event.items():
                if "messages" in node_data:
                    last_message = node_data["messages"][-1]
                    
                    if hasattr(last_message, 'type') and last_message.type == 'tool':
                        continue
                    
                    if last_message.content:
                        if not prefix_printed:
                            sys.stdout.write("\r" + " " * 20 + "\r")
                            print("Response:\n", end="", flush=True)
                            prefix_printed = True
                        
                        print(last_message.content, end="", flush=True)

        print("\n") # Ensure a clean break before the next prompt
        print("-" * 50) # Visual separator for the next question

    # Handle Ctrl+C gracefully
    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user. Shutting down.")
        break