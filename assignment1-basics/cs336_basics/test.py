import threading
import multiprocessing
import time
import math


def cpu_intensive_work(n):
    """
    A CPU-intensive task that does mathematical calculations.
    This represents the type of work that SHOULD benefit from multiple cores.
    """
    result = 0
    for i in range(n):
        result += math.sqrt(i) * math.sin(i)
    return result


def demonstrate_threading_with_gil():
    """
    This shows how Python threading is affected by the GIL.
    Even with multiple threads, they can't use multiple CPU cores for CPU-bound work.
    """
    print("=== Threading with GIL (CPU-bound task) ===")

    # Define the work amount
    work_amount = 100000000
    num_threads = 4

    # Single-threaded execution
    print("Running single-threaded...")
    start_time = time.time()
    result = cpu_intensive_work(work_amount)
    single_thread_time = time.time() - start_time
    print(f"Single thread time: {single_thread_time:.2f} seconds")

    # Multi-threaded execution (affected by GIL)
    print(f"\nRunning with {num_threads} threads...")
    start_time = time.time()

    # Create and start threads
    threads = []
    for i in range(num_threads):
        # Each thread does 1/4 of the work
        thread = threading.Thread(
            target=cpu_intensive_work, args=(work_amount // num_threads,)
        )
        threads.append(thread)

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    multi_thread_time = time.time() - start_time
    print(f"Multi-thread time: {multi_thread_time:.2f} seconds")

    # Show the (lack of) speedup
    if single_thread_time > 0:
        speedup = single_thread_time / multi_thread_time
        print(f"Threading speedup: {speedup:.2f}x")
        if speedup < 1.5:  # Less than significant improvement
            print("❌ Threading provided little to no speedup for CPU-bound work!")
            print("   This is because of the GIL - threads can't use multiple cores.")

    return single_thread_time


def demonstrate_multiprocessing_without_gil():
    """
    This shows how multiprocessing bypasses the GIL by using separate processes.
    Each process has its own Python interpreter and can use a different CPU core.
    """
    print("\n=== Multiprocessing without GIL (CPU-bound task) ===")

    work_amount = 100000000
    num_processes = 4

    print(f"Running with {num_processes} processes...")
    start_time = time.time()

    # Create a pool of processes
    with multiprocessing.Pool(processes=num_processes) as pool:
        # Split the work among processes
        work_per_process = work_amount // num_processes
        work_chunks = [work_per_process] * num_processes

        # Each process runs on a separate CPU core (if available)
        pool.map(cpu_intensive_work, work_chunks)

    multi_process_time = time.time() - start_time
    print(f"Multi-process time: {multi_process_time:.2f} seconds")

    return multi_process_time


def demonstrate_io_bound_threading():
    """
    This shows where Python threading DOES help - with I/O-bound tasks.
    The GIL is released during I/O operations, allowing other threads to run.
    """
    print("\n=== Threading with I/O-bound tasks (GIL released) ===")

    def io_bound_work():
        """Simulates I/O-bound work like network requests or file reading"""
        # time.sleep() simulates I/O wait time (network, disk, etc.)
        # During sleep, the GIL is released!
        time.sleep(0.5)
        return "Task completed"

    num_tasks = 8

    # Sequential I/O operations
    print("Running I/O tasks sequentially...")
    start_time = time.time()
    for i in range(num_tasks):
        io_bound_work()
    sequential_time = time.time() - start_time
    print(f"Sequential time: {sequential_time:.2f} seconds")

    # Concurrent I/O operations with threading
    print(f"\nRunning {num_tasks} I/O tasks with threading...")
    start_time = time.time()

    threads = []
    for i in range(num_tasks):
        thread = threading.Thread(target=io_bound_work)
        threads.append(thread)

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    threaded_time = time.time() - start_time
    print(f"Threaded time: {threaded_time:.2f} seconds")

    speedup = sequential_time / threaded_time
    print(f"I/O threading speedup: {speedup:.2f}x")
    print("✅ Threading works great for I/O-bound tasks!")
    print("   The GIL is released during I/O operations.")


def main():
    """
    Demonstrates the relationship between threads, CPU cores, and the GIL
    """
    print("Demonstrating Python's GIL impact on threading vs multiprocessing")
    print(f"Your system has {multiprocessing.cpu_count()} CPU cores")
    print("=" * 60)

    # Show how GIL affects CPU-bound threading
    single_thread_time = demonstrate_threading_with_gil()
    multi_process_time = demonstrate_multiprocessing_without_gil()

    # Compare the results
    if single_thread_time > 0 and multi_process_time > 0:
        multiprocessing_speedup = single_thread_time / multi_process_time
        print(f"\n📊 SUMMARY:")
        print(
            f"Multiprocessing speedup vs single thread: {multiprocessing_speedup:.2f}x"
        )
        if multiprocessing_speedup > 2:
            print("✅ Multiprocessing successfully used multiple CPU cores!")

    # Show where threading DOES help
    demonstrate_io_bound_threading()

    print("\n🎓 KEY TAKEAWAY:")
    print(
        "• Python threads can't use multiple CPU cores for CPU-bound work (due to GIL)"
    )
    print(
        "• Python processes CAN use multiple CPU cores (each has its own interpreter)"
    )
    print("• Python threads ARE useful for I/O-bound work (GIL released during I/O)")


if __name__ == "__main__":
    main()
