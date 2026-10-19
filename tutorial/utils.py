from collections.abc import Callable
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
from triton.testing import do_bench

BENCHMARK_DIR = Path(__file__).resolve().parent.parent / "benchmark"


def bench_sweep(
    providers: dict[str, Callable],
    xs: list[int],
    make_inputs: Callable,
    name: str,
    xlabel: str,
):
    """
    Time each provider over xs and save a plot to benchmark/<name>.png.

    Args:
        providers: A dictionary of provider names and their functions.
        xs: A list of input sizes.
        make_inputs: A function that makes input arguments for the provider functions.
        name: The name of the benchmark.
        xlabel: The label of the x-axis.

    Save the plot to benchmark/<name>.png.
    """

    results = {label: [] for label in providers}
    for x in xs:
        args = make_inputs(x)
        for label, fn in providers.items():
            results[label].append(do_bench(partial(fn, *args)) * 1e3)  # ms -> us
        print(x, {label: round(us[-1], 2) for label, us in results.items()})

    # Plot the results
    for label, us in results.items():
        plt.plot(xs, us, marker="o", label=label)
    plt.xscale("log", base=2)
    plt.xticks(xs, [str(x) for x in xs], rotation=45)
    plt.xlabel(xlabel)
    plt.ylabel("time (us)")
    plt.title(name)
    plt.legend()
    plt.grid(alpha=0.3)

    # Save the plot
    BENCHMARK_DIR.mkdir(exist_ok=True)
    plt.tight_layout()
    plt.savefig(BENCHMARK_DIR / f"{name}.png", dpi=150)
    plt.close()
