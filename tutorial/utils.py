from collections.abc import Callable
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
from triton.testing import do_bench

BENCHMARK_DIR = Path(__file__).resolve().parent.parent / "benchmark"

# Plot styles
STYLES = [
    {
        "color": "tab:blue",
        "marker": "o",
        "linestyle": "-",
        "markersize": 9,
        "alpha": 0.85,
    },
    {
        "color": "tab:red",
        "marker": "s",
        "linestyle": "--",
        "markersize": 7,
        "alpha": 0.85,
    },
    {
        "color": "tab:green",
        "marker": "^",
        "linestyle": "-",
        "markersize": 5,
        "alpha": 0.85,
    },
    {
        "color": "tab:purple",
        "marker": "D",
        "linestyle": "-.",
        "markersize": 7,
        "alpha": 0.85,
    },
    {
        "color": "tab:orange",
        "marker": "v",
        "linestyle": ":",
        "markersize": 7,
        "alpha": 0.85,
    },
]


def bench_sweep(
    providers: dict[str, Callable],
    x_sweep: list[int],
    make_inputs: Callable,
    xlabel: str,
    title: str,
    filename: str,
):
    """
    Time each provider over x_sweep and save a plot to benchmark/<filename>.png.

    Args:
        providers: A dictionary of provider names and their functions.
        x_sweep: A list of input sizes.
        make_inputs: A function that makes input arguments for the provider functions.
        xlabel: The label of the x-axis.
        title: The title of the benchmark.
        filename: The filename of the benchmark.

    Save the plot to benchmark/<filename>.png.
    """
    assert 0 < len(providers) <= len(STYLES), (
        "Number of providers must be greater than 0 and less than or equal to the number of styles"
    )

    # Generate the results
    results = {label: [] for label in providers}
    for x in x_sweep:
        # generate the inputs
        args = make_inputs(x)

        # time the providers
        for label, fn in providers.items():
            results[label].append(do_bench(partial(fn, *args)) * 1e3)  # ms -> us

        print(x, {label: round(us[-1], 2) for label, us in results.items()})

    # Plot the results
    for i, (label, us) in enumerate(results.items()):
        plt.plot(x_sweep, us, label=label, **STYLES[i % len(STYLES)])
    plt.xscale("log", base=2)
    plt.xticks(x_sweep, [str(x) for x in x_sweep], rotation=45)
    plt.xlabel(xlabel)
    plt.ylabel("time (us)")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)

    # Save the plot
    BENCHMARK_DIR.mkdir(exist_ok=True)
    plt.tight_layout()
    plt.savefig(BENCHMARK_DIR / f"{filename}.png", dpi=150)
    plt.close()
