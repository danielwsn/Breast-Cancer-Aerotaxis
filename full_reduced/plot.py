"""Plot matched full and reduced profiles for Figure 2 (five times, three fields)."""

import matplotlib.pyplot as plt

from .experiment import FIELDS


def plot(data):
    fig, axes = plt.subplots(5, 3, figsize=(10.5, 12), sharex=True, layout="constrained")
    colors = ("#d95319", "#0072bd", "#edb120", "#7e2f8e", "#009973")
    # Display limits of the manuscript panel grid. The saved densities include the full peaks.
    limits = (0.8, 1.05 * 9.650660242812263, 1.05 * 4.281445318882152)
    labels = ("Cell density", "Oxygen", "EGF")
    for i, time in enumerate(data["time"]):
        for j, (field, symbol) in enumerate(zip(FIELDS, ("N", "u", "v"))):
            ax = axes[i, j]
            ax.plot(data["radius"], data["reduced_" + field][i], color="black", lw=1.7, label="Reduced")
            for k, relaxation in enumerate(data["lambda_values"]):
                ax.plot(data["radius"], data["full_" + field][k, i], lw=1.15,
                        color=colors[k], label=rf"$\lambda={relaxation:g}$")
            ax.set(xlim=(0, 10), ylim=(0, limits[j]), ylabel=labels[j],
                   title=rf"${symbol}(r,{time:g})$", xticks=(0, 2, 4, 6, 8, 10))
            ax.grid(alpha=0.22)
            if i == len(data["time"]) - 1:
                ax.set_xlabel("Radius $r$")
    handles, legends = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legends, loc="outside upper center", ncols=6, frameon=False)
    return {"figure_2": fig}
