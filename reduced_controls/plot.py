"""Plot the profiles used in Figures 3, 4, 5 and C.1."""

import matplotlib.pyplot as plt

from .experiment import ACTIVATION, ACTIVITY, BASE, RADII, case_name


def format_axes(ax, title, ylabel, ceiling):
    ax.set(title=title, xlabel="Radius $r$", ylabel=ylabel, xlim=(0, 10), ylim=(0, ceiling),
           xticks=(0, 2.5, 5, 7.5, 10))
    ax.grid(color="0.88", linewidth=0.65)
    ax.set_axisbelow(True)


def figure_3(data):
    fig, axes = plt.subplots(1, 2, figsize=(10.7, 3.6), layout="constrained")
    r = data["radius"]
    axes[0].plot(r, data["reference_density"][0], color="#696969", ls="--", label="Initial density")
    for ax, field in zip(axes, ("density", "oxygen")):
        for case, label, color, style in (
            ("reference", "Reference", "#ca2b30", "-"),
            ("reaction_off", "Reaction removed", "#1c5cab", "-"),
            ("diffusion_only", "Reaction and aerotaxis removed", "#daa520", "--"),
        ):
            ax.plot(r, data[case + "_" + field][-1], color=color, ls=style, label=label)
    format_axes(axes[0], "(a) Cell density at $T=5$", "$N$", 0.62)
    format_axes(axes[1], "(b) Oxygen at $T=5$", "$u$", 10.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncols=4, frameon=False)
    return fig


def sweep(data, prefix, levels, colors, geometry, panels, legend_title):
    fig, axes = plt.subplots(*geometry, figsize=(11.0, 3.7) if geometry[0] == 1 else (9, 6.5),
                             layout="constrained")
    styles = ("--", "-.", ":", (0, (5, 2, 1, 2)), "-")
    for ax, (field, title, ylabel, ceiling) in zip(axes.flat, panels):
        for level, color, style in zip(levels, colors, styles):
            name = case_name(prefix, level)
            if field == "activated_egfr":
                profile = (level * BASE.k3 / BASE.d2) * data[name + "_egf"][-1]
            else:
                profile = data[name + "_" + field][-1]
            ax.plot(data["radius"], profile, color=color, ls=style, lw=1.8, label=f"{level:g}")
        format_axes(ax, title, ylabel, ceiling)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, title=legend_title, loc="outside upper center", ncols=5, frameon=False)
    return fig


def figure_c1(data):
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.0), layout="constrained")
    r = data["radius"]
    time_styles = (("#94a1b8", ":"), ("#e6a600", "--"), ("#1c4dd6", "-"))
    for ax, letter, radius in zip(axes[0], "abc", (1.5, 3.0, 5.0)):
        name = case_name("radius", radius)
        ax.axvline(radius, color="0.77", ls=":", lw=0.9)
        for index, (color, style) in enumerate(time_styles):
            ax.plot(r, data[name + "_density"][index], color=color, ls=style,
                    label=rf"$t={data['time'][index]:g}$")
        format_axes(ax, rf"({letter}) Initial radius $R_0={radius:g}$", "$N$", 0.65)
    colors = ("#2461eb", "#0d9c8a", "#e6a600", "#d945ed", "#db2424")
    for ax, field, title, ylabel, ceiling in zip(
        axes[1], ("density", "oxygen", "egf"), ("(d) Final density", "(e) Final oxygen", "(f) Final EGF"),
        ("$N$", "$u$", "$v$"), (0.65, 10.5, 3.2),
    ):
        for radius, color in zip(RADII, colors):
            ax.plot(r, data[case_name("radius", radius) + "_" + field][-1], color=color,
                    label=rf"$R_0={radius:g}$")
        format_axes(ax, title, ylabel, ceiling)
    for ax, loc, ncols in ((axes[0, 0], "outside upper center", 3),
                           (axes[1, 0], "outside lower center", 5)):
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc=loc, ncols=ncols, frameon=False)
    return fig


def plot(data):
    oxygen = ("oxygen", "(a) Oxygen", "$u$", 10.2)
    return {
        "figure_3": figure_3(data),
        "figure_4": sweep(data, "theta", ACTIVITY,
                         ("#3b4cc0", "#2a7ab9", "#2a9d8f", "#e9a23b", "#c83e4d"), (1, 3),
                         (oxygen, ("egf", "(b) Extracellular EGF", "$v$", 3.5),
                          ("density", "(c) Cell density", "$N$", 0.6)), r"Respiratory activity $\theta$"),
        "figure_5": sweep(data, "activation", ACTIVATION,
                         ("#d62728", "#2468b4", "#e5a000", "#cc79a7", "#111111"), (2, 2),
                         (oxygen, ("egf", "(b) Extracellular EGF", "$v$", 4),
                          ("activated_egfr", "(c) Activated EGFR", "$z^*$", 16),
                          ("density", "(d) Cell density", "$N$", 0.6)), r"Residual EGFR activation $\vartheta_E$"),
        "figure_C1": figure_c1(data),
    }
