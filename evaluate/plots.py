import argparse

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
import seaborn as sns


class FigureThreePlotter:
    def get_colors(self, color, n):
        """Generate a list of `n` colors from a matplotlib colormap."""
        if n < 1:
            return []
        if n == 1:
            return [plt.get_cmap(color)(0.5)]
        if n == 2:
            return [plt.get_cmap(color)(0.25), plt.get_cmap(color)(0.75)]
        if n == 3:
            return [plt.get_cmap(color)(0.2), plt.get_cmap(color)(0.5), plt.get_cmap(color)(0.8)]
        return [plt.get_cmap(color)(i / (n - 1)) for i in range(n)]

    def radial_bins(self, x):
        """Map radial distance to plotting bin center.

        Args:
            x: Radial distance in planetary radii.

        Returns:
            Bin center value or NaN if outside plotting range.
        """
        if 1 <= x < 1.2:
            return 1.1
        if 1.2 <= x < 1.4:
            return 1.3
        if 1.4 <= x < 1.6:
            return 1.5
        if 1.6 <= x < 1.8:
            return 1.7
        if 1.8 <= x < 2:
            return 1.9
        if 2 <= x < 3:
            return 2.5
        if 3 <= x < 4:
            return 3.5
        if 4 <= x <= 5:
            return 4.5
        if 5 < x <= 6:
            return 5.5
        return np.nan

    def set_radial_bounds(self, df):
        """Prepare long-form radial error dataframe for the bottom panel.

        Args:
            df: Results dataframe containing radius and `Bx/By/Bz_diff` columns.

        Returns:
            Long-form dataframe with radial bins, component labels and reference envelopes.
        """
        radial_df = df.copy()
        if "R" in radial_df.columns:
            radial_source_col = "R"
        elif "R1" in radial_df.columns:
            radial_source_col = "R1"
        else:
            raise ValueError("Expected radius column 'R' or 'R1' in results dataframe.")

        radial_df["R_bins"] = radial_df[radial_source_col].apply(self.radial_bins)
        radial_df = radial_df.dropna(subset=["R_bins"])

        rad_df_long = radial_df.melt(
            id_vars=["R_bins"],
            value_vars=["Bx_diff", "By_diff", "Bz_diff"],
            var_name="DiffType",
            value_name="DiffValue",
        )
        rad_df_long = rad_df_long.sort_values(by="R_bins", inplace=False)
        
        return rad_df_long

    def plot_combined_hist_radial(self, test_df, radial_df_long, save_path):
        """Create and save the combined histogram/radial error figure.

        Args:
            test_df: Results dataframe containing true/predicted fields and diffs.
            radial_df_long: Output of `set_radial_bounds`.
            save_path: Output filepath for figure.
        """
        # Figure layout: top row has 4 histograms, bottom row has radial error boxplots.
        fig = plt.figure(figsize=(10, 6))
        gs = fig.add_gridspec(2, 4, height_ratios=[0.5, 1])
        ax_top = [fig.add_subplot(gs[0, i]) for i in range(4)]
        ax_bottom = fig.add_subplot(gs[1, :])

        # Top panel styling and summary statistics.
        true, pred = self.get_colors("viridis_r", 2)
        kde_bool = True

        # Decor + labels.
        Bx_lab = r'$B_x$'
        By_lab = r'$B_y$'
        Bz_lab = r'$B_z$'
        B_lab = r'$|B|$'
        sum_traj = r'$\sum T$'
        Bx_MAE = np.mean(np.abs(test_df['Bx_diff']))
        By_MAE = np.mean(np.abs(test_df['By_diff']))
        Bz_MAE = np.mean(np.abs(test_df['Bz_diff']))
        B_MAE = np.sqrt(Bx_MAE**2 + By_MAE**2 + Bz_MAE**2)
        #B_MAE = np.mean(np.abs(test_df['B_diff']))

        # Top row: predicted vs true distributions for each component and magnitude.
        sns.histplot(data=test_df, x="Bx_pred", label="Bx Predicted", color=pred, kde=kde_bool, stat="count", ax=ax_top[0])
        sns.histplot(data=test_df, x="Bx_true", label="Bx True", color=true, kde=kde_bool, stat="count", ax=ax_top[0])
        ax_top[0].set_title(f"MAE: {Bx_MAE:.1f} nT", fontsize=10)
        ax_top[0].set_ylabel(sum_traj)
        ax_top[0].set_xlabel(f"Δ {Bx_lab} [nT]")

        sns.histplot(data=test_df, x="By_pred", label="By Predicted", color=pred, kde=kde_bool, stat="count", ax=ax_top[1])
        sns.histplot(data=test_df, x="By_true", label="By True", color=true, kde=kde_bool, stat="count", ax=ax_top[1])
        ax_top[1].set_title(f"MAE: {By_MAE:.1f} nT", fontsize=10)
        ax_top[1].set_ylabel(sum_traj)
        ax_top[1].set_xlabel(f"Δ {By_lab} [nT]")

        sns.histplot(data=test_df, x="Bz_pred", label="Bz Predicted", color=pred, kde=kde_bool, stat="count", ax=ax_top[2])
        sns.histplot(data=test_df, x="Bz_true", label="Bz True", color=true, kde=kde_bool, stat="count", ax=ax_top[2])
        ax_top[2].set_title(f"MAE: {Bz_MAE:.1f} nT", fontsize=10)
        ax_top[2].set_ylabel(sum_traj)
        ax_top[2].set_xlabel(f"Δ {Bz_lab} [nT]")

        sns.histplot(data=test_df, x="B_pred", label="B Predicted", color=pred, kde=kde_bool, stat="count", ax=ax_top[3])
        sns.histplot(data=test_df, x="B_true", label="B True", color=true, kde=kde_bool, stat="count", ax=ax_top[3])
        ax_top[3].set_title(f"MAE: {B_MAE:.1f} nT", fontsize=10)
        ax_top[3].set_ylabel(sum_traj)
        ax_top[3].set_xlabel(f"Δ {B_lab} [nT]")

        # Y-axis count scaling for readability.
        scale_factor = 10000
        scale_factor_label = r'$10^{4}$'
        y_label_denotation = f"Count, {scale_factor_label}"
        y_formatter = FuncFormatter(lambda y, _: f"{y / scale_factor:.1f}")
        for ax_ in ax_top:
            ax_.yaxis.set_major_formatter(y_formatter)
            ax_.set_ylabel("")
        ax_top[0].set_ylabel(y_label_denotation)


        # Shared legend and panel cosmetics.
        handles, _ = ax_top[0].get_legend_handles_labels()
        fig.legend(handles, ["Predicted", "True"], loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.6), fontsize=10, frameon=False)

        for a in ax_top:
            a.spines["top"].set_visible(False)
            a.spines["right"].set_visible(False)

        ax_top[0].text(-0.2, 1.15, "a", transform=ax_top[0].transAxes, fontsize=16, fontweight="bold", va="top")
        ax_bottom.text(-0.04, 1.15, "b", transform=ax_bottom.transAxes, fontsize=16, fontweight="bold", va="top")

        # Bottom row: radial-bin error distributions by field component.
        radial_df_long["R_bins"] = radial_df_long["R_bins"].astype(str)

        bx, by, bz = self.get_colors("plasma", 3)
        sns.boxplot(
            x="R_bins",
            y="DiffValue",
            hue="DiffType",
            data=radial_df_long,
            showfliers=False,
            ax=ax_bottom,
            zorder=1,
            palette={"Bx_diff": bx, "By_diff": by, "Bz_diff": bz},
            width=0.7,
            fliersize=0.2,
            linewidth=1,
            notch=True,
            showcaps=True,
        )

        ax_bottom.set_xticks(range(9))
        ax_bottom.set_xticklabels([
            "1.0 - 1.2",
            "1.2 - 1.4",
            "1.4 - 1.6",
            "1.6 - 1.8",
            "1.8 - 2.0",
            "2.0 - 3.0",
            "3.0 - 4.0",
            "4.0 - 5.0",
            "5.0 - 6.0",
        ])
        re_symbol = r"$\mathrm{R_E}$"
        ax_bottom.set_xlabel(f"Distance from Europa's Center ({re_symbol})", labelpad=15)

        ax_bottom.set_yscale("symlog")
        ax_bottom.set_yticks([-50, -10, -5, -1, 0, 1, 5, 10, 50])
        ax_bottom.set_yticklabels(["-50", "-10", "-5", "-1", "0", "1", "5", "10", "50"])
        ax_bottom.set_ylabel("Prediction Error\n(Difference) [nT]")
        ax_bottom.set_ylim(-60, 60)

        #grid line at 1, 0 and -1
        line_style = '-'
        line_width = 0.5
        #ax_bottom.axhline(1, color="gray", linestyle=line_style, linewidth=line_width, zorder=0)
        ax_bottom.axhline(0, color="gray", linestyle=line_style, linewidth=line_width, zorder=0)
        #ax_bottom.axhline(-1, color="gray", linestyle=line_style, linewidth=line_width, zorder=0)

        # Bottom legend.
        box_handles, _ = ax_bottom.get_legend_handles_labels()
        ax_bottom.legend(
            list(box_handles[:3]),
            ["Bx", "By", "Bz"],
            loc="upper center",
            bbox_to_anchor=(0.5, -0.27),
            ncol=4,
            frameon=False,
            fontsize=10,
        )

        # Final render.
        sns.despine(ax=ax_bottom, top=True, right=True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=600, bbox_inches="tight")


def parse_args():
    """Parse CLI args for standalone plotting script."""
    parser = argparse.ArgumentParser(description="Plot LEAP evaluation figures")
    parser.add_argument("--results-path", type=str, default="data/out/test_results.csv")
    parser.add_argument("--save-path", type=str, default="assets/figure3.png")
    return parser.parse_args()


def main():
    """Entry-point for standalone plotting."""
    args = parse_args()
    test_set = pd.read_csv(args.results_path)
    fig3 = FigureThreePlotter()
    radial_bounds = fig3.set_radial_bounds(test_set)
    fig3.plot_combined_hist_radial(test_set, radial_bounds, args.save_path)


if __name__ == "__main__":
    main()
