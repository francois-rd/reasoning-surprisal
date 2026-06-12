import os

from statsmodels.formula.api import mixedlm
from statsmodels.api import qqplot
from coma import command
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np

from ....llms import Nickname, flatten
from ....io import (
    ConditionalPrinter,
    PathConfig,
    ensure_path,
    walk_files,
)

from .base import Config
from .postprocess import DFCols


def all_category_orders() -> dict[str, list]:
    return {
        "RelationType": [
            "AtLocation",
            "Causes",
            "PartOf",
            "IsA",
            "UsedFor",
            "HasPrerequisite",
        ]
    }


class Analyze:
    def __init__(self, cfg: Config, df: pd.DataFrame):
        self.cfg = cfg
        self.df = df

        # Reorder the variant category, so that control condition is first.
        self.df[DFCols.VARIANT_ID.value] = pd.Categorical(
            self.df[DFCols.VARIANT_ID.value],
            categories=self.cfg.collate_variants,
            ordered=True,
        )

    def run(self) -> tuple[pd.DataFrame, str]:
        dfs, summary = [], ""
        for shots, group_df in self.df.groupby(by=DFCols.PROMPT_SHOTS.value):
            shots_df, shots_summary = self._do_run(shots, group_df)
            dfs.append(shots_df)
            summary += shots_summary + "\n\n"
        return pd.concat(dfs), summary

    def _do_run(self, shots, df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
        # The inclusion of these vc parameters depends on real data.
        vc_formula = {
            # variant_id: f"0 + C({variant_id})",  # This is CLEARLY pathological in the residuals plot.
            # relation_type: f"0 + C({relation_type})", # This is probably worse to include, but not awful.
        }
        vc_formula = vc_formula if vc_formula else None

        surprisal = DFCols.SURPRISAL.value
        variant_id = DFCols.VARIANT_ID.value
        relation_type = DFCols.RELATION_TYPE.value
        group_id = DFCols.GROUP_ID.value
        model = mixedlm(
            formula=f"{surprisal} ~ C({variant_id})*C({relation_type})",
            data=df,
            vc_formula=vc_formula,
            groups=group_id,
        )
        result = model.fit()

        if self.cfg.show_assumption_plots:
            self.assumption_plots(result)

        summary = f"********** {shots} shots **********\n{result.summary()}"
        var_ids = df[variant_id].unique().tolist()
        var_ids.remove(self.cfg.factual_variant_id)
        r_types = df[relation_type].unique()
        return self._post_hoc_tests(shots, var_ids, r_types, result), summary

    @staticmethod
    def assumption_plots(result):
        # 1. Residuals vs Fitted (Homoscedasticity).
        plt.scatter(result.fittedvalues, result.resid)
        plt.axhline(0, linestyle="--", color="red")
        plt.xlabel("Fitted Values")
        plt.ylabel("Residuals")
        plt.title("Residuals vs Fitted (Homoscedasticity)")
        plt.show()

        # 2. Q-Q Plot (Normality).
        qqplot(result.resid, line="s")
        plt.title("Q-Q Plot (Normality)")
        plt.show()

    @staticmethod
    def _post_hoc_tests(shots, variant_ids, relation_types, result) -> pd.DataFrame:
        data = {}
        for relation in relation_types:
            for variant in variant_ids:
                # We have some random effects variables, but this matrix should
                # be 1 x num_fixed_effects. By manual inspection, we know that
                # fixed effects appear in "results.params" before random effects,
                # so we won't have any offset errors in indexing (it's the end of
                # the array -- the part that we won't index -- that would be off).
                r_matrix = np.zeros((1, result.k_fe))

                # Main effect.
                main_index = result.params.index.get_loc(f"C(VariantID)[T.{variant}]")
                r_matrix[0, main_index] = 1
                # For non-reference relation type: main effect + interaction
                if relation != relation_types[0]:
                    interaction_index = result.params.index.get_loc(
                        f"C(VariantID)[T.{variant}]:C(RelationType)[T.{relation}]"
                    )
                    r_matrix[0, interaction_index] = 1
                test_result = result.t_test(r_matrix, use_t=True)
                p_value = test_result.pvalue
                data.setdefault(DFCols.PROMPT_SHOTS.value, []).append(shots)
                data.setdefault(DFCols.RELATION_TYPE.value, []).append(relation)
                data.setdefault(DFCols.VARIANT_ID.value, []).append(variant)
                data.setdefault(DFCols.P_VALUE.value, []).append(p_value)
        return pd.DataFrame(data)


class PlotSaver:
    def __init__(self, path: PathConfig, cfg: Config, nickname: Nickname | None):
        self.path = path
        self.cfg = cfg
        self.llm = nickname

    def save(
        self,
        dir_ids: tuple[str, ...],
        file_ids: tuple[str, ...],
        fig: go.Figure,
        **kwargs,
    ) -> None:
        from kaleido._kaleido_tab import KaleidoError  # noqa

        out_dir = self.cfg.plots_dir(self.path.cnet_exp_dir)
        if self.llm is not None:
            out_dir = os.path.join(out_dir, flatten(self.llm))
        file_base = "-".join(file_ids) + ".pdf"
        file_path = ensure_path(os.path.join(out_dir, *dir_ids, file_base))
        try:
            fig.write_image(file_path, **kwargs)
        except KaleidoError:
            pass


class ViolinPlots:
    def __init__(self, cfg: Config, main_df: pd.DataFrame, analysis_df: pd.DataFrame):
        self.cfg = cfg
        self.df = main_df
        self.analysis_df = analysis_df

    def run(self, saver: PlotSaver) -> None:
        for shots, group_df in self.df.groupby(by=DFCols.PROMPT_SHOTS.value):
            for file_ids, fig in self._do_run(shots, group_df).items():
                saver.save(dir_ids=("violin",), file_ids=file_ids, fig=fig)

    def _do_run(self, shots, df: pd.DataFrame) -> dict[tuple, go.Figure]:
        df = self._prepare(df)
        plots, v_id_key = {}, DFCols.VARIANT_ID.value
        for variant_id, group_df in df.groupby(by=v_id_key, observed=True):
            plots[(shots, variant_id)] = self._make_plot(shots, variant_id, group_df)
        return plots

    def _make_plot(self, shots, variant_id, df: pd.DataFrame) -> go.Figure:
        fig = px.violin(
            data_frame=df,
            y=DFCols.SURPRISAL.value,
            color=DFCols.RELATION_TYPE.value,
            box=True,
            category_orders=all_category_orders(),
            template="simple_white",
            width=700,  # default is 700
            height=500,  # default is 500
        )

        fig.update_layout(
            font=dict(family="Times New Roman", weight="bold"),
            xaxis=dict(
                title=dict(text="Relation Type", font_size=20),
                showticklabels=False,
                ticklen=0,
            ),
            yaxis=dict(title=dict(text=f"Relative Surprisal Difference", font_size=20)),
            margin=dict(l=0, r=0, b=40, t=0),
            legend=dict(
                title=dict(text="   Legend", font_size=16),
                # xanchor="right", yanchor="top", # This + width=500 above for compact.
            ),
        )
        fig.add_hline(y=0.0, opacity=0.5, line_width=2, line_dash="dash")
        fig.update_traces(meanline_visible=True)
        self._add_stat_sig_markers(shots, variant_id, df, fig)
        return fig

    def _add_stat_sig_markers(
        self, shots, variant_id, df: pd.DataFrame, fig: go.Figure
    ) -> None:
        # Currently broken: annotations are being added next to violin plots as new
        # traces rather than above each one.
        return

        # # Get corresponding p-values for each relation type.
        # shots_mask = self.analysis_df[DFCols.PROMPT_SHOTS.value] == shots
        # vid_mask = self.analysis_df[DFCols.VARIANT_ID.value] == variant_id
        # subset_df = self.analysis_df[shots_mask & vid_mask]
        # p_values = {}
        # for r in subset_df.to_dict(orient="records"):
        #     p_values[r[DFCols.RELATION_TYPE.value]] = r[DFCols.P_VALUE.value].item()
        #
        # # Calculate a y-position slightly above the maximum data value.
        # y_pos = df[DFCols.SURPRISAL.value].max() * 1.05
        # for relation_type, p_value in p_values.items():
        #     if p_value < 0.001:
        #         text = "****"
        #     elif p_value < 0.005:
        #         text = "***"
        #     elif p_value < 0.01:
        #         text = "**"
        #     elif p_value < 0.05:
        #         text = "*"
        #     else:
        #         text = ""
        #     fig.add_annotation(
        #         x=relation_type,  # This is not working. Add 'next to' rather than 'above'.
        #         y=y_pos,  # Position on y-axis (above the violin).
        #         text=text,  # Significance symbol.
        #         showarrow=False,
        #         font=dict(
        #             family="Times New Roman", size=16, weight="bold", color="black"
        #         ),
        #         xref="x",  # Reference x to the data axis.
        #         yref="y",  # Reference y to the data axis.
        #     )

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        v_id_key = DFCols.VARIANT_ID.value
        g_id_key = DFCols.GROUP_ID.value
        s_key = DFCols.SURPRISAL.value
        shots_key = DFCols.PROMPT_SHOTS.value

        # Split treatments from control.
        df_control = df[df[v_id_key] == self.cfg.factual_variant_id].copy()
        df_treatments = df[df[v_id_key] != self.cfg.factual_variant_id].copy()

        # Merge treatments with control on GroupID.
        df_merged = df_treatments.merge(
            df_control[[g_id_key, s_key]],
            on=g_id_key,
            suffixes=("_treat", "_control"),
            how="inner",  # Keeps only subjects that have both control and treatment.
        )

        # Calculate paired differences.
        variant_ids = self.cfg.collate_variants.copy()
        variant_ids.remove(self.cfg.factual_variant_id)
        for variant_id in variant_ids:
            df_merged[f"diff_{variant_id}"] = df_merged.apply(
                lambda row: (
                    row[f"{s_key}_treat"] - row[f"{s_key}_control"]
                    if row[v_id_key] == variant_id
                    else None
                ),
                axis=1,
            )

        # Sum across columns to merge. Works because NaN + value = value.
        diff_cols = [f"diff_{v_id}" for v_id in variant_ids]
        df_merged[s_key] = df_merged[diff_cols].sum(axis=1)

        # Drop intermediary calculation columns.
        return df_merged.drop(
            columns=[f"{s_key}_treat", f"{s_key}_control", shots_key] + diff_cols
        )


class BarPlots:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.dfs = {}

    def add(self, nickname: Nickname, df: pd.DataFrame):
        self.dfs[flatten(nickname)] = df

    def run(self, saver: PlotSaver) -> None:
        # Prepare all.
        dfs = []
        for nickname, df in self.dfs.items():
            for shots, group_df in df.groupby(by=DFCols.PROMPT_SHOTS.value):
                group_df = self._prepare(group_df)
                group_df[DFCols.LLM.value] = nickname
                dfs.append(group_df)
        df = pd.concat(dfs)

        # Run prepared collection.
        for shots, group_df in df.groupby(by=DFCols.PROMPT_SHOTS.value):
            for file_ids, fig in self._do_run(shots, group_df).items():
                saver.save(dir_ids=("bar",), file_ids=file_ids, fig=fig)

    def _do_run(self, shots, df: pd.DataFrame) -> dict[tuple, go.Figure]:
        plots, v_id_key = {}, DFCols.VARIANT_ID.value
        for variant_id, group_df in df.groupby(by=v_id_key, observed=True):
            plots[(shots, variant_id)] = self._make_plot(group_df)
        return plots

    @staticmethod
    def _make_plot(df: pd.DataFrame) -> go.Figure:
        # Aggregate raw data.
        agg_df = (
            df.groupby(by=[DFCols.RELATION_TYPE.value, DFCols.LLM.value])
            .agg(
                Mean=(DFCols.SURPRISAL.value, "mean"),
                CI=(DFCols.SURPRISAL.value, "sem"),
            )
            .reset_index()
        )
        agg_df["CI"] *= 1.96

        fig = px.bar(
            data_frame=agg_df,
            x=DFCols.LLM.value,
            y="Mean",
            color=DFCols.RELATION_TYPE.value,
            error_y="CI",
            barmode="group",
            category_orders=all_category_orders(),
            template="simple_white",
            width=700,  # default is 700
            height=500,  # default is 500
        )

        fig.update_layout(
            font=dict(family="Times New Roman", weight="bold"),
            xaxis=dict(title=None),
            yaxis=dict(title=dict(text=f"Relative Surprisal Difference", font_size=20)),
            margin=dict(l=0, r=0, b=60, t=0),
            legend=dict(
                title=dict(text="   Legend", font_size=16),
                # xanchor="right", yanchor="top", # This + width=500 above for compact.
            ),
        )
        fig.update_traces(opacity=0.8, marker_line_width=2)
        return fig

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        v_id_key = DFCols.VARIANT_ID.value
        g_id_key = DFCols.GROUP_ID.value
        s_key = DFCols.SURPRISAL.value

        # Split treatments from control.
        df_control = df[df[v_id_key] == self.cfg.factual_variant_id].copy()
        df_treatments = df[df[v_id_key] != self.cfg.factual_variant_id].copy()

        # Merge treatments with control on GroupID.
        df_merged = df_treatments.merge(
            df_control[[g_id_key, s_key]],
            on=g_id_key,
            suffixes=("_treat", "_control"),
            how="inner",  # Keeps only subjects that have both control and treatment.
        )

        # Calculate paired differences.
        variant_ids = self.cfg.collate_variants.copy()
        variant_ids.remove(self.cfg.factual_variant_id)
        for variant_id in variant_ids:
            df_merged[f"diff_{variant_id}"] = df_merged.apply(
                lambda row: (
                    row[f"{s_key}_treat"] - row[f"{s_key}_control"]
                    if row[v_id_key] == variant_id
                    else None
                ),
                axis=1,
            )

        # Sum across columns to merge. Works because NaN + value = value.
        diff_cols = [f"diff_{v_id}" for v_id in variant_ids]
        df_merged[s_key] = df_merged[diff_cols].sum(axis=1)

        # Drop intermediary calculation columns.
        return df_merged.drop(
            columns=[f"{s_key}_treat", f"{s_key}_control"] + diff_cols
        )


@command(name="cnet.analyze")
class ConceptNetAnalyze:
    def __init__(self, path: PathConfig, conceptnet: Config):
        self.path = path
        self.cfg = conceptnet
        self.print = ConditionalPrinter(self.cfg.verbose)

    def _skip(self, test_nickname: Nickname) -> bool:
        for nickname in self.cfg.llms:
            if flatten(test_nickname) == flatten(nickname):
                return False
        return True

    def _save_analysis(
        self, nickname: Nickname, analysis_df: pd.DataFrame, analysis_summary: str
    ) -> None:
        out_dir = self.cfg.analysis_dir(self.path.cnet_exp_dir)
        df_file = ensure_path(os.path.join(out_dir, nickname + ".csv"))
        summary_file = ensure_path(os.path.join(out_dir, nickname + ".txt"))
        analysis_df.to_csv(df_file, index=False)
        with open(summary_file, "w", encoding="utf-8") as f:
            f.write(analysis_summary)

    def run(self):
        bar_plots = BarPlots(self.cfg)
        for walk in walk_files(self.cfg.postprocess_dir(self.path.cnet_exp_dir)):
            inference_path, nickname = walk.path, flatten(walk.no_ext())
            if self._skip(nickname):
                continue
            self.print("Analyzing results of model:", nickname)
            self.print("    Loading data...")
            df = pd.read_csv(inference_path)
            bar_plots.add(nickname, df)
            self.print("    Running statistical analysis...")
            analysis_df, analysis_summary = Analyze(self.cfg, df).run()
            self._save_analysis(nickname, analysis_df, analysis_summary)
            self.print("    Building violin plots...")
            saver = PlotSaver(self.path, self.cfg, nickname)
            ViolinPlots(self.cfg, df, analysis_df=analysis_df).run(saver)
            self.print("    Done.")
        self.print("Building collective bar plots...")
        bar_plots.run(PlotSaver(self.path, self.cfg, nickname=None))
        self.print("Done.")
