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


def all_category_orders(llm_order: list[str]) -> dict[str, list]:
    return {
        DFCols.RELATION_TYPE.value: [
            "AtLocation",
            "Causes",
            "PartOf",
            "IsA",
            "UsedFor",
            "HasPrerequisite",
        ],
        DFCols.LLM.value: llm_order.copy(),
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
            data=self.df,
            vc_formula=vc_formula,
            groups=group_id,
        )
        result = model.fit()

        if self.cfg.show_assumption_plots:
            self.assumption_plots(result)

        var_ids = self.df[variant_id].unique().tolist()
        var_ids.remove(self.cfg.factual_variant_id)
        r_types = self.df[relation_type].unique()
        return self._post_hoc_tests(var_ids, r_types, result), str(result.summary())

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
    def _post_hoc_tests(variant_ids, relation_types, result) -> pd.DataFrame:
        data = {}
        v_id_key, r_key = DFCols.VARIANT_ID.value, DFCols.RELATION_TYPE.value
        for relation in relation_types:
            for variant in variant_ids:
                # We have some random effects variables, but this matrix should
                # be 1 x num_fixed_effects. By manual inspection, we know that
                # fixed effects appear in "results.params" before random effects,
                # so we won't have any offset errors in indexing (it's the end of
                # the array -- the part that we won't index -- that would be off).
                r_matrix = np.zeros((1, result.k_fe))

                # Main effect.
                main_index = result.params.index.get_loc(f"C({v_id_key})[T.{variant}]")
                r_matrix[0, main_index] = 1
                # For non-reference relation type: main effect + interaction
                if relation != relation_types[0]:
                    interaction_index = result.params.index.get_loc(
                        f"C({v_id_key})[T.{variant}]:C({r_key})[T.{relation}]"
                    )
                    r_matrix[0, interaction_index] = 1
                test_result = result.t_test(r_matrix, use_t=True)
                p_value = test_result.pvalue
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
        for file_ids, fig in self._do_run().items():
            saver.save(dir_ids=("violin",), file_ids=file_ids, fig=fig)

    def _do_run(self) -> dict[tuple, go.Figure]:
        df = self._prepare(self.df)
        plots, v_id_key = {}, DFCols.VARIANT_ID.value
        for variant_id, group_df in df.groupby(by=v_id_key, observed=True):
            plots[(variant_id,)] = self._make_plot(group_df)
        return plots

    def _make_plot(self, df: pd.DataFrame) -> go.Figure:
        fig = px.violin(
            data_frame=df,
            y=DFCols.SURPRISAL.value,
            color=DFCols.RELATION_TYPE.value,
            box=True,
            category_orders=all_category_orders(self.cfg.llms),
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
            legend=dict(title=dict(text="   Legend", font_size=16)),
        )
        fig.add_hline(y=0.0, opacity=0.5, line_width=2, line_dash="dash")
        fig.update_traces(meanline_visible=True)
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
            df = self._prepare(df)
            df[DFCols.LLM.value] = nickname
            dfs.append(df)

        # Run prepared collection.
        for file_ids, fig in self._do_run(pd.concat(dfs)).items():
            saver.save(dir_ids=("bar",), file_ids=file_ids, fig=fig)

    def _do_run(self, df: pd.DataFrame) -> dict[tuple, go.Figure]:
        plots, v_id_key = {}, DFCols.VARIANT_ID.value
        for variant_id, group_df in df.groupby(by=v_id_key, observed=True):
            plots[(variant_id,)] = self._make_plot(group_df)
        return plots

    def _make_plot(self, df: pd.DataFrame) -> go.Figure:
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
            category_orders=all_category_orders(self.cfg.llms),
            template="simple_white",
            width=700,  # default is 700
            height=500,  # default is 500
        )

        fig.update_layout(
            font=dict(family="Times New Roman", weight="bold"),
            xaxis=dict(title=None, tickfont=dict(size=8), tickangle=15),
            yaxis=dict(title=dict(text=f"Relative Surprisal Difference", font_size=20)),
            margin=dict(l=0, r=0, b=60, t=0),
            legend=dict(title=dict(text="   Legend", font_size=16)),
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
            if self.cfg.create_violin_plots:
                self.print("    Building violin plots...")
                saver = PlotSaver(self.path, self.cfg, nickname)
                ViolinPlots(self.cfg, df, analysis_df=analysis_df).run(saver)
                self.print("    Done.")
        self.print("Building collective bar plots...")
        bar_plots.run(PlotSaver(self.path, self.cfg, nickname=None))
        self.print("Done.")
