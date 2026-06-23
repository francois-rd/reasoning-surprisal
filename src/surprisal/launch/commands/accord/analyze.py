from enum import Enum
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

from .base import AccordSubset, Config
from .postprocess import DFCols


def all_category_orders(llm_order: list[str]) -> dict[str, list]:
    subsets = [s.value for s in AccordSubset if s != AccordSubset.BASELINE]
    return {DFCols.SUBSET.value: subsets, DFCols.LLM.value: llm_order.copy()}


class ConditionType(Enum):
    CONTROL = "CONTROL"
    TREATMENT = "TREATMENT"


class Comparator(Enum):
    F_REL_BASELINE = "F_REL_BASELINE"
    NF_CSQA_REL_BASELINE = "NF_CSQA_REL_BASELINE"
    NF_ACCORD_REL_BASELINE = "NF_ACCORD_REL_BASELINE"
    F_REL_NF_CSQA = "F_REL_NF_CSQA"
    F_REL_NF_ACCORD = "F_REL_NF_ACCORD"

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self == Comparator.F_REL_BASELINE:
            return self._transform_x_rel_baseline(
                df, factuality=True, label_key=DFCols.CSQA_LABEL
            )
        elif self == Comparator.NF_CSQA_REL_BASELINE:
            return self._transform_x_rel_baseline(
                df, factuality=False, label_key=DFCols.CSQA_LABEL
            )
        elif self == Comparator.NF_ACCORD_REL_BASELINE:
            return self._transform_x_rel_baseline(
                df, factuality=False, label_key=DFCols.ACCORD_LABEL
            )
        elif self == Comparator.F_REL_NF_CSQA:
            return self._transform_f_rel_nf(df, label_key=DFCols.CSQA_LABEL)
        elif self == Comparator.F_REL_NF_ACCORD:
            return self._transform_f_rel_nf(df, label_key=DFCols.ACCORD_LABEL)
        else:
            raise ValueError(f"Unsupported: {self}")

    @staticmethod
    def _select_surprisal_and_drop(df: pd.DataFrame, selection_col: str) -> None:
        df[DFCols.SURPRISAL.value] = df.apply(
            lambda row: row[DFCols.SURPRISAL.value + str(row[selection_col])], axis=1
        )
        df.drop(
            columns=[
                DFCols.SURPRISAL_A.value,
                DFCols.SURPRISAL_B.value,
                DFCols.SURPRISAL_C.value,
                DFCols.SURPRISAL_D.value,
                DFCols.SURPRISAL_E.value,
            ],
            inplace=True,
        )

    @staticmethod
    def _add_condition_merge_and_drop(
        df_control: pd.DataFrame, df_treat: pd.DataFrame
    ) -> pd.DataFrame:
        # Add new CONDITION column to each component df.
        df_control[DFCols.CONDITION.value] = ConditionType.CONTROL.value
        df_treat[DFCols.CONDITION.value] = ConditionType.TREATMENT.value

        # Merge components dfs back into one and drop unnecessary columns.
        df = pd.concat([df_control, df_treat], ignore_index=True).drop(
            columns=[
                DFCols.VARIANT_ID.value,
                DFCols.ACCORD_ID.value,
                DFCols.CSQA_ID.value,
                DFCols.IS_FACTUAL.value,
                DFCols.ACCORD_LABEL.value,
                DFCols.CSQA_LABEL.value,
            ]
        )

        # Reorder the CONDITION category, so that the control is first.
        df[DFCols.CONDITION.value] = pd.Categorical(
            df[DFCols.CONDITION.value],
            categories=[c.value for c in ConditionType],
            ordered=True,
        )

        # Reorder the SUBSET category, to be in numerical order.
        df[DFCols.SUBSET.value] = pd.Categorical(
            df[DFCols.SUBSET.value],
            categories=[s.value for s in AccordSubset if s != AccordSubset.BASELINE],
            ordered=True,
        )
        return df

    def _transform_x_rel_baseline(
        self, df: pd.DataFrame, factuality: bool, label_key: DFCols
    ) -> pd.DataFrame:
        # Recurring keys.
        subset_key, factual_key = DFCols.SUBSET.value, DFCols.IS_FACTUAL.value
        g_id_key, csqa_id_key = DFCols.GROUP_ID.value, DFCols.CSQA_ID.value

        # Split treatment from control. Drop unused factuality.
        baseline = AccordSubset.BASELINE.value
        df_control = df[df[subset_key] == baseline].copy()
        df_treatment = df[df[subset_key] != baseline]
        df_treatment = df_treatment[df_treatment[factual_key] == factuality].copy()

        # Select the CSQA or ACCORD surprisal.
        self._select_surprisal_and_drop(df_control, label_key.value)
        self._select_surprisal_and_drop(df_treatment, label_key.value)

        # For each row in df_treatment, find the equivalent CSQA ID row in df_control.
        # This is a many-to-one mapping. From df_treatment, keep only: CSQA ID (for
        # merge key) and GROUP ID & SUBSET (as these are the values we want to replace
        # in df_control).
        # Similarly, drop GROUP ID & SUBSET from df_control to enable that replacement.
        # All other column values are taken from df_control.
        expanded_baseline_df = pd.merge(
            left=df_treatment[[g_id_key, csqa_id_key, subset_key]],
            right=df_control.drop(columns=[g_id_key, subset_key]),
            on=csqa_id_key,
            how="left",
        )

        # With this merge, each original row from df_treatment now has a second row
        # where the column data comes from df_control, except GROUP ID & SUBSET, which
        # are copied from the original df_treatment row. Each component df also gets
        # a distinguishing CONDITION column. Extra columns are dropped.
        return self._add_condition_merge_and_drop(expanded_baseline_df, df_treatment)

    def _transform_f_rel_nf(self, df: pd.DataFrame, label_key: DFCols) -> pd.DataFrame:
        # Drop baseline. Split treatment from control.
        baseline = AccordSubset.BASELINE.value
        df = df[df[DFCols.SUBSET.value] != baseline]
        df_control = df[df[DFCols.IS_FACTUAL.value]].copy()
        df_treatment = df[~df[DFCols.IS_FACTUAL.value]].copy()

        # Select the CSQA or ACCORD surprisal.
        self._select_surprisal_and_drop(df_control, label_key.value)
        self._select_surprisal_and_drop(df_treatment, label_key.value)

        # Add condition col. Merge. Drop the columns that are no longer needed. Order.
        return self._add_condition_merge_and_drop(df_control, df_treatment)


class Analyze:
    def __init__(self, cfg: Config, df: pd.DataFrame):
        self.cfg = cfg
        self.df = df

    def run(self) -> tuple[pd.DataFrame, str]:
        # The inclusion of these vc parameters depends on real data.
        vc_formula = {}
        vc_formula = vc_formula if vc_formula else None

        surprisal = DFCols.SURPRISAL.value
        condition = DFCols.CONDITION.value
        subset = DFCols.SUBSET.value
        group_id = DFCols.GROUP_ID.value
        model = mixedlm(
            formula=f"{surprisal} ~ C({condition})*C({subset})",
            data=self.df,
            vc_formula=vc_formula,
            groups=group_id,
        )
        result = model.fit()

        if self.cfg.show_assumption_plots:
            self.assumption_plots(result)

        subsets = sorted(self.df[subset].unique())
        return self._post_hoc_tests(subsets, result), str(result.summary())

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
    def _post_hoc_tests(subsets, result) -> pd.DataFrame:
        data = {}
        cond_key, subset_key = DFCols.CONDITION.value, DFCols.SUBSET.value
        condition = ConditionType.TREATMENT.value
        for subset in subsets:
            # We have some random effects variables, but this matrix should
            # be 1 x num_fixed_effects. By manual inspection, we know that
            # fixed effects appear in "results.params" before random effects,
            # so we won't have any offset errors in indexing (it's the end of
            # the array -- the part that we won't index -- that would be off).
            r_matrix = np.zeros((1, result.k_fe))

            # Main effect.
            main_index = result.params.index.get_loc(f"C({cond_key})[T.{condition}]")
            r_matrix[0, main_index] = 1

            # For non-reference subset: main effect + interaction
            if subset != subsets[0]:
                interaction_index = result.params.index.get_loc(
                    f"C({cond_key})[T.{condition}]:C({subset_key})[T.{subset}]"
                )
                r_matrix[0, interaction_index] = 1

            # Test and save result.
            test_result = result.t_test(r_matrix, use_t=True)
            data.setdefault(DFCols.SUBSET.value, []).append(subset)
            data.setdefault(DFCols.P_VALUE.value, []).append(test_result.pvalue)
        return pd.DataFrame(data)


class PlotSaver:
    def __init__(self, path: PathConfig, cfg: Config, comparator: Comparator):
        self.path = path
        self.cfg = cfg
        self.comp = comparator.value

    def save(
        self,
        file_ids: tuple[str, ...],
        fig: go.Figure,
        **kwargs,
    ) -> None:
        from kaleido._kaleido_tab import KaleidoError  # noqa

        out_dir = os.path.join(self.cfg.plots_dir(self.path.accord_exp_dir), self.comp)
        file_path = ensure_path(os.path.join(out_dir, "-".join(file_ids) + ".pdf"))
        try:
            fig.write_image(file_path, **kwargs)
        except KaleidoError:
            pass


class ViolinPlots:
    def __init__(self, cfg: Config, main_df: pd.DataFrame):
        self.cfg = cfg
        self.df = main_df

    def run(self, saver: PlotSaver) -> None:
        llm = self.df[DFCols.LLM.value].unique()[0]
        saver.save(file_ids=(llm,), fig=self._make_plot())

    def _make_plot(self) -> go.Figure:
        df = self._prepare(self.df)
        fig = px.violin(
            data_frame=df,
            y=DFCols.SURPRISAL.value,
            color=DFCols.SUBSET.value,
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

    @staticmethod
    def _prepare(df: pd.DataFrame) -> pd.DataFrame:
        g_id_key = DFCols.GROUP_ID.value
        cond_key = DFCols.CONDITION.value
        surprisal_key = DFCols.SURPRISAL.value

        # Split treatments from control.
        df_control = df[df[cond_key] == ConditionType.CONTROL.value].copy()
        df_treatment = df[df[cond_key] == ConditionType.TREATMENT.value].copy()

        # Merge treatments with control on GroupID.
        df_merged = df_treatment.merge(
            df_control[[g_id_key, surprisal_key]],
            on=g_id_key,
            suffixes=("_treat", "_control"),
            how="inner",  # Keeps only subjects that have both control and treatment.
        )

        # Calculate paired differences.
        df_merged["diff"] = df_merged.apply(
            lambda row: (
                row[f"{surprisal_key}_treat"] - row[f"{surprisal_key}_control"]
                if row[cond_key] == ConditionType.TREATMENT.value
                else None
            ),
            axis=1,
        )

        # Sum across columns to merge. Works because NaN + value = value.
        df_merged[surprisal_key] = df_merged["diff"].sum()

        # Drop intermediary calculation columns.
        return df_merged.drop(
            columns=[f"{surprisal_key}_treat", f"{surprisal_key}_control", "diff"]
        )


class BarPlots:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.dfs = []

    def add(self, df: pd.DataFrame):
        self.dfs.append(df)

    def run(self, saver: PlotSaver) -> None:
        df = pd.concat([self._prepare(df) for df in self.dfs])
        saver.save(file_ids=("bar",), fig=self._make_plot(df))

    def _make_plot(self, df: pd.DataFrame) -> go.Figure:
        # Aggregate raw data.
        agg_df = (
            df.groupby(by=[DFCols.SUBSET.value, DFCols.LLM.value], observed=True)
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
            color=DFCols.SUBSET.value,
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

    @staticmethod
    def _prepare(df: pd.DataFrame) -> pd.DataFrame:
        g_id_key = DFCols.GROUP_ID.value
        cond_key = DFCols.CONDITION.value
        surprisal_key = DFCols.SURPRISAL.value

        # Split treatments from control.
        df_control = df[df[cond_key] == ConditionType.CONTROL.value].copy()
        df_treatment = df[df[cond_key] == ConditionType.TREATMENT.value].copy()

        # Merge treatments with control on GroupID.
        df_merged = df_treatment.merge(
            df_control[[g_id_key, surprisal_key]],
            on=g_id_key,
            suffixes=("_treat", "_control"),
            how="inner",  # Keeps only subjects that have both control and treatment.
        )

        # Calculate paired differences.
        df_merged["diff"] = df_merged.apply(
            lambda row: (
                row[f"{surprisal_key}_treat"] - row[f"{surprisal_key}_control"]
                if row[cond_key] == ConditionType.TREATMENT.value
                else None
            ),
            axis=1,
        )

        # Sum across columns to merge. Works because NaN + value = value.
        df_merged[surprisal_key] = df_merged["diff"].sum()

        # Drop intermediary calculation columns.
        return df_merged.drop(
            columns=[f"{surprisal_key}_treat", f"{surprisal_key}_control", "diff"]
        )


@command(name="accord.analyze")
class AccordAnalyze:
    def __init__(self, path: PathConfig, accord: Config):
        self.path = path
        self.cfg = accord
        self.print = ConditionalPrinter(self.cfg.verbose)

    def _skip(self, test_nickname: Nickname) -> bool:
        for nickname in self.cfg.llms:
            if flatten(test_nickname) == flatten(nickname):
                return False
        return True

    def _save_analysis(
        self,
        comparator: Comparator,
        nickname: Nickname,
        analysis_df: pd.DataFrame,
        analysis_summary: str,
    ) -> None:
        out_dir = self.cfg.analysis_dir(self.path.accord_exp_dir)
        out_dir = os.path.join(out_dir, comparator.value)
        df_file = ensure_path(os.path.join(out_dir, nickname + ".csv"))
        summary_file = ensure_path(os.path.join(out_dir, nickname + ".txt"))
        analysis_df.to_csv(df_file, index=False)
        with open(summary_file, "w", encoding="utf-8") as f:
            f.write(analysis_summary)

    def run(self):
        for comparator in Comparator:
            self.print("Analysis for comparator:", comparator)
            self._do_run(comparator)
        self.print("Done.")

    def _do_run(self, comparator: Comparator):
        bar_plots = BarPlots(self.cfg)
        for walk in walk_files(self.cfg.postprocess_dir(self.path.accord_exp_dir)):
            inference_path, nickname = walk.path, flatten(walk.no_ext())
            if self._skip(nickname):
                continue
            self.print("    Analyzing results of model:", nickname)
            self.print("        Loading data...")
            df = comparator.transform(pd.read_csv(inference_path))
            bar_plots.add(df)
            self.print("        Running statistical analysis...")
            analysis_df, analysis_summary = Analyze(self.cfg, df).run()
            self._save_analysis(comparator, nickname, analysis_df, analysis_summary)
            if self.cfg.create_violin_plots:
                self.print("        Building violin plots...")
                saver = PlotSaver(self.path, self.cfg, comparator)
                ViolinPlots(self.cfg, df).run(saver)
                self.print("        Done.")
        self.print("    Building collective bar plots...")
        bar_plots.run(PlotSaver(self.path, self.cfg, comparator))
        self.print("    Done.")
