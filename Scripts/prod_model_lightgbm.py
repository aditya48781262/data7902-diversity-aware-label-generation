from sklearn.model_selection import StratifiedKFold
import pandas as pd
import lightgbm as lgb
import scipy
import os
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer
from sklearn.metrics import classification_report, accuracy_score
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric
from fairlearn.metrics import demographic_parity_ratio, demographic_parity_difference, MetricFrame, selection_rate, true_positive_rate, false_positive_rate, equalized_odds_difference, equal_opportunity_ratio
import matplotlib.pyplot as plt
import numpy as np

llm_label_mapping = {
    'TRUE': 1,
    'mostly true': 1,
    'half true': 2,
    'mostly false': 0,
    'FALSE': 0,
    'implausible': 0
}

politifact_label_mapping = {
    'TRUE': 1,
    'mostly-true': 1,
    'half-true': 2,
    'mostly-false': 0,
    'FALSE': 0,
    'pants-fire': 0
}


def workflow(llm, prompt_type, df, llm_verdict, include_source=False, n_splits=5):
    df = df.copy()
    df['party'] = LabelEncoder().fit_transform(df['party'])
    df['topic'] = LabelEncoder().fit_transform(df['topic'])
    df['average_views'] = LabelEncoder().fit_transform(df['average_views'])
    df['politician_name'] = LabelEncoder().fit_transform(df['politician_name'])
    df['statement_date'] = pd.to_datetime(df['statement_date']).dt.year

    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(df['statement'].tolist(), show_progress_bar=True)
    if include_source:
        feature_cols = ['party', 'topic', 'average_views',
                        'statement_date', 'politician_name']
    else:
        feature_cols = ['party', 'topic', 'average_views', 'statement_date']
    features = df[feature_cols].values

    X = scipy.sparse.hstack((scipy.sparse.csr_matrix(
        embeddings), scipy.sparse.csr_matrix(features)))
    y = df[llm_verdict].values

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    all_reports = []
    all_fairness = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, df['party'])):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        df_test = df.iloc[test_idx]

        lgb_model = lgb.LGBMClassifier(
            objective='multiclass', num_class=3, metric='multi_logloss',
            class_weight='balanced', verbose=-1)
        lgb_model.fit(X_train, y_train)
        y_pred = lgb_model.predict(X_test)

        report = classification_report(y_test, y_pred, output_dict=True)
        print(
            f"\nFold {fold+1} Results for {llm} {prompt_type} with source={include_source}")
        print(classification_report(y_test, y_pred))
        print("Accuracy:", accuracy_score(y_test, y_pred))

        # Save report for aggregation
        df_report = pd.DataFrame(report).transpose()
        df_report['llm'] = llm
        df_report['prompt_type'] = prompt_type
        df_report['include_source'] = include_source
        df_report['fold'] = fold + 1
        all_reports.append(df_report)

        sensitive_attribute = df_test['party'].values
        privileged_groups = [{'party': 0}]
        unprivileged_groups = [{'party': 2}]

        fairness_metrics = {
            'class_id': [],
            'statistical_parity_difference': [],
            'disparate_impact': [],
            'equalised_odds_difference': [],
            'fold': []
        }

        for class_id in [0, 1, 2]:
            y_true_binary = (y_test == class_id).astype(int)
            y_pred_binary = (y_pred == class_id).astype(int)

            df_true = pd.DataFrame(
                {'label': y_true_binary, 'party': sensitive_attribute})
            df_pred = pd.DataFrame(
                {'label': y_pred_binary, 'party': sensitive_attribute})

            dataset_true = BinaryLabelDataset(
                favorable_label=1,
                unfavorable_label=0,
                df=df_true,
                label_names=['label'],
                protected_attribute_names=['party'],
            )
            dataset_pred = BinaryLabelDataset(
                favorable_label=1,
                unfavorable_label=0,
                df=df_pred,
                label_names=['label'],
                protected_attribute_names=['party'],
            )

            metric = ClassificationMetric(
                dataset_true, dataset_pred,
                unprivileged_groups=unprivileged_groups,
                privileged_groups=privileged_groups
            )

            fairness_metrics['class_id'].append(class_id)
            fairness_metrics['statistical_parity_difference'].append(
                metric.statistical_parity_difference())
            fairness_metrics['disparate_impact'].append(
                metric.disparate_impact())
            fairness_metrics['equalised_odds_difference'].append(
                metric.equalized_odds_difference())
            fairness_metrics['fold'].append(fold + 1)

            print(f"\nFairness metrics for class {class_id} (Fold {fold+1}):")
            print("Statistical parity difference:",
                  metric.statistical_parity_difference())
            print("Disparate impact:", metric.disparate_impact())
            print("Equalised Odds Difference:",
                  metric.equalized_odds_difference())

        df_fairness = pd.DataFrame(fairness_metrics)
        df_fairness['llm'] = llm
        df_fairness['prompt_type'] = prompt_type
        df_fairness['include_source'] = include_source
        all_fairness.append(df_fairness)

    # Aggregate and save all folds
    all_reports_df = pd.concat(all_reports, ignore_index=True)
    all_fairness_df = pd.concat(all_fairness, ignore_index=True)

    if os.path.exists("classification_fairness_metrics.csv"):
        all_reports_df.to_csv(
            "classification_fairness_metrics.csv", mode='a', header=True, index=False)
        all_fairness_df.to_csv(
            "classification_fairness_metrics.csv", mode='a', header=True, index=False)
    else:
        all_reports_df.to_csv(
            "classification_fairness_metrics.csv", mode='w', header=True, index=False)
        all_fairness_df.to_csv(
            "classification_fairness_metrics.csv", mode='a', header=True, index=False)

    all_reports_df = pd.concat(all_reports, ignore_index=True)
    all_fairness_df = pd.concat(all_fairness, ignore_index=True)

    reports_filename = "classification_reports.csv"
    fairness_filename = "classification_fariness_metrics.csv"

    if os.path.exists(reports_filename):
        all_reports_df.to_csv(reports_filename, mode='a',
                              header=False, index=False)
    else:
        all_reports_df.to_csv(reports_filename, mode='w',
                              header=True, index=False)

    if os.path.exists(fairness_filename):
        all_fairness_df.to_csv(
            fairness_filename, mode='a', header=False, index=False)
    else:
        all_fairness_df.to_csv(
            fairness_filename, mode='w', header=True, index=False)

    return all_reports_df, all_fairness_df


def plot_fairness_metrics(all_fairness_df, save_prefix="fairness_metrics"):
    metrics = {
        'statistical_parity_difference': 'Statistical Parity Difference',
        'equalised_odds_difference': 'Equalised Odds Difference',
        'disparate_impact': 'Disparate Impact'
    }

    all_fairness_df['model_prompt'] = all_fairness_df['llm'] + ' ' + \
        all_fairness_df['prompt_type'] + \
        all_fairness_df['include_source'].map({True: ' +src', False: ''})

    model_prompts = sorted(all_fairness_df['model_prompt'].unique())
    classes = [1, 2, 0]
    class_names = {1: 'True', 2: 'Mixed', 0: 'False'}

    colours = {
        1: '#90caf9',
        2: '#bdbdbd',
        0: '#ef9a9a'
    }

    for metric_key, metric_title in metrics.items():
        pivot = all_fairness_df.groupby(['model_prompt', 'class_id'])[
            metric_key].mean().unstack(fill_value=0)
        pivot = pivot.reindex(index=model_prompts, fill_value=0)

        for c in classes:
            if c not in pivot.columns:
                pivot[c] = 0
        pivot = pivot[classes]
        pivot = pivot.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        y = np.arange(len(pivot))
        fig, ax = plt.subplots(figsize=(12, max(6, len(model_prompts) * 0.35)))

        max_pos = 0.0
        max_neg = 0.0

        for i, row in enumerate(pivot.values):
            pos_total = 0.0
            neg_total = 0.0
            for j, cls in enumerate(classes):
                val = float(row[j])
                if np.isnan(val):
                    val = 0.0
                if val >= 0:
                    left = pos_total
                    width = val
                    pos_total += width
                    ax.barh(i, width, left=left,
                            color=colours[cls], edgecolor='white', height=0.6, label=class_names[cls] if i == 0 else "")
                    if width != 0 and abs(width) >= 0.02:
                        ax.text(
                            left + width/2, i, f"{width:.2f}", ha='center', va='center', color='black', fontsize=8)
                else:
                    width = abs(val)
                    left = -(neg_total + width)
                    neg_total += width
                    ax.barh(i, width, left=left,
                            color=colours[cls], edgecolor='white', height=0.6, label=class_names[cls] if i == 0 else "")
                    if width != 0 and abs(width) >= 0.02:
                        ax.text(
                            left + width/2, i, f"-{width:.2f}", ha='center', va='center', color='black', fontsize=8)
            if np.isfinite(pos_total):
                max_pos = max(max_pos, pos_total)
            if np.isfinite(neg_total):
                max_neg = max(max_neg, neg_total)

        ax.set_yticks(y)
        ax.set_yticklabels(pivot.index, fontsize=9)
        ax.invert_yaxis()
        ax.axvline(0, color='black', linewidth=0.8)

        max_extent = max(max_pos, max_neg, 0.0)
        if max_extent == 0.0 or not np.isfinite(max_extent):
            left_lim, right_lim = -0.5, 0.5
        else:
            pad = max_extent * 0.15
            left_lim = -max_neg - pad
            right_lim = max_pos + pad
            if not np.isfinite(left_lim) or not np.isfinite(right_lim) or left_lim == right_lim:
                left_lim, right_lim = -max_extent * 1.2, max_extent * 1.2

        ax.set_xlim(left_lim, right_lim)

        ax.set_xlabel(metric_title)
        ax.set_title(
            f"{metric_title} - stacked by class", fontsize=12)
        ax.legend(title='Class',
                  bbox_to_anchor=(1.02, 1), loc='upper left')

        plt.tight_layout()
        fname = f"{save_prefix}_{metric_key}.png"
        plt.savefig(fname, dpi=300, bbox_inches='tight')
        plt.close(fig)


def save_report(report, filename, llm, prompt_type, include_source):
    df_report = pd.DataFrame(report).transpose()
    df_report['llm'] = llm
    df_report['prompt_type'] = prompt_type
    df_report['include_source'] = include_source
    if os.path.exists(filename):
        df_report.to_csv(filename, mode='a', header=True)
    else:
        df_report.to_csv(filename, mode='w', header=True)


def save_fairness_metrics(metrics_dict, filename, llm, prompt_type, include_source):
    df_metrics = pd.DataFrame(metrics_dict)
    df_metrics['llm'] = llm
    df_metrics['prompt_type'] = prompt_type
    df_metrics['include_source'] = include_source
    if os.path.exists(filename):
        df_metrics.to_csv(filename, mode='a', header=True)
    else:
        df_metrics.to_csv(filename, mode='w', header=True)


'''Mistral 7b instruct'''
mistral_df_zs = pd.read_csv(
    "politifact_kaggle_dataset_sample_mistral_labelled.csv")
mistral_df_zs_src = pd.read_csv(
    "politifact_kaggle_dataset_sample_mistral_labelled_with_source.csv")
mistral_df_cot = pd.read_csv(
    "politifact_kaggle_dataset_sample_mistral_labelled_cot_no_source.csv")
mistral_df_cot_src = pd.read_csv(
    "politifact_kaggle_dataset_sample_mistral_labelled_cot_with_source.csv"
)
mistral_df_fs = pd.read_csv(
    "politifact_kaggle_dataset_sample_mistral_labelled_fs.csv"
)
mistral_df_fs_src = pd.read_csv(
    "politifact_kaggle_dataset_sample_mistral_labelled_fs_src.csv"
)
##
mistral_df_zs['mistral_verdict'] = mistral_df_zs['mistral_verdict'].apply(
    lambda x: llm_label_mapping.get(x, 3)
)
mistral_df_zs['verdict'] = mistral_df_zs['verdict'].apply(
    lambda x: politifact_label_mapping.get(x, 3)
)
##
##
mistral_df_zs_src['mistral_verdict'] = mistral_df_zs_src['mistral_verdict'].apply(
    lambda x: llm_label_mapping.get(x, 3)
)
mistral_df_zs_src['verdict'] = mistral_df_zs_src['verdict'].apply(
    lambda x: politifact_label_mapping.get(x, 3)
)
##
##
mistral_df_cot['mistral_verdict'] = mistral_df_cot['mistral_verdict'].apply(
    lambda x: llm_label_mapping.get(x, 3)
)
mistral_df_cot['verdict'] = mistral_df_cot['verdict'].apply(
    lambda x: politifact_label_mapping.get(x, 3)
)
##
mistral_df_cot_src['mistral_verdict'] = mistral_df_cot_src['mistral_verdict'].apply(
    lambda x: llm_label_mapping.get(x, 3)
)
mistral_df_cot_src['verdict'] = mistral_df_cot_src['verdict'].apply(
    lambda x: politifact_label_mapping.get(x, 3)
)
mistral_df_fs['mistral_verdict'] = mistral_df_fs['mistral_verdict'].apply(
    lambda x: llm_label_mapping.get(x, 3)
)
mistral_df_fs['verdict'] = mistral_df_fs['verdict'].apply(
    lambda x: politifact_label_mapping.get(x, 3)
)
mistral_df_fs_src['mistral_verdict'] = mistral_df_fs_src['mistral_verdict'].apply(
    lambda x: llm_label_mapping.get(x, 3)
)
mistral_df_fs_src['verdict'] = mistral_df_fs_src['verdict'].apply(
    lambda x: politifact_label_mapping.get(x, 3)
)

# ---------------------
mistral_df_zs.drop(
    mistral_df_zs[mistral_df_zs['verdict']
                  == 3].index, inplace=True)
mistral_df_zs.drop(
    mistral_df_zs[mistral_df_zs['mistral_verdict']
                  == 3].index, inplace=True)

mistral_df_zs_src.drop(
    mistral_df_zs_src[mistral_df_zs_src['verdict']
                      == 3].index, inplace=True)
mistral_df_zs_src.drop(
    mistral_df_zs_src[mistral_df_zs_src['mistral_verdict']
                      == 3].index, inplace=True)

mistral_df_cot.drop(
    mistral_df_cot[mistral_df_cot['verdict']
                   == 3].index, inplace=True)
mistral_df_cot.drop(
    mistral_df_cot[mistral_df_cot['mistral_verdict']
                   == 3].index, inplace=True)

mistral_df_cot_src.drop(
    mistral_df_cot_src[mistral_df_cot_src['verdict']
                       == 3].index, inplace=True)
mistral_df_cot_src.drop(
    mistral_df_cot_src[mistral_df_cot_src['mistral_verdict']
                       == 3].index, inplace=True)

mistral_df_fs.drop(
    mistral_df_fs[mistral_df_fs['verdict']
                  == 3].index, inplace=True)
mistral_df_fs.drop(
    mistral_df_fs[mistral_df_fs['mistral_verdict']
                  == 3].index, inplace=True)
mistral_df_fs_src.drop(
    mistral_df_fs_src[mistral_df_fs_src['verdict']
                      == 3].index, inplace=True)
mistral_df_fs_src.drop(
    mistral_df_fs_src[mistral_df_fs_src['mistral_verdict']
                      == 3].index, inplace=True)


# ---------------------

'''Llama 3.3 70b'''
llama_df_zs = pd.read_csv(
    "politifact_kaggle_dataset_sample_llama_3.3_70b_labelled.csv"
)
llama_df_zs_src = pd.read_csv(
    "politifact_kaggle_dataset_sample_llama_3.3_70b_labelled_with_source.csv"
)
llama_df_cot = pd.read_csv(
    "politifact_kaggle_dataset_sample_llama_3.3_70b_labelled_cot_no_source.csv"
)
llama_df_cot_src = pd.read_csv(
    "politifact_kaggle_dataset_sample_llama_3.3_70b_labelled_cot_with_source.csv"
)
llama_df_fs = pd.read_csv(
    "politifact_kaggle_dataset_sample_llama_3.3_70b_labelled_fs.csv"
)
llama_df_fs_src = pd.read_csv(
    "politifact_kaggle_dataset_sample_llama_3.3_70b_labelled_fs_src.csv"
)

##
llama_df_zs['llama_verdict'] = llama_df_zs['llama_verdict'].apply(
    lambda x: llm_label_mapping.get(x, 3)
)
llama_df_zs['verdict'] = llama_df_zs['verdict'].apply(
    lambda x: politifact_label_mapping.get(x, 3)
)
##
##
llama_df_zs_src['llama_verdict'] = llama_df_zs_src['llama_verdict'].apply(
    lambda x: llm_label_mapping.get(x, 3)
)
llama_df_zs_src['verdict'] = llama_df_zs_src['verdict'].apply(
    lambda x: politifact_label_mapping.get(x, 3)
)
##
##
llama_df_cot['llama_verdict'] = llama_df_cot['llama_verdict'].apply(
    lambda x: llm_label_mapping.get(x, 3)
)
llama_df_cot['verdict'] = llama_df_cot['verdict'].apply(
    lambda x: politifact_label_mapping.get(x, 3)
)
##
llama_df_cot_src['llama_verdict'] = llama_df_cot_src['llama_verdict'].apply(
    lambda x: llm_label_mapping.get(x, 3)
)
llama_df_cot_src['verdict'] = llama_df_cot_src['verdict'].apply(
    lambda x: politifact_label_mapping.get(x, 3)
)

llama_df_fs['llama_verdict'] = llama_df_fs['llama_verdict'].apply(
    lambda x: llm_label_mapping.get(x, 3)
)
llama_df_fs['verdict'] = llama_df_fs['verdict'].apply(
    lambda x: politifact_label_mapping.get(x, 3)
)

llama_df_fs_src['llama_verdict'] = llama_df_fs_src['llama_verdict'].apply(
    lambda x: llm_label_mapping.get(x, 3)
)
llama_df_fs_src['verdict'] = llama_df_fs_src['verdict'].apply(
    lambda x: politifact_label_mapping.get(x, 3)
)

# ---------------------
llama_df_zs.drop(
    llama_df_zs[llama_df_zs['verdict']
                == 3].index, inplace=True)
llama_df_zs.drop(
    llama_df_zs[llama_df_zs['llama_verdict']
                == 3].index, inplace=True)

llama_df_zs_src.drop(
    llama_df_zs_src[llama_df_zs_src['verdict']
                    == 3].index, inplace=True)
llama_df_zs_src.drop(
    llama_df_zs_src[llama_df_zs_src['llama_verdict']
                    == 3].index, inplace=True)

llama_df_cot.drop(
    llama_df_cot[llama_df_cot['verdict']
                 == 3].index, inplace=True)
llama_df_cot.drop(
    llama_df_cot[llama_df_cot['llama_verdict']
                 == 3].index, inplace=True)

llama_df_cot_src.drop(
    llama_df_cot_src[llama_df_cot_src['verdict']
                     == 3].index, inplace=True)
llama_df_cot_src.drop(
    llama_df_cot_src[llama_df_cot_src['llama_verdict']
                     == 3].index, inplace=True)

llama_df_fs.drop(
    llama_df_fs[llama_df_fs['verdict'] == 3].index, inplace=True)
llama_df_fs.drop(
    llama_df_fs[llama_df_fs['llama_verdict'] == 3].index, inplace=True)

llama_df_fs_src.drop(
    llama_df_fs_src[llama_df_fs_src['verdict'] == 3].index, inplace=True)
llama_df_fs_src.drop(
    llama_df_fs_src[llama_df_fs_src['llama_verdict'] == 3].index, inplace=True)
# ---------------------

if __name__ == '__main__':
    reports_list = []
    fairness_list = []
    mistral_fairness_list = []
    llama_fairness_list = []

    for df, prompt_type, include_source in [
        (mistral_df_zs, "zero-shot", False),
        (mistral_df_zs_src, "zero-shot", True),
        (mistral_df_fs, "few-shot", False),
        (mistral_df_fs_src, "few-shot", True),
        (mistral_df_cot, "COT", False),
        (mistral_df_cot_src, "COT", True)
    ]:
        r, f = workflow(llm="Mistral", prompt_type=prompt_type, df=df,
                        llm_verdict='mistral_verdict', include_source=include_source)
        reports_list.append(r)
        fairness_list.append(f)
        mistral_fairness_list.append(f)

    mistral_fairness = pd.concat(mistral_fairness_list, ignore_index=True)
    plot_fairness_metrics(
        mistral_fairness, save_prefix="mistral_fairness")

    for df, prompt_type, include_source in [
        (llama_df_zs, "zero-shot", False),
        (llama_df_zs_src, "zero-shot", True),
        (llama_df_fs, "few-shot", False),
        (llama_df_fs_src, "few-shot", True),
        (llama_df_cot, "COT", False),
        (llama_df_cot_src, "COT", True)
    ]:
        r, f = workflow(llm="Llama", prompt_type=prompt_type, df=df,
                        llm_verdict='llama_verdict', include_source=include_source)
        reports_list.append(r)
        fairness_list.append(f)
        llama_fairness_list.append(f)

    llama_fairness = pd.concat(llama_fairness_list, ignore_index=True)
    plot_fairness_metrics(
        llama_fairness, save_prefix="llama_fairness")

    combined_reports = pd.concat(reports_list, ignore_index=True)
    combined_fairness = pd.concat(fairness_list, ignore_index=True)

    combined_reports.to_csv("classification_reports_all_runs.csv", index=False)
    combined_fairness.to_csv(
        "classification_fairness_all_runs.csv", index=False)

    plot_fairness_metrics(
        combined_fairness, save_prefix="fairness_metrics_all_runs")
