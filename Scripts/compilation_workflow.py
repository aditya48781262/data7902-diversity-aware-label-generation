import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import krippendorff as kp
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, ConfusionMatrixDisplay, mutual_info_score
from fairlearn.metrics import demographic_parity_ratio, demographic_parity_difference, MetricFrame, selection_rate, true_positive_rate, false_positive_rate, equalized_odds_difference, equal_opportunity_ratio
from scipy.stats import entropy, chi2
from statsmodels.stats.contingency_tables import mcnemar
import regex as re
import math

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


'''Pre processing'''
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

# print(f"Politifact: {mistral_df_zs['verdict'].value_counts()}"
#       f"Mistral ZS: {mistral_df_zs['mistral_verdict'].value_counts()}"
#       f"Mistral ZS Source: {mistral_df_zs_src['mistral_verdict'].value_counts()}"
#       f"Mistral FS: {mistral_df_fs['mistral_verdict'].value_counts()}"
#       f"Mistral FS Source: {mistral_df_fs_src['mistral_verdict'].value_counts()}"
#       f"Mistral COT: {mistral_df_cot['mistral_verdict'].value_counts()}"
#       f"Mistral COT Source: {mistral_df_cot_src['mistral_verdict'].value_counts()}")
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

# print(f"Politifact: {llama_df_zs['verdict'].value_counts()}"
#       f"Llama ZS: {llama_df_zs['llama_verdict'].value_counts()}"
#       f"Llama ZS Source: {llama_df_zs_src['llama_verdict'].value_counts()}"
#       f"Llama FS: {llama_df_fs['llama_verdict'].value_counts()}"
#       f"Llama FS Source: {llama_df_fs_src['llama_verdict'].value_counts()}"
#       f"Llama COT: {llama_df_zs_cot['llama_verdict'].value_counts()}"
#       f"Llama COT Source: {llama_df_zs_cot_src['llama_verdict'].value_counts()}")

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


def plot_dropped_labels():
    def count_dropped_labels(df, verdict_col):
        return len(df[df[verdict_col] == 3])

    dropped_counts = {
        'Mistral ZS': count_dropped_labels(mistral_df_zs, 'mistral_verdict'),
        'Mistral ZS+src': count_dropped_labels(mistral_df_zs_src, 'mistral_verdict'),
        'Mistral FS': count_dropped_labels(mistral_df_fs, 'mistral_verdict'),
        'Mistral FS+src': count_dropped_labels(mistral_df_fs_src, 'mistral_verdict'),
        'Mistral COT': count_dropped_labels(mistral_df_cot, 'mistral_verdict'),
        'Mistral COT+src': count_dropped_labels(mistral_df_cot_src, 'mistral_verdict'),
        'Llama ZS': count_dropped_labels(llama_df_zs, 'llama_verdict'),
        'Llama ZS+src': count_dropped_labels(llama_df_zs_src, 'llama_verdict'),
        'Llama FS': count_dropped_labels(llama_df_fs, 'llama_verdict'),
        'Llama FS+src': count_dropped_labels(llama_df_fs_src, 'llama_verdict'),
        'Llama COT': count_dropped_labels(llama_df_cot, 'llama_verdict'),
        'Llama COT+src': count_dropped_labels(llama_df_cot_src, 'llama_verdict')
    }

    fig, ax = plt.subplots(figsize=(12, 6))

    models = list(dropped_counts.keys())
    counts = list(dropped_counts.values())

    y_pos = np.arange(len(models))

    bars = ax.barh(y_pos, counts, color='#ef9a9a')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(models)
    ax.set_xlabel('Number of errors / unknown labels')
    ax.set_title('Number of Errors / Unknown Labels per Dataset')

    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + 0.5, bar.get_y() + bar.get_height() / 2,
                f'{int(width)}', ha='left', va='center')

    plt.tight_layout()
    plt.savefig('dropped_labels.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    plt.clf()


plot_dropped_labels()

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


# print(mistral_df_zs_no_src_cot.head())
# print(mistral_df_source.head())
# print(mistral_df_cot.head())

politifact_df_label_count = mistral_df_zs['verdict'].value_counts(
)

mistral_df_zs_llm_label_count = mistral_df_zs['mistral_verdict'].value_counts(
)
mistral_df_zs_politifact_label_count = mistral_df_zs['verdict'].value_counts(
)

mistral_df_zs_src_llm_label_count = mistral_df_zs_src['mistral_verdict'].value_counts(
)

mistral_df_cot_llm_label_count = mistral_df_cot['mistral_verdict'].value_counts(
)

mistral_df_cot_src_llm_label_count = mistral_df_cot_src['mistral_verdict'].value_counts(
)

mistral_df_fs_llm_label_count = mistral_df_fs['mistral_verdict'].value_counts(
)

mistral_df_fs_src_llm_label_count = mistral_df_fs_src['mistral_verdict'].value_counts(
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


llama_df_zs_llm_label_count = llama_df_zs['llama_verdict'].value_counts(
)
llama_df_zs_politifact_label_count = llama_df_zs['verdict'].value_counts(
)

llama_df_zs_src_llm_label_count = llama_df_zs_src['llama_verdict'].value_counts(
)
llama_df_zs_src_politifact_label_count = llama_df_zs_src['verdict'].value_counts(
)

llama_df_cot_llm_label_count = llama_df_cot['llama_verdict'].value_counts(
)
llama_df_cot_politifact_label_count = llama_df_cot['verdict'].value_counts(
)

llama_df_cot_src_llm_label_count = llama_df_cot_src['llama_verdict'].value_counts(
)
llama_df_cot_src_politifact_label_count = llama_df_cot_src['verdict'].value_counts(
)

llama_df_fs_llm_label_count = llama_df_fs['llama_verdict'].value_counts(
)
llama_df_fs_politifact_label_count = llama_df_fs['verdict'].value_counts(
)

llama_df_fs_src_llm_label_count = llama_df_fs_src['llama_verdict'].value_counts(
)
llama_df_fs_src_politifact_label_count = llama_df_fs_src['verdict'].value_counts(
)

# ---------------------


mistral_zs_ground_truth_labels = mistral_df_zs['verdict'].values
mistral_zs_labels = mistral_df_zs['mistral_verdict'].values
mistral_zs_cm = confusion_matrix(
    mistral_zs_ground_truth_labels, mistral_zs_labels)

mistral_zs_accuracy = accuracy_score(
    mistral_zs_ground_truth_labels, mistral_zs_labels)
mistral_zs_recall = recall_score(
    mistral_zs_ground_truth_labels, mistral_zs_labels, average='macro')
mistral_zs_precision = precision_score(
    mistral_zs_ground_truth_labels, mistral_zs_labels, average='macro')
mistral_zs_f1 = f1_score(mistral_zs_ground_truth_labels,
                         mistral_zs_labels, average='macro')
print(f"Accuracy: {mistral_zs_accuracy:.4f}\n"
      f"Precision: {mistral_zs_precision:.4f}\n"
      f"Recall: {mistral_zs_recall:.4f}\n"
      f"F1 Score: {mistral_zs_f1:.4f}")

print('\n\n')


mistral_zs_src_ground_truth_labels = mistral_df_zs_src['verdict'].values
mistral_zs_src_labels = mistral_df_zs_src['mistral_verdict'].values
mistral_src_cm = confusion_matrix(
    mistral_zs_src_ground_truth_labels, mistral_zs_src_labels)

mistral_src_accuracy = accuracy_score(
    mistral_zs_src_ground_truth_labels, mistral_zs_src_labels)
mistral_src_recall = recall_score(
    mistral_zs_src_ground_truth_labels, mistral_zs_src_labels, average='macro')
mistral_src_precision = precision_score(
    mistral_zs_src_ground_truth_labels, mistral_zs_src_labels, average='macro')
mistral_src_f1 = f1_score(
    mistral_zs_src_ground_truth_labels, mistral_zs_src_labels, average='macro')
print(f"Accuracy: {mistral_src_accuracy:.4f}\n"
      f"Precision: {mistral_src_precision:.4f}\n"
      f"Recall: {mistral_src_recall:.4f}\n"
      f"F1 Score: {mistral_src_f1:.4f}")

print('\n\n')

mistral_zs_cot_ground_truth_labels = mistral_df_cot['verdict'].values
mistral_cot_labels = mistral_df_cot['mistral_verdict'].values
mistral_cot_cm = confusion_matrix(
    mistral_zs_cot_ground_truth_labels, mistral_cot_labels)


mistral_cot_accuracy = accuracy_score(
    mistral_zs_cot_ground_truth_labels, mistral_cot_labels)
mistral_cot_recall = recall_score(
    mistral_zs_cot_ground_truth_labels, mistral_cot_labels, average='macro')
mistral_cot_precision = precision_score(
    mistral_zs_cot_ground_truth_labels, mistral_cot_labels, average='macro')
mistral_cot_f1 = f1_score(
    mistral_zs_cot_ground_truth_labels, mistral_cot_labels, average='macro')
print(f"Accuracy: {mistral_cot_accuracy:.4f}\n"
      f"Precision: {mistral_cot_precision:.4f}\n"
      f"Recall: {mistral_cot_recall:.4f}\n"
      f"F1 Score: {mistral_cot_f1:.4f}")

mistral_zs_cot_src_ground_truth_labels = mistral_df_cot_src['verdict'].values
mistral_cot_src_labels = mistral_df_cot_src['mistral_verdict'].values
mistral_cot_src_cm = confusion_matrix(
    mistral_zs_cot_src_ground_truth_labels, mistral_cot_src_labels)


mistral_cot_src_accuracy = accuracy_score(
    mistral_zs_cot_src_ground_truth_labels, mistral_cot_src_labels)
mistral_cot_src_recall = recall_score(
    mistral_zs_cot_src_ground_truth_labels, mistral_cot_src_labels, average='macro')
mistral_cot_src_precision = precision_score(
    mistral_zs_cot_src_ground_truth_labels, mistral_cot_src_labels, average='macro')
mistral_cot_src_f1 = f1_score(
    mistral_zs_cot_src_ground_truth_labels, mistral_cot_src_labels, average='macro')
print(f"Accuracy: {mistral_cot_src_accuracy:.4f}\n"
      f"Precision: {mistral_cot_src_precision:.4f}\n"
      f"Recall: {mistral_cot_src_recall:.4f}\n"
      f"F1 Score: {mistral_cot_src_f1:.4f}")

mistral_fs_ground_truth_labels = mistral_df_fs['verdict'].values
mistral_fs_labels = mistral_df_fs['mistral_verdict'].values
mistral_fs_cm = confusion_matrix(
    mistral_fs_ground_truth_labels, mistral_fs_labels).ravel()

mistral_fs_accuracy = accuracy_score(
    mistral_fs_ground_truth_labels, mistral_fs_labels)
mistral_fs_recall = recall_score(
    mistral_fs_ground_truth_labels, mistral_fs_labels, average='macro')
mistral_fs_precision = precision_score(
    mistral_fs_ground_truth_labels, mistral_fs_labels, average='macro')
mistral_fs_f1 = f1_score(mistral_fs_ground_truth_labels,
                         mistral_fs_labels, average='macro')
print(f"Accuracy: {mistral_fs_accuracy:.4f}\n"
      f"Precision: {mistral_fs_precision:.4f}\n"
      f"Recall: {mistral_fs_recall:.4f}\n"
      f"F1 Score: {mistral_fs_f1:.4f}")

print('\n\n')

mistral_fs_src_ground_truth_labels = mistral_df_fs_src['verdict'].values
mistral_fs_src_labels = mistral_df_fs_src['mistral_verdict'].values
mistral_fs_src_cm = confusion_matrix(
    mistral_fs_src_ground_truth_labels, mistral_fs_src_labels)

mistral_fs_src_accuracy = accuracy_score(
    mistral_fs_src_ground_truth_labels, mistral_fs_src_labels)
mistral_fs_src_recall = recall_score(
    mistral_fs_src_ground_truth_labels, mistral_fs_src_labels, average='macro')
mistral_fs_src_precision = precision_score(
    mistral_fs_src_ground_truth_labels, mistral_fs_src_labels, average='macro')
mistral_fs_src_f1 = f1_score(
    mistral_fs_src_ground_truth_labels, mistral_fs_src_labels, average='macro')
print(f"Accuracy: {mistral_fs_src_accuracy:.4f}\n"
      f"Precision: {mistral_fs_src_precision:.4f}\n"
      f"Recall: {mistral_fs_src_recall:.4f}\n"
      f"F1 Score: {mistral_fs_src_f1:.4f}")


# mistral_zs_cm = ConfusionMatrixDisplay(confusion_matrix(
#     mistral_zs_ground_truth_labels, mistral_zs_labels), display_labels=["False", "Mixed", "True"])
# mistral_zs_cm.plot(cmap='Blues')
# plt.title("Mistral 7b Instruct - Zero Shot without Source or COT")
# plt.savefig('mistral_zs_cm.png')
# plt.clf()

# mistral_zs_src_cm = ConfusionMatrixDisplay(confusion_matrix(
#     mistral_zs_src_ground_truth_labels, mistral_zs_src_labels), display_labels=["False", "Mixed", "True"])
# mistral_zs_src_cm.plot(cmap='Blues')
# plt.title("Mistral 7b Instruct - Zero Shot with Source")
# plt.savefig('mistral_zs_src_cm.png')
# plt.clf()

# mistral_cot_cm = ConfusionMatrixDisplay(confusion_matrix(
#     mistral_zs_cot_ground_truth_labels, mistral_cot_labels), display_labels=["False", "Mixed", "True"])
# mistral_cot_cm.plot(cmap='Blues')
# plt.title("Mistral 7b Instruct - Annotations derived from Chain-of-Thought Prompting")
# plt.savefig('mistral_cot_cm.png')
# plt.clf()

# mistral_cot_src_cm = ConfusionMatrixDisplay(confusion_matrix(
#     mistral_zs_cot_src_ground_truth_labels, mistral_cot_src_labels), display_labels=["False", "Mixed", "True"])
# mistral_cot_src_cm.plot(cmap='Blues')
# plt.title("Mistral 7b Instruct - Annotations derived from Chain-of-Thought Prompting with Source")
# plt.savefig('mistral_cot_src_cm.png')
# plt.clf()

# mistral_fs_cm = ConfusionMatrixDisplay(confusion_matrix(
#     mistral_fs_ground_truth_labels, mistral_fs_labels), display_labels=["False", "Mixed", "True"])
# mistral_fs_cm.plot(cmap='Blues')
# plt.title(
#     "Mistral 7b Instruct - Annotations derived from Few-Shot Prompting without Source")
# plt.savefig('mistral_fs_cm.png')
# plt.clf()

# mistral_fs_src_cm = ConfusionMatrixDisplay(confusion_matrix(
#     mistral_fs_src_ground_truth_labels, mistral_fs_src_labels), display_labels=["False", "Mixed", "True"])
# mistral_fs_src_cm.plot(cmap='Blues')
# plt.title(
#     "Mistral 7b Instruct - Annotations derived from Few-Shot Prompting with Source")
# plt.savefig('mistral_fs_src_cm.png')
# plt.clf()

print(f"Llama:\n"
      f"Zero Shot Prompt, No Source or COT:\n"
      f"Llama: {llama_df_zs_llm_label_count}\n"
      f"Politifact: {llama_df_zs_politifact_label_count}"
      f"\n\n"
      f"Zero Shot Prompt with source:\n"
      f"Llama: {llama_df_zs_src_llm_label_count}\n"
      f"Politifact: {llama_df_zs_src_politifact_label_count}"
      f"\n\n"
      f"COT Prompt with no source:\n"
      f"Llama: {llama_df_cot_llm_label_count}\n"
      f"Politifact: {llama_df_cot_politifact_label_count}"
      f"\n\n"
      f"COT Prompt with source:\n"
      f"Llama: {llama_df_cot_src_llm_label_count}\n"
      f"Politifact: {llama_df_cot_src_politifact_label_count}"
      f"Few-Shot prompt without source:"
      f"Llama: {llama_df_fs_llm_label_count}\n"
      f"Politifact: {llama_df_fs_politifact_label_count}"
      f"(\n\n)"
      f"Llama: {llama_df_fs_src_llm_label_count}\n"
      f"Politifact: {llama_df_fs_src_politifact_label_count}"
      )

llama_zs_ground_truth_labels = llama_df_zs['verdict'].values
llama_zs_labels = llama_df_zs['llama_verdict'].values
llama_zs_cm = confusion_matrix(
    llama_zs_ground_truth_labels, llama_zs_labels)

llama_zs_accuracy = accuracy_score(
    llama_zs_ground_truth_labels, llama_zs_labels)
llama_zs_recall = recall_score(
    llama_zs_ground_truth_labels, llama_zs_labels, average='macro')
llama_zs_precision = precision_score(
    llama_zs_ground_truth_labels, llama_zs_labels, average='macro')
llama_zs_f1 = f1_score(llama_zs_ground_truth_labels,
                       llama_zs_labels, average='macro')
print(f"Accuracy: {llama_zs_accuracy:.4f}\n"
      f"Precision: {llama_zs_precision:.4f}\n"
      f"Recall: {llama_zs_recall:.4f}\n"
      f"F1 Score: {llama_zs_f1:.4f}")

print('\n\n')


llama_zs_src_ground_truth_labels = llama_df_zs_src['verdict'].values
llama_zs_src_labels = llama_df_zs_src['llama_verdict'].values
llama_src_cm = confusion_matrix(
    llama_zs_src_ground_truth_labels, llama_zs_src_labels)

llama_src_accuracy = accuracy_score(
    llama_zs_src_ground_truth_labels, llama_zs_src_labels)
llama_src_recall = recall_score(
    llama_zs_src_ground_truth_labels, llama_zs_src_labels, average='macro')
llama_src_precision = precision_score(
    llama_zs_src_ground_truth_labels, llama_zs_src_labels, average='macro')
llama_src_f1 = f1_score(llama_zs_src_ground_truth_labels,
                        llama_zs_src_labels, average='macro')
print(f"Accuracy: {llama_src_accuracy:.4f}\n"
      f"Precision: {llama_src_precision:.4f}\n"
      f"Recall: {llama_src_recall:.4f}\n"
      f"F1 Score: {llama_src_f1:.4f}")

print('\n\n')

llama_zs_cot_ground_truth_labels = llama_df_cot['verdict'].values
llama_cot_labels = llama_df_cot['llama_verdict'].values
llama_cot_cm = confusion_matrix(
    llama_zs_cot_ground_truth_labels, llama_cot_labels)

llama_cot_accuracy = accuracy_score(
    llama_zs_cot_ground_truth_labels, llama_cot_labels)
llama_cot_recall = recall_score(
    llama_zs_cot_ground_truth_labels, llama_cot_labels, average='macro')
llama_cot_precision = precision_score(
    llama_zs_cot_ground_truth_labels, llama_cot_labels, average='macro')
llama_cot_f1 = f1_score(llama_zs_cot_ground_truth_labels,
                        llama_cot_labels, average='macro')
print(f"Accuracy: {llama_cot_accuracy:.4f}\n"
      f"Precision: {llama_cot_precision:.4f}\n"
      f"Recall: {llama_cot_recall:.4f}\n"
      f"F1 Score: {llama_cot_f1:.4f}")

llama_zs_cot_src_ground_truth_labels = llama_df_cot_src['verdict'].values
llama_cot_src_labels = llama_df_cot_src['llama_verdict'].values
llama_cot_src_cm = confusion_matrix(
    llama_zs_cot_src_ground_truth_labels, llama_cot_src_labels)

llama_cot_src_accuracy = accuracy_score(
    llama_zs_cot_src_ground_truth_labels, llama_cot_src_labels)
llama_cot_src_recall = recall_score(
    llama_zs_cot_src_ground_truth_labels, llama_cot_src_labels, average='macro')
llama_cot_src_precision = precision_score(
    llama_zs_cot_src_ground_truth_labels, llama_cot_src_labels, average='macro')
llama_cot_src_f1 = f1_score(
    llama_zs_cot_src_ground_truth_labels, llama_cot_src_labels, average='macro')
print(f"Accuracy: {llama_cot_src_accuracy:.4f}\n"
      f"Precision: {llama_cot_src_precision:.4f}\n"
      f"Recall: {llama_cot_src_recall:.4f}\n"
      f"F1 Score: {llama_cot_src_f1:.4f}")

llama_fs_ground_truth_labels = llama_df_fs['verdict'].values
llama_fs_labels = llama_df_fs['llama_verdict'].values
llama_fs_cm = confusion_matrix(
    llama_fs_ground_truth_labels, llama_fs_labels)

llama_fs_accuracy = accuracy_score(
    llama_fs_ground_truth_labels, llama_fs_labels)
llama_fs_recall = recall_score(
    llama_fs_ground_truth_labels, llama_fs_labels, average='macro')
llama_fs_precision = precision_score(
    llama_fs_ground_truth_labels, llama_fs_labels, average='macro')
llama_fs_f1 = f1_score(llama_fs_ground_truth_labels,
                       llama_fs_labels, average='macro')
print(f"Accuracy: {llama_fs_accuracy:.4f}\n"
      f"Precision: {llama_fs_precision:.4f}\n"
      f"Recall: {llama_fs_recall:.4f}\n"
      f"F1 Score: {llama_fs_f1:.4f}")

print('\n\n')

llama_fs_src_ground_truth_labels = llama_df_fs_src['verdict'].values
llama_fs_src_labels = llama_df_fs_src['llama_verdict'].values
llama_fs_src_cm = confusion_matrix(
    llama_fs_src_ground_truth_labels, llama_fs_src_labels)

llama_fs_src_accuracy = accuracy_score(
    llama_fs_src_ground_truth_labels, llama_fs_src_labels)
llama_fs_src_recall = recall_score(
    llama_fs_src_ground_truth_labels, llama_fs_src_labels, average='macro')
llama_fs_src_precision = precision_score(
    llama_fs_src_ground_truth_labels, llama_fs_src_labels, average='macro')
llama_fs_src_f1 = f1_score(
    llama_fs_src_ground_truth_labels, llama_fs_src_labels, average='macro')
print(f"Accuracy: {llama_fs_src_accuracy:.4f}\n"
      f"Precision: {llama_fs_src_precision:.4f}\n"
      f"Recall: {llama_fs_src_recall:.4f}\n"
      f"F1 Score: {llama_fs_src_f1:.4f}")

print('\n\n')


# llama_zs_cm = ConfusionMatrixDisplay(confusion_matrix(
#     llama_zs_ground_truth_labels, llama_zs_labels), display_labels=["False", "Mixed", "True"])
# llama_zs_cm.plot(cmap='Blues')
# plt.title("Llama 3.3 70b - Zero Shot without Source or COT")
# plt.savefig('llama_zs_cm.png')
# plt.clf()

# llama_zs_src_cm = ConfusionMatrixDisplay(confusion_matrix(
#     llama_zs_src_ground_truth_labels, llama_zs_src_labels), display_labels=["False", "Mixed", "True"])
# llama_zs_src_cm.plot(cmap='Blues')
# plt.title("Llama 3.3 70b - Zero Shot with Source")
# plt.savefig('llama_zs_src_cm.png')
# plt.clf()

# llama_cot_cm = ConfusionMatrixDisplay(confusion_matrix(
#     llama_zs_cot_ground_truth_labels, llama_cot_labels), display_labels=["False", "Mixed", "True"])
# llama_cot_cm.plot(cmap='Blues')
# plt.title("Llama 3.3 70b - Annotations derived from Chain-of-Thought Prompting")
# plt.savefig('llama_cot_cm.png')
# plt.clf()

# llama_cot_src_cm = ConfusionMatrixDisplay(confusion_matrix(
#     llama_zs_cot_src_ground_truth_labels, llama_cot_src_labels), display_labels=["False", "Mixed", "True"])
# llama_cot_src_cm.plot(cmap='Blues')
# plt.title(
#     "Llama 3.3 70b - Annotations derived from Chain-of-Thought Prompting with Source")
# plt.savefig('llama_cot_src_cm.png')
# plt.clf()

# llama_fs_cm = ConfusionMatrixDisplay(confusion_matrix(
#     llama_fs_ground_truth_labels, llama_fs_labels), display_labels=["False", "Mixed", "True"])
# llama_fs_cm.plot(cmap='Blues')
# plt.title(
#     "Llama 3.3 70b - Annotations derived from Few-Shot Prompting without Source")
# plt.savefig('llama_fs_cm.png')
# plt.clf()

# llama_fs_src_cm = ConfusionMatrixDisplay(confusion_matrix(
#     llama_fs_src_ground_truth_labels, llama_fs_src_labels), display_labels=["False", "Mixed", "True"])
# llama_fs_src_cm.plot(cmap='Blues')
# plt.title(
#     "Llama 3.3 70b - Annotations derived from Few-Shot Prompting with Source")
# plt.savefig('llama_fs_src_cm.png')
# plt.clf()

zs_metrics = {
    'Accuracy': (round(mistral_zs_accuracy*100, 2), round(llama_zs_accuracy*100, 2)),
    'Recall': (round(mistral_zs_recall*100, 2), round(llama_zs_recall*100, 2)),
    'Precision': (round(mistral_zs_precision*100, 2), round(llama_zs_precision*100, 2)),
    'F1 Score': (round(mistral_zs_f1*100, 2), round(llama_zs_f1*100, 2))
}

zs_src_metrics = {
    'Accuracy': (round(mistral_src_accuracy*100, 2), round(llama_src_accuracy*100, 2)),
    'Recall': (round(mistral_src_recall*100, 2), round(llama_src_recall*100, 2)),
    'Precision': (round(mistral_src_precision*100, 2), round(llama_src_precision*100, 2)),
    'F1 Score': (round(mistral_src_f1*100, 2), round(llama_src_f1*100, 2))
}

cot_metrics = {
    'Accuracy': (round(mistral_cot_accuracy*100, 2), round(llama_cot_accuracy*100, 2)),
    'Recall': (round(mistral_cot_recall*100, 2), round(llama_cot_recall*100, 2)),
    'Precision': (round(mistral_cot_precision*100, 2), round(llama_cot_precision*100, 2)),
    'F1 Score': (round(mistral_cot_f1*100, 2), round(llama_cot_f1*100, 2))
}

cot_src_metrics = {
    'Accuracy': (round(mistral_cot_src_accuracy*100, 2), round(llama_cot_src_accuracy*100, 2)),
    'Recall': (round(mistral_cot_src_recall*100, 2), round(llama_cot_src_recall*100, 2)),
    'Precision': (round(mistral_cot_src_precision*100, 2), round(llama_cot_src_precision*100, 2)),
    'F1 Score': (round(mistral_cot_src_f1*100, 2), round(llama_cot_src_f1*100, 2))
}

fs_metrics = {
    'Accuracy': (round(mistral_fs_accuracy*100, 2), round(llama_fs_accuracy*100, 2)),
    'Recall': (round(mistral_fs_recall*100, 2), round(llama_fs_recall*100, 2)),
    'Precision': (round(mistral_fs_precision*100, 2), round(llama_fs_precision*100, 2)),
    'F1 Score': (round(mistral_fs_f1*100, 2), round(llama_fs_f1*100, 2))
}

fs_src_metrics = {
    'Accuracy': (round(mistral_fs_src_accuracy*100, 2), round(llama_fs_src_accuracy*100, 2)),
    'Recall': (round(mistral_fs_src_recall*100, 2), round(llama_fs_src_recall*100, 2)),
    'Precision': (round(mistral_fs_src_precision*100, 2), round(llama_fs_src_precision*100, 2)),
    'F1 Score': (round(mistral_fs_src_f1*100, 2), round(llama_fs_src_f1*100, 2))
}

bar_colors = {
    'Accuracy': '#90caf9',    # light blue
    'Recall': '#a5d6a7',      # light green
    'Precision': '#fff59d',   # light yellow
    'F1 Score': '#ffccbc'     # light orange
}

models = ("Mistral-7b-Instruct", "Llama-3.3-70b")


def plot_metrics(metrics, title, filename):
    fig, ax = plt.subplots(layout='constrained')

    x = np.arange(0, len(models) * 2, 2)  # the label locations
    width = 0.4
    multiplier = 0
    for attribute, measurements in metrics.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurements, width, label=attribute,
                       color=bar_colors.get(attribute, '#e0e0e0'), edgecolor='lightgrey', linewidth=1.5)
        ax.bar_label(rects, labels=[
                     f"{val}%" for val in measurements], padding=3)
        multiplier += 1

    ax.set_ylabel('Scores (%)')
    ax.set_title(title)
    ax.set_xticks(x + width * 1.5, models)
    ax.legend(loc='upper left', ncols=4)
    ax.set_ylim(0, 100)

    plt.savefig(f"{filename}.png")


plot_metrics(
    zs_metrics, 'Zero-Shot without Source', 'model_comparison_zs_metrics')
plt.clf()
plot_metrics(zs_src_metrics, 'Zero-Shot with Source',
             'model_comparison_zs_src_metrics')
plt.clf()
plot_metrics(cot_metrics, 'Chain-of-Thought Prompting without Source',
             'model_comparison_cot_metrics')
plt.clf()
plot_metrics(cot_src_metrics, 'Chain-of-Thought Prompting with Source',
             'model_comparison_cot_src_metrics')
plt.clf()
plot_metrics(fs_metrics, 'Few-Shot Prompting without Source',
             'model_comparison_fs_metrics')
plt.clf()
plot_metrics(fs_src_metrics, 'Few-Shot Prompting with Source',
             'model_comparison_fs_src_metrics')


"""Demographics"""
datasets = [
    mistral_df_zs,
    mistral_df_zs,
    mistral_df_zs_src,
    mistral_df_fs,
    mistral_df_fs_src,
    mistral_df_cot,
    mistral_df_cot_src,
    llama_df_zs,
    llama_df_zs_src,
    llama_df_fs,
    llama_df_fs_src,
    llama_df_cot,
    llama_df_cot_src,
]
mistral_datasets = {
    'Mistral ZS': mistral_df_zs,
    'Mistral ZS Src': mistral_df_zs_src,
    'Mistral FS': mistral_df_fs,
    'Mistral FS Src': mistral_df_fs_src,
    'Mistral COT': mistral_df_cot,
    'Mistral COT Src': mistral_df_cot_src
}
mistral_datasets_no_src = {
    'Mistral ZS': mistral_df_zs,
    'Mistral FS': mistral_df_fs,
    'Mistral COT': mistral_df_cot
}
mistral_datasets_src = {
    'Mistral ZS Src': mistral_df_zs_src,
    'Mistral FS Src': mistral_df_fs_src,
    'Mistral COT Src': mistral_df_cot_src
}
llama_datasets = {
    'Llama ZS': llama_df_zs,
    'Llama ZS Src': llama_df_zs_src,
    'Llama FS': llama_df_fs,
    'Llama FS Src': llama_df_fs_src,
    'Llama COT': llama_df_cot,
    'Llama COT Src': llama_df_cot_src
}
llama_datasets_no_src = {
    'Llama ZS': llama_df_zs,
    'Llama FS': llama_df_fs,
    'Llama COT': llama_df_cot
}
llama_datasets_src = {
    'Llama ZS Src': llama_df_zs_src,
    'Llama FS Src': llama_df_fs_src,
    'Llama COT Src': llama_df_cot_src
}
label_cols = [
    'verdict',
    'mistral_verdict',
    'mistral_verdict',
    'mistral_verdict',
    'mistral_verdict',
    'mistral_verdict',
    'mistral_verdict',
    'llama_verdict',
    'llama_verdict',
    'llama_verdict',
    'llama_verdict',
    'llama_verdict',
    'llama_verdict'
]
dataset_names = [
    "Politifact",
    "Mistral ZS",
    "Mistral ZS Source",
    "Mistral FS",
    "Mistral FS Source",
    "Mistral COT",
    "Mistral COT Source",
    "Llama ZS",
    "Llama ZS Source",
    "Llama FS",
    "Llama FS Source",
    "Llama COT",
    "Llama COT Source",
]


def get_color_gradient(value, colormap='Blues'):
    norm_value = min(max(value, 0), 1)
    return plt.cm.get_cmap(colormap)(0.3 + 0.7 * norm_value)


def label_disparity(
    datasets: dict, verdict_col,
    title, filename, plot_average=True
):
    parties = ['R', 'D']
    party_names = ['Republican', 'Democrat']

    def compute_f1(df, party, verdict_col):
        y_true = df[df['party'] == party]['verdict'].values
        y_pred = df[df['party'] == party][verdict_col].values
        if len(y_true) == 0:
            return 0.0
        return f1_score(y_true=y_true, y_pred=y_pred, average='macro')

    if plot_average:
        scores_list = []
        for df in datasets.values():
            scores_list.append([compute_f1(df, p, verdict_col)
                               for p in parties])
        # shape (n_datasets, n_parties)
        arr = np.array(scores_list, dtype=float)

        # average across datasets, ignore NaNs
        avg_scores = np.nanmean(arr, axis=0)

        # plot only the average
        fig, ax = plt.subplots(figsize=(8, 6))
        x = np.arange(len(parties))
        colors = [get_color_gradient(score, 'Blues') for score in avg_scores]
        bars = ax.bar(x, avg_scores * 100, width=0.6,
                      color=colors, edgecolor='black')
        ax.bar_label(bars, labels=[f"{v*100:.1f}%" if not np.isnan(v)
                     else "" for v in avg_scores], padding=4, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(party_names)
        ax.set_ylabel("Average F1 Score (%)")
        ax.set_title(f"{title} — Average F1 by Party")
        ax.set_ylim(0, 100)
        plt.tight_layout()
        plt.savefig(f"{filename}_average.png", dpi=200, bbox_inches='tight')
        plt.close(fig)
    else:
        results = {}
        for dataset_name, df in datasets.items():
            results[dataset_name] = [compute_f1(
                df, p, verdict_col) for p in parties]

        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(parties))
        n = len(results)
        width = 0.8 / max(n, 1)
        colors = plt.cm.Set3(np.linspace(0, 1, n))

        for i, (dataset_name, scores) in enumerate(results.items()):
            offset = (i - (n - 1) / 2) * width
            bars = ax.bar(x + offset, np.array(scores) * 100, width,
                          label=dataset_name, color=colors[i], edgecolor='grey', linewidth=0.8)
            ax.bar_label(bars, labels=[f"{v*100:.1f}%" for v in scores],
                         padding=3, fontsize=9, rotation=0)

        ax.set_xticks(x)
        ax.set_xticklabels(party_names)
        ax.set_ylabel("F1 Score (%)")
        ax.set_title(f"{title} — F1 Scores by Party")
        ax.set_ylim(0, 100)
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(f"{filename}.png", dpi=200, bbox_inches='tight')
        plt.close(fig)


plt.clf()
label_disparity(mistral_datasets, 'mistral_verdict',
                'Mistral-7B-Instruct', 'mistral_f1_scores_party', plot_average=True)
plt.clf()
label_disparity(llama_datasets, 'llama_verdict',
                'Llama-3.3-70B', 'llama_f1_scores_party', plot_average=True)
plt.clf()


def label_disparity_topic(
    datasets: dict, verdict_col,
    title, filename, plot_average=True
):
    topics = ['economy', 'crime', 'war', 'transparency',
              'environment', 'education', 'race', 'health', 'immigration']

    def compute_f1(df, topic, verdict_col):
        y_true = (df[df['topic'] == topic]['verdict']).values
        y_pred = (df[df['topic'] == topic][verdict_col]).values
        if len(y_true) == 0:
            return 0
        return f1_score(y_true=y_true, y_pred=y_pred, average='macro')

    if plot_average:
        scores_list = []
        for df in datasets.values():
            scores_list.append([compute_f1(df, topic, verdict_col)
                               for topic in topics])

        arr = np.array(scores_list, dtype=float)

        avg_scores = np.nanmean(arr, axis=0)

        fig, ax = plt.subplots(figsize=(14, 6))
        x = np.arange(len(topics))
        colors = [get_color_gradient(score, 'Blues') for score in avg_scores]
        bars = ax.bar(x, avg_scores * 100, width=0.7,
                      color=colors, edgecolor='black')
        ax.bar_label(bars, labels=[f"{v*100:.1f}" if not np.isnan(
            v) else "" for v in avg_scores], padding=3, rotation=90, fontsize=8)
        ax.set_ylabel('Average F1 Score (%)')
        ax.set_title(f'{title} - Average F1 Scores by Topic')
        ax.set_xticks(x)
        ax.set_xticklabels(topics, rotation=45, ha='right')
        ax.set_ylim(0, 100)
        plt.tight_layout()
        plt.savefig(f'{filename}_average.png', dpi=200, bbox_inches='tight')
        plt.close(fig)

    else:
        results = {}

        for dataset_name, df in datasets.items():
            results[dataset_name] = [compute_f1(
                df, topic, verdict_col) for topic in topics]

        fig, ax = plt.subplots(figsize=(15, 8))
        x = np.arange(len(topics))
        width = 0.12
        multiplier = 0

        colours = plt.cm.Set3(np.linspace(0, 1, len(datasets)))

        for dataset_name, scores in results.items():
            offset = width * multiplier
            bars = ax.bar(x + offset, np.array(scores)*100, width, label=dataset_name,
                          color=colours[multiplier], edgecolor='grey', linewidth=1)
            ax.bar_label(bars, labels=[f"{v*100:.1f}" for v in scores],
                         padding=3, rotation=90, fontsize=8)
            multiplier += 1

        ax.set_ylabel('F1 Score (%)')
        ax.set_title(f'{title} - F1 Scores by Topic')
        ax.set_xticks(x + width * (len(datasets) - 1) / 2)
        ax.set_xticklabels(topics, rotation=45, ha='right')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.set_ylim(0, 100)

        plt.tight_layout()
        plt.savefig(f'{filename}.png', dpi=200, bbox_inches='tight')
        plt.close(fig)


def plot_label_distribution_bar(datasets, label_cols, dataset_names):
    # Calculate percentages for each label in each dataset
    percentages = []
    for df, col in zip(datasets, label_cols):
        counts = df[col].value_counts(normalize=True) * 100
        # Ensure we have all labels (0,1,2) with 0% if missing
        formatted_counts = {
            1: counts.get(1, 0),  # True
            2: counts.get(2, 0),  # Mixed
            0: counts.get(0, 0)   # False
        }
        percentages.append(formatted_counts)

    # Prepare data for plotting
    true_pcts = [p[1] for p in percentages]
    mixed_pcts = [p[2] for p in percentages]
    false_pcts = [p[0] for p in percentages]

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot stacked bars
    x = np.arange(len(dataset_names))
    width = 0.8

    ax.bar(x, true_pcts, width, label='True', color='#6ea8fe')
    ax.bar(x, mixed_pcts, width, bottom=true_pcts,
           label='Mixed', color='#b2beb5')
    ax.bar(x, false_pcts, width, bottom=[sum(x) for x in zip(true_pcts, mixed_pcts)],
           label='False', color='#ff8787')

    # Customize plot
    ax.set_ylabel('Percentage (%)')
    ax.set_title('Label Distribution Across Datasets')
    ax.set_xticks(x)
    ax.set_xticklabels(dataset_names, rotation=45, ha='right')

    # Add legend
    ax.legend(title='Label')

    # Add percentage labels on bars
    for i in range(len(dataset_names)):
        # True percentage
        if true_pcts[i] > 0:
            ax.text(i, true_pcts[i]/2, f'{true_pcts[i]:.1f}%',
                    ha='center', va='center')

        # Mixed percentage
        if mixed_pcts[i] > 0:
            ax.text(i, true_pcts[i] + mixed_pcts[i]/2,
                    f'{mixed_pcts[i]:.1f}%', ha='center', va='center')

        # False percentage
        if false_pcts[i] > 0:
            ax.text(i, true_pcts[i] + mixed_pcts[i] + false_pcts[i]/2,
                    f'{false_pcts[i]:.1f}%', ha='center', va='center')

    plt.tight_layout()
    plt.savefig('label_distribution_bar.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


plot_label_distribution_bar(datasets, label_cols, dataset_names)

plt.clf()
label_disparity_topic(mistral_datasets, 'mistral_verdict',
                      'Mistral-7B-Instruct', 'topic_f1_scores_mistral', plot_average=True)
plt.clf()
label_disparity_topic(llama_datasets, 'llama_verdict',
                      'Llama-3.3-70B', 'topic_f1_scores_llama', plot_average=True)
plt.clf()


def label_disparity_popularity(
    datasets: dict, verdict_col, title, filename, plot_average=True
):
    bins = ['<100', '100-500', '500-1000', '1000-5000', '5000-10000',
            '10000-50000', '50000-100000', '>100000']

    def compute_f1(df, bin_val, verdict_col):
        y_true = df[df['average_views'] == bin_val]['verdict'].values
        y_pred = df[df['average_views'] == bin_val][verdict_col].values
        if len(y_true) == 0:
            return 0.0
        return f1_score(y_true=y_true, y_pred=y_pred, average='macro')

    if plot_average:
        scores_list = []
        for df in datasets.values():
            scores_list.append([compute_f1(df, b, verdict_col)
                               for b in bins])
        arr = np.array(scores_list, dtype=float)

        avg_scores = np.nanmean(arr, axis=0)

        fig, ax = plt.subplots(figsize=(14, 6))
        x = np.arange(len(bins))
        colors = [get_color_gradient(score, 'Blues') for score in avg_scores]
        bars = ax.bar(x, avg_scores * 100, width=0.7,
                      color=colors, edgecolor='black')
        ax.bar_label(bars, labels=[f"{v*100:.1f}%" if not np.isnan(
            v) else "" for v in avg_scores], padding=3, rotation=90, fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(bins, rotation=45, ha='right')
        ax.set_ylabel("Average F1 Score (%)")
        ax.set_title(f"{title} — Average F1 by Popularity Bin")
        ax.set_ylim(0, 100)
        plt.tight_layout()
        plt.savefig(f"{filename}_average.png", dpi=200, bbox_inches='tight')
        plt.close(fig)
    else:
        results = {}
        for dataset_name, df in datasets.items():
            results[dataset_name] = [compute_f1(
                df, b, verdict_col) for b in bins]

        arr = np.array(list(results.values()), dtype=float)
        avg_scores = np.nanmean(arr, axis=0) if arr.size else np.array(
            [np.nan]*len(bins))
        results['Average'] = avg_scores.tolist()

        fig, ax = plt.subplots(figsize=(14, 7))
        x = np.arange(len(bins))
        n = len(results)
        width = 0.8 / max(n, 1)
        colors = plt.cm.Set3(np.linspace(0, 1, n))

        for i, (dataset_name, scores) in enumerate(results.items()):
            offset = (i - (n - 1) / 2) * width
            bars = ax.bar(x + offset, np.array(scores) * 100, width,
                          label=dataset_name, color=colors[i], edgecolor='grey', linewidth=0.8)
            ax.bar_label(bars, labels=[f"{v*100:.1f}%" for v in scores],
                         padding=2, fontsize=8, rotation=90)

        ax.set_xticks(x)
        ax.set_xticklabels(bins, rotation=45, ha='right')
        ax.set_ylabel("F1 Score (%)")
        ax.set_title(f"{title} — F1 by Popularity Bin")
        ax.set_ylim(0, 100)
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', ncol=1)
        plt.tight_layout()
        plt.savefig(f"{filename}.png", dpi=200, bbox_inches='tight')
        plt.close(fig)


plt.clf()
label_disparity_popularity(mistral_datasets, 'mistral_verdict',
                           'Mistral-7B-Instruct', 'popularity_f1_scores_mistral', plot_average=True)
plt.clf()
label_disparity_popularity(llama_datasets, 'llama_verdict',
                           'Llama-3.3-70B', 'popularity_f1_scores_llama', plot_average=True)
plt.clf()


def kl_divergence(p_counts, q_counts, label_set=[0, 1, 2]):
    p = np.array([p_counts.get(l, 0) for l in label_set], dtype=float)
    q = np.array([q_counts.get(l, 0) for l in label_set], dtype=float)

    p = p / p.sum() if p.sum() > 0 else np.ones(len(label_set)) / len(label_set)
    q = q / q.sum() if q.sum() > 0 else np.ones(len(label_set)) / len(label_set)

    epsilon = 1e-10
    p = np.clip(p, epsilon, 1)
    q = np.clip(q, epsilon, 1)
    return entropy(p, q)


mistral_zs_kl = kl_divergence(
    mistral_df_zs_llm_label_count, mistral_df_zs_politifact_label_count)
mistral_zs_src_kl = kl_divergence(
    mistral_df_zs_src_llm_label_count, politifact_df_label_count)
mistral_fs_kl = kl_divergence(
    mistral_df_fs_llm_label_count, politifact_df_label_count)
mistral_fs_src_kl = kl_divergence(
    mistral_df_fs_src_llm_label_count, politifact_df_label_count)
mistral_cot_kl = kl_divergence(
    mistral_df_cot_llm_label_count, politifact_df_label_count)
mistral_cot_src_kl = kl_divergence(
    mistral_df_cot_src_llm_label_count, politifact_df_label_count)

llama_zs_kl = kl_divergence(
    llama_df_zs_llm_label_count, politifact_df_label_count)
llama_zs_src_kl = kl_divergence(
    llama_df_zs_src_llm_label_count, politifact_df_label_count)
llama_fs_kl = kl_divergence(
    llama_df_fs_llm_label_count, politifact_df_label_count)
llama_fs_src_kl = kl_divergence(
    llama_df_fs_src_llm_label_count, politifact_df_label_count)
llama_cot_kl = kl_divergence(
    llama_df_cot_llm_label_count, politifact_df_label_count)
llama_cot_src_kl = kl_divergence(
    llama_df_cot_src_llm_label_count, politifact_df_label_count)

print(
    f"Mistral zero-shot KL divergence: {mistral_zs_kl}\n"
    f"Mistral zero-shot source KL divergence: {mistral_zs_src_kl}\n"
    f"Mistral few-shot KL divergence: {mistral_fs_kl}\n"
    f"Mistral few-shot source KL divergence: {mistral_fs_src_kl}\n"
    f"Mistral COT KL divergence: {mistral_cot_kl}\n"
    f"Mistral COT source KL divergence: {mistral_cot_src_kl}\n"

    f"Llama zero-shot KL divergence: {llama_zs_kl}\n"
    f"Llama zero-shot source KL divergence: {llama_zs_src_kl}\n"
    f"Llama few-shot KL divergence: {llama_fs_kl}\n"
    f"Llama few-shot source KL divergence: {llama_fs_src_kl}\n"
    f"Llama COT KL divergence: {llama_cot_kl}\n"
    f"Llama COT source KL divergence: {llama_cot_src_kl}\n"
)

kl_values = [
    mistral_zs_kl,
    mistral_zs_src_kl,
    mistral_fs_kl,
    mistral_fs_src_kl,
    mistral_cot_kl,
    mistral_cot_src_kl,
    llama_zs_kl,
    llama_zs_src_kl,
    llama_fs_kl,
    llama_fs_src_kl,
    llama_cot_kl,
    llama_cot_src_kl
]

kl_names = [
    "Mistral ZS",
    "Mistral ZS Source",
    "Mistral FS",
    "Mistral FS Source",
    "Mistral COT",
    "Mistral COT Source",
    "Llama ZS",
    "Llama ZS Source",
    "Llama FS",
    "Llama FS Source",
    "Llama COT",
    "Llama COT Source"
]

# plt.figure(figsize=(12, 6))
# bars = plt.bar(kl_names, kl_values, color="#90caf9",
#                edgecolor="grey", linewidth=1.5)
# plt.ylabel("KL Divergence", fontsize=14)
# plt.title("KL Divergence between Generated Labels and Politifact")
# plt.xticks(rotation=45, ha='right', fontsize=12)
# plt.ylim(0, max(kl_values)*1.15)
# plt.tight_layout()

# for bar, val in zip(bars, kl_values):
#     plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f"{val:.3f}",
#              ha='center', va='bottom', fontsize=11, fontweight='bold')

# plt.savefig("kl_divergence_barchart.png", dpi=200)

"""All LLMs and Politifact Agreement"""


def krippendorff_calc(mistral_csv, llama_csv, title, type: str, llm: str):
    mistral_df = pd.read_csv(
        mistral_csv
    )
    llama_df = pd.read_csv(
        llama_csv
    )

    df_type = pd.DataFrame()

    if type == "inter_llm":
        mistral_df['mistral_verdict'] = mistral_df['mistral_verdict'].apply(
            lambda x: llm_label_mapping.get(x, np.nan)
        ).astype(float)
        llama_df['llama_verdict'] = llama_df['llama_verdict'].apply(
            lambda x: llm_label_mapping.get(x, np.nan)
        ).astype(float)
        df_type = pd.DataFrame({
            'mistral_verdict': mistral_df['mistral_verdict'],
            'llama_verdict': llama_df['llama_verdict']
        }).astype(float)
    elif type == "politifact_llm":
        if llm == "mistral":
            mistral_df['verdict'] = mistral_df['verdict'].apply(
                lambda x: politifact_label_mapping.get(x, np.nan)
            ).astype(float)
            mistral_df['mistral_verdict'] = mistral_df['mistral_verdict'].apply(
                lambda x: llm_label_mapping.get(x, np.nan)
            ).astype(float)
            df_type = pd.DataFrame({
                'politifact_verdict': mistral_df['verdict'],
                'mistral_verdict': mistral_df['mistral_verdict']
            })
        else:
            llama_df['verdict'] = llama_df['verdict'].apply(
                lambda x: politifact_label_mapping.get(x, np.nan)
            ).astype(float)
            llama_df['llama_verdict'] = llama_df['llama_verdict'].apply(
                lambda x: llm_label_mapping.get(x, np.nan)
            ).astype(float)
            df_type = pd.DataFrame({
                'politifact_verdict': llama_df['verdict'],
                'llama_verdict': llama_df['llama_verdict']
            })
    else:
        mistral_df['verdict'] = mistral_df['verdict'].apply(
            lambda x: politifact_label_mapping.get(x, np.nan)
        ).astype(float)
        mistral_df['mistral_verdict'] = mistral_df['mistral_verdict'].apply(
            lambda x: llm_label_mapping.get(x, np.nan)
        ).astype(float)
        llama_df['llama_verdict'] = llama_df['llama_verdict'].apply(
            lambda x: llm_label_mapping.get(x, np.nan)
        ).astype(float)
        df_type = pd.DataFrame({
            'politifact_verdict': mistral_df['verdict'],
            'mistral_verdict': mistral_df['mistral_verdict'],
            'llama_verdict': llama_df['llama_verdict'],
        })

    print(df_type)

    # Krippendorff's alpha
    zs_alpha = kp.alpha(df_type.values.T, level_of_measurement='nominal')
    print(f"{title}: {zs_alpha}")
    return zs_alpha


zs_alpha = krippendorff_calc(
    "politifact_kaggle_dataset_sample_mistral_labelled.csv",
    "politifact_kaggle_dataset_sample_llama_3.3_70b_labelled.csv",
    "Zero-Shot without Source",
    type="None",
    llm="None"
)
zs_src_alpha = krippendorff_calc(
    "politifact_kaggle_dataset_sample_mistral_labelled_with_source.csv",
    "politifact_kaggle_dataset_sample_llama_3.3_70b_labelled_with_source.csv",
    "Zero-Shot with Source",
    type="None",
    llm="None"
)
fs_alpha = krippendorff_calc(
    "politifact_kaggle_dataset_sample_mistral_labelled_fs.csv",
    "politifact_kaggle_dataset_sample_llama_3.3_70b_labelled_fs.csv",
    "Few-Shot without Source",
    type="None",
    llm="None"
)
fs_src_alpha = krippendorff_calc(
    "politifact_kaggle_dataset_sample_mistral_labelled_fs_src.csv",
    "politifact_kaggle_dataset_sample_llama_3.3_70b_labelled_fs_src.csv",
    "Few-Shot with Source",
    type="None",
    llm="None"
)
cot_alpha = krippendorff_calc(
    "politifact_kaggle_dataset_sample_mistral_labelled_cot_no_source.csv",
    "politifact_kaggle_dataset_sample_llama_3.3_70b_labelled_cot_no_source.csv",
    "COT without Source",
    type="None",
    llm="None"
)
cot_src_alpha = krippendorff_calc(
    "politifact_kaggle_dataset_sample_mistral_labelled_cot_with_source.csv",
    "politifact_kaggle_dataset_sample_llama_3.3_70b_labelled_cot_with_source.csv",
    "COT with Source",
    type="None",
    llm="None"
)

"""Inter LLM Agreement"""
zs_alpha_llm_only = krippendorff_calc(
    "politifact_kaggle_dataset_sample_mistral_labelled.csv",
    "politifact_kaggle_dataset_sample_llama_3.3_70b_labelled.csv",
    "Inter-LLM Zero-Shot without Source",
    type="inter_llm",
    llm="None"
)
zs_src_alpha_llm_only = krippendorff_calc(
    "politifact_kaggle_dataset_sample_mistral_labelled_with_source.csv",
    "politifact_kaggle_dataset_sample_llama_3.3_70b_labelled_with_source.csv",
    "Inter-LLM Zero-Shot with Source",
    type="inter_llm",
    llm="None"
)
fs_alpha_llm_only = krippendorff_calc(
    "politifact_kaggle_dataset_sample_mistral_labelled_fs.csv",
    "politifact_kaggle_dataset_sample_llama_3.3_70b_labelled_fs.csv",
    "Inter-LLM Few-Shot without Source",
    type="inter_llm",
    llm="None"
)
fs_src_alpha_llm_only = krippendorff_calc(
    "politifact_kaggle_dataset_sample_mistral_labelled_fs_src.csv",
    "politifact_kaggle_dataset_sample_llama_3.3_70b_labelled_fs_src.csv",
    "Inter-LLM Few-Shot with Source",
    type="inter_llm",
    llm="None"
)
cot_alpha_llm_only = krippendorff_calc(
    "politifact_kaggle_dataset_sample_mistral_labelled_cot_no_source.csv",
    "politifact_kaggle_dataset_sample_llama_3.3_70b_labelled_cot_no_source.csv",
    "Inter-LLM COT without Source",
    type="inter_llm",
    llm="None"
)
cot_src_alpha_llm_only = krippendorff_calc(
    "politifact_kaggle_dataset_sample_mistral_labelled_cot_with_source.csv",
    "politifact_kaggle_dataset_sample_llama_3.3_70b_labelled_cot_with_source.csv",
    "Inter-LLM COT with Source",
    type="inter_llm",
    llm="None"
)

"""Politifact and LLM Agreement"""
zs_alpha_politifact_mistral = krippendorff_calc(
    "politifact_kaggle_dataset_sample_mistral_labelled.csv",
    "politifact_kaggle_dataset_sample_llama_3.3_70b_labelled.csv",
    "Politifact and Mistral Zero-Shot without Source",
    type="politifact_llm",
    llm="mistral"
)
zs_alpha_politifact_llama = krippendorff_calc(
    "politifact_kaggle_dataset_sample_mistral_labelled.csv",
    "politifact_kaggle_dataset_sample_llama_3.3_70b_labelled.csv",
    "Politifact and Llama Zero-Shot without Source",
    type="politifact_llm",
    llm="llama"
)
zs_src_alpha_politifact_mistral = krippendorff_calc(
    "politifact_kaggle_dataset_sample_mistral_labelled_with_source.csv",
    "politifact_kaggle_dataset_sample_llama_3.3_70b_labelled_with_source.csv",
    "Politifact and Mistral Zero-Shot with Source",
    type="politifact_llm",
    llm="mistral"
)
zs_src_alpha_politifact_llama = krippendorff_calc(
    "politifact_kaggle_dataset_sample_mistral_labelled_with_source.csv",
    "politifact_kaggle_dataset_sample_llama_3.3_70b_labelled_with_source.csv",
    "Politifact and Llama Zero-Shot with Source",
    type="politifact_llm",
    llm="llama"
)
fs_alpha_politifact_mistral = krippendorff_calc(
    "politifact_kaggle_dataset_sample_mistral_labelled_fs.csv",
    "politifact_kaggle_dataset_sample_llama_3.3_70b_labelled_fs.csv",
    "Politifact and Mistral Few-Shot without Source",
    type="politifact_llm",
    llm="mistral"
)
fs_alpha_politifact_llama = krippendorff_calc(
    "politifact_kaggle_dataset_sample_mistral_labelled_fs.csv",
    "politifact_kaggle_dataset_sample_llama_3.3_70b_labelled_fs.csv",
    "Politifact and Llama Few-Shot without Source",
    type="politifact_llm",
    llm="llama"
)
fs_src_alpha_politifact_mistral = krippendorff_calc(
    "politifact_kaggle_dataset_sample_mistral_labelled_fs_src.csv",
    "politifact_kaggle_dataset_sample_llama_3.3_70b_labelled_fs_src.csv",
    "Politifact and Mistral Few-Shot with Source",
    type="politifact_llm",
    llm="mistral"
)
fs_src_alpha_politifact_llama = krippendorff_calc(
    "politifact_kaggle_dataset_sample_mistral_labelled_fs_src.csv",
    "politifact_kaggle_dataset_sample_llama_3.3_70b_labelled_fs_src.csv",
    "Politifact and Llama Few-Shot with Source",
    type="politifact_llm",
    llm="llama"
)
cot_alpha_politifact_mistral = krippendorff_calc(
    "politifact_kaggle_dataset_sample_mistral_labelled_cot_no_source.csv",
    "politifact_kaggle_dataset_sample_llama_3.3_70b_labelled_cot_no_source.csv",
    "Politifact and Mistral COT without Source",
    type="politifact_llm",
    llm="mistral"
)
cot_alpha_politifact_llama = krippendorff_calc(
    "politifact_kaggle_dataset_sample_mistral_labelled_cot_no_source.csv",
    "politifact_kaggle_dataset_sample_llama_3.3_70b_labelled_cot_no_source.csv",
    "Politifact and Llama COT without Source",
    type="politifact_llm",
    llm="llama"
)
cot_src_alpha_politifact_mistral = krippendorff_calc(
    "politifact_kaggle_dataset_sample_mistral_labelled_cot_with_source.csv",
    "politifact_kaggle_dataset_sample_llama_3.3_70b_labelled_cot_with_source.csv",
    "Politifact and Mistral COT with Source",
    type="politifact_llm",
    llm="mistral"
)
cot_src_alpha_politifact_llama = krippendorff_calc(
    "politifact_kaggle_dataset_sample_mistral_labelled_cot_with_source.csv",
    "politifact_kaggle_dataset_sample_llama_3.3_70b_labelled_cot_with_source.csv",
    "Politifact and Llama COT with Source",
    type="politifact_llm",
    llm="llama"
)


def tpr_fpr_calc(prompt_type, dataframe, model, model_col, senstive_feature):
    classes = [0, 1, 2]
    for cls in classes:
        y_true = (dataframe['verdict'] == cls).astype(int)
        y_pred = (dataframe[model_col] == cls).astype(int)
        tpr = MetricFrame(metrics=true_positive_rate,
                          y_true=y_true,
                          y_pred=y_pred,
                          sensitive_features=dataframe[senstive_feature])
        print(
            f"\n{cls} {model} {prompt_type} True Positive Rate by Party", tpr.by_group)

        fpr = MetricFrame(metrics=false_positive_rate,
                          y_true=y_true,
                          y_pred=y_pred,
                          sensitive_features=dataframe[senstive_feature])
    print(f"{cls} {model} {prompt_type} False Positive Rate by Party", fpr.by_group)


def dp_calc(prompt_type, dataframe, model, model_col, senstive_feature):
    dp = MetricFrame(metrics=selection_rate,
                     y_true=dataframe['verdict'],
                     y_pred=dataframe[model_col],
                     sensitive_features=dataframe[senstive_feature])
    print(f"\n{model} {prompt_type} Selection Rate by Group", dp.by_group)
    dp_diff = demographic_parity_difference(
        y_true=dataframe['verdict'],
        y_pred=dataframe[model_col],
        sensitive_features=dataframe[senstive_feature]
    )
    dp_ratio = demographic_parity_ratio(
        y_true=dataframe['verdict'],
        y_pred=dataframe[model_col],
        sensitive_features=dataframe[senstive_feature]
    )
    print(f"{model} {prompt_type} {senstive_feature} Demographic Parity Difference:", dp_diff)

    print(f"{model} {prompt_type} {senstive_feature} Demographic Parity Ratio:", dp_ratio)


def mcnemar_calc():
    mistral_df_keys = list(mistral_datasets.keys())
    mistral_df_values = list(mistral_datasets.values())
    llama_df_keys = list(llama_datasets.keys())
    llama_df_values = list(llama_datasets.values())

    for n in range(len(mistral_datasets)):
        mistral_df = mistral_df_values[n].copy()
        llama_df = llama_df_values[n].copy()

        mistral_correct = (
            mistral_df['mistral_verdict'] == mistral_df['verdict']).astype(int)
        llama_correct = (llama_df['llama_verdict'] ==
                         llama_df['verdict']).astype(int)

        table = pd.crosstab(mistral_correct, llama_correct)

        result = mcnemar(table, exact=True)

        print(
            f"McNemar's test statistic for {mistral_df_keys[n]} vs {llama_df_keys[n]}: {result.statistic}, p-value: {result.pvalue}")
        if result.pvalue < 0.05:
            print("Difference is statistically significant (p < 0.05)\n\n")
        else:
            print("No statistically significant difference (p >= 0.05)\n\n")


def mcnemar_calc_source():
    mistral_dataset_no_src_keys = list(mistral_datasets_no_src.keys())
    mistral_dataset_no_src_values = list(mistral_datasets_no_src.values())

    mistral_dataset_src_keys = list(mistral_datasets_src.keys())
    mistral_dataset_src_values = list(mistral_datasets_src.values())

    llama_dataset_no_src_keys = list(llama_datasets_no_src.keys())
    llama_dataset_no_src_values = list(llama_datasets_no_src.values())

    llama_dataset_src_keys = list(llama_datasets_src.keys())
    llama_dataset_src_values = list(llama_datasets_src.values())

    for n in range(len(mistral_datasets_no_src)):
        mistral_df_no_src = mistral_dataset_no_src_values[n].copy()
        mistral_df_src = mistral_dataset_src_values[n].copy()

        llama_df_no_src = llama_dataset_no_src_values[n].copy()
        llama_df_src = llama_dataset_src_values[n].copy()

        mistral_no_src_correct = (
            mistral_df_no_src['mistral_verdict'] == mistral_df_no_src['verdict']).astype(int)
        mistral_src_correct = (
            mistral_df_src['mistral_verdict'] == mistral_df_src['verdict']).astype(int)

        llama_no_src_correct = (
            llama_df_no_src['llama_verdict'] == llama_df_no_src['verdict']).astype(int)
        llama_src_correct = (
            llama_df_src['llama_verdict'] == llama_df_src['verdict']).astype(int)

        mistral_table = pd.crosstab(
            mistral_no_src_correct, mistral_src_correct)
        llama_table = pd.crosstab(llama_no_src_correct, llama_src_correct)

        mistral_result = mcnemar(mistral_table, exact=True)
        llama_result = mcnemar(llama_table, exact=True)

        print(
            f"McNemar's test statistic for {mistral_dataset_no_src_keys[n]} vs {mistral_dataset_src_keys[n]}: {mistral_result.statistic}, p-value: {mistral_result.pvalue}")
        print(
            f"McNemar's test statistic for {llama_dataset_no_src_keys[n]} vs {llama_dataset_src_keys[n]}: {llama_result.statistic}, p-value: {llama_result.pvalue}")
        if mistral_result.pvalue < 0.05:
            print("Mistral Difference is statistically significant (p < 0.05)\n\n")
        elif mistral_result.pvalue > 0.05:
            print("No statistically significant difference for Mistral (p >= 0.05)\n\n")

        if llama_result.pvalue < 0.05:
            print("Llama Difference is statistically significant (p < 0.05)\n\n")
        elif llama_result.pvalue > 0.05:
            print(
                "No statistically significant difference for Llama's source (p >= 0.05)\n\n")


mcnemar_calc()
# mcnemar_calc_source()


def bowker_test(y_pred1, y_pred2, labels=None):
    if labels is None:
        labels = np.unique(np.concatenate([y_pred1, y_pred2]))
    n = len(labels)
    table = np.zeros((n, n), dtype=int)
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    for a, b in zip(y_pred1, y_pred2):
        i = label_to_idx[a]
        j = label_to_idx[b]
        table[i, j] += 1

    stat = 0
    df = 0
    for i in range(n):
        for j in range(i+1, n):
            nij = table[i, j]
            nji = table[j, i]
            if nij + nji > 0:
                stat += (nij - nji) ** 2 / (nij + nji)
                df += 1

    p_value = 1 - chi2.cdf(stat, df)
    return stat, df, p_value


# for n in range(len(mistral_datasets)):
#     mistral_dataset = mistral_datasets[n]
#     llama_dataset = llama_datasets[n]

#     y_pred1 = mistral_dataset['mistral_verdict'].values
#     y_pred2 = llama_dataset['llama_verdict'].values

#     stat, df, p = bowker_test(y_pred1=y_pred1, y_pred2=y_pred2)

#     print(f"Bowker test statistic: {stat:.4f}, df: {df}, p-value: {p:.4f}")
#     if p < 0.05:
#         print("Statistically significant difference (p < 0.05)")
#     else:
#         print("No statistically significant difference")


dataset_dict = {
    # "Politifact": mistral_df_zs,  # or any ground truth df
    "Mistral ZS": mistral_df_zs,
    "Mistral ZS Source": mistral_df_zs_src,
    "Mistral FS": mistral_df_fs,
    "Mistral FS Source": mistral_df_fs_src,
    "Mistral COT": mistral_df_cot,
    "Mistral COT Source": mistral_df_cot_src,
    "Llama ZS": llama_df_zs,
    "Llama ZS Source": llama_df_zs_src,
    "Llama FS": llama_df_fs,
    "Llama FS Source": llama_df_fs_src,
    "Llama COT": llama_df_cot,
    "Llama COT Source": llama_df_cot_src
}

feature_cols = ['average_views', 'topic', 'party', 'politician_name']

mi_matrix = pd.DataFrame(index=feature_cols, columns=dataset_dict.keys())


def plot_gini_entropy(dataset_mapping=None, save_path="diversity_metrics.png"):
    """
    Compute Shannon entropy (base 2) and Gini impurity for each dataset's label column,
    normalize both to 0-100 (%) by their theoretical maxima for the number of classes
    and plot them as grouped bars.

    dataset_mapping: optional dict name -> (df, label_col). If None, uses `dataset_dict`
                     and infers label column from the dataset name.
    """
    if dataset_mapping is None:
        mapping = {}
        for name, df in dataset_dict.items():
            if "Politifact" in name:
                col = 'verdict'
            elif "Mistral" in name:
                col = 'mistral_verdict'
            elif "Llama" in name:
                col = 'llama_verdict'
            else:
                col = 'verdict'
            mapping[name] = (df, col)
    else:
        mapping = dataset_mapping

    rows = []
    for name, (df, col) in mapping.items():
        if col not in df.columns:
            continue
        counts = df[col].value_counts().sort_index()
        # consider only non-mixed classes if present: keep classes that appear (0,1,2)
        probs = (counts / counts.sum()).values
        # number of distinct classes present (use max 3)
        k = max(2, len(probs))  # avoid log2(1)
        # Shannon entropy in bits
        ent_bits = entropy(probs, base=2) if probs.sum() > 0 else 0.0
        # Gini impurity
        gini = 1.0 - np.sum(probs ** 2) if probs.size > 0 else 0.0
        # normalize to percentage of theoretical max for k classes
        ent_max = math.log2(k)
        ent_pct = (ent_bits / ent_max * 100.0) if ent_max > 0 else 0.0
        gini_max = 1.0 - 1.0 / k
        gini_pct = (gini / gini_max * 100.0) if gini_max > 0 else 0.0

        rows.append({
            'dataset': name,
            'entropy_bits': ent_bits,
            'entropy_pct_of_max': ent_pct,
            'gini': gini,
            'gini_pct_of_max': gini_pct,
            'n': int(counts.sum())
        })

    if not rows:
        print("No datasets to plot in plot_gini_entropy()")
        return

    df_metrics = pd.DataFrame(rows).sort_values('dataset')
    y = np.arange(len(df_metrics))
    height = 0.35

    fig, ax = plt.subplots(figsize=(10, max(6, len(df_metrics)*0.4)))
    bars1 = ax.barh(y - height/2, df_metrics['entropy_pct_of_max'], height,
                    label='Entropy (% of max)', color='#6ea8fe', edgecolor='white')
    bars2 = ax.barh(y + height/2, df_metrics['gini_pct_of_max'], height,
                    label='Gini (% of max)', color='#ffb86b', edgecolor='white')

    ax.set_yticks(y)
    ax.set_yticklabels(df_metrics['dataset'], fontsize=10)
    ax.set_xlabel("Normalized diversity (% of theoretical max)", fontsize=12)
    ax.set_title("Label Diversity: Shannon Entropy and Gini Impurity (normalized)",
                 fontsize=14, weight='bold')
    ax.set_xlim(0, 110)
    ax.legend(loc='upper right')

    # annotate bars with raw and normalized values
    for b, ent_b, ent_bits, n in zip(bars1, df_metrics['entropy_pct_of_max'], df_metrics['entropy_bits'], df_metrics['n']):
        w = b.get_width()
        ax.text(w + 1, + b.get_y() + b.get_height()/2,
                f"{ent_b:.1f}%", va='center', fontsize=8)

    for b, gini_b, gini_raw in zip(bars2, df_metrics['gini_pct_of_max'], df_metrics['gini']):
        w = b.get_width()
        ax.text(w + 1, + b.get_y() + b.get_height()/2,
                f"{gini_b:.1f}%", va='center', fontsize=8)

    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.show()
    plt.close(fig)


plot_gini_entropy()

"""Check for regurgitation"""


def plot_politifact_mentions():
    def count_politifact_mentions(df, column):
        count = 0
        for reason in df[column].dropna():
            if re.search('politifact', str(reason), re.IGNORECASE):
                count += 1
        return count

    mapping = {
        'Mistral ZS': (mistral_df_zs, 'mistral_reason'),
        'Mistral ZS+src': (mistral_df_zs_src, 'mistral_reason'),
        'Mistral FS': (mistral_df_fs, 'mistral_reason'),
        'Mistral FS+src': (mistral_df_fs_src, 'mistral_reason'),
        'Mistral COT': (mistral_df_cot, 'mistral_reason'),
        'Mistral COT+src': (mistral_df_cot_src, 'mistral_reason'),
        'Llama ZS': (llama_df_zs, 'llama_reason'),
        'Llama ZS+src': (llama_df_zs_src, 'llama_reason'),
        'Llama FS': (llama_df_fs, 'llama_reason'),
        'Llama FS+src': (llama_df_fs_src, 'llama_reason'),
        'Llama COT': (llama_df_cot, 'llama_reason'),
        'Llama COT+src': (llama_df_cot_src, 'llama_reason')
    }

    models = []
    mention_counts = []
    percents = []
    totals = []

    for name, (df, col) in mapping.items():
        if col not in df.columns:
            cnt = 0
        else:
            cnt = count_politifact_mentions(df, col)
        total = len(df)
        pct = (cnt / total * 100) if total > 0 else 0.0

        models.append(name)
        mention_counts.append(cnt)
        percents.append(pct)
        totals.append(total)

    fig, ax = plt.subplots(figsize=(10, 8))
    y_pos = np.arange(len(models))
    bars = ax.barh(y_pos, mention_counts, color='#ef9a9a', edgecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(models)
    ax.set_xlabel('Number of PolitiFact Mentions')
    ax.set_title(
        'PolitiFact Mentions in Reasoning by Model and Prompt Type (raw counts)')
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + max(1, max(mention_counts) * 0.01), bar.get_y() + bar.get_height() / 2,
                f'{int(width)} (n={totals[i]})', ha='left', va='center')
    plt.tight_layout()
    plt.savefig('politifact_mentions_counts.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(y_pos, percents, color='#6ea8fe', edgecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(models)
    ax.set_xlabel('Percentage of Statements mentioning "PolitiFact" (%)')
    ax.set_title('PolitiFact Mentions as % of Total Statements by Dataset')
    # small left padding if all zeros
    ax.set_xlim(0, max(5, max(percents) * 1.15))
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + max(0.1, max(percents) * 0.01), bar.get_y() + bar.get_height() / 2,
                f'{width:.2f}% (n={mention_counts[i]})', ha='left', va='center')
    plt.tight_layout()
    plt.savefig('politifact_mentions_percent.png',
                dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


plot_politifact_mentions()
