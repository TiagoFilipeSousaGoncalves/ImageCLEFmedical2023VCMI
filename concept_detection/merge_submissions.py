# Imports
import argparse
import numpy as np
import pandas as pd



# CLI Arguments
# Command Line Interface
parser = argparse.ArgumentParser()
parser.add_argument("--submission_csv_1", type=str, required=True, help="Submission file #1.")
parser.add_argument("--submission_csv_2", type=str, required=True, help="Submission file #2.")
args = parser.parse_args()

# Submission file nr 1
submission_csv_1 = args.submission_csv_1
submission_csv_1_df = pd.read_csv(submission_csv_1, header=None, sep="|")
submission_1_images = np.array(submission_csv_1_df.iloc[:, 0])
submission_1_concepts = np.array(submission_csv_1_df.iloc[:, 1])
ml_index = np.argsort(submission_1_images)
submission_1_images = submission_1_images[ml_index]
submission_1_concepts = submission_1_concepts[ml_index]

# Submission file nr 2
submission_csv_2 = args.submission_csv_2
submission_csv_2_df = pd.read_csv(submission_csv_2, header=None, sep="|")
submission_csv_2_images = np.array(submission_csv_2_df.iloc[:, 0])
submission_csv_2_concepts = np.array(submission_csv_2_df.iloc[:, 1])
ml_index = np.argsort(submission_csv_2_images)
submission_csv_2_images = submission_csv_2_images[ml_index]
submission_csv_2_concepts = submission_csv_2_concepts[ml_index]

final_concepts_NAN = []
nan_count = 0
for idx in range(len(submission_1_images)):
    if submission_1_images[idx] != submission_csv_2_images[idx]:
        print('Different image order')
        exit(-1)

    if pd.isna(submission_1_concepts[idx]):
        nan_count += 1
        final_concepts_NAN.append(submission_csv_2_concepts[idx])
    else:
        final_concepts_NAN.append(submission_1_concepts[idx])

eval_set = dict()
eval_set["ID"] = submission_csv_2_images
eval_set["cuis"] = final_concepts_NAN

evaluation_df = pd.DataFrame(data=eval_set)
evaluation_df.to_csv('merged_submissions.csv', sep='|', index=False, header=False)

print('NAN count: ' + str(nan_count))
