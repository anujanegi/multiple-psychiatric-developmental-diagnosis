import pandas as pd
import numpy as np
from sklearn.metrics import average_precision_score, roc_curve, auc, hamming_loss, classification_report

def evaluate(test_annotation_file, user_submission_file, phase_codename, **kwargs):
    print("Starting Evaluation.....")

    print(kwargs["submission_metadata"])
    y_true = np.array(pd.read_csv(test_annotation_file).drop('IDs', axis=1))
    y_pred = np.array(pd.read_csv(user_submission_file, header=None))

    try:
        assert(y_true.shape == y_pred.shape)
    except:
        raise ValueError("Submitted data doesn't match the required shape of %s"%str(y_true.shape))
    
    auprc = average_precision_score(y_true, y_pred, average="macro")
    
    n_classes = y_true.shape[1]
    fpr = dict()
    tpr = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes
    auroc = auc(all_fpr, mean_tpr)
    
    brier_score = np.mean(np.mean((y_pred - y_true)**2, axis=1))
    hamming_score = hamming_loss(y_true, y_pred.round())
    
    report = classification_report(y_true, y_pred.round(), zero_division=1, output_dict=True)
    micro_avg_f1_score = report['micro avg']['f1-score']
    
    output = {}
    if phase_codename == "pseudo-test":
        print("Evaluating for Pseudo Test Phase")
        output["result"] = [
            {
                "test_split": {
                    "AUPRC": auprc,
                    "AUROC": auroc,
                    "Brier Score": brier_score,
                    "Hamming Score": hamming_score,
                    "Micro Avg F1 score": micro_avg_f1_score
                }
            }
        ]
        # To display the results in the result file
        output["submission_result"] = output["result"][0]["test_split"]
        print("Completed evaluation for Pseudo Test Phase")
        
    elif phase_codename == "dev":
        print("Evaluating for Dev Phase")
        output["result"] = [
            {
                "test_split": {
                    "AUPRC": auprc,
                    "AUROC": auroc,
                    "Brier Score": brier_score,
                    "Hamming Score": hamming_score,
                    "Micro Avg F1 score": micro_avg_f1_score
                }
            }
        ]
        # To display the results in the result file
        output["submission_result"] = output["result"][0]["test_split"]
        print("Completed evaluation for Dev Phase")
        
    print(output)
    return output