print("confusion matrix:")
for y_idx in range(4):
    print("---------------- for y_" + str(y_idx+1))
    print(confusion_matrix(y_test.values[:, y_idx], y_pred[:, y_idx]))
    
print("\nclassification report:")
print(classification_report(y_test, y_pred))
print("AUC = {}".format(roc_auc_score(y_test, y_pred, average='weighted')))