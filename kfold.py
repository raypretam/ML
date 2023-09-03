from sklearn.metrics import precision_score, recall_score

def k_fold_cross_validation(data, attributes, k):
    kf = KFold(n_splits=k)
    accuracies = []
    macro_precisions = []
    macro_recalls = []

    for train_index, validation_index in kf.split(data):
        train_data = data.iloc[train_index]
        validation_data = data.iloc[validation_index]

        root = id3(train_data, train_data['PlayTennis'], attributes, min_samples=10)

        reduced_error_pruning(root, validation_data)

        accuracy = evaluate_tree(root, validation_data)
        accuracies.append(accuracy)

        y_true = validation_data['PlayTennis']
        y_pred = [predict_tree(root, row) for _, row in validation_data.iterrows()]
        macro_precisions.append(precision_score(y_true, y_pred, average='macro'))
        macro_recalls.append(recall_score(y_true, y_pred, average='macro'))

    mean_accuracy = np.mean(accuracies)
    mean_precision = np.mean(macro_precisions)
    mean_recall = np.mean(macro_recalls)

    print("Cross-Validation Accuracies:", accuracies)
    print("Mean Macro Accuracy:", mean_accuracy)
    print("Mean Macro Precision:", mean_precision)
    print("Mean Macro Recall:", mean_recall)

k_fold_cross_validation(data, attributes, k)
