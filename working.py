def nested_cross_val(df, name, hidden_layer_sizes, learning_rates, weight_decays, epochs, n_splits_outer, n_splits_inner):

    oxides = ['SiO2', 'TiO2', 'Al2O3', 'FeOt', 'MnO', 'MgO', 'CaO', 'Na2O', 'K2O', 'Cr2O3']
    label = ['Mineral']

    X = df[oxides].fillna(0).to_numpy()
    y = pd.Categorical(df['Mineral']).codes

    mapping = dict(zip(pd.Categorical(df['Mineral']).codes, pd.Categorical(df['Mineral'])))
    sort_mapping= dict(sorted(mapping.items(), key=lambda item: item[0])) 

    outer_cv = StratifiedKFold(n_splits=n_splits_outer, shuffle=True, random_state=42)
    inner_cv = StratifiedKFold(n_splits=n_splits_inner, shuffle=True, random_state=42)

    train_reports = []
    test_reports = []

    for train_idx, test_idx in outer_cv.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        best_val_score = 0
        best_hidden_layers = None
        best_learning_rate = None
        best_weight_decay = None
        best_epochs = None 
        nn_count = 0 

        for hidden_layers in hidden_layer_sizes:
            for lr in learning_rates:
                for wd in weight_decays:
                    for ep in epochs: 
                        val_scores = []

                        for train_inner_idx, test_inner_idx in inner_cv.split(X_train, y_train):

                            X_train_inner, X_test_inner = X_train[train_inner_idx], X_train[test_inner_idx]
                            y_train_inner, y_test_inner = y_train[train_inner_idx], y_train[test_inner_idx]
                            
                            print('nn_count = {}, hidden_layers = {}, learning rate = {:.4f}, weight_decay = {:.4f}, epochs = {}'.format(nn_count, hidden_layers, lr, wd, ep))
                            train_pred_classes_inner, test_pred_classes_inner, train_report_inner, test_report_inner = neuralnetwork_kfold(X_train_inner, y_train_inner, X_test_inner, y_test_inner, name, hidden_layers, lr, wd, ep)

                            val_f1_score = f1_score(y_test_inner, test_pred_classes_inner, average='weighted')
                            val_scores.append(val_f1_score)

                            nn_count += 1
                        avg_val_score = np.mean(val_scores)

                        if avg_val_score > best_val_score:
                            best_val_score = avg_val_score
                            best_hidden_layers = hidden_layers
                            best_learning_rate = lr
                            best_weight_decay = wd
                            best_epochs = ep


        train_pred_classes, test_pred_classes, train_report, test_report = neuralnetwork_kfold(X_train, y_train, X_test, y_test, name, best_hidden_layers, best_learning_rate, best_weight_decay, best_epochs)

        train_reports.append({
            'true_labels': y_train,
            'predicted_labels': train_pred_classes,
            'label_names': list(sort_mapping.values())
        })
        test_reports.append({
            'true_labels': y_test,
            'predicted_labels': test_pred_classes,
            'label_names': list(sort_mapping.values())
        })

    avg_train_report = average_classification_reports(train_reports)
    avg_test_report = average_classification_reports(test_reports)

    best_params = np.array([best_hidden_layers, best_learning_rate, best_weight_decay, best_epochs])

    return avg_train_report, avg_test_report, best_params


def neuralnetwork_kfold(X_train, y_train, X_test, y_test, name, hidden_layer_sizes, learning_rate, weight_decay, epochs): 
    ss = StandardScaler()
    array_norm_train = ss.fit_transform(X_train)
    array_norm_test = ss.transform(X_test)

    # Define datasets to be used with PyTorch - see autoencoder file for details
    feature_dataset = FeatureDataset_nn(array_norm_train, y_train)
    test_dataset = FeatureDataset_nn(array_norm_test, y_test)

    mapping = dict(zip(pd.Categorical(df['Mineral']).codes, pd.Categorical(df['Mineral'])))
    sort_mapping= dict(sorted(mapping.items(), key=lambda item: item[0])) 

    # Autoencoder params:
    # lr = 2.5e-3
    # wd = 1e-4 
    batch_size = 256
    input_size = len(feature_dataset.__getitem__(0)[0])

    # Define data loaders
    feature_loader = DataLoader(feature_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    np.savez('nn_parametermatrix/' + name + '_nn_features.npz', feature_loader = feature_loader, test_loader = test_loader)

    # Initialize model
    model = MultiClassClassifier(input_dim=input_size, hidden_layer_sizes=hidden_layer_sizes).to(device) # dropout_rate = dr

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer=torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay = weight_decay)

    model_path = 'nn_parametermatrix/' + name + "_nn_params.pt"
    save_model(model, optimizer, model_path)

    # Train model using pre-defined function
    train_output, test_output, train_loss, test_loss = train_nn(model, optimizer, label, feature_loader, test_loader, epochs, criterion)
    np.savez('nn_parametermatrix/' + name + '_nn_loss.npz', train_loss = train_loss, test_loss = test_loss)

    # Predict classes for entire training and test datasets
    train_pred_classes = model.predict(feature_dataset.x)
    test_pred_classes = model.predict(test_dataset.x)

    # Calculate classification metrics
    train_report = classification_report(y_train, train_pred_classes, target_names = sort_mapping.values(), zero_division=0)
    test_report = classification_report(y_test, test_pred_classes, target_names = sort_mapping.values(), zero_division=0)

    fig, ax = plt.subplots(1, 1, figsize = (8, 8))
    ax.plot(np.linspace(1, epochs, epochs), train_loss, '.-', label = 'Train Loss')
    ax.plot(np.linspace(1, epochs, epochs), test_loss, '.-', label = 'Test Loss')
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.legend(prop={'size': 10})


    return train_pred_classes, test_pred_classes, train_report, test_report 


# Define a function to average classification reports

def average_classification_reports(reports):
    metrics = []
    
    for report in reports:
        precision, recall, fscore, _ = precision_recall_fscore_support(
            report['true_labels'],
            report['predicted_labels'],
            average=None
        )
        metrics.append(np.array([precision, recall, fscore]))

    # Calculate the mean values of precision, recall, and F1-score
    avg_metrics = np.mean(metrics, axis=0)

    # Create a DataFrame to store the results
    avg_report_df = pd.DataFrame(
        avg_metrics.T,
        columns=['precision', 'recall', 'fscore'],
        index=reports[0]['label_names']
    )

    return avg_report_df