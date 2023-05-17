


class VariationalLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(VariationalLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features))
        
        self.softplus = nn.Softplus()
        self.reset_parameters()
        
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight_mu.size(1))
        self.weight_mu.data.uniform_(-stdv, stdv)
        self.weight_rho.data.uniform_(-stdv, stdv)
        self.bias_mu.data.uniform_(-stdv, stdv)
        self.bias_rho.data.uniform_(-stdv, stdv)
        
    def forward(self, input):
        weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        bias_sigma = torch.log1p(torch.exp(self.bias_rho))
        
        weight_epsilon = torch.normal(mean=0., std=1., size=weight_sigma.size(), device=input.device)
        bias_epsilon = torch.normal(mean=0., std=1., size=bias_sigma.size(), device=input.device)
        
        weight_sample = self.weight_mu + weight_epsilon * weight_sigma
        bias_sample = self.bias_mu + bias_epsilon * bias_sigma
        
        output = F.linear(input, weight_sample, bias_sample)
        return output
    
    def kl_divergence(self):
        weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        bias_sigma = torch.log1p(torch.exp(self.bias_rho))
        
        kl_div = -0.5 * torch.sum(1 + torch.log(weight_sigma.pow(2)) - self.weight_mu.pow(2) - weight_sigma.pow(2))
        kl_div += -0.5 * torch.sum(1 + torch.log(bias_sigma.pow(2)) - self.bias_mu.pow(2) - bias_sigma.pow(2))
        return kl_div



class MultiClassClassifier(nn.Module):
    def __init__(self, input_dim=10, classes=17, dropout_rate=0.1, hidden_layer_sizes=[8]):
        super(MultiClassClassifier, self).__init__()
        self.input_dim = input_dim
        self.classes = classes
        self.dropout_rate = dropout_rate
        self.hls = hidden_layer_sizes

        def element(in_channel, out_channel, is_last=False):
            layers = [
                VariationalLayer(in_channel, out_channel),
                nn.BatchNorm1d(out_channel),  # Add batch normalization
                nn.LeakyReLU(0.02),
            ]
            if not is_last:
                layers.append(nn.Dropout(self.dropout_rate))  # Add dropout
            return layers

        if len(self.hls) == 1:
            encoder = element(self.input_dim, self.hls[0], is_last=True)
            encoder += [VariationalLayer(self.hls[-1], classes)]
        else:
            encoder = element(self.input_dim, self.hls[0])
            for i in range(len(self.hls) - 1):
                is_last = (i == len(self.hls) - 2)
                encoder += element(self.hls[i], self.hls[i + 1], is_last)
            encoder += [VariationalLayer(self.hls[-1], classes)]

        self.encode = nn.Sequential(*encoder)
        self.apply(weights_init)

    def encoded(self, x):
        return self.encode(x)

    def forward(self, x):
        en = self.encoded(x)
        return en

    def predict(self, x):
        # Get predicted scores
        scores = self.forward(x)
        # Get predicted class indices
        class_indices = scores.argmax(dim=1)
        return class_indices



def train_nn(model, optimizer, label, train_loader, test_loader, n_epoch, criterion, patience=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    

    avg_train_loss = []
    avg_test_loss = []
    best_test_loss = float('inf')
    best_model_state = None
    best_epoch = 0
    patience_counter = 0

    for epoch in range(n_epoch):
        model.train()
        t = time.time()
        train_loss = []
        for i, (data, labels) in enumerate(train_loader):
            x = data.to(device)
            y = labels.to(device)
            train_output = model(x)
            loss = criterion(train_output, y)

            # Add KL divergence
            kl_div = 0.
            for module in model.modules():
                if isinstance(module, VariationalLayer):
                    kl_div += module.kl_divergence()
            loss += kl_div / len(train_loader.dataset)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss.append(loss.detach().item())
        
        # Validation
        model.eval()
        test_loss = []
        with torch.no_grad():
            for i, (data, labels) in enumerate(test_loader):
                x = data.to(device)
                y = labels.to(device)
                test_output = model(x)
                loss = criterion(test_output, y)
                test_loss.append(loss.detach().item())

        # Logging
        avg_train = sum(train_loss) / len(train_loss)
        avg_test = sum(test_loss) / len(test_loss)
        avg_train_loss.append(avg_train)
        avg_test_loss.append(avg_test)
        
        training_time = time.time() - t
 
        print(f'[{epoch+1:03}/{n_epoch:03}] train_loss: {avg_train:.6f}, test_loss: {avg_test:.6f}, time: {training_time:.2f} s')

        # Early stopping
        if avg_test < best_test_loss:
            best_test_loss = avg_test
            best_epoch = epoch
            patience_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())  # Save the best model weights

        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Validation loss hasn't improved for {patience} epochs. Stopping early.")
                break

    return train_output, test_output, avg_train_loss, avg_test_loss, best_model_state



def predict_with_uncertainty(model, input_data, n_iterations=100):
    model.eval()
    output_list = []
    for i in range(n_iterations):
        with torch.no_grad():
            output = model(input_data)
            output_list.append(torch.nn.functional.softmax(output, dim=1).detach().cpu().numpy())

    output_list = np.array(output_list)
    
    # Calculate mean and standard deviation
    prediction_mean = output_list.mean(axis=0)
    prediction_stddev = output_list.std(axis=0)
    return prediction_mean, prediction_stddev


def neuralnetwork(df, name, WhichClassifier, hidden_layer_sizes, lr, wd, dr, epochs, n, balanced): 

    # Your existing code...

    # train model using pre-defined function
    train_output, test_output, train_loss, test_loss, best_model_state = train_nn(model, optimizer, label, feature_loader, test_loader, epochs, criterion)
    np.savez('nn_parametermatrix/' + name + '_nn_loss.npz', train_loss = train_loss, test_loss = test_loss)
    
    model_path = 'nn_parametermatrix/' + name + "_nn_params.pt"
    save_model_nn(model, optimizer, model_path, best_model_state)

    # Predict classes for entire training and test datasets with uncertainties
    train_pred_mean, train_pred_stddev = predict_with_uncertainty(model, feature_dataset.x)
    test_pred_mean, test_pred_stddev = predict_with_uncertainty(model, test_dataset.x)

    # Get the most probable classes
    train_pred_classes = np.argmax(train_pred_mean, axis=1)
    test_pred_classes = np.argmax(test_pred_mean, axis=1)

    # Calculate classification metrics
    train_report = classification_report(train_data_y, train_pred_classes, target_names = sort_mapping.values(), zero_division=0)
    test_report = classification_report(test_data_y, test_pred_classes, target_names = sort_mapping.values(), zero_division=0)

    return train_pred_classes, test_pred_classes, train_report, test_report, best_model_state
