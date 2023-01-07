import os
import ast
import argparse
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

class ModelTrain:
    def __init__(self, source_files, plagiat_files, plagiat2_files, model_file):
        self.source_files = source_files
        self.plagiat_files = plagiat_files
        self.plagiat2_files = plagiat2_files
        self.model_file = model_file

    def run(self):
        source_features = np.array(self.get_dataset(self.source_files))
        plagiat_features = np.array(self.get_dataset(self.plagiat_files))
        plagiat2_features =  np.array(self.get_dataset(self.plagiat2_files))

        # Train dataset
        X_train1 = np.hstack((source_features, plagiat_features)) # Plagiat
        X_train2 = np.hstack((plagiat_features, source_features))
        X_train3 = np.hstack((source_features, source_features))
        X_train4 = np.hstack((plagiat_features, plagiat_features))
        X_train = np.vstack((X_train1, X_train2, X_train3, X_train4))
        Y_ones = np.ones(X_train1.shape[0])
        Y_train = np.hstack((Y_ones, Y_ones, Y_ones, Y_ones))

        Y_zeros = np.zeros(X_train1.shape[0])
        rand_idx = np.random.choice(len(source_features), len(source_features), False) # Not plagiat
        for i in range(10):
            rand_idx = np.random.choice(len(source_features), len(source_features), False)
            X_train5 = np.hstack((source_features, plagiat_features[rand_idx]))
            X_train6 = np.hstack((plagiat_features[rand_idx], source_features))
            X_train = np.vstack((X_train, X_train5, X_train6))
            Y_train = np.hstack((Y_train, Y_zeros, Y_zeros))
        
        
        X_mean = np.mean(X_train, axis=0)
        X_std = np.std(X_train, axis=0)
        train_dataset = SomeDataset(X_train, Y_train[:, np.newaxis], X_mean, X_std)
        self.train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)

        # Validation dataset
        X_val = np.hstack((source_features, plagiat2_features))
        Y_val = np.ones(X_val.shape[0])
        val_dataset = SomeDataset(X_val, Y_val[:, np.newaxis], X_mean, X_std)
        self.val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)
        
        self.model = AntiPlagiatNet()

        self.loss_fn = nn.BCELoss() # Binary classification
        learning_rate = 1e-3
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        # Model training
        self.train(n_epoch=30)
        # Model saving
        torch.save(self.model.state_dict(), self.model_file)

        print("Model saved: ", self.model_file)

    def train(self, n_epoch=20):
        for epoch in range(n_epoch):
            self.model.train(True)
            losses = []
            accuracies = []
            print("Epoch:", epoch+1)
            
            for i, batch in enumerate(self.train_dataloader):
                # Get batch
                X_batch, Y_batch = batch 
                
                # forward pass
                logits = self.model(X_batch) 
                
                # Loss calculation
                loss = self.loss_fn(logits, Y_batch) 
                losses.append(loss.item())
                
                loss.backward() # Backpropagation
                self.optimizer.step() # Update model weights
                self.optimizer.zero_grad() # Set the gradients to zero
                
                # Accuracy calculation
                model_answers = torch.round(logits)
                train_accuracy = torch.sum(Y_batch == model_answers.cpu()) / len(Y_batch)
                accuracies.append(train_accuracy)

            losses_mean = np.mean(losses)
            accuracies_mean = np.mean(accuracies)
            print("Mean train loss and accuracy on ", epoch + 1, " epoch: ", 
                  losses_mean, accuracies_mean, end='\n')
            
            # Validation accuracy calculation        
            self.eval()

    def eval(self):
        self.model.eval()
        batch_accuracies = []
        
        for i, batch in enumerate(self.val_dataloader):
            X_batch, Y_batch = batch 
            
            logits = self.model(X_batch) 
    
            model_answers = torch.round(logits)
            val_accuracy = torch.sum(Y_batch == model_answers.cpu()) / len(Y_batch)
            batch_accuracies.append(val_accuracy)

        accuracies_mean = np.mean(batch_accuracies)
        print("Mean validation accuracy: ", accuracies_mean, end='\n')

    def extract_features(self, code):
        features_num = 9
        features = np.zeros(features_num)

        features[0] = code.count("\n") + 1

        # Parse the code into an AST
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return features

        # Extract the frequency of each keyword
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                features[1] += 1
            if isinstance(node, ast.For):
                features[2] += 1
            if isinstance(node, ast.While):
                features[3] += 1
            if isinstance(node, ast.Try):
                features[4] += 1
            if isinstance(node, ast.FunctionDef):
                features[5] += 1
            if isinstance(node, ast.ClassDef):
                features[6] += 1
            if isinstance(node, ast.Import):
                features[7] += 1
            if isinstance(node, ast.Assign):
                features[8] += 1

        # Return the extracted features
        return features

    def read_text_from_file(self, file_path):
        with open(file_path, 'r', encoding="utf8") as f:
            return f.read()

    def get_dataset(self, files_dir):
        dataset = []
        if not os.path.exists(files_dir):
            print("Files directory not found!")
            raise ValueError('Files directory not found!')

        for dirpath, dirnames, filenames in os.walk(files_dir):
            for fname in filenames:
                if fname.endswith(".py"):
                    fpath = os.path.join(dirpath, fname)
                    source_code = self.read_text_from_file(fpath)
                    features_vector = self.extract_features(source_code)
                    dataset.append(features_vector)

        return dataset

class AntiPlagiatNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(18, 6)
        self.fc2 = nn.Linear(6, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))

        return x

class SomeDataset(Dataset):
    def __init__(self, X, Y, X_mean, X_std):
        self.X = torch.FloatTensor(X)
        self.Y = torch.FloatTensor(Y)
        X_mean = torch.FloatTensor(X_mean)
        X_std = torch.FloatTensor(X_std)
        self.X = (self.X - X_mean) / X_std

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return (self.X[index], self.Y[index])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("files", type=str, help="Path to the source files")
    parser.add_argument("plagiat", type=str, help="Path to the plagiat1 files")
    parser.add_argument("plagiat2", type=str, help="Path to the plagiat2 files")
    parser.add_argument("--model", type=str, required=True, help="Path to the model file")
    args = parser.parse_args()
    model_train = ModelTrain(args.files, args.plagiat, args.plagiat2, args.model)
    model_train.run()