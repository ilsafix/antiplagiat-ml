import os
import ast
import argparse
import torch
from torch import nn

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

class Comparison:
    def __init__(self, input_file, output_file, model_file):
        self.input_file = input_file
        self.output_file = output_file
        self.model_file = model_file
        
        if not os.path.exists(model_file):
            print("Model file not found!")
            raise ValueError('Model file not found!')

        self.model = AntiPlagiatNet()
        self.model.load_state_dict(torch.load(model_file))
        self.model.eval()

    def compare(self):
        with open(self.input_file, 'r', encoding="utf8") as input_f:
            with open(self.output_file, 'w', encoding="utf8") as output_f:
                # Processing data from input_file and writing results to output_file
                for line in input_f:
                    file1, file2 = line.strip().split()
                    # Reading text from doc1 and doc2
                    code1 = self.read_text_from_file(file1)
                    code2 = self.read_text_from_file(file2)
                    # Feature extracting
                    code1_features = torch.FloatTensor(self.extract_features(code1))
                    code2_features = torch.FloatTensor(self.extract_features(code2))
                    codes_features = torch.hstack((code1_features, code2_features))
                    # Similarity prediction
                    model_pred = self.model(codes_features)
                    # Writing similarity score to output_file
                    output_f.write(f"{model_pred.item():.3f}\n")

    def read_text_from_file(self, file_path):
        with open(file_path, 'r', encoding="utf8") as f:
            return f.read()
    
    def extract_features(self, code):
        features_num = 9
        features = [0] * features_num

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str, help="Path to the input file")
    parser.add_argument("output_file", type=str, help="Path to the output file")
    parser.add_argument("--model", type=str, required=True, help="Path to the model file")

    args = parser.parse_args()
    compare = Comparison(args.input_file, args.output_file, args.model)
    compare.compare()