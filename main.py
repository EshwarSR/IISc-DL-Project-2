import argparse
import utils
import torch
import os
repo_path = os.path.dirname(os.path.abspath(__file__))


def write_software_1_output(test_file):
    # Writing the Software1.txt file
    with open(test_file, "r") as infile:
        nums = infile.read().strip().split()
        nums = [int(item) for item in nums]

        with open(repo_path + "/Software1.txt", "w") as outfile:
            for num in nums:
                out = utils.rule_output(num)
                if out == 0:
                    out_str = "fizz"
                elif out == 1:
                    out_str = "buzz"
                elif out == 2:
                    out_str = "fizzbuzz"
                else:
                    out_str = str(num)
                outfile.write(out_str+"\n")


def write_software_2_output(test_file):
    dataset = utils.InferenceDataset(test_file)

    layers = [
        torch.nn.Linear(10, 100),
        torch.nn.Tanh(),
        torch.nn.Linear(100, 100),
        torch.nn.Tanh(),
        torch.nn.Linear(100, 4)
    ]
    model = torch.nn.Sequential(*layers)
    model.load_state_dict(torch.load("model/2L_100H_100H_tanh.pt"))
    model.eval()
    nums = dataset[:][0]
    features = dataset[:][1]
    out = model(features)
    _, classes = torch.max(out, 1)
    classes = classes.numpy().tolist()

    with open(repo_path + "/Software2.txt", "w") as outfile:
        for num, pred_class in zip(nums, classes):
            if pred_class == 0:
                out_str = "fizz"
            elif pred_class == 1:
                out_str = "buzz"
            elif pred_class == 2:
                out_str = "fizzbuzz"
            else:
                out_str = str(num)
            outfile.write(out_str+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Main Program')
    parser.add_argument('-t', '--test-data',
                        help='Test File', required=True)
    args = parser.parse_args()
    print("Reading test data from file:", args.test_data)

    # Writing Software1.txt file
    write_software_1_output(args.test_data)
    print("Done writing Software1.txt file")

    # Writing Software2.txt file
    write_software_2_output(args.test_data)
    print("Done writing Software2.txt file")
