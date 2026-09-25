import argparse
import json

import torch

from clamp_diffusion.eval import SGRScorer, score_arm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_file", required=True)
    parser.add_argument("--base_root", required=True, help="baseline's validation_images dir")
    parser.add_argument("--immunized_root", required=True, help="immunized model's validation_images dir")
    parser.add_argument("--epochs", nargs="+", type=int, required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    scorer = SGRScorer(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    results = score_arm(scorer, args.csv_file, args.base_root, args.immunized_root, args.epochs)

    for epoch, value in results.items():
        print(f"epoch {epoch}: SGR_G = {value:.2f}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
