import argparse
import json
from preprocessor.preprocessor import Preprocessor


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="path to preprocess.yaml")
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        s = f.read()
    config = json.loads(s)

    preprocessor = Preprocessor(config)
    preprocessor.build_from_path()
