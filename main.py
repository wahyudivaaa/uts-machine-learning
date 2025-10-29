import argparse
from src.data.generate import generate_and_save
from src.models.train import train_and_select
from src.models.evaluate import evaluate_and_plot

def main():
    parser = argparse.ArgumentParser(description="UTS Polynomial Regression Project")
    parser.add_argument("--mode", choices=["generate", "train", "evaluate"], required=True)
    parser.add_argument("--n", type=int, default=400, help="Jumlah sampel untuk data sintetis")
    parser.add_argument("--noise", type=float, default=3.0, help="Besaran noise data sintetis")
    args = parser.parse_args()

    if args.mode == "generate":
        generate_and_save(n_samples=args.n, noise=args.noise)
    elif args.mode == "train":
        train_and_select()
    elif args.mode == "evaluate":
        evaluate_and_plot()
    else:
        raise ValueError("Mode tidak dikenali. Pilih generate/train/evaluate.")

if __name__ == "__main__":
    main()