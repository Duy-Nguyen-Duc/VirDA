import argparse

from yacs.config import CfgNode as CN

from burn_in import run_bi_step
from domain_adapt import run_da_step
from utils import setup


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the YAML config file"
    )
    args, _ = parser.parse_known_args()
    cfg = CN(new_allowed=True)
    cfg.merge_from_file(args.config)
    exp_save_dir = setup(cfg)

    # Run burn-in step
    best_ckpt = run_bi_step(cfg, exp_save_dir=exp_save_dir)
    print("Loading best checkpoint from burn-in step:", best_ckpt)
    # Run domain adaptation step
    run_da_step(cfg, exp_save_dir=exp_save_dir, best_bi_ckpt=best_ckpt)


if __name__ == "__main__":
    main()
