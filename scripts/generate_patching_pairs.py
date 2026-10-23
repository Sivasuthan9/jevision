"""Generate the counterfactual image pairs of experiment 2 (cache patching).

Sets:
  main         : 4 skills x 200 pairs (presence, text, count, spatial)
  size-control : 4 skills x 2 evidence sizes x 100 pairs
"""
import argparse

from jevision.stimuli.counterfactual import PAIR_SETS, generate_pair_set


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--set", required=True, choices=list(PAIR_SETS))
    p.add_argument("--out", required=True, help="output directory (img/ and items.json are written here)")
    p.add_argument("--pairs-per-skill", type=int, default=None, help="override the set's default (for quick tests)")
    args = p.parse_args()
    items = generate_pair_set(args.out, args.set, args.pairs_per_skill)
    print(f"{len(items)} pairs -> {args.out}")


if __name__ == "__main__":
    main()
