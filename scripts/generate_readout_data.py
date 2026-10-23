"""Generate the synthetic stimuli of experiment 1 (readout and probes).

Sets:
  main         : 800 counting + 800 fixed-code text items (fonts 8-14 px)
  text-hard    : 800 fixed-code text items at 5-8 px
  random-codes : 1,000 yes/no items with a random code per image at 5-8 px
"""
import argparse

from jevision.stimuli.readout_tasks import FIXED_CLASS_SETS, generate_fixed_class_set, generate_random_code_set


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--set", required=True, choices=[*FIXED_CLASS_SETS, "random-codes"])
    p.add_argument("--out", required=True, help="output directory (img/ and items.json are written here)")
    args = p.parse_args()
    if args.set == "random-codes":
        items = generate_random_code_set(args.out)
    else:
        items = generate_fixed_class_set(args.out, **FIXED_CLASS_SETS[args.set])
    print(f"{len(items)} items -> {args.out}")


if __name__ == "__main__":
    main()
