#!/usr/bin/env python3
import random
import argparse
import sys

def generate_input(num_data, num_queries, num_attrs, attr_min, attr_max, minK, maxK, num_labels):
    """
    Generate input text matching the format expected by the MPI KNN program.
    """
    lines = []
    lines.append(f"{num_data} {num_queries} {num_attrs}")
    
    for i in range(num_data):
        label = random.randint(0, num_labels - 1)
        attrs = [f"{random.uniform(attr_min, attr_max):.6f}" for _ in range(num_attrs)]
        lines.append(f"{label} " + " ".join(attrs))
    
    for i in range(num_queries):
        k = random.randint(minK, min(maxK, num_data))
        attrs = [f"{random.uniform(attr_min, attr_max):.6f}" for _ in range(num_attrs)]
        lines.append("Q " + f"{k} " + " ".join(attrs))
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate input for MPI KNN program.")
    parser.add_argument("--num_data", type=int, required=True, help="Number of data points")
    parser.add_argument("--num_queries", type=int, required=True, help="Number of queries")
    parser.add_argument("--num_attrs", type=int, required=True, help="Number of attributes per point/query")
    parser.add_argument("--min", type=float, required=True, help="Minimum value for attribute range")
    parser.add_argument("--max", type=float, required=True, help="Maximum value for attribute range")
    parser.add_argument("--minK", type=int, required=True, help="Minimum K for queries")
    parser.add_argument("--maxK", type=int, required=True, help="Maximum K for queries")
    parser.add_argument("--num_labels", type=int, required=True, help="Number of possible labels")
    parser.add_argument("--output", type=str, required=True, help="Output file name")
    parser.add_argument("--seed", type=int, required=False, default=42,
                        help="Seed for RNG ( Default : 42 )")
    
    args = parser.parse_args()

    # Basic validation
    if args.num_data <= 0:
        sys.exit("Error: --num_data must be positive")
    if args.num_queries <= 0:
        sys.exit("Error: --num_queries must be positive")
    if args.num_attrs <= 0:
        sys.exit("Error: --num_attrs must be positive")
    if args.min >= args.max:
        sys.exit("Error: --min must be less than --max")
    if args.minK <= 0:
        sys.exit("Error: --minK must be positive")
    if args.minK > args.maxK:
        sys.exit("Error: --minK must be ≤ --maxK")
    if args.minK > args.num_data:
        sys.exit("Error: --minK must be ≤ --num_data")
    if args.num_labels <= 0:
        sys.exit("Error: --num_labels must be positive")
   
    random.seed(args.seed)

    text = generate_input(
        args.num_data,
        args.num_queries,
        args.num_attrs,
        args.min,
        args.max,
        args.minK,
        args.maxK,
        args.num_labels
    )

    with open(args.output, "w") as f:
        f.write(text + "\n")

    print(f"✅ Input file '{args.output}' generated successfully.")


if __name__ == "__main__":
    main()
