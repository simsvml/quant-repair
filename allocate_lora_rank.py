import itertools
import json
import math
import sys
from quant_repair.architecture import Llama3Arch
from quant_repair import model_util as QRM


def main():
    assert len(sys.argv) == 3
    json_path = sys.argv[1]
    out_json_path = sys.argv[2]

    arch = Llama3Arch.llama3_8b()
    dims = QRM.llama3_lora.linear_dimensions(arch)

    kl_div = json.load(open(json_path))

    all_modules = sorted(key for key in kl_div.keys() if key != '')

    def calc_score(key):
        """
        Assign an importance score to module `key`.
        """
        n, m = dims.get(key)
        improvement = kl_div[''] - kl_div[key]
        # `improvement` is how much the KL-div improved when module `key` was
        # perfectly repaired by replacing the quantized version of this module
        # with the unquantized original.  `n + m` is a measure of the cost of
        # allocating rank to this module: a rank-r LoRA for this module would
        # have `n*r + m*r` weights.
        #
        # In some cases, `improvement` is actually negative, meaning the
        # qunatized layer performs better than the original.  A high negative
        # impact indicates that the module is somehow highly relevant to the
        # overall behavior, so we take the absolute value to count negative
        # impact and positive impact the same.
        return abs(improvement / (n + m))

    sorted_modules = sorted(all_modules, key = calc_score)


    ratios_low = [
        #(8, 1),
        #(16, 3),
        #(16, 1),
        #(32, 3),
        (8, 1),
        (16, 4),
        (32, 16),
    ]
    ratios_high = [
        (32, 16),
        (64, 4),
        (128, 1),
        #(32, 5),
        #(64, 3),
        #(128, 1),
    ]
    # We actually want an average rank of 32, but setting the target to 33 gets
    # us closer to that goal.
    target_average = 32.4

    avg_low = sum(r * a for r, a in ratios_low) / sum(a for r, a in ratios_low)
    avg_high = sum(r * a for r, a in ratios_high) / sum(a for r, a in ratios_high)

    assert avg_low <= target_average <= avg_high
    # We subdivide the unit interval and assign a portion to each rank.  First,
    # the region `0 <= x <= split_point` is assigned to the low ranks and
    # `split_point <= x <= 1` to the high ranks.
    split_point = 1 - (target_average - avg_low) / (avg_high - avg_low)

    print('after split, avg = ', avg_low * split_point + avg_high * (1 - split_point))

    # Now subdivide the low region among the low ranks and the high region
    # among the high ranks.
    rank_starts = []

    alpha_divisor = sum(a for r, a in ratios_low)
    alpha_sum = 0
    for rank, alpha in ratios_low:
        start = alpha_sum
        alpha_sum += alpha / alpha_divisor * split_point
        rank_starts.append((rank, start))

    alpha_divisor = sum(a for r, a in ratios_high)
    alpha_sum = split_point
    for rank, alpha in ratios_high:
        start = alpha_sum
        alpha_sum += alpha / alpha_divisor * (1 - split_point)
        rank_starts.append((rank, start))

    # Round off the floating-point subdivision to get a subdivision of the
    # discrete modules.
    rank_ranges = []
    num_modules = len(all_modules)
    prev_cutoff = 0
    for i, (rank, _) in enumerate(rank_starts):
        if i + 1 < len(rank_starts):
            _, next_rank_start = rank_starts[i + 1]
            # We use `ceil` here so that rounding favors lower-rank modules.
            # This ensures the total stays under the rank/bpw target.
            cutoff = math.ceil(next_rank_start * num_modules)
        else:
            cutoff = num_modules
        rank_ranges.append((rank, prev_cutoff, cutoff))
        prev_cutoff = cutoff


    module_ranks = itertools.chain(*(itertools.repeat(rank, end - start)
        for rank, start, end in rank_ranges))


    rank_map = {}
    base_weights = 0
    lora_weights = 0
    total_linear_dimension = 0
    for key, rank in zip(sorted_modules, module_ranks):
        print('%4d  %.6e  %s' % (rank, calc_score(key), key))
        rank_map[key] = rank
        n, m = dims.get(key)
        base_weights += n * m
        lora_weights += n * rank + m * rank
        total_linear_dimension += n + m

    print('adds %d lora weights' % lora_weights)
    print('average lora rank: %f' % (lora_weights / total_linear_dimension))
    print('effective bpw (fp16): %+f' % (16 * lora_weights / base_weights))

    json.dump(rank_map, open(out_json_path, 'w'))
    print('wrote rank map to %s' % out_json_path)

if __name__ == '__main__':
    main()
