from main_multiple_runs import params_symmetric_increasing_taus, params_asymetric_increasing_taus, \
    params_asymetric_increasing_gammas
from noise import NoiseType

empty_noise = {'e': '-',
               'tau1': '-',
               'tau2': '-',
               'noiseType': {'type': '-', 'gamma1': '-',
                             'gamma2': '-',}
               }

def d_to_list(d):
    print(d)
    return list_items_to_str([d['e'], d['tau1'], d['tau2'], d['noiseType']['gamma1'] if d['noiseType']['type'] == NoiseType.RED else '-',
                              d['noiseType']['gamma2'] if d['noiseType']['type'] == NoiseType.RED else '-'])

def build_pairs(set):
    rn = [d for d in set if d['noiseType']['type'] == NoiseType.RED]
    wn = [d for d in set if d['noiseType']['type'] == NoiseType.WHITE]

    if len(rn) > 0:
        return [[find_wp(rs), rs] for rs in rn]
    else:
        return [[ws, empty_noise] for ws in wn]


def find_wp(r_set):
    candidates = [d for d in set if
            d['noiseType']['type'] == NoiseType.WHITE and d['e'] == r_set['e'] and d['tau1'] ==
            r_set['tau1'] and d['tau2'] == r_set['tau2']]
    return candidates[0] if len(candidates) > 0 else empty_noise


def list_items_to_str(d):
    return [i if type(i) == str else f"{i:.2}" for i in d]


def pairs_to_str_list(w_r_pair_list):
    return [d_to_list(wp) + [' '] + d_to_list(rp) for wp, rp in w_r_pair_list]

def sets_to_table(set):
    return [' & '.join(s) + ' \\\\\n' for s in set]

all_sets = [params_symmetric_increasing_taus,
            params_asymetric_increasing_taus,
            params_asymetric_increasing_gammas]


for i, set in enumerate(all_sets):
    noise_pairs = build_pairs(set)

    pr_set = sets_to_table(pairs_to_str_list(noise_pairs))

    file_path = f'./results/param_set_{i}.tex'
    print(file_path)
    with open(file_path, 'w') as f:
        f.writelines(pr_set)

    with open(file_path, 'r') as f:
        print(f.read())
