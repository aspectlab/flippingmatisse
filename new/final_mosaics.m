load ../output
[uniqueEds, editions, flip] = find_flips(D, fnames);
load permutation2
mkdir mosaics

% hackish suggestion: before running this line, comment out lines 11 and 55 in make_mosaics to avoid skipping any "no flip" cases.
make_mosaics('../data/matisse_color_small/', 'mosaics/', fnames, uniqueEds, editions, flip_all)
