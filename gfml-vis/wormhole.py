from parsers import parse_args
args = parse_args()

mass_pos = args.mass_pos # full_fix full head tail

lamb_gra = args.lamb_gra # 0.001 ~ 5.0
dist_mode = args.dist_mode # l2, l1

mix_on = args.mix_on # 0, 1
mix_mode = args.mix_mode # split, normal
lamb_mix = args.lamb_mix # 0.1 ~ 0.9
num_split = args.num_split # 1 ~ num of heads
split_mix = args.split_mix # 0, 1

score_norm = 1
