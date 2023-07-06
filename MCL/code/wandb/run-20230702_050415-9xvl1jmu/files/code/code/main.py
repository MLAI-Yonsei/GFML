from dataloader import Loader
from model import LightGCN
import numpy as np
from os.path import join
from parse import parse_args
import Procedure
# from tensorboardX import SummaryWriter
# from torch.utils.tensorboard import SummaryWriter
import time
import torch
import utils
import wandb

if __name__ == '__main__':
    # set seed
    args = parse_args()
    utils.set_seed(args.seed)
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else "cpu")

    # create model and load dataset
    dataset = Loader(args, device, path="../data/" + args.dataset)
    model = LightGCN(args, device, dataset)
    model = model.to(device)
    metric = utils.MetricLoss(model, args)

    # save/load file
    weight_file = utils.getFileName(args)
    print(f"load and save to {weight_file}")

    wandb.login()
    if args.gfml:
        wandb.init(project='GFML_Revision_MCL',
                   name=f'{weight_file}',
                   config=args)
    else:
        wandb.init(project='GFML_Revision_MCL',
                   name=f'{weight_file}',
                   config=args)

    # wandb.config.update(args)

    wandb.watch(model)

    if args.load:
        try:
            model.load_state_dict(torch.load(weight_file, map_location=torch.device('cpu')))
            print(f"loaded model weights from {weight_file}") 
        except FileNotFoundError:
            print(f"{weight_file} not exists, start from beginning")

    # init tensorboard
    if args.tensorboard:
        # w : SummaryWriter = SummaryWriter(
        #                         join("./runs", time.strftime("%m-%d-%Hh%Mm%Ss-"))
        #                     )
        w = None
    else:
        w = None

    # init sampler
    sampler = utils.WarpSampler(dataset, args.batch_size, args.num_neg)

    # training
    try:
        topks = eval(args.topks)
        best_result = np.zeros(2*len(topks))
        BEST_NDCG10 = 1e-10
        for epoch in range(1, args.epochs+1):
            print(f'Epoch {epoch}/{args.epochs}')
            start = time.time()
            if epoch % 10 == 0:
                result = Procedure.Test(args, dataset, model, epoch, device, w, args.multicore)
                if np.sum(np.append(result['recall'], result['ndcg'])) > np.sum(best_result):
                    best_result = np.append(result['recall'], result['ndcg'])
                    torch.save(model.state_dict(), weight_file)
                    wandb.log(
                        {
                            'Recall@5_total_best' : result['recall'][0],
                            'Recall@10_total_best': result['recall'][1],
                            'Recall@20_total_best': result['recall'][2],
                            'ndcg@5_total_best': result['ndcg'][0],
                            'ndcg@10_total_best': result['ndcg'][1],
                            'ndcg@20_total_best': result['ndcg'][2],
                        }
                    )
                if result['ndcg'][1] > BEST_NDCG10:
                    BEST_NDCG10 = result['ndcg'][1]
                    wandb.log(
                        {
                            'EPOCH_ndcg_best' : epoch,
                            'Recall@5_ndcg_best': result['recall'][0],
                            'Recall@10_ndcg_best': result['recall'][1],
                            'Recall@20_ndcg_best': result['recall'][2],
                            'ndcg@5_ndcg_best': result['ndcg'][0],
                            'ndcg@10_ndcg_best': result['ndcg'][1],
                            'ndcg@20_ndcg_best': result['ndcg'][2],
                            'BEST_NDCG@10': BEST_NDCG10
                        }
                    )
                print("Best so far:", best_result)

            output_information = Procedure.Metric_train_original(args, dataset, model, metric, epoch, sampler, w)

            print(f'{output_information}')
            print(f"Total time {time.time() - start}")
        
        result = Procedure.Test(args, dataset, model, epoch, device, w, args.multicore)
        if np.sum(np.append(result['recall'], result['ndcg'])) > np.sum(best_result):
            best_result = np.append(result['recall'], result['ndcg'])
            torch.save(model.state_dict(), weight_file)
            wandb.log(
                {
                    'Recall@5_total_best': result['recall'][0],
                    'Recall@10_total_best': result['recall'][1],
                    'Recall@20_total_best': result['recall'][2],
                    'ndcg@5_total_best': result['ndcg'][0],
                    'ndcg@10_total_best': result['ndcg'][1],
                    'ndcg@20_total_best': result['ndcg'][2],
                }
            )
        if result['ndcg'][1] > BEST_NDCG10:
            BEST_NDCG10 = result['ndcg'][1]
            wandb.log(
                {
                    'EPOCH_ndcg_best': epoch,
                    'Recall@5_ndcg_best': result['recall'][0],
                    'Recall@10_ndcg_best': result['recall'][1],
                    'Recall@20_ndcg_best': result['recall'][2],
                    'ndcg@5_ndcg_best': result['ndcg'][0],
                    'ndcg@10_ndcg_best': result['ndcg'][1],
                    'ndcg@20_ndcg_best': result['ndcg'][2],
                    'BEST_NDCG@10': BEST_NDCG10
                }
            )
        print("Best overall:", best_result)

        wandb.finish()

    finally:
        sampler.close()
        if args.tensorboard:
            w.close()