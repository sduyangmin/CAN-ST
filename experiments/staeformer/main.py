import os
import argparse
import numpy as np
import sys

sys.path.append(os.path.abspath(__file__ + "/../../.."))

import torch

torch.set_num_threads(3)

from src.models.staeformer import STAEformer
from src.base.engine import BaseEngine
from src.utils.args import get_public_config
from src.utils.dataloader import load_dataset, load_adj_from_numpy, get_dataset_info
from src.utils.graph_algo import normalize_adj_mx, calculate_cheb_poly
from src.utils.metrics import masked_mae
from src.utils.logging import get_logger
from src.models.Model import Model
from models.norm.CAN_ST import CAN_ST

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


def get_config():
    parser = get_public_config()
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lrate", type=float, default=1e-3)
    parser.add_argument("--wdecay", type=float, default=1e-4)
    parser.add_argument("--clip_grad_value", type=float, default=0)
    args = parser.parse_args()
    args.model_name = __file__.split("/")[-2]
    folder_name = "{}_{}_CANST".format(args.dataset, args.years)
    log_dir = "./experiments/{}/{}/".format(args.model_name, folder_name)
    logger = get_logger(log_dir, __name__, "record_s{}.log".format(args.seed))
    logger.info(args)

    return args, log_dir, logger


def main():
    args, log_dir, logger = get_config()
    set_seed(args.seed)
    device = torch.device(args.device)

    data_path, adj_path, node_num = get_dataset_info(args.dataset)
    logger.info("Adj path: " + adj_path)

    dataloader, scaler = load_dataset(data_path, args, logger)

    forcast_model = STAEformer(
        node_num=node_num,
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        dropout=args.dropout,
    )

    norm_model = CAN_ST(node_num,args.clusters)

    
    

    model = Model(forcast_model, norm_model)
    loss_fn = masked_mae
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lrate, weight_decay=args.wdecay
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[25, 45, 65], gamma=0.1
    )

    engine = BaseEngine(
        device=device,
        model=model,
        dataloader=dataloader,
        scaler=scaler,
        sampler=None,
        loss_fn=loss_fn,
        lrate=args.lrate,
        optimizer=optimizer,
        scheduler=scheduler,
        clip_grad_value=args.clip_grad_value,
        max_epochs=args.max_epochs,
        patience=args.patience,
        log_dir=log_dir,
        logger=logger,
        seed=args.seed,
    )

    if args.mode == "train":
        engine.train()
    else:
        engine.evaluate("test")
        engine.evaluate("shift")


if __name__ == "__main__":
    main()
