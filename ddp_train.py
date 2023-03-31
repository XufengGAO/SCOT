r"""Beam search for hyperpixel layers"""

import datetime
import argparse
import os

from torch.utils.data.dataloader import DataLoader

import torch

from PIL import Image
from data import download
from model import scot_CAM, util, geometry
import numpy as np
from model.objective import Objective
from common import supervision as sup
from common import utils
from common.logger import AverageMeter, Logger
from common.evaluation import Evaluator
import torch.optim as optim
from model.base.geometry import Geometry
from pprint import pprint
import wandb
import matplotlib.pyplot as plt

# DDP package
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import tempfile
wandb.login()


# DDP setup
def dist_setup(rank, world_size):
    pass


# function to train model
def train(epoch, model, dataloader, strategy, optimizer, training, args):
    r"""Code for training SCOT"""

    if training:
        model.backbone.eval()  # keeps freezing backbone
        model.learner.train()
        # model.train()
    else:
        model.eval()

    average_meter = AverageMeter(
        dataloader.dataset.benchmark, cls=dataloader.dataset.cls
    )

    steps_in_batch = len(dataloader)

    for step, batch in enumerate(dataloader):
        src_img, trg_img = strategy.get_image_pair(
            batch
        )  # batch['src_img'], batch['trg_img']

        if "src_mask" in batch.keys():
            src_mask = (batch["src_mask"],)
            trg_mask = (batch["trg_mask"],)
        else:
            src_mask = None
            trg_mask = None

        confidence_ts, src_box, trg_box = model(
            src_img,
            trg_img,
            args.sim,
            args.exp1,
            args.exp2,
            args.eps,
            args.classmap,
            src_mask,
            trg_mask,
            args.backbone,
        )
        # print("model result", confidence_ts.size(), src_box.size(), trg_box.size())
        # model result torch.Size([4, 4096, 4096]) torch.Size([4096, 4]) torch.Size([4096, 4])
        # print("geometry", Geometry.rf_center.size())

        prd_kps = geometry.predict_kps(
            src_box,
            trg_box,
            batch["src_kps"],
            batch["n_pts"],
            strategy.get_correlation(confidence_ts),
        )
        # print("prd_kps", prd_kps.size()) # prd_kps torch.Size([4, 2, 400])

        # 3. Evaluate predictions
        eval_result = Evaluator.evaluate(prd_kps, batch)
        # eval_result = {'easy_match': easy_match,
        #                'hard_match': hard_match,
        #                'pck': pck,
        #                'pck_ids': pck_ids}

        loss = strategy.compute_loss(confidence_ts, eval_result, batch)

        step_ratio = 0.0
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step_ratio = torch.norm(
                model.learner.layerweight.grad.detach().clone().view(-1)
            ) / (
                torch.norm(model.learner.layerweight.detach().clone().view(-1)) + 1e-20
            )

            if args.use_wandb:
                wandb.log(
                    {
                        "step_ratio": step_ratio.item(),
                        "total_step": step + epoch * steps_in_batch,
                    }
                )

        # print(model.learner.layerweight.size())
        # print(eval_result['pck'])
        # print(batch["pair_class"], batch["n_pts"])
        # print(prd_kps[0][:,:10])
        # print(batch['trg_kps'][0][:,:10])
        # print(eval_result['easy_match'])
        # print(eval_result['hardmatch'])

        # log pck, loss
        average_meter.update(
            eval_result,
            batch["pair_class"],
            loss.item(),
        )
        # log batch loss, batch pck
        average_meter.write_process(step, len(dataloader), epoch)
        Logger.tbd_writer.add_histogram(
            tag="learner_grad",
            values=model.learner.layerweight.grad.detach().clone().view(-1),
            global_step=step,
        )

        # print("pck", eval_result["pck"], torch.tensor(eval_result["pck"]))

        # log running step loss
        if training and args.use_wandb and (step % 20 == 0):
            running_avg_loss = utils.mean(average_meter.loss_buffer)
            running_avg_pck = utils.mean(average_meter.buffer["pck"])
            wandb.log(
                {
                    "running_trn_avg_loss": running_avg_loss,
                    "running_trn_avg_pck": running_avg_pck,
                }
            )

        log_freq = 100
        if training and (step % log_freq == 0):
            # 1. Draw weight map
            weight_map_path = os.path.join(Logger.logpath, "weight_map")
            os.makedirs(weight_map_path, exist_ok=True)

            weight_pth = utils.draw_weight_map(
                model.learner.layerweight.detach().clone().view(-1),
                epoch,
                step,
                weight_map_path,
            )
            if args.use_wandb:
                wandb.log(
                    {
                        "weight_map": wandb.Image(
                            Image.open(weight_pth).convert("RGB")
                        ),
                        "draw_step": step + epoch * steps_in_batch,
                    }
                )
            # wandb.log({"Sigmoid weight":wandb.Image(fig)})
            # plt.close(fig)

            # 2. Draw matches
            min_pck, min_pck_idx = torch.tensor(eval_result["pck"]).min(dim=0)
            # print(min_pck, min_pck_idx)

            src_img = dataloader.dataset.get_image(batch["src_imname"][min_pck_idx])
            trg_img = dataloader.dataset.get_image(batch["trg_imname"][min_pck_idx])

            draw_match_path = os.path.join(Logger.logpath, "draw_match")
            os.makedirs(draw_match_path, exist_ok=True)

            match_pth = utils.draw_matches_on_image(
                epoch,
                step,
                min_pck_idx.item(),
                min_pck,
                src_img,
                trg_img,
                batch,
                pred_trg_kps=prd_kps[min_pck_idx],
                origin=True,
                color_ids=eval_result["pck_ids"][min_pck_idx],
                draw_match_path=draw_match_path,
            )
            if args.use_wandb:
                wandb.log(
                    {
                        "match_map": wandb.Image(Image.open(match_pth).convert("RGB")),
                        "draw_step": step + epoch * steps_in_batch,
                    }
                )

    draw_class_pck_path = os.path.join(Logger.logpath, "draw_class_pck")
    os.makedirs(draw_class_pck_path, exist_ok=True)
    # 3. Draw class pck
    class_pth = utils.draw_class_pck(
        average_meter.sel_buffer, draw_class_pck_path, epoch, step
    )
    if args.use_wandb:
        wandb.log(
            {
                "class_pck": wandb.Image(Image.open(class_pth).convert("RGB")),
                "draw_step": step + epoch * steps_in_batch,
            }
        )

    # log epoch loss, epoch pck
    average_meter.write_result("Training" if training else "Validation", epoch)

    avg_loss = utils.mean(average_meter.loss_buffer)
    avg_pck = utils.mean(average_meter.buffer["pck"])
    return avg_loss, avg_pck


# main function
# local_rank
def main(local_rank, world_size, args):

    if torch.cuda.is_available() is False:
        raise EnvironmentError("not find GPU device for training.")
    
    # 1. DDP Initializations
    args.local_rank = local_rank
    args.world_size = world_size
    global_rank = args.node_rank * args.gpus + local_rank
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12345"
    dist.init_process_group(backend="nccl", world_size=world_size, rank=global_rank)
    torch.cuda.set_device(local_rank)

    device = torch.device("cuda", local_rank)
    util.fix_randseed(seed=0)


    dist.barrier() # Wait for all processes to complete

    if local_rank == 0:
        # print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
        Logger.initialize(args)



    # fmt: on
    # 1. CUDA and reproducibility
    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    

    # 3. Model intialization
    img_side = (256, 256)
    n_layers = {"resnet50": 17, "resnet101": 34, "fcn101": 34}
    hyperpixels = list(range(n_layers[args.backbone]))
    model = scot_CAM.SCOT_CAM(
        args.backbone,
        hyperpixels,
        args.benchmark,
        device,
        args.cam,
        args.use_xavier,
        img_side=img_side,
    )
    model.cuda(local_rank)

    # Check checkpoint
    if os.path.exists(args.checkpoint_path):
        weights_dict = torch.load(args.checkpoint_path, map_location=device)
        model.load_state_dict(weights_dict, strict=False)
    else:
        checkpoint_path = os.path.join(tempfile.gettempdir(), "initial_weights.pt")
        # save the weights for 0th process, than load it to other processes, to make sure the
        # weight initializations are consistent
        if rank == 0:
            torch.save(model.state_dict(), checkpoint_path)

        dist.barrier()
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    # convert it to DDP version
    model = DDP(model, device_ids=[local_rank])

    # 4. Objective and Optimizer
    Objective.initialize(target_rate=0.5, alpha=args.alpha)
    strategy = (
        sup.WeakSupStrategy() if args.supervision == "weak" else sup.StrongSupStrategy()
    )
    param_model = [
        param for name, param in model.named_parameters() if "backbone" not in name
    ]
    param_backbone = [
        param for name, param in model.named_parameters() if "backbone" in name
    ]

    if args.optimizer == "sgd":
        optimizer = optim.SGD(
            [
                {"params": param_model, "lr": args.lr},
                {"params": param_backbone, "lr": args.lr_backbone},
            ],
            momentum=args.momentum if args.use_regularization else 0.00,
        )
    elif args.optimizer == "adam":
        optimizer = optim.AdamW(
            [
                {"params": param_model, "lr": args.lr},
                {"params": param_backbone, "lr": args.lr_backbone},
            ],
            weight_decay=args.weight_decay if args.use_regularization else 0.00,
        )
    else:
        assert args.optimizer in ["sgd", "adam"], "Unrecognized model type"

    # print(type(model.learner))

    if args.use_wandb:
        run = wandb.init(project="train SCOT", config=args)
        wandb.watch(model.learner, log="all", log_freq=100)


    # 5. Dataset download & initialization

    trn_ds = download.load_dataset(
        args.benchmark, args.datapath, args.thres, device, "trn", img_side=img_side
    )
    val_ds = download.load_dataset(
        args.benchmark, args.datapath, args.thres, device, "val", img_side=img_side
    )

    # DDP sampler
    num_workers = min(
        [os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8]
    )
    pin_memory = True
    trn_sampler = DistributedSampler(
        dataset=trn_ds,
        num_replicas=args.world_size,
        rank=dist.get_rank(),  # global rank
        shuffle=True,
    )
    val_sampler = DistributedSampler(
        dataset=val_ds,
        num_replicas=args.world_size,
        rank=dist.get_rank(),
        shuffle=True,
    )
    trn_dl = DataLoader(
        dataset=trn_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        sampler=trn_sampler,
        pin_memory=pin_memory,
    )
    val_dl = DataLoader(
        dataset=val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        sampler=val_sampler,
        pin_memory=pin_memory,
    )

    # 6. Evaluator
    Evaluator.initialize(args.benchmark, args.alpha)

    # 7. Load checkpoints
    # TODO, change later
    rank = 0
    if args.checkpoint_path:
        checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
        state_dict = checkpoint["model"]
        model.load_state_dict(state_dict)
        model = model.to(rank)
        optimizer.load_state_dict(checkpoint["optimizer"])
    model = model.to(rank)
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    # 8. Train SCOT
    best_val_pck = float("-inf")
    log_benchmark = {}
    for epoch in range(args.niter):
        trn_loss, trn_pck = train(
            epoch, model, trn_dl, strategy, optimizer, training=True, args=args
        )
        log_benchmark["trn_loss"] = trn_loss
        log_benchmark["trn_pck"] = trn_pck

        with torch.no_grad():
            val_loss, val_pck = train(
                epoch, model, val_dl, strategy, optimizer, training=False, args=args
            )
            log_benchmark["val_loss"] = val_loss
            log_benchmark["val_pck"] = val_pck

        log_benchmark["epoch"] = epoch
        if args.use_wandb:
            wandb.log(log_benchmark)

        # Save the best model
        if val_pck > best_val_pck:
            best_val_pck = val_pck
            Logger.save_model(model, epoch, val_pck)

        Logger.tbd_writer.add_scalars(
            "data/loss", {"trn_loss": trn_loss, "val_loss": val_loss}, epoch
        )
        Logger.tbd_writer.add_scalars(
            "data/pck", {"trn_pck": trn_pck, "val_pck": val_pck}, epoch
        )

    Logger.tbd_writer.close()
    Logger.info("==================== Finished training ====================")


if __name__ == "__main__":
    # Arguments parsing
    # fmt: off
    parser = argparse.ArgumentParser(description="SCOT Training Script")

    parser.add_argument('--gpu', type=str, default='0', help='GPU id')
    parser.add_argument('--datapath', type=str, default='./Datasets_SCOT')
    parser.add_argument('--benchmark', type=str, default='pfpascal')
    parser.add_argument('--backbone', type=str, default='resnet50')
    parser.add_argument('--thres', type=str, default='auto', choices=['auto', 'img', 'bbox'])
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--logpath', type=str, default='')
    parser.add_argument('--split', type=str, default='trn', help='trn,val.test') 

    # Training parameters
    parser.add_argument('--supervision', type=str, default='strong', choices=['weak', 'strong'])
    parser.add_argument('--lr', type=float, default=0.01) 
    parser.add_argument('--lr_backbone', type=float, default=0.0) 
    parser.add_argument('--niter', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--optimizer', type=str, default="sgd", choices=["sgd", "adam"])
    parser.add_argument('--weight_decay', type=float, default=0.00,help='weight decay (default: 0.00)')
    parser.add_argument('--momentum', type=float, default=0.9,help='momentum (default: 0.9)')
    parser.add_argument('--use_regularization', type= utils.boolean_string, nargs="?", default=False)
    parser.add_argument('--checkpoint_path', type=str, default='')

    # DDP arguments
    parser.add_argument("--local_rank", type=int)
    parser.add_argument('--num_gpus', type=int, default=1, help='#GPUs to use. 0 means CPU mode')
    parser.add_argument('--world_size', type=int, default=2, help='#distributed processes, #GPUs in one machine')

    # Algorithm parameters
    parser.add_argument('--sim', type=str, default='OTGeo', help='Similarity type: OT, OTGeo, cos, cosGeo')
    parser.add_argument('--exp1', type=float, default=1.0, help='exponential factor on initial cosine cost')
    parser.add_argument('--exp2', type=float, default=1.0, help='exponential factor on final OT scores')
    parser.add_argument('--eps', type=float, default=0.05, help='epsilon for Sinkhorn Regularization')
    parser.add_argument('--classmap', type=int, default=0, help='class activation map: 0 for none, 1 for using CAM')
    parser.add_argument('--cam', type=str, default='', help='activation map folder, empty for end2end computation')

    parser.add_argument("--use_wandb", type= utils.boolean_string, nargs="?", const=True, default=True)
    parser.add_argument("--use_xavier", type= utils.boolean_string, nargs="?", const=True, default=True)

    # default is the value that the attribute gets when the argument is absent. const is the value it gets when given.

    opts = parser.parse_args()
    


    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    world_size = opts.world_size
    torch.multiprocessing.spawn(main, args=(world_size, opts), nprocs=world_size, join=True)
