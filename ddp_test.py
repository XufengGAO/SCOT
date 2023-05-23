
import datetime
import argparse
import os
import sys
from torch.utils.data import DataLoader
import torch
import time
from PIL import Image
from data.download import load_dataset
from model import scot_CAM, geometry
from common.loss import StrongCrossEntropyLoss, StrongFlowLoss, WeakDiscMatchLoss
from common.utils import match_idx, mean, draw_weight_map, boolean_string, fix_randseed, parse_string, NewAverageMeter, ProgressMeter, reduce_tensor, Summary
from common.evaluation import Evaluator
import torch.optim as optim
from pprint import pprint
import wandb
import re
import torch.backends.cudnn as cudnn
from torch.utils.data import Subset
import numpy as np

# DDP package
from torch.utils.data.distributed import DistributedSampler
from torch import distributed as dist

wandb.login()

def test(args, model, dataloader):
    # 1. Meter
    pck_meter = NewAverageMeter('pck', ':4.2f')

    model.backbone.eval()
    model.learner.train()
    total_steps = len(dataloader)
    for step, data in enumerate(dataloader):
        bsz = data["src_img"].size(0)
        num_negatives = 0
        src_img = data["src_img"].cuda(non_blocking=True)
        trg_img = data["trg_img"].cuda(non_blocking=True)

        if "src_mask" in data.keys():
            src_mask = data["src_mask"].cuda(non_blocking=True)
            trg_mask = data["trg_mask"].cuda(non_blocking=True)
        else:
            src_mask, trg_mask = None, None

        data["src_kps"] = data["src_kps"].cuda(non_blocking=True)
        data["n_pts"] = data["n_pts"].cuda(non_blocking=True)
        data["trg_kps"] = data["trg_kps"].cuda(non_blocking=True)
        data["pckthres"] = data["pckthres"].cuda(non_blocking=True)   

        src, trg = model(
            src_img,
            trg_img,
            args.classmap,
            src_mask,
            trg_mask,
            args.backbone,
            'test',
        )

        batch_pck = 0
        src_size = (src["feats"].size()[0], src["feats"].size()[1])
        trg_size = (trg["feats"].size()[0], trg["feats"].size()[1])
        votes, _, _ = model.calculate_votes(
            src["feats"], trg["feats"], args.epsilon, args.exp2, src_size, trg_size, src["weights"], trg["weights"], False, bsz)
        votes_geo, _, _ = model.calculate_votesGeo(
            votes, None, None, src["imsize"], trg["imsize"], src["box"], trg["box"])
        prd_kps = geometry.predict_kps(
            src["box"], trg["box"], data["src_kps"],
            data["n_pts"], votes_geo,
        )  # predicted points
        eval_result = Evaluator.eval_kps_transfer(
            prd_kps, data, None, pck_only=True
        )  # return dict results
        # prd_kps_list[key] = prd_kps
        batch_pck = mean(eval_result["pck"])
        del votes, votes_geo, eval_result, prd_kps

        pck_meter.update(batch_pck, bsz)
        
        if step%50 == 0:
            print('step[%d/%d], pck = %4.2f'%(step, total_steps, batch_pck))

        del batch_pck
   


    # 1. Draw weight map
    weight_map_path = os.path.join('./save', "weight_map")
    os.makedirs(weight_map_path, exist_ok=True)
    weight_pth = draw_weight_map(
        model.learner.layerweight.detach().clone(),
        -1, weight_map_path)
    if args.use_wandb:
        wandb.log({"weight_map": wandb.Image(Image.open(weight_pth).convert("RGB"))})
    
    return pck_meter.avg

def main(args):
    local_rank = 0
    device = torch.device("cuda:{}".format(local_rank))


    fix_randseed(seed=(0))
    cudnn.benchmark = False
    cudnn.deterministic = True
    
    # ============ 2. Make Dataloader ... ============
    num_workers = 16 if torch.cuda.is_available() else 8
    pin_memory = True if torch.cuda.is_available() else False

    print("Loading %s dataset" % (args.benchmark))

    test_dataset = load_dataset(
        args.benchmark,
        args.datapath,
        args.thres,
        args.split,
        args.cam,
        img_side=args.img_side,
        use_resize=True,
        use_batch=args.use_batch,
    )

    test_loader = DataLoader(
        dataset=test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory)
    
    print(
        f"Data loaded: there are {len(test_loader.dataset)} test images."
    )
    
    # ============ 3. Model Intialization ... ============
    assert args.backbone in ["resnet50", "resnet101", "fcn101"], "Unknown backbone"
    print(f">>>>>>>>>> Creating model:{args.backbone} + {args.selfsup} <<<<<<<<<<")
    n_layers = {"resnet50": 17, "resnet101": 34, "fcn101": 34}
    hyperpixels = list(range(n_layers[args.backbone]))
    model = scot_CAM.SCOT_CAM(args, hyperpixels)
    model.cuda()

    # freeze the backbone
    for name, param in model.named_parameters():
        if "backbone" in name:
            param.requires_grad = False

    # check # of training params
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f">>>>>>>>>> number of training params: {n_parameters}")

    # resume the model    
    print(f">>>>>>>>>> Resuming from {args.resume} ..........")
    checkpoint = torch.load(args.resume, map_location="cpu")
    msg = model.load_state_dict(checkpoint["model"])
    print(msg)

    # ============ 6. Evaluator, Wandb ... ============
    Evaluator.initialize(args.alpha)

    pck_name = "%s_pck"%(args.split)
    if args.use_wandb:
        _wandb = wandb.init(project=args.wandb_proj, config=args, id=args.run_id, resume="allow", name=args.wandb_name)
        wandb.define_metric("epochs")
        wandb.define_metric(pck_name, step_metric="epochs")

    # ============ 7. Start training ... ============
    log_benchmark = {}
    print(">>>>>>>>>> Start training")
    for epoch in range(50):
        # train
        start_time = time.time()
        if epoch == 0:
            with torch.no_grad():
                test_pck = test(args, model, test_loader)
            log_benchmark[pck_name] = test_pck
        end_time = (time.time()-start_time)/60
        # print(epoch, model.module.learner.layerweight.detach().clone().view(-1))
        print('Epoch = %d, pck = %4.2f'%(epoch, test_pck))

        if args.use_wandb:
            wandb.log({"epochs": epoch})
            wandb.log(log_benchmark)
            
        time_message = (
            ">>>>> Test %d epochs took:%4.3f"%(epoch + 1, end_time,)+" minutes"
        )
        print(time_message)

    print("==================== Finished testing ====================")


if __name__ == "__main__":
    # Arguments parsing
    # fmt: off
    parser = argparse.ArgumentParser(description="SCOT Training Script")

    parser.add_argument('--gpu', type=str, default='0', help='GPU id')
    parser.add_argument('--datapath', type=str, default='./Datasets_SCOT') 
    parser.add_argument('--benchmark', type=str, default='pfpascal')
    parser.add_argument('--backbone', type=str, default='resnet50')
    parser.add_argument('--selfsup', type=str, default='sup', choices=['sup', 'dino', 'denseCL'], help='supervised or self-supervised backbone')
    parser.add_argument('--thres', type=str, default='auto', choices=['auto', 'img', 'bbox'])
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--logpath', type=str, default='')
    # parser.add_argument('--split', type=str, default='trn', help='trn, val, test, old_trn, trn_val') 

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--split', type=str, default='test', help='trn,val,test')

    parser.add_argument("--use_wandb", type= boolean_string, nargs="?", default=False)
    parser.add_argument("--use_xavier", type= boolean_string, nargs="?", default=False)
    parser.add_argument('--resume', default='', type=str,help='path to latest checkpoint (default: none)')
    parser.add_argument('--backbone_path', type=str, default='./backbone')
    
    parser.add_argument('--img_side', type=str, default='(300)')
    parser.add_argument('--weight_thres', type=float, default=0.00, help='weight_thres (default: 0.00)')
    parser.add_argument('--select_all', type=float, default=1.01, help='selec all probability (default: 1.0)')
    # Algorithm parameters
    parser.add_argument('--sim', type=str, default='OTGeo', help='Similarity type: OT, OTGeo, cos, cosGeo')
    parser.add_argument('--exp1', type=float, default=1.0, help='exponential factor on initial cosine cost')
    parser.add_argument('--exp2', type=float, default=1.0, help='exponential factor on final OT scores')
    parser.add_argument('--epsilon', type=float, default=0.05, help='epsilon for Sinkhorn Regularization')
    parser.add_argument('--classmap', type=int, default=0, help='class activation map: 0 for none, 1 for using CAM')
    parser.add_argument('--cam', type=str, default='', help='activation map folder, empty for end2end computation')
    # default is the value that the attribute gets when the argument is absent. const is the value it gets when given.

    parser.add_argument('--run_id', type=str, default='', help='run_id')
    parser.add_argument('--wandb_proj', type=str, default='', help='wandb project name')
    parser.add_argument('--wandb_name', type=str, default='', help='wandb project name')
    



    args = parser.parse_args()


    if args.use_wandb and args.run_id == '':
        args.run_id = wandb.util.generate_id()
    args.img_side = parse_string(args.img_side)
    if isinstance(args.img_side, int):
        # use target rhf center if only scale the max_side
        args.trg_cen = True
        args.use_batch = False
    else:
        args.trg_cen = False
        args.use_batch = True

    main(args)
















