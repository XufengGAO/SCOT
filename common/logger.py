r"""Logging"""
import datetime
import logging
import os

from tensorboardX import SummaryWriter
import torch
from torch import distributed as dist

class Logger:
    r"""Writes results of training/testing"""
    @classmethod
    def initialize(cls, args, training=True):
        if training:
            if args.logpath == "":
                logpath = "%.e_%s_%s"%(args.lr, args.loss_stage, args.optimizer)
                
                if args.optimizer == "sgd":
                    logpath = logpath + "_m%.2f"%(args.momentum)
                if args.scheduler != 'none':
                    logpath = logpath + "_%s"%(args.scheduler)
    
                logpath = logpath + "_bsz%d"%(args.batch_size)

                cls.logpath = os.path.join('logs', 'ddp', 'train', args.backbone, args.selfsup, args.criterion, args.benchmark + "_%s"%(args.alpha), logpath)
                filemode = 'w'
            else:
                cls.logpath = args.logpath
                filemode = 'a'
        else:
            # logtime = datetime.datetime.now().__format__('_%m%d_%H%M%S')
            cls.logpath = os.path.join('logs', 'test', args.backbone, args.selfsup, args.criterion, args.benchmark, args.logpath)
            filemode = 'w'
        
        if dist.get_rank() == 0:
            os.makedirs(cls.logpath, exist_ok=True)
        dist.barrier()
  

        logging.basicConfig(filemode=filemode,
                            filename=os.path.join(cls.logpath, 'log.txt'),
                            level=logging.INFO if dist.get_rank()==0 else logging.WARN,
                            format='%(message)s',
                            datefmt='%m-%d %H:%M:%S')
        # if dist.get_rank()==0 else logging.WARN
        # Console log config
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

        # Tensorboard writer
#         cls.tbd_writer = SummaryWriter(os.path.join(cls.logpath, 'tbd/runs'))
        
        # Log arguments
        logging.info('\n+=========== SCOT arguments ============+')
        for arg_key in args.__dict__:
            logging.info('| %20s: %-24s |' % (arg_key, str(args.__dict__[arg_key])))
        logging.info('+================================================+\n')

    @classmethod
    def info(cls, msg):
        r"""Writes message to .txt"""
        logging.info(msg)

    @classmethod
    def save_epoch(cls, model, epoch, val_pck):
        torch.save(model.state_dict(), os.path.join(cls.logpath, 'eooch_%d.pt'%(epoch)))
        cls.info('Epoch Model saved @%d w/ val. PCK: %5.4f on [%s]\n' % (epoch, val_pck, os.path.join(cls.logpath, 'eooch_%d.pt'%(epoch))))
    
    @classmethod
    def save_model(cls, model, epoch, val_pck, best_val_pck):
        torch.save(model.state_dict(), os.path.join(cls.logpath, 'best_model.pt'))
        cls.info('Best Model saved @%d w/ val. PCK: %5.4f -> %5.4f on [%s]\n' % (epoch, best_val_pck, val_pck, os.path.join(cls.logpath, 'best_model.pt')))


class AverageMeter:
    r"""Stores loss, evaluation results, selected layers"""
    def __init__(self, benchmark, cls=None):
        r"""Constructor of AverageMeter"""

        self.eval_buf = {
            'pfwillow': {'pck': [], 'cls_pck': dict()},
            'pfpascal': {'pck': [], 'cls_pck': dict()},
            'spair':    {'pck': [], 'cls_pck': dict()}
        }
        self.eval_buf = self.eval_buf[benchmark]
        
        # buffer for average pck
        self.pck_buffer = []

        # buffer for class-pck
        self.cls_buffer = {}
        if cls is not None:
            for sub_cls in cls:                
                self.cls_buffer[sub_cls] = []

        # buffer for loss
        self.loss_buffer = {}
                
    def eval_pck(self, prd_kps, data, alpha=0.1):
        r"""Compute percentage of correct key-points (PCK) based on prediction"""
        pckthres = data['pckthres'][0] 
        # ncorrt = correct_kps(data['trg_kps'].cuda(), prd_kps, pckthres, data['alpha'])
        ncorrt = correct_kps(data['trg_kps'][0].to(prd_kps.device), prd_kps, pckthres.to(prd_kps.device), alpha)
        # print(int(ncorrt), data['n_pts'], data['trg_kps'].size(), data['n_pts'].is_cuda)
        pair_pck = int(ncorrt) / int(data['n_pts'].item())
        
        self.eval_buf['pck'].append(pair_pck)

        if self.eval_buf['cls_pck'].get(data['pair_class'][0]) is None:
            self.eval_buf['cls_pck'][data['pair_class'][0]] = []
        self.eval_buf['cls_pck'][data['pair_class'][0]].append(pair_pck)
        
        return pair_pck

    def log_pck(self):
        r"""Log percentage of correct key-points (PCK)"""

        pck = sum(self.eval_buf['pck']) / len(self.eval_buf['pck'])
        for cls in self.eval_buf['cls_pck']:
            cls_avg = sum(self.eval_buf['cls_pck'][cls]) / len(self.eval_buf['cls_pck'][cls])
            Logger.info('%15s: %3.3f' % (cls, cls_avg))
        Logger.info(' * Testing Average: %3.3f' % pck)

        return pck
    
    def update(self, batch_pck, loss=None):
      
        self.pck_buffer.append(batch_pck)

        # eval_name = ['sim', 'votes', 'votes_geo']
        # for key in self.cls_buffer: # per-class pck term
        #     for idx, eval_result in enumerate(eval_result_list):
        #         for eval_cls, cls in zip(eval_result[key], category):
        #             if self.cls_buffer.get(cls) is None:
        #                 self.cls_buffer[eval_name[idx]][cls] = []
        #             self.cls_buffer[eval_name[idx]][cls] += [eval_cls]
        
        if loss is not None: # mean loss term
            for key, value in loss.items():
                if self.loss_buffer.get(key) is None:
                    self.loss_buffer[key] = []
                self.loss_buffer[key].append(value)

    def write_result(self, split, epoch=-1):
        msg = '\n*** %s ' % split
        msg += '[@Epoch %02d] ' % epoch if epoch > -1 else ''

        if len(self.loss_buffer) > 0:
            for key, value in self.loss_buffer.items():
                msg += '%s: %4.2f  ' % (key, mean(value))

        msg += 'PCK in buf: %4.2f' % (mean(self.pck_buffer))

        msg += '***\n'
        Logger.info(msg)

    def write_process(self, batch_idx, datalen, epoch=-1):
        msg = '[Epoch: %02d] ' % epoch if epoch > -1 else ''
        msg += '[Batch: %04d/%04d] ' % (batch_idx+1, datalen)
        if len(self.loss_buffer) > 0:
            # msg += 'Last sample: '
            # for key, value in self.loss_buffer.items():
            #     msg += '%s: %4.2f  ' % (key, value[-1])
            msg += 'Avg: '
            for key, value in self.loss_buffer.items():
                msg += '%s: %4.2f  ' % (key, mean(value))

        msg += 'Avg PCK: %4.4f' % (sum(self.pck_buffer) / len(self.pck_buffer))

        Logger.info(msg)
         
def correct_kps(trg_kps, prd_kps, pckthres, alpha=0.1):
    r"""Compute the number of correctly transferred key-points"""
    # print(trg_kps.is_cuda, prd_kps.is_cuda, pckthres.is_cuda)
    l2dist = torch.pow(torch.sum(torch.pow(trg_kps - prd_kps, 2), 0), 0.5)
    thres = pckthres.expand_as(l2dist).float()
    correct_pts = torch.le(l2dist, thres * alpha)

    return torch.sum(correct_pts)


def mean(x):
    r"""Computes average of a list"""
    return sum(x) / len(x) if len(x) > 0 else 0.0