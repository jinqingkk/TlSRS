import argparse, os, sys, time, gc, datetime
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from datasets import find_dataset_def
from models import *
from utils import *
import torch.distributed as dist

cudnn.benchmark = False
cudnn.deterministic = True

parser = argparse.ArgumentParser(description='A PyTorch Implementation of Cascade Cost Volume MVSNet')
parser.add_argument('--mode', default='train', help='train or test', choices=['train', 'test', 'profile'])
parser.add_argument('--model', default='mvsnet', help='select model')
parser.add_argument('--device', default='cuda', help='select model')

parser.add_argument('--dataset', default='dtu_yao', help='select dataset')
parser.add_argument('--trainpath', help='train datapath')
parser.add_argument('--testpath', help='test datapath')
parser.add_argument('--trainlist', help='train list')
parser.add_argument('--testlist', help='test list')

parser.add_argument('--epochs', type=int, default=16, help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--lrepochs', type=str, default="10,12,14:2", help='epoch ids to downscale lr and the downscale rate')
parser.add_argument('--wd', type=float, default=0.0, help='weight decay')

parser.add_argument('--batch_size', type=int, default=1, help='train batch size')
parser.add_argument('--numdepth', type=int, default=192, help='the number of depth values')
parser.add_argument('--interval_scale', type=float, default=1.06, help='the number of depth values')

parser.add_argument('--loadckpt', default=None, help='load a specific checkpoint')
parser.add_argument('--logdir', default='./checkpoints/debug', help='the directory to save checkpoints/logs')
parser.add_argument('--resume', action='store_true', help='continue to train the model')

parser.add_argument('--summary_freq', type=int, default=50, help='print and summary frequency')
parser.add_argument('--save_freq', type=int, default=1, help='save checkpoint frequency')
parser.add_argument('--eval_freq', type=int, default=1, help='eval freq')

parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed')
parser.add_argument('--pin_m', action='store_true', help='data loader pin memory')
parser.add_argument("--local_rank", type=int, default=0)

parser.add_argument('--share_cr', action='store_true', help='whether share the cost volume regularization')
parser.add_argument('--ndepths', type=str, default="48,32,8", help='ndepths')
parser.add_argument('--depth_inter_r', type=str, default="4,2,1", help='depth_intervals_ratio')
parser.add_argument('--dlossw', type=str, default="0.5,1.0,2.0", help='depth loss weight for different stage')
parser.add_argument('--normal_smooth_loss_weight', type=float, default=0.02, help='normal smoothness loss weight')
parser.add_argument('--curv_loss_weight', type=float, default=0.005, help='curvature continuity loss weight')
parser.add_argument('--edge_smooth_loss_weight', type=float, default=0.005, help='edge-aware smooth loss weight')
parser.add_argument('--depth_normal_loss_weight', type=float, default=0.03, help='depth-normal consistency loss weight')
parser.add_argument('--depth_normal_conf_threshold', type=float, default=0.8, help='confidence threshold for depth-normal consistency')
parser.add_argument('--edge_grad_threshold', type=float, default=0.05, help='image gradient threshold for non-edge normal consistency')
parser.add_argument('--geometry_conf_mid', type=float, default=0.65, help='confidence midpoint for soft geometry weight')
parser.add_argument('--geometry_k_conf', type=float, default=10.0, help='confidence sigmoid slope for soft geometry weight')
parser.add_argument('--geometry_edge_mid', type=float, default=0.25, help='image gradient midpoint for soft geometry weight')
parser.add_argument('--geometry_k_edge', type=float, default=10.0, help='edge sigmoid slope for soft geometry weight')
parser.add_argument('--geometry_w_min', type=float, default=0.05, help='minimum soft geometry weight on valid depth')
parser.add_argument('--use_dual_region_curvature', action='store_true', default=True, help='use Test9 hard/soft dual-region curvature constraint')
parser.add_argument('--disable_dual_region_curvature', dest='use_dual_region_curvature', action='store_false', help='fall back to Test8 soft geometry curvature constraint')
parser.add_argument('--region_lambda_a', type=float, default=1.5, help='hard Region A curvature loss weight')
parser.add_argument('--region_lambda_b', type=float, default=1.0, help='soft Region B curvature loss weight')
parser.add_argument('--region_edge_threshold', type=float, default=0.25, help='Sobel image edge threshold for Region A')
parser.add_argument('--region_depth_threshold', type=float, default=0.02, help='normalized depth gradient threshold for Region A')
parser.add_argument('--region_curv_threshold', type=float, default=0.02, help='normalized high curvature threshold for Region A')
parser.add_argument('--region_smooth_k', type=float, default=2.0, help='image-gradient decay slope for Region B soft weight')
parser.add_argument('--cr_base_chs', type=str, default="8,8,8", help='cost regularization base channels')
parser.add_argument('--grad_method', type=str, default="detach", choices=["detach", "undetach"], help='grad method')
parser.add_argument('--use_sger', action='store_true', help='enable per-stage SGER depth refinement')
parser.add_argument('--use_sger_lite', action='store_true', help='enable Stage3-only SGER-Lite depth refinement')
parser.add_argument('--sger_share', action='store_true', help='share the SGER core across stages')
parser.add_argument('--sger_feature_channels', type=int, default=8, help='projected reference feature channels for SGER')
parser.add_argument('--sger_hidden_channels', type=int, default=32, help='SGER residual head channels')
parser.add_argument('--sger_gate_channels', type=int, default=16, help='SGER gate head channels')
parser.add_argument('--sger_max_residual_ratio', type=float, default=0.25, help='maximum residual in stage depth intervals')
parser.add_argument('--detach_refined_feedback', action='store_true', default=True, help='detach refined depth before next-stage sampling')
parser.add_argument('--allow_refined_feedback_grad', dest='detach_refined_feedback', action='store_false', help='allow gradients through refined-depth cascade feedback')
parser.add_argument('--raw_depth_loss_weight', type=float, default=1.0, help='auxiliary raw stage depth loss weight')
parser.add_argument('--refined_depth_loss_weight', type=float, default=1.0, help='refined stage depth loss weight')
parser.add_argument('--residual_loss_weight', type=float, default=0.01, help='normalized SGER residual regularization weight')
parser.add_argument('--gate_loss_weight', type=float, default=0.0, help='legacy SGER gate mean penalty weight')
parser.add_argument('--safe_refine_loss_weight', type=float, default=0.1, help='penalty when refined depth is worse than raw depth')
parser.add_argument('--safe_refine_margin', type=float, default=0.0, help='raw-to-refined safety loss margin')
parser.add_argument('--residual_target_loss_weight', type=float, default=0.05, help='bounded SGER residual proposal supervision weight')
parser.add_argument('--gate_benefit_loss_weight', type=float, default=0.05, help='benefit-supervised SGER gate loss weight')
parser.add_argument('--residual_target_ratio', type=float, default=0.25, help='maximum residual target in stage depth intervals')
parser.add_argument('--benefit_margin_ratio', type=float, default=0.05, help='minimum proposal improvement in stage depth intervals')
parser.add_argument('--freeze_backbone_epochs', type=int, default=8, help='freeze non-SGER-Lite backbone for the first N epochs')
parser.add_argument('--backbone_lr_scale', type=float, default=0.1, help='SGER-Lite backbone learning-rate multiplier after unfreezing')
parser.add_argument('--sger_warmup_start_epoch', type=int, default=7, help='first epoch with a nonzero SGER residual scale')
parser.add_argument('--sger_warmup_end_epoch', type=int, default=10, help='first epoch with the full SGER residual scale')

parser.add_argument('--using_apex', action='store_true', help='using apex, need to install apex')
parser.add_argument('--sync_bn', action='store_true',help='enabling apex sync BN.')
parser.add_argument('--opt-level', type=str, default="O0")
parser.add_argument('--keep-batchnorm-fp32', type=str, default=None)
parser.add_argument('--loss-scale', type=str, default=None)


num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
is_distributed = num_gpus > 1


def loss_kwargs(sample_cuda, args):
    return {
        "imgs": sample_cuda["imgs"],
        "proj_matrices": sample_cuda["proj_matrices"],
        "dlossw": [float(e) for e in args.dlossw.split(",") if e],
        "raw_depth_loss_weight": args.raw_depth_loss_weight,
        "refined_depth_loss_weight": args.refined_depth_loss_weight,
        "residual_loss_weight": args.residual_loss_weight,
        "gate_loss_weight": args.gate_loss_weight,
        "safe_refine_loss_weight": args.safe_refine_loss_weight,
        "safe_refine_margin": args.safe_refine_margin,
        "residual_target_loss_weight": args.residual_target_loss_weight,
        "gate_benefit_loss_weight": args.gate_benefit_loss_weight,
        "residual_target_ratio": args.residual_target_ratio,
        "benefit_margin_ratio": args.benefit_margin_ratio,
        "residual_target_loss_scale": getattr(
            args, "current_residual_target_loss_scale", 1.0),
        "gate_benefit_loss_scale": getattr(
            args, "current_gate_benefit_loss_scale", 1.0),
        "sger_loss_scale": getattr(args, "current_sger_warmup_scale", 1.0),
        "normal_smooth_loss_weight": args.normal_smooth_loss_weight,
        "curv_loss_weight": args.curv_loss_weight,
        "edge_smooth_loss_weight": args.edge_smooth_loss_weight,
        "depth_normal_loss_weight": args.depth_normal_loss_weight,
        "depth_normal_conf_threshold": args.depth_normal_conf_threshold,
        "edge_grad_threshold": args.edge_grad_threshold,
        "geometry_conf_mid": args.geometry_conf_mid,
        "geometry_k_conf": args.geometry_k_conf,
        "geometry_edge_mid": args.geometry_edge_mid,
        "geometry_k_edge": args.geometry_k_edge,
        "geometry_w_min": args.geometry_w_min,
        "use_dual_region_curvature": args.use_dual_region_curvature,
        "region_lambda_a": args.region_lambda_a,
        "region_lambda_b": args.region_lambda_b,
        "region_edge_threshold": args.region_edge_threshold,
        "region_depth_threshold": args.region_depth_threshold,
        "region_curv_threshold": args.region_curv_threshold,
        "region_smooth_k": args.region_smooth_k,
        "return_extra": True,
    }


def validate_checkpoint_keys(missing_keys, unexpected_keys, use_sger, use_sger_lite=False):
    if not (use_sger or use_sger_lite):
        return
    allowed_missing_prefixes = (
        "sger_blocks.",
        "shared_sger.",
        "sger_feature_adapters.",
    )
    invalid_missing = [
        key for key in missing_keys
        if not key.startswith(allowed_missing_prefixes)
    ]
    if invalid_missing or unexpected_keys:
        raise RuntimeError(
            "checkpoint is incompatible with SGER initialization; "
            "unrelated missing keys: {}; unexpected keys: {}".format(
                invalid_missing, list(unexpected_keys)))


def compute_sger_warmup_scale(epoch_idx, start_epoch, end_epoch):
    if start_epoch < 0 or end_epoch < 0:
        raise ValueError("SGER warm-up epochs must be non-negative")
    if end_epoch < start_epoch:
        raise ValueError("SGER warm-up end must be >= start")
    if epoch_idx < start_epoch:
        return 0.0
    if epoch_idx >= end_epoch:
        return 1.0
    return float(epoch_idx - start_epoch + 1) / float(
        end_epoch - start_epoch + 1)


def compute_experiment14_loss_scales(
        epoch_idx, residual_start_epoch=4,
        gate_start_epoch=7, gate_end_epoch=10):
    if residual_start_epoch < 0 or gate_start_epoch < 0 or gate_end_epoch < 0:
        raise ValueError("Experiment14 schedule epochs must be non-negative")
    if gate_end_epoch < gate_start_epoch:
        raise ValueError("benefit gate end epoch must be >= start epoch")
    residual_scale = 1.0 if epoch_idx >= residual_start_epoch else 0.0
    if epoch_idx < gate_start_epoch:
        gate_scale = 0.0
    elif epoch_idx >= gate_end_epoch:
        gate_scale = 1.0
    else:
        gate_scale = float(epoch_idx - gate_start_epoch + 1) / float(
            gate_end_epoch - gate_start_epoch + 1)
    return residual_scale, gate_scale


def set_experiment14_loss_state(args, epoch_idx):
    residual_scale, gate_scale = compute_experiment14_loss_scales(epoch_idx)
    if not args.use_sger_lite:
        residual_scale, gate_scale = 1.0, 1.0
    args.current_residual_target_loss_scale = residual_scale
    args.current_gate_benefit_loss_scale = gate_scale
    return residual_scale, gate_scale


def set_sger_warmup_state(model, args, epoch_idx):
    module = model.module if hasattr(model, "module") else model
    scale = 1.0
    if args.use_sger_lite:
        scale = compute_sger_warmup_scale(
            epoch_idx,
            args.sger_warmup_start_epoch,
            args.sger_warmup_end_epoch)
    module.set_sger_residual_scale(scale)
    args.current_sger_warmup_scale = scale
    return scale


def validate_sger_warmup_checkpoint(checkpoint, args):
    start_epoch = checkpoint.get("sger_warmup_start_epoch")
    end_epoch = checkpoint.get("sger_warmup_end_epoch")
    if start_epoch is None and end_epoch is None:
        return
    expected = (
        args.sger_warmup_start_epoch,
        args.sger_warmup_end_epoch)
    actual = (start_epoch, end_epoch)
    if actual != expected:
        raise RuntimeError(
            "SGER warm-up configuration mismatch: checkpoint {} != "
            "current {}".format(actual, expected))


def is_sger_lite_parameter(module, name):
    trainable_prefixes = (
        "normal_head.{}.".format(module.num_stage - 1),
        "sger_blocks.",
        "shared_sger.",
        "sger_feature_adapters.{}.".format(module.num_stage - 1),
    )
    return name.startswith(trainable_prefixes)


def build_optimizer_param_groups(model, base_lr, backbone_lr_scale):
    module = model.module if hasattr(model, "module") else model
    if not getattr(module, "use_sger_lite", False):
        return [{
            "params": list(module.parameters()),
            "lr": base_lr,
            "name": "model",
        }]
    if not 0.0 < backbone_lr_scale <= 1.0:
        raise ValueError("backbone_lr_scale must be in (0, 1]")

    sger_params = []
    backbone_params = []
    for name, param in module.named_parameters():
        if is_sger_lite_parameter(module, name):
            sger_params.append(param)
        else:
            backbone_params.append(param)
    return [
        {"params": sger_params, "lr": base_lr, "name": "sger_lite"},
        {
            "params": backbone_params,
            "lr": base_lr * backbone_lr_scale,
            "name": "backbone",
        },
    ]


def set_sger_lite_freeze(model, freeze_backbone):
    module = model.module if hasattr(model, "module") else model
    if not getattr(module, "use_sger_lite", False):
        return
    for name, param in module.named_parameters():
        param.requires_grad = (
            (not freeze_backbone) or is_sger_lite_parameter(module, name))


# main function
def train(model, model_loss, optimizer, TrainImgLoader, TestImgLoader, start_epoch, args):
    milestones = [len(TrainImgLoader) * int(epoch_idx) for epoch_idx in args.lrepochs.split(':')[0].split(',')]
    lr_gamma = 1 / float(args.lrepochs.split(':')[1])
    lr_scheduler = WarmupMultiStepLR(optimizer, milestones, gamma=lr_gamma, warmup_factor=1.0/3, warmup_iters=500,
                                                        last_epoch=len(TrainImgLoader) * start_epoch - 1)

    for epoch_idx in range(start_epoch, args.epochs):
        print('Epoch {}:'.format(epoch_idx))
        sger_warmup_scale = set_sger_warmup_state(model, args, epoch_idx)
        residual_target_scale, benefit_gate_scale = (
            set_experiment14_loss_state(args, epoch_idx))
        if (not is_distributed) or (dist.get_rank() == 0):
            print("SGER residual/loss scale: {:.4f}".format(
                sger_warmup_scale))
            print(
                "Experiment14 residual-target/gate-benefit scales: "
                "{:.4f}/{:.4f}".format(
                    residual_target_scale, benefit_gate_scale))
        freeze_backbone = args.use_sger_lite and epoch_idx < args.freeze_backbone_epochs
        set_sger_lite_freeze(model, freeze_backbone)
        if freeze_backbone and ((not is_distributed) or (dist.get_rank() == 0)):
            print("SGER-Lite freeze: training Stage3 NormalHead and SGERBlock only")
        global_step = len(TrainImgLoader) * epoch_idx

        # training
        for batch_idx, sample in enumerate(TrainImgLoader):
            start_time = time.time()
            global_step = len(TrainImgLoader) * epoch_idx + batch_idx
            do_summary = global_step % args.summary_freq == 0
            loss, scalar_outputs, image_outputs = train_sample(model, model_loss, optimizer, sample, args)
            lr_scheduler.step()
            if (not is_distributed) or (dist.get_rank() == 0):
                if do_summary:
                    save_scalars(logger, 'train', scalar_outputs, global_step)
                    save_images(logger, 'train', image_outputs, global_step)
                    lr_text = ",".join(
                        "{}={:.6f}".format(
                            group.get("name", index), group["lr"])
                        for index, group in enumerate(optimizer.param_groups))
                    print(
                       "Epoch {}/{}, Iter {}/{}, lr {}, train loss = {:.3f}, depth loss = {:.3f}, time = {:.3f}".format(
                           epoch_idx, args.epochs, batch_idx, len(TrainImgLoader),
                           lr_text, loss,
                           scalar_outputs['depth_loss'],
                           time.time() - start_time))
                del scalar_outputs, image_outputs

        # checkpoint
        if (not is_distributed) or (dist.get_rank() == 0):
            if (epoch_idx + 1) % args.save_freq == 0:
                torch.save({
                    'epoch': epoch_idx,
                    'model': model.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'sger_warmup_start_epoch': args.sger_warmup_start_epoch,
                    'sger_warmup_end_epoch': args.sger_warmup_end_epoch},
                    "{}/model_{:0>6}.ckpt".format(args.logdir, epoch_idx))
        gc.collect()

        # testing
        if (epoch_idx % args.eval_freq == 0) or (epoch_idx == args.epochs - 1):
            avg_test_scalars = DictAverageMeter()
            for batch_idx, sample in enumerate(TestImgLoader):
                start_time = time.time()
                global_step = len(TrainImgLoader) * epoch_idx + batch_idx
                do_summary = global_step % args.summary_freq == 0
                loss, scalar_outputs, image_outputs = test_sample_depth(model, model_loss, sample, args)
                if (not is_distributed) or (dist.get_rank() == 0):
                    if do_summary:
                        save_scalars(logger, 'test', scalar_outputs, global_step)
                        save_images(logger, 'test', image_outputs, global_step)
                        print("Epoch {}/{}, Iter {}/{}, test loss = {:.3f}, depth loss = {:.3f}, time = {:3f}".format(
                                                                            epoch_idx, args.epochs,
                                                                            batch_idx,
                                                                            len(TestImgLoader), loss,
                                                                            scalar_outputs["depth_loss"],
                                                                            time.time() - start_time))
                    avg_test_scalars.update(scalar_outputs)
                    del scalar_outputs, image_outputs

            if (not is_distributed) or (dist.get_rank() == 0):
                save_scalars(logger, 'fulltest', avg_test_scalars.mean(), global_step)
                print("avg_test_scalars:", avg_test_scalars.mean())
            gc.collect()


def test(model, model_loss, TestImgLoader, args):
    avg_test_scalars = DictAverageMeter()
    for batch_idx, sample in enumerate(TestImgLoader):
        start_time = time.time()
        loss, scalar_outputs, image_outputs = test_sample_depth(model, model_loss, sample, args)
        avg_test_scalars.update(scalar_outputs)
        del scalar_outputs, image_outputs
        if (not is_distributed) or (dist.get_rank() == 0):
            print('Iter {}/{}, test loss = {:.3f}, time = {:3f}'.format(batch_idx, len(TestImgLoader), loss,
                                                                        time.time() - start_time))
            if batch_idx % 100 == 0:
                print("Iter {}/{}, test results = {}".format(batch_idx, len(TestImgLoader), avg_test_scalars.mean()))
    if (not is_distributed) or (dist.get_rank() == 0):
        print("final", avg_test_scalars.mean())


def train_sample(model, model_loss, optimizer, sample, args):
    model.train()
    optimizer.zero_grad()

    sample_cuda = tocuda(sample)
    depth_gt_ms = sample_cuda["depth"]
    mask_ms = sample_cuda["mask"]

    num_stage = len([int(nd) for nd in args.ndepths.split(",") if nd])
    depth_gt = depth_gt_ms["stage{}".format(num_stage)]
    mask = mask_ms["stage{}".format(num_stage)]

    outputs = model(sample_cuda["imgs"], sample_cuda["proj_matrices"], sample_cuda["depth_values"])
    depth_est = outputs["depth"]

    loss, depth_loss, normal_smooth_loss_final, curvature_loss_final, edge_aware_smooth_loss_final, loss_extra = model_loss(outputs, depth_gt_ms, mask_ms, **loss_kwargs(sample_cuda, args))


    if is_distributed and args.using_apex:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
    else:
        loss.backward()

    optimizer.step()

    scalar_outputs = {"loss": loss,
                      "depth_loss": depth_loss,
                      "normal_smooth_loss": normal_smooth_loss_final,
                      "curvature_loss": curvature_loss_final,
                      "edge_aware_smooth_loss": edge_aware_smooth_loss_final,
                      "abs_depth_error": AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5),
                      "thres2mm_error": Thres_metrics(depth_est, depth_gt, mask > 0.5, 2),
                      "thres4mm_error": Thres_metrics(depth_est, depth_gt, mask > 0.5, 4),
                      "thres8mm_error": Thres_metrics(depth_est, depth_gt, mask > 0.5, 8),}
    scalar_outputs.update({k: v for k, v in loss_extra.items() if k != "smooth_mask"})

    image_outputs = {"depth_est": depth_est * mask,
                     "depth_est_nomask": depth_est,
                     "depth_gt": sample["depth"]["stage1"],
                     "ref_img": sample["imgs"][:, 0],
                     "mask": sample["mask"]["stage1"],
                     "errormap": (depth_est - depth_gt).abs() * mask,
                     }
    if "normal" in outputs:
        image_outputs["normal_pred"] = (outputs["normal"] + 1.0) * 0.5
    if "depth_raw" in outputs:
        image_outputs["depth_raw"] = outputs["depth_raw"] * mask
    if "depth_residual" in outputs:
        image_outputs["depth_residual"] = outputs["depth_residual"]
    if "geometry_gate" in outputs:
        image_outputs["geometry_gate"] = outputs["geometry_gate"]
    if "region_a" in outputs:
        image_outputs["region_a"] = outputs["region_a"].float()
    if "smooth_mask" in loss_extra:
        image_outputs["smooth_mask"] = loss_extra["smooth_mask"].unsqueeze(1).float()

    if is_distributed:
        scalar_outputs = reduce_scalar_outputs(scalar_outputs)

    return tensor2float(scalar_outputs["loss"]), tensor2float(scalar_outputs), tensor2numpy(image_outputs)


@make_nograd_func
def test_sample_depth(model, model_loss, sample, args):
    if is_distributed:
        model_eval = model.module
    else:
        model_eval = model
    model_eval.eval()

    sample_cuda = tocuda(sample)
    depth_gt_ms = sample_cuda["depth"]
    mask_ms = sample_cuda["mask"]

    num_stage = len([int(nd) for nd in args.ndepths.split(",") if nd])
    depth_gt = depth_gt_ms["stage{}".format(num_stage)]
    mask = mask_ms["stage{}".format(num_stage)]

    outputs = model_eval(sample_cuda["imgs"], sample_cuda["proj_matrices"], sample_cuda["depth_values"])
    depth_est = outputs["depth"]

    loss, depth_loss, normal_smooth_loss_final, curvature_loss_final, edge_aware_smooth_loss_final, loss_extra = model_loss(outputs, depth_gt_ms, mask_ms, **loss_kwargs(sample_cuda, args))

    scalar_outputs = {"loss": loss,
                      "depth_loss": depth_loss,
                      "abs_depth_error": AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5),
                      "thres2mm_error": Thres_metrics(depth_est, depth_gt, mask > 0.5, 2),
                      "thres4mm_error": Thres_metrics(depth_est, depth_gt, mask > 0.5, 4),
                      "thres8mm_error": Thres_metrics(depth_est, depth_gt, mask > 0.5, 8),
                      "thres14mm_error": Thres_metrics(depth_est, depth_gt, mask > 0.5, 14),
                      "thres20mm_error": Thres_metrics(depth_est, depth_gt, mask > 0.5, 20),

                      "thres2mm_abserror": AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5, [0, 2.0]),
                      "thres4mm_abserror": AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5, [2.0, 4.0]),
                      "thres8mm_abserror": AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5, [4.0, 8.0]),
                      "thres14mm_abserror": AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5, [8.0, 14.0]),
                      "thres20mm_abserror": AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5, [14.0, 20.0]),
                      "thres>20mm_abserror": AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5, [20.0, 1e5]),
                    }
    scalar_outputs.update({k: v for k, v in loss_extra.items() if k != "smooth_mask"})

    image_outputs = {"depth_est": depth_est * mask,
                     "depth_est_nomask": depth_est,
                     "depth_gt": sample["depth"]["stage1"],
                     "ref_img": sample["imgs"][:, 0],
                     "mask": sample["mask"]["stage1"],
                     "errormap": (depth_est - depth_gt).abs() * mask}
    if "normal" in outputs:
        image_outputs["normal_pred"] = (outputs["normal"] + 1.0) * 0.5
    if "depth_raw" in outputs:
        image_outputs["depth_raw"] = outputs["depth_raw"] * mask
    if "depth_residual" in outputs:
        image_outputs["depth_residual"] = outputs["depth_residual"]
    if "geometry_gate" in outputs:
        image_outputs["geometry_gate"] = outputs["geometry_gate"]
    if "region_a" in outputs:
        image_outputs["region_a"] = outputs["region_a"].float()
    if "smooth_mask" in loss_extra:
        image_outputs["smooth_mask"] = loss_extra["smooth_mask"].unsqueeze(1).float()

    if is_distributed:
        scalar_outputs = reduce_scalar_outputs(scalar_outputs)

    return tensor2float(scalar_outputs["loss"]), tensor2float(scalar_outputs), tensor2numpy(image_outputs)

def profile():
    warmup_iter = 5
    iter_dataloader = iter(TestImgLoader)

    @make_nograd_func
    def do_iteration():
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        test_sample(next(iter_dataloader), detailed_summary=True)
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        return end_time - start_time

    for i in range(warmup_iter):
        t = do_iteration()
        print('WarpUp Iter {}, time = {:.4f}'.format(i, t))

    with torch.autograd.profiler.profile(enabled=True, use_cuda=True) as prof:
        for i in range(5):
            t = do_iteration()
            print('Profile Iter {}, time = {:.4f}'.format(i, t))
            time.sleep(0.02)

    if prof is not None:
        # print(prof)
        trace_fn = 'chrome-trace.bin'
        prof.export_chrome_trace(trace_fn)
        print("chrome trace file is written to: ", trace_fn)


if __name__ == '__main__':
    # parse arguments and check
    args = parser.parse_args()

    # using sync_bn by using nvidia-apex, need to install apex.
    if args.sync_bn:
        assert args.using_apex, "must set using apex and install nvidia-apex"
    if args.using_apex:
        try:
            from apex.parallel import DistributedDataParallel as DDP
            from apex.fp16_utils import *
            from apex import amp, optimizers
            from apex.multi_tensor_apply import multi_tensor_applier
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")

    if args.resume:
        assert args.mode == "train"
        assert args.loadckpt is None
    if args.use_sger and args.use_sger_lite:
        raise ValueError("--use_sger and --use_sger_lite are mutually exclusive")
    if args.testpath is None:
        args.testpath = args.trainpath

    if is_distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    set_random_seed(args.seed)
    device = torch.device(args.device)

    if (not is_distributed) or (dist.get_rank() == 0):
        # create logger for mode "train" and "testall"
        if args.mode == "train":
            if not os.path.isdir(args.logdir):
                os.makedirs(args.logdir)
            current_time_str = str(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
            print("current time", current_time_str)
            print("creating new summary file")
            logger = SummaryWriter(args.logdir)
        print("argv:", sys.argv[1:])
        print_args(args)

    # model, optimizer
    model = CascadeMVSNet(refine=False, ndepths=[int(nd) for nd in args.ndepths.split(",") if nd],
                          depth_interals_ratio=[float(d_i) for d_i in args.depth_inter_r.split(",") if d_i],
                          share_cr=args.share_cr,
                          cr_base_chs=[int(ch) for ch in args.cr_base_chs.split(",") if ch],
                          grad_method=args.grad_method,
                          use_sger=args.use_sger,
                          use_sger_lite=args.use_sger_lite,
                          sger_share=args.sger_share,
                          sger_feature_channels=args.sger_feature_channels,
                          sger_hidden_channels=args.sger_hidden_channels,
                          sger_gate_channels=args.sger_gate_channels,
                          sger_max_residual_ratio=args.sger_max_residual_ratio,
                          detach_refined_feedback=args.detach_refined_feedback,
                          sger_gate_kwargs={
                              "threshold_edge": args.region_edge_threshold,
                              "threshold_depth": args.region_depth_threshold,
                              "threshold_curv": args.region_curv_threshold,
                              "conf_mid": args.geometry_conf_mid,
                              "k_conf": args.geometry_k_conf,
                              "smooth_k": args.region_smooth_k,
                          })
    model.to(device)
    model_loss = cas_mvsnet_loss

    if args.sync_bn:
        import apex
        print("using apex synced BN")
        model = apex.parallel.convert_syncbn_model(model)

    optimizer_groups = build_optimizer_param_groups(
        model, args.lr, args.backbone_lr_scale)
    optimizer = optim.Adam(
        optimizer_groups, lr=args.lr, betas=(0.9, 0.999),
        weight_decay=args.wd)

    # load parameters
    start_epoch = 0
    if args.resume:
        saved_models = [fn for fn in os.listdir(args.logdir) if fn.endswith(".ckpt")]
        saved_models = sorted(saved_models, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        # use the latest checkpoint file
        loadckpt = os.path.join(args.logdir, saved_models[-1])
        print("resuming", loadckpt)
        state_dict = torch.load(loadckpt, map_location=torch.device("cpu"))
        validate_sger_warmup_checkpoint(state_dict, args)
        model.load_state_dict(state_dict['model'])
        optimizer.load_state_dict(state_dict['optimizer'])
        start_epoch = state_dict['epoch'] + 1
    elif args.loadckpt:
        # load checkpoint file specified by args.loadckpt
        print("loading model {}".format(args.loadckpt))
        state_dict = torch.load(args.loadckpt, map_location=torch.device("cpu"))
        missing_keys, unexpected_keys = model.load_state_dict(state_dict['model'], strict=False)
        validate_checkpoint_keys(missing_keys, unexpected_keys, args.use_sger, args.use_sger_lite)
        if missing_keys:
            print("missing keys when loading checkpoint:", missing_keys)
        if unexpected_keys:
            print("unexpected keys when loading checkpoint:", unexpected_keys)

    if (not is_distributed) or (dist.get_rank() == 0):
        print("start at epoch {}".format(start_epoch))
        print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    if args.using_apex:
        # Initialize Amp
        model, optimizer = amp.initialize(model, optimizer,
                                          opt_level=args.opt_level,
                                          keep_batchnorm_fp32=args.keep_batchnorm_fp32,
                                          loss_scale=args.loss_scale
                                          )

    if is_distributed:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank,
            # find_unused_parameters=False,
            # this should be removed if we update BatchNorm stats
            # broadcast_buffers=False,
        )
    else:
        if torch.cuda.is_available():
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)

    # dataset, dataloader
    MVSDataset = find_dataset_def(args.dataset)
    train_dataset = MVSDataset(args.trainpath, args.trainlist, "train", 3, args.numdepth, args.interval_scale)
    test_dataset = MVSDataset(args.testpath, args.testlist, "test", 5, args.numdepth, args.interval_scale)

    if is_distributed:
        train_sampler = torch.utils.data.DistributedSampler(train_dataset, num_replicas=dist.get_world_size(),
                                                            rank=dist.get_rank())
        test_sampler = torch.utils.data.DistributedSampler(test_dataset, num_replicas=dist.get_world_size(),
                                                           rank=dist.get_rank())

        TrainImgLoader = DataLoader(train_dataset, args.batch_size, sampler=train_sampler, num_workers=1,
                                    drop_last=True,
                                    pin_memory=args.pin_m)
        TestImgLoader = DataLoader(test_dataset, args.batch_size, sampler=test_sampler, num_workers=1, drop_last=False,
                                   pin_memory=args.pin_m)
    else:
        TrainImgLoader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=1, drop_last=True,
                                    pin_memory=args.pin_m)
        TestImgLoader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=1, drop_last=False,
                                   pin_memory=args.pin_m)


    if args.mode == "train":
        train(model, model_loss, optimizer, TrainImgLoader, TestImgLoader, start_epoch, args)
    elif args.mode == "test":
        test(model, model_loss, TestImgLoader, args)
    elif args.mode == "profile":
        profile()
    else:
        raise NotImplementedError
