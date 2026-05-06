"""
Cross-session distillation: REALM Teacher -> REALM Student.

Teacher (pretrained, frozen):
  - BiMamba-2, bidirectional, expand=2, 10 layers, ~10.9M params
Student (trained from scratch / spatial-init):
  - Mamba-2, causal, expand=1, 8 layers, ~2.1M params

Two distillation approaches:
  A) CrossModalDistill (Erturk et al. NeurIPS 2025):
     L = L_task + lambda_repr * L_cosine + lambda_ae * L_ae
  B) ReprDynamics (ours):
     L = alpha * L_task + beta * L_repr(CKA) + gamma * L_dynamics(SSM states)

Usage:
    # CrossModalDistill loss (default)
    python -m REALM.scripts.distill_crosssession \
        --pretrained output/pretrain_REALM_teacher_best.pt \
        --loss_type realm

    # ReprDynamics loss
    python -m REALM.scripts.distill_crosssession \
        --pretrained output/pretrain_REALM_teacher_best.pt \
        --loss_type reprdyn
"""

import argparse
import json
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.dataset import (
    create_test_dataloaders, compute_r2,
    discover_sessions, HELDOUT_SESSIONS,
    HELDOUT_MAKIN, HELDOUT_FLINT, MAX_CHANNELS,
)
from models.decoder import REALMDecoder
from models.configs import (
    STUDENT_REALM_S_KWARGS, STUDENT_REALM_KWARGS, STUDENT_REALM_L_KWARGS, STUDENT_REALM_XL_KWARGS,
    STUDENT_REALM_BI_L_KWARGS,
    STUDENT_REALM_UXL_KWARGS, STUDENT_REALM_XLBI_KWARGS,
    STUDENT_REALM_BI_KWARGS, STUDENT_REALM_LBI_KWARGS,
)
from utils.distill_losses import (
    CrossModalDistillLoss, ReprDynamicsDistillLoss,
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_teacher(ckpt_path):
    """Load pretrained BiMamba-2 teacher (frozen)."""
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    encoder_kwargs = ckpt['encoder_kwargs'].copy()
    encoder_kwargs['max_sessions'] = 200

    teacher = REALMDecoder(
        encoder_kwargs=encoder_kwargs, output_dim=2,
    ).to(DEVICE)

    state = {k: v for k, v in ckpt['encoder_state_dict'].items()
             if 'session_embed' not in k and 'pos_embed' not in k}
    teacher.encoder.load_state_dict(state, strict=False)

    # Freeze all teacher parameters
    for p in teacher.parameters():
        p.requires_grad = False
    teacher.eval()

    total = sum(p.numel() for p in teacher.parameters())
    print(f"  Teacher: {total:,} params (all frozen)")
    print(f"  Teacher config: d_model={encoder_kwargs['d_model']}, "
          f"n_layers={encoder_kwargs['n_layers']}, "
          f"expand={encoder_kwargs['expand']}, "
          f"bidirectional={encoder_kwargs.get('bidirectional', True)}")
    return teacher, encoder_kwargs


def create_student_model(teacher, student_size='realm', disable_eca=False,
                         head_layers=1, head_dropout=0.0):
    """Create student and optionally copy spatial layers from teacher."""
    # New canonical names (preferred)
    if student_size in ('realm_s', 'realm'):
        student_kwargs = {**(STUDENT_REALM_S_KWARGS if student_size == 'realm_s'
                              else STUDENT_REALM_KWARGS), 'max_sessions': 200}
    elif student_size == 'realm_l':
        student_kwargs = {**STUDENT_REALM_L_KWARGS, 'max_sessions': 200}
    elif student_size == 'realm_bi':
        student_kwargs = {**STUDENT_REALM_BI_KWARGS, 'max_sessions': 200}
    elif student_size == 'realm_bi_l':
        student_kwargs = {**STUDENT_REALM_BI_L_KWARGS, 'max_sessions': 200}
    # Legacy aliases (still resolvable via aliases in configs.py)
    elif student_size == 'realm_xl':
        student_kwargs = {**STUDENT_REALM_XL_KWARGS, 'max_sessions': 200}
    elif student_size == 'realm_uxl':
        student_kwargs = {**STUDENT_REALM_UXL_KWARGS, 'max_sessions': 200}
    elif student_size == 'realm_xlbi':
        student_kwargs = {**STUDENT_REALM_XLBI_KWARGS, 'max_sessions': 200}
    elif student_size == 'realm_lbi':
        student_kwargs = {**STUDENT_REALM_LBI_KWARGS, 'max_sessions': 200}
    else:
        raise ValueError(f"Unknown student_size: {student_size}")

    if disable_eca:
        student_kwargs['disable_eca'] = True

    student = REALMDecoder(
        encoder_kwargs=student_kwargs, output_dim=2,
        head_layers=head_layers, head_dropout=head_dropout,
    ).to(DEVICE)

    # Copy spatial layers from teacher (same architecture if d_channel matches)
    t_enc = teacher.encoder
    s_enc = student.encoder

    try:
        s_enc.temporal_conv.load_state_dict(t_enc.temporal_conv.state_dict())
        s_enc.eca_conv.load_state_dict(t_enc.eca_conv.state_dict())
        s_enc.spatial_norm.load_state_dict(t_enc.spatial_norm.state_dict())

        # spatial_proj: only works if n_channels * d_channel -> d_model match
        if hasattr(t_enc, 'spatial_proj') and hasattr(s_enc, 'spatial_proj'):
            t_shape = t_enc.spatial_proj.weight.shape
            s_shape = s_enc.spatial_proj.weight.shape
            if t_shape == s_shape:
                s_enc.spatial_proj.load_state_dict(
                    t_enc.spatial_proj.state_dict())
                print("  Copied spatial_proj from teacher")
            else:
                print(f"  spatial_proj shape mismatch: teacher={t_shape}, "
                      f"student={s_shape}, skipping")
        print("  Copied spatial layers (temporal_conv, eca_conv, spatial_norm) "
              "from teacher")
    except Exception as e:
        print(f"  Warning: could not copy spatial layers: {e}")

    total     = sum(p.numel() for p in student.parameters())
    trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print(f"  Student: {total:,} params ({trainable:,} trainable)")
    print(f"  Student config: d_model={student_kwargs['d_model']}, "
          f"n_layers={student_kwargs['n_layers']}, "
          f"expand={student_kwargs['expand']}, "
          f"bidirectional={student_kwargs.get('bidirectional', False)}")
    return student, student_kwargs


def collect_session_data(loader):
    """Collect all segments from a session loader into tensors."""
    all_lfp, all_cmask, all_tgt = [], [], []
    for batch in loader:
        all_lfp.append(batch['lfp'])
        all_cmask.append(batch['channel_mask'])
        all_tgt.append(batch['target'])
    return (torch.cat(all_lfp,   dim=0),
            torch.cat(all_cmask, dim=0),
            torch.cat(all_tgt,   dim=0))


def load_session_segments(info, segment_length):
    """Load a single session's LFP+target into non-overlapping segments."""
    data = np.load(info['path'], allow_pickle=True)
    lfp = data['lfp_data'].astype(np.float32)     # (n_ch, 1, T)
    targets = data['targets'].astype(np.float32)   # (T, 2)
    n_ch = lfp.shape[0]

    if n_ch < MAX_CHANNELS:
        pad = np.zeros((MAX_CHANNELS - n_ch, lfp.shape[1], lfp.shape[2]),
                       dtype=np.float32)
        lfp = np.concatenate([lfp, pad], axis=0)

    T = lfp.shape[2]
    lfp_segs, cmask_segs, tgt_segs = [], [], []
    ch_mask = torch.zeros(MAX_CHANNELS)
    ch_mask[:n_ch] = 1.0

    for start in range(0, T - segment_length + 1, segment_length):
        end = start + segment_length
        lfp_segs.append(torch.from_numpy(lfp[:, :, start:end].copy()))
        tgt_segs.append(torch.from_numpy(targets[start:end].copy()))
        cmask_segs.append(ch_mask.clone())

    return (torch.stack(lfp_segs), torch.stack(cmask_segs), torch.stack(tgt_segs))


def compute_val_loss(teacher, student, criterion, val_loader, args, need_states):
    """Compute validation loss (distillation loss on held-out data)."""
    student.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for lfp, cmask, tgt, sid_b in val_loader:
            lfp   = lfp.to(DEVICE)
            cmask = cmask.to(DEVICE)
            tgt   = tgt.to(DEVICE)
            s_ids = sid_b.to(DEVICE) if args.use_session_embed else None

            # Teacher forward (always without session_ids — teacher is frozen)
            t_out = teacher(lfp, session_ids=None, channel_mask=cmask,
                            return_intermediates=True)
            teacher_repr = t_out['intermediates'][-1]
            if args.loss_type != 'realm' or args.align_layers > 1 or args.align_layer_from_end is not None:
                teacher_intermediates = t_out['intermediates']
            if need_states:
                teacher_states = teacher.get_captured_states()

            # Student forward
            s_out = student(lfp, session_ids=s_ids, channel_mask=cmask,
                            return_intermediates=True)
            student_pred = s_out['prediction']
            student_repr = s_out['intermediates'][-1]

            if args.loss_type == 'realm':
                cm_kwargs = dict(
                    student_pred=student_pred, target=tgt,
                    student_repr=student_repr, teacher_repr=teacher_repr,
                    lfp_data=lfp,
                )
                if args.align_layers > 1 or args.align_layer_from_end is not None:
                    cm_kwargs['student_intermediates'] = s_out['intermediates']
                    cm_kwargs['teacher_intermediates'] = teacher_intermediates
                loss_dict = criterion(**cm_kwargs)
            else:
                student_intermediates = s_out['intermediates']
                student_states = student.get_captured_states() if need_states else None
                loss_dict = criterion(
                    student_pred=student_pred, target=tgt,
                    student_intermediates=student_intermediates,
                    teacher_intermediates=teacher_intermediates,
                    student_ssm_states=student_states,
                    teacher_ssm_states=teacher_states if need_states else None,
                )

            total_loss += loss_dict['loss'].item()
            n_batches += 1

    return total_loss / max(n_batches, 1)


def distill_cross_session(teacher, student, train_sessions_data, val_sessions_data, args):
    """Distillation training on Makin+Flint (non-heldout), val on heldout sessions."""

    # Combine all train session data with per-session IDs
    all_train_lfp, all_train_cmask, all_train_tgt, all_train_sid = [], [], [], []
    for sid, (lfp, cmask, tgt) in enumerate(train_sessions_data):
        all_train_lfp.append(lfp)
        all_train_cmask.append(cmask)
        all_train_tgt.append(tgt)
        all_train_sid.append(torch.full((len(lfp),), sid, dtype=torch.long))

    # Combine all val (heldout) session data
    all_val_lfp, all_val_cmask, all_val_tgt, all_val_sid = [], [], [], []
    n_train_sessions = len(train_sessions_data)
    for sid, (lfp, cmask, tgt) in enumerate(val_sessions_data):
        all_val_lfp.append(lfp)
        all_val_cmask.append(cmask)
        all_val_tgt.append(tgt)
        all_val_sid.append(torch.full((len(lfp),), n_train_sessions + sid, dtype=torch.long))

    train_lfp   = torch.cat(all_train_lfp,   dim=0)
    train_cmask = torch.cat(all_train_cmask, dim=0)
    train_tgt   = torch.cat(all_train_tgt,   dim=0)
    val_lfp     = torch.cat(all_val_lfp,     dim=0)
    val_cmask   = torch.cat(all_val_cmask,   dim=0)
    val_tgt     = torch.cat(all_val_tgt,     dim=0)

    n_train = len(train_lfp)
    n_val   = len(val_lfp)
    print(f"  Train: {n_train} segments from {len(train_sessions_data)} non-heldout sessions")
    print(f"  Val:   {n_val} segments from {len(val_sessions_data)} heldout sessions")

    if args.use_session_embed:
        sid_train = torch.cat(all_train_sid, dim=0)
        sid_val   = torch.cat(all_val_sid, dim=0)
        print(f"  Session embedding enabled: {n_train_sessions} train + {len(val_sessions_data)} val sessions")
    else:
        sid_train = torch.zeros(n_train, dtype=torch.long)
        sid_val   = torch.zeros(n_val,   dtype=torch.long)
    train_ds  = TensorDataset(train_lfp, train_cmask, train_tgt, sid_train)
    val_ds    = TensorDataset(val_lfp,   val_cmask,   val_tgt,   sid_val)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False)

    # Set up distillation loss
    if args.loss_type == 'realm':
        criterion = CrossModalDistillLoss(
            lambda_task=args.lambda_task,
            lambda_repr=args.lambda_repr,
            lambda_ae=args.lambda_ae,
            d_model=args.student_d_model,
            n_channels=96,
            n_bands=1,
            ae_mask_ratio=args.ae_mask_ratio,
            align_layers=args.align_layers,
            align_layer_from_end=args.align_layer_from_end,
        ).to(DEVICE)
        need_states = False
        if args.align_layer_from_end is not None:
            layers_str = f"from_end_{args.align_layer_from_end}"
        elif args.align_layers > 1:
            layers_str = f"last_{args.align_layers}_layers"
        else:
            layers_str = "last_layer"
        print(f"  Loss: REALM distill (lambda_repr={args.lambda_repr}, "
              f"lambda_ae={args.lambda_ae}, {layers_str})")
    else:
        student_d_inner = args.student_d_model * args.student_expand
        teacher_d_inner = args.teacher_d_model * args.teacher_expand
        criterion = ReprDynamicsDistillLoss(
            alpha=args.alpha,
            beta=args.beta,
            gamma=args.gamma,
            n_layers=min(args.student_n_layers, args.teacher_n_layers),
            student_d_inner=student_d_inner,
            teacher_d_inner=teacher_d_inner,
            repr_loss_type='cka',
        ).to(DEVICE)
        need_states = (args.gamma > 0)
        print(f"  Loss: ReprDynamics (alpha={args.alpha}, beta={args.beta}, "
              f"gamma={args.gamma}, need_states={need_states})")

    # Optimizer: student params + loss params (ae_decoder, dynamics projectors)
    opt_params = list(student.parameters()) + list(criterion.parameters())
    optimizer = optim.AdamW(
        [p for p in opt_params if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01,
    )

    best_val_loss = float('inf')
    best_state    = None
    patience      = 0

    if need_states:
        teacher.enable_state_capture(True)
        student.enable_state_capture(True)

    for epoch in range(1, args.epochs + 1):
        student.train()
        total_loss, total_task, total_repr, total_extra = 0.0, 0.0, 0.0, 0.0
        n_batches = 0

        for lfp, cmask, tgt, sid_b in train_loader:
            lfp   = lfp.to(DEVICE)
            cmask = cmask.to(DEVICE)
            tgt   = tgt.to(DEVICE)
            s_ids = sid_b.to(DEVICE) if args.use_session_embed else None

            optimizer.zero_grad(set_to_none=True)

            # Teacher forward (no grad, always without session_ids — teacher is frozen)
            with torch.no_grad():
                t_out = teacher(lfp, session_ids=None, channel_mask=cmask,
                                return_intermediates=True)
                teacher_repr = t_out['intermediates'][-1].detach()
                if args.loss_type != 'realm' or args.align_layers > 1 or args.align_layer_from_end is not None:
                    teacher_intermediates = [h.detach() for h in t_out['intermediates']]
                if need_states:
                    teacher_states = teacher.get_captured_states()

            # Student forward
            s_out = student(lfp, session_ids=s_ids, channel_mask=cmask,
                            return_intermediates=True)
            student_pred = s_out['prediction']
            student_repr = s_out['intermediates'][-1]

            # Compute loss
            if args.loss_type == 'realm':
                cm_kwargs = dict(
                    student_pred=student_pred,
                    target=tgt,
                    student_repr=student_repr,
                    teacher_repr=teacher_repr,
                    lfp_data=lfp,
                )
                if args.align_layers > 1 or args.align_layer_from_end is not None:
                    cm_kwargs['student_intermediates'] = s_out['intermediates']
                    cm_kwargs['teacher_intermediates'] = teacher_intermediates
                loss_dict = criterion(**cm_kwargs)
                total_repr += loss_dict['l_repr']
                total_extra += loss_dict['l_ae']
            else:
                student_intermediates = s_out['intermediates']
                student_states = student.get_captured_states() if need_states else None
                loss_dict = criterion(
                    student_pred=student_pred,
                    target=tgt,
                    student_intermediates=student_intermediates,
                    teacher_intermediates=teacher_intermediates,
                    student_ssm_states=student_states,
                    teacher_ssm_states=teacher_states if need_states else None,
                )
                total_repr += loss_dict['l_repr']
                total_extra += loss_dict['l_dynamics']

            loss = loss_dict['loss']
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_task += loss_dict['l_task']
            n_batches += 1

        scheduler.step()

        avg_loss = total_loss / max(n_batches, 1)
        avg_task = total_task / max(n_batches, 1)
        avg_repr = total_repr / max(n_batches, 1)
        avg_extra = total_extra / max(n_batches, 1)

        # Validation loss on heldout splits
        val_loss = compute_val_loss(
            teacher, student, criterion, val_loader, args, need_states)

        extra_name = 'ae' if args.loss_type == 'realm' else 'dyn'
        is_best_str = " *" if val_loss < best_val_loss else ""
        print(f"  Epoch {epoch:3d}/{args.epochs} | "
              f"train={avg_loss:.5f} val={val_loss:.5f} | "
              f"task={avg_task:.4f} repr={avg_repr:.4f} {extra_name}={avg_extra:.4f} | "
              f"best_val={min(best_val_loss, val_loss):.5f}{is_best_str}")

        tag = f"distill_{args.loss_type}_{args.student_size}"
        if args.tag:
            tag = f"{tag}_{args.tag}"

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone()
                          for k, v in student.state_dict().items()}
            patience = 0
            # Save best checkpoint
            best_ckpt = OUTPUT_DIR / f"{tag}_student_best.pt"
            torch.save({
                'student_state_dict': student.state_dict(),
                'encoder_kwargs': args._student_kwargs,
                'epoch': epoch,
                'val_loss': best_val_loss,
            }, best_ckpt)
        else:
            patience += 1
            if patience >= args.patience:
                print(f"\n  Early stop at epoch {epoch}  "
                      f"best_val_loss={best_val_loss:.5f}")
                break

        # Periodic checkpoint
        if epoch % args.save_every == 0 or epoch == args.epochs:
            periodic_ckpt = OUTPUT_DIR / f"{tag}_student_epoch{epoch:03d}.pt"
            torch.save({
                'student_state_dict': student.state_dict(),
                'encoder_kwargs': args._student_kwargs,
                'epoch': epoch,
                'val_loss': val_loss,
            }, periodic_ckpt)
            print(f"\n  Saved checkpoint: {periodic_ckpt.name}")


    if need_states:
        teacher.enable_state_capture(False)
        student.enable_state_capture(False)

    if best_state is not None:
        student.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    return student


@torch.no_grad()
def evaluate_session(model, test_lfp, test_cmask, test_tgt):
    """Evaluate R^2 on test split."""
    model.eval()
    sid = torch.zeros(len(test_lfp), dtype=torch.long)
    ds  = TensorDataset(test_lfp, test_cmask, test_tgt, sid)
    loader = DataLoader(ds, batch_size=64, shuffle=False)

    preds, targets = [], []
    for lfp, cmask, tgt, _ in loader:
        out = model(lfp.to(DEVICE), session_ids=None,
                    channel_mask=cmask.to(DEVICE))
        preds.append(out['prediction'].cpu())
        targets.append(tgt)

    preds   = torch.cat(preds,   dim=0).reshape(-1, 2)
    targets = torch.cat(targets, dim=0).reshape(-1, 2)
    return compute_r2(preds, targets)


def main():
    parser = argparse.ArgumentParser(
        description="Cross-session distillation: REALM Teacher -> REALM Student")
    parser.add_argument('--pretrained', type=str,
                        default='output/pretrain_REALM_teacher_best.pt')
    parser.add_argument('--loss_type', type=str, default='realm',
                        choices=['realm', 'reprdyn'])
    parser.add_argument('--student_size', type=str, default='realm',
                        choices=['realm_s', 'realm', 'realm_l',
                                 'realm_bi', 'realm_bi_l',
                                 'realm_xl', 'realm_uxl', 'realm_xlbi', 'realm_lbi'])

    # REALM distill hyperparams
    parser.add_argument('--lambda_task', type=float, default=1.0,
                        help="Task loss weight (0=unsupervised distillation)")
    parser.add_argument('--lambda_repr', type=float, default=1.0)
    parser.add_argument('--lambda_ae',   type=float, default=0.1)
    parser.add_argument('--ae_mask_ratio', type=float, default=0.3,
                        help="Mask ratio for AE reconstruction loss")
    parser.add_argument('--align_layers', type=int, default=2,
                        help="Number of last encoder layers to align (1=last layer, 2=last 2 layers, etc.)")
    parser.add_argument('--align_layer_from_end', type=int, default=None,
                        help="If set (0=last, 1=2nd-to-last, 2=3rd-to-last, ...), "
                             "align ONLY that single layer. Overrides --align_layers.")

    # ReprDynamics hyperparams
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--beta',  type=float, default=0.5)
    parser.add_argument('--gamma', type=float, default=0.1)

    # Training
    parser.add_argument('--epochs',       type=int,   default=300)
    parser.add_argument('--lr',           type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--patience',     type=int,   default=30)
    parser.add_argument('--batch_size',   type=int,   default=32)
    parser.add_argument('--train_ratio',  type=float, default=0.8)
    parser.add_argument('--segment_length', type=int, default=500)
    parser.add_argument('--seed',         type=int,   default=42)
    parser.add_argument('--save_every',   type=int,   default=25,
                        help="Save checkpoint every N epochs")
    parser.add_argument('--tag',          type=str,   default=None,
                        help="Extra tag for checkpoint naming (e.g. 'ep010')")
    parser.add_argument('--use_session_embed', action='store_true',
                        help="Enable per-session embedding during distillation")
    parser.add_argument('--disable_eca', action='store_true',
                        help="Disable ECA channel attention in student")
    parser.add_argument('--datasets',     type=str,   nargs='+',
                        default=['makin', 'flint'],
                        choices=['makin', 'flint'],
                        help="Which datasets to use for distillation (default: both)")
    parser.add_argument('--head_layers',  type=int,   default=1,
                        help="Number of layers in student out_proj head (1=linear, 3=MLP)")
    parser.add_argument('--head_dropout', type=float, default=0.0,
                        help="Dropout probability inside multi-layer head (only used if head_layers>1)")
    # Stacking ensemble: disjoint K-fold session subset
    parser.add_argument('--fold_idx',  type=int, default=-1,
                        help="If >=0, drop this fold (~1/n_folds of train sessions) for ensemble base model")
    parser.add_argument('--n_folds',   type=int, default=5,
                        help="Total folds for disjoint K-fold session subsetting")
    parser.add_argument('--fold_seed', type=int, default=42,
                        help="Seed for shuffling sessions before fold assignment (fixed across folds)")
    args = parser.parse_args()

    set_seed(args.seed)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Device: {DEVICE}")
    print(f"Loss: {args.loss_type}  Student: {args.student_size}")

    base_dir  = Path(__file__).resolve().parent.parent
    ckpt_path = base_dir / args.pretrained
    if not ckpt_path.exists():
        print(f"ERROR: {ckpt_path} not found")
        return

    # Load teacher
    print("\n=== Loading Teacher ===")
    teacher, teacher_kwargs = load_teacher(ckpt_path)

    # Store teacher config for loss computation
    args.teacher_d_model  = teacher_kwargs['d_model']
    args.teacher_expand   = teacher_kwargs['expand']
    args.teacher_n_layers = teacher_kwargs['n_layers']

    # Create student
    print("\n=== Creating Student ===")
    student, student_kwargs = create_student_model(teacher, args.student_size,
                                                    disable_eca=args.disable_eca,
                                                    head_layers=args.head_layers,
                                                    head_dropout=args.head_dropout)
    if args.head_layers > 1:
        n_head = sum(p.numel() for p in student.out_proj.parameters())
        print(f"  Student head: {args.head_layers}-layer MLP, {n_head:,} params, dropout={args.head_dropout}")
    args.student_d_model  = student_kwargs['d_model']
    args.student_expand   = student_kwargs['expand']
    args.student_n_layers = student_kwargs['n_layers']

    # Load data: train = non-heldout, val/eval = heldout
    print(f"\n=== Loading Data (datasets={args.datasets}) ===")
    all_sessions = discover_sessions(datasets=args.datasets)

    # Filter heldout by selected datasets
    active_heldout = set()
    if 'makin' in args.datasets:
        active_heldout.update(HELDOUT_MAKIN)
    if 'flint' in args.datasets:
        active_heldout.update(HELDOUT_FLINT)

    train_sessions_data = []
    train_session_names = []
    heldout_sessions_data = {}

    for sess_name, info in sorted(all_sessions.items()):
        lfp, cmask, tgt = load_session_segments(info, args.segment_length)
        if sess_name in active_heldout:
            heldout_sessions_data[sess_name] = (lfp, cmask, tgt)
            print(f"  [heldout] {sess_name:<25} N={len(lfp)}")
        else:
            train_sessions_data.append((lfp, cmask, tgt))
            train_session_names.append(sess_name)
            print(f"  [train]   {sess_name:<25} N={len(lfp)}")

    print(f"\nTrain: {len(train_sessions_data)} sessions, "
          f"Heldout: {len(heldout_sessions_data)} sessions")

    # Disjoint K-fold session subset for stacking ensemble
    fold_dropped_names = []
    fold_kept_names = list(train_session_names)
    if args.fold_idx >= 0:
        rng = np.random.RandomState(args.fold_seed)
        order = list(range(len(train_session_names)))
        rng.shuffle(order)
        fold_size = len(order) // args.n_folds
        folds = [order[k * fold_size:(k + 1) * fold_size] for k in range(args.n_folds)]
        for j, idx in enumerate(order[args.n_folds * fold_size:]):
            folds[j].append(idx)
        drop_set = set(folds[args.fold_idx])
        fold_dropped_names = [train_session_names[i] for i in sorted(drop_set)]
        keep_data = [d for i, d in enumerate(train_sessions_data) if i not in drop_set]
        keep_names = [n for i, n in enumerate(train_session_names) if i not in drop_set]
        print(f"\n=== Stacking fold {args.fold_idx}/{args.n_folds}: "
              f"dropping {len(drop_set)} sessions, keeping {len(keep_data)} ===")
        print(f"  Dropped: {fold_dropped_names}")
        train_sessions_data = keep_data
        train_session_names = keep_names
        fold_kept_names = keep_names

    # Store student kwargs for checkpoint saving
    args._student_kwargs = student_kwargs

    # Distillation
    print(f"\n=== Distillation ({args.loss_type}) ===")
    t0 = time.time()
    student = distill_cross_session(
        teacher, student,
        train_sessions_data,
        list(heldout_sessions_data.values()),
        args,
    )
    elapsed = time.time() - t0
    print(f"  Training done in {elapsed:.1f}s")

    # Per-session evaluation on heldout sessions (full data)
    print("\n=== Evaluation (Heldout Sessions) ===")
    per_session_r2 = {}
    for sess_name, (lfp, cmask, tgt) in sorted(heldout_sessions_data.items()):
        r2 = evaluate_session(student, lfp, cmask, tgt)
        per_session_r2[sess_name] = r2
        print(f"  {sess_name:<25} R2={r2:.4f}")

    makin_r2s    = [per_session_r2[s] for s in per_session_r2 if s in HELDOUT_MAKIN]
    flint_r2s    = [per_session_r2[s] for s in per_session_r2 if s in HELDOUT_FLINT]
    makin_mean   = float(np.mean(makin_r2s))  if makin_r2s  else 0.0
    flint_mean   = float(np.mean(flint_r2s))  if flint_r2s  else 0.0
    overall_mean = float(np.mean(list(per_session_r2.values())))

    print(f"\n  Makin mean R2:   {makin_mean:.4f}")
    print(f"  Flint mean R2:   {flint_mean:.4f}")
    print(f"  Overall mean R2: {overall_mean:.4f}")

    # Save results
    result = {
        'per_session_r2': per_session_r2,
        'makin_mean':   makin_mean,
        'flint_mean':   flint_mean,
        'overall_mean': overall_mean,
        'loss_type':    args.loss_type,
        'student_size': args.student_size,
        'lambda_repr':  args.lambda_repr,
        'lambda_ae':    args.lambda_ae,
        'alpha':        args.alpha,
        'beta':         args.beta,
        'gamma':        args.gamma,
        'epochs':       args.epochs,
        'lr':           args.lr,
        'elapsed_s':    elapsed,
        'datasets':     args.datasets,
        'fold_idx':     args.fold_idx,
        'n_folds':      args.n_folds,
        'fold_seed':    args.fold_seed,
        'fold_dropped': fold_dropped_names,
        'fold_kept':    fold_kept_names,
    }
    tag = f"distill_{args.loss_type}_{args.student_size}"
    if args.tag:
        tag = f"{tag}_{args.tag}"
    out_json = OUTPUT_DIR / f"{tag}_result.json"
    with open(out_json, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved results to {out_json}")

    # Save student checkpoint (full decoder)
    ckpt_out = OUTPUT_DIR / f"{tag}_student.pt"
    torch.save({
        'student_state_dict': student.state_dict(),
        'encoder_kwargs': student_kwargs,
        'result': result,
    }, ckpt_out)
    print(f"Saved student checkpoint to {ckpt_out}")

    # Save encoder-only checkpoint (for MAE finetune + probe)
    encoder_ckpt = OUTPUT_DIR / f"{tag}_student_encoder.pt"
    torch.save({
        'encoder_state_dict': {
            k.replace('encoder.', ''): v
            for k, v in student.state_dict().items()
            if k.startswith('encoder.')
        },
        'encoder_kwargs': student_kwargs,
        'result': result,
    }, encoder_ckpt)
    print(f"Saved student encoder to {encoder_ckpt}")


if __name__ == '__main__':
    main()
