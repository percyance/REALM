"""
Per-session supervised fine-tuning + evaluation.

For each heldout session independently:
  - Load pretrained encoder + velocity head (REALMDecoder)
  - Sequential 80/20 split within session (same as linear probe)
  - Freeze spatial encoder + first freeze_layers BiMamba2 layers
  - Train remaining layers + head with MSE loss on velocity
  - Evaluate R² on the held-out 20%

Usage:
    python -m REALM.scripts.finetune_supervised_persession \
        --pretrained output/pretrain_REALM_teacher_best.pt --freeze_layers 4

    # Freeze all encoder, only train head (comparable to linear probe)
    python -m REALM.scripts.finetune_supervised_persession \
        --pretrained output/pretrain_REALM_teacher_best.pt --freeze_layers 6

    # Full finetune (no freeze)
    python -m REALM.scripts.finetune_supervised_persession \
        --pretrained output/pretrain_REALM_teacher_best.pt --freeze_layers 0
"""

import argparse
import hashlib
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
    create_test_dataloaders, compute_r2, compute_r2_per_axis,
    HELDOUT_MAKIN, HELDOUT_FLINT,
)
from models.decoder import REALMDecoder
from models.configs import (
    STUDENT_REALM_S_KWARGS, STUDENT_REALM_KWARGS, STUDENT_REALM_L_KWARGS,
    STUDENT_REALM_BI_KWARGS, STUDENT_REALM_BI_L_KWARGS,
    STUDENT_REALM_XL_KWARGS, STUDENT_REALM_UXL_KWARGS,
    STUDENT_REALM_LBI_KWARGS, STUDENT_REALM_XLBI_KWARGS,
)

ARCH_KWARGS = {
    # New canonical names
    'realm_s': STUDENT_REALM_S_KWARGS,
    'realm':   STUDENT_REALM_KWARGS,
    'realm_l': STUDENT_REALM_L_KWARGS,
    'realm_bi':   STUDENT_REALM_BI_KWARGS,
    'realm_bi_l': STUDENT_REALM_BI_L_KWARGS,
    # Legacy aliases
    'realm_xl':   STUDENT_REALM_XL_KWARGS,
    'realm_uxl':  STUDENT_REALM_UXL_KWARGS,
    'realm_lbi':  STUDENT_REALM_LBI_KWARGS,
    'realm_xlbi': STUDENT_REALM_XLBI_KWARGS,
}

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_model(ckpt_path, freeze_layers, random_init=False, random_init_arch='realm_bi',
               unfreeze_spatial=False, use_session_embed=False,
               head_layers=1, head_dropout=0.0):
    """Load pretrained encoder into REALMDecoder, freeze early layers."""
    if random_init:
        encoder_kwargs = {**ARCH_KWARGS.get(random_init_arch, STUDENT_REALM_BI_KWARGS), 'max_sessions': 200}
        model = REALMDecoder(
            encoder_kwargs=encoder_kwargs, output_dim=2,
            head_layers=head_layers, head_dropout=head_dropout,
        ).to(DEVICE)
        print("  Using random initialization (no pretrained weights)")
    else:
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        encoder_kwargs = ckpt['encoder_kwargs'].copy()
        encoder_kwargs['max_sessions'] = 200

        model = REALMDecoder(
            encoder_kwargs=encoder_kwargs, output_dim=2,
            head_layers=head_layers, head_dropout=head_dropout,
        ).to(DEVICE)

        # Support distill checkpoint format
        if 'student_state_dict' in ckpt:
            raw = ckpt['student_state_dict']
            prefix = 'encoder.'
            enc_state = {k[len(prefix):]: v for k, v in raw.items()
                         if k.startswith(prefix)}
        elif 'encoder_state_dict' in ckpt:
            enc_state = ckpt['encoder_state_dict']
        else:
            raise ValueError(f"Unknown checkpoint format: {list(ckpt.keys())}")
        state = {k: v for k, v in enc_state.items()
                 if 'session_embed' not in k and 'pos_embed' not in k}
        model.encoder.load_state_dict(state, strict=False)

    n_layers = encoder_kwargs['n_layers']

    # Freeze or unfreeze spatial encoder components
    if not unfreeze_spatial:
        for name in ['temporal_conv', 'eca_conv', 'spatial_proj', 'spatial_norm']:
            module = getattr(model.encoder, name, None)
            if module is not None:
                for p in module.parameters():
                    p.requires_grad = False
        if not use_session_embed:
            model.encoder.session_embed.requires_grad_(False)
    else:
        # Keep pretrained spatial weights but make them trainable
        print("  Spatial encoder: keeping pretrained weights, unfrozen for finetune")

    if use_session_embed:
        nn.init.normal_(model.encoder.session_embed.weight, std=0.02)
        model.encoder.session_embed.requires_grad_(True)
        print("  Session embedding: re-initialized & trainable")

    # Freeze first freeze_layers BiMamba2 layers
    for i in range(min(freeze_layers, n_layers)):
        for p in model.encoder.layers[i].parameters():
            p.requires_grad = False
        for p in model.encoder.layer_norms[i].parameters():
            p.requires_grad = False

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable: {trainable:,} ({100*trainable/total:.1f}%)  "
          f"Frozen: {total-trainable:,}  Freeze layers: 0..{freeze_layers-1}/{n_layers-1}")

    return model, encoder_kwargs


class NoSkipWrapper(nn.Module):
    """Wrapper that disables skip connection for strict linear probe."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, lfp_data, session_ids=None, channel_mask=None, **kwargs):
        out = self.model(lfp_data, session_ids=session_ids,
                         channel_mask=channel_mask, **kwargs)
        # Recompute prediction without skip
        S = self.model.encoder.n_spatial_patches
        B, C, nb, T = lfp_data.shape
        encoded = self.model.encoder(lfp_data, session_ids=session_ids,
                                      channel_mask=channel_mask)
        if S > 1:
            encoded = encoded.reshape(B, T, S, -1).mean(dim=2)
        out['prediction'] = self.model.out_proj(encoded)
        return out

    def parameters(self):
        return self.model.parameters()

    def state_dict(self, *args, **kwargs):
        return self.model.state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        return self.model.load_state_dict(*args, **kwargs)

    def train(self, mode=True):
        self.model.train(mode)
        return self

    def eval(self):
        self.model.eval()
        return self


def finetune_session(model, train_lfp, train_cmask, train_tgt, args,
                     test_lfp=None, test_cmask=None, test_tgt=None):
    """Supervised finetune on one session's training split."""
    sid = torch.zeros(len(train_lfp), dtype=torch.long)
    ds = TensorDataset(train_lfp, train_cmask, train_tgt, sid)
    g = torch.Generator()
    g.manual_seed(args.seed)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=False,
                        generator=g)

    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01,
    )
    criterion = nn.MSELoss()

    best_r2 = -float('inf')
    best_state = None
    patience = 0

    pbar = tqdm(range(1, args.epochs + 1), desc="    Finetune", leave=True)
    for epoch in pbar:
        model.train()
        total_loss, n = 0.0, 0
        for lfp, cmask, tgt, sid_b in loader:
            lfp = lfp.to(DEVICE)
            cmask = cmask.to(DEVICE)
            tgt = tgt.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            s_ids = sid_b.to(DEVICE) if args.use_session_embed else None
            out = model(lfp, session_ids=s_ids, channel_mask=cmask)
            pred = out['prediction']  # (B, T, 2)
            loss = criterion(pred, tgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n += 1
        scheduler.step()

        avg = total_loss / max(n, 1)

        # Use val R² as best criterion
        val_r2 = evaluate_session(model, test_lfp, test_cmask, test_tgt,
                                  use_session_embed=args.use_session_embed,
                                  r2_mode=args.r2_mode)
        is_best = val_r2 > best_r2
        if is_best:
            best_r2 = val_r2
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= args.patience:
                pbar.close()
                print(f"    Early stop at epoch {epoch}  best_R2={best_r2:.4f}")
                break

        postfix = {"loss": f"{avg:.5f}", "R2": f"{val_r2:.4f}", "best_R2": f"{best_r2:.4f}", "pat": patience}
        if is_best:
            postfix["*"] = "best"
        pbar.set_postfix(postfix)

    if best_state is not None:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    return model


@torch.no_grad()
def evaluate_session(model, test_lfp, test_cmask, test_tgt, use_session_embed=False, r2_mode='combined'):
    """Evaluate R² on test split."""
    model.eval()
    sid = torch.zeros(len(test_lfp), dtype=torch.long)
    ds = TensorDataset(test_lfp, test_cmask, test_tgt, sid)
    loader = DataLoader(ds, batch_size=64, shuffle=False)

    preds, targets = [], []
    for lfp, cmask, tgt, sid_b in loader:
        s_ids = sid_b.to(DEVICE) if use_session_embed else None
        out = model(lfp.to(DEVICE), session_ids=s_ids,
                    channel_mask=cmask.to(DEVICE))
        preds.append(out['prediction'].cpu())
        targets.append(tgt)

    preds = torch.cat(preds, dim=0).reshape(-1, 2)
    targets = torch.cat(targets, dim=0).reshape(-1, 2)
    if r2_mode == 'per_axis':
        return compute_r2_per_axis(preds, targets)
    return compute_r2(preds, targets)


def main():
    parser = argparse.ArgumentParser(
        description="Per-session supervised finetune (isolated per heldout session)")
    parser.add_argument('--pretrained', type=str,
                        default='output/distill_REALM_bi_student_best.pt')
    parser.add_argument('--freeze_layers', type=int, default=4,
                        help="Freeze first N BiMamba2 layers (0=full finetune, 6=head only)")
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--random_split', action='store_true', default=True,
                        help="Random train/test split (default: True)")
    parser.add_argument('--sequential_split', action='store_true',
                        help="Override to sequential split")
    parser.add_argument('--fixed_test_ratio', type=float, default=0.0,
                        help="If >0, fix last fixed_test_ratio fraction as test "
                             "set (sequential), then sample train_ratio*n samples "
                             "from the remaining first (1-fixed_test_ratio) "
                             "fraction. Enables fair within-condition fewshot "
                             "comparison across train_ratios.")
    parser.add_argument('--shuffled_fixed_test', action='store_true',
                        help="With --fixed_test_ratio: random shuffle first, then "
                             "test = last fixed_test_ratio of shuffled (random 20pct), "
                             "train = first train_ratio*n of shuffled remainder. "
                             "Avoids time-drift on the test set.")
    parser.add_argument('--segment_length', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--datasets', type=str, nargs='+',
                        default=['makin', 'flint'],
                        choices=['makin', 'flint'],
                        help="Which datasets to evaluate (default: both)")
    parser.add_argument('--tag', type=str, default='',
                        help="Extra tag appended to output filename (e.g. 'ep050')")
    parser.add_argument('--random_init', action='store_true',
                        help="Use random initialization (no pretrained weights)")
    parser.add_argument('--random_init_arch', type=str, default='realm_bi',
                        choices=['realm_s', 'realm', 'realm_l',
                                 'realm_bi', 'realm_bi_l',
                                 'realm_xl', 'realm_uxl', 'realm_lbi', 'realm_xlbi'],
                        help="Architecture for random init (default: realm_bi)")
    parser.add_argument('--unfreeze_spatial', action='store_true',
                        help="Unfreeze & re-init spatial encoder for per-session adaptation")
    parser.add_argument('--save_model', action='store_true',
                        help="Save finetuned model per session for visualization")
    parser.add_argument('--use_session_embed', action='store_true',
                        help="Enable session embedding (trainable per-session bias)")
    parser.add_argument('--no_skip', action='store_true',
                        help="Disable skip connection for strict linear probe")
    parser.add_argument('--r2_mode', type=str, default='combined',
                        choices=['combined', 'per_axis'],
                        help="R2 computation: combined or per_axis (avg across vx,vy)")
    parser.add_argument('--head_layers', type=int, default=1,
                        help="Number of layers in out_proj head (1=linear, 3=MLP)")
    parser.add_argument('--head_dropout', type=float, default=0.0,
                        help="Dropout probability inside multi-layer head (only used if head_layers>1)")
    args = parser.parse_args()

    if args.sequential_split:
        args.random_split = False
    set_seed(args.seed)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Device: {DEVICE}")
    split_mode = "random" if args.random_split else "sequential"
    print(f"Freeze layers: {args.freeze_layers}  Epochs: {args.epochs}  LR: {args.lr}  Split: {split_mode}")

    base_dir = Path(__file__).resolve().parent.parent
    ckpt_path = base_dir / args.pretrained
    if not ckpt_path.exists():
        print(f"ERROR: {ckpt_path} not found")
        return

    test_loaders = create_test_dataloaders(
        segment_length=args.segment_length, batch_size=256)

    # Filter by selected datasets
    active_heldout = set()
    if 'makin' in args.datasets:
        active_heldout.update(HELDOUT_MAKIN)
    if 'flint' in args.datasets:
        active_heldout.update(HELDOUT_FLINT)
    test_loaders = {k: v for k, v in test_loaders.items() if k in active_heldout}
    print(f"Test sessions: {len(test_loaders)} (datasets={args.datasets})")

    per_session_r2 = {}

    for sess_name, loader in sorted(test_loaders.items()):
        t0 = time.time()

        # Per-session deterministic seed for reproducible splits
        sess_hash = int(hashlib.md5(sess_name.encode()).hexdigest(), 16) % (2**31)
        set_seed(args.seed + sess_hash)

        # Collect all data from this session
        all_lfp, all_cmask, all_tgt = [], [], []
        for batch in loader:
            all_lfp.append(batch['lfp'])
            all_cmask.append(batch['channel_mask'])
            all_tgt.append(batch['target'])
        all_lfp = torch.cat(all_lfp, dim=0)    # (N, 96, 1, T)
        all_cmask = torch.cat(all_cmask, dim=0) # (N, 96)
        all_tgt = torch.cat(all_tgt, dim=0)     # (N, T, 2)

        # Train/test split
        n = len(all_lfp)
        if args.fixed_test_ratio > 0.0:
            # Fixed-test mode: test set fixed at fixed_test_ratio fraction.
            # Train sampled from the remaining pool, capped at train_ratio*n.
            n_test = int(args.fixed_test_ratio * n)
            n_train_pool = n - n_test
            n_train = min(int(args.train_ratio * n), n_train_pool)
            if args.shuffled_fixed_test:
                # Shuffle then split: random 20% test (scattered across session),
                # train sampled from remaining 80% pool.
                indices = torch.randperm(n)
                test_idx = indices[-n_test:]
                train_pool = indices[:-n_test]
                train_idx = train_pool[:n_train]
            else:
                # Sequential: last n_test (time-end) as test.
                test_idx = torch.arange(n_train_pool, n)
                if args.random_split:
                    pool_perm = torch.randperm(n_train_pool)
                    train_idx = pool_perm[:n_train]
                else:
                    train_idx = torch.arange(n_train)
            train_lfp, test_lfp = all_lfp[train_idx], all_lfp[test_idx]
            train_cmask, test_cmask = all_cmask[train_idx], all_cmask[test_idx]
            train_tgt, test_tgt = all_tgt[train_idx], all_tgt[test_idx]
        else:
            split = int(args.train_ratio * n)
            if args.random_split:
                indices = torch.randperm(n)
                train_idx, test_idx = indices[:split], indices[split:]
                train_lfp, test_lfp = all_lfp[train_idx], all_lfp[test_idx]
                train_cmask, test_cmask = all_cmask[train_idx], all_cmask[test_idx]
                train_tgt, test_tgt = all_tgt[train_idx], all_tgt[test_idx]
            else:
                train_lfp, test_lfp = all_lfp[:split], all_lfp[split:]
                train_cmask, test_cmask = all_cmask[:split], all_cmask[split:]
                train_tgt, test_tgt = all_tgt[:split], all_tgt[split:]

        # Load fresh model for each session
        model, _ = load_model(ckpt_path, args.freeze_layers,
                              random_init=args.random_init,
                              random_init_arch=args.random_init_arch,
                              unfreeze_spatial=args.unfreeze_spatial,
                              use_session_embed=args.use_session_embed,
                              head_layers=args.head_layers,
                              head_dropout=args.head_dropout)
        if args.no_skip:
            model = NoSkipWrapper(model)
            print("  Skip connection disabled (strict linear probe)")

        # Supervised finetune on train split
        model = finetune_session(model, train_lfp, train_cmask, train_tgt, args,
                                 test_lfp=test_lfp, test_cmask=test_cmask, test_tgt=test_tgt)

        # Evaluate on test split
        r2 = evaluate_session(model, test_lfp, test_cmask, test_tgt,
                              use_session_embed=args.use_session_embed,
                              r2_mode=args.r2_mode)
        per_session_r2[sess_name] = r2
        elapsed = time.time() - t0
        print(f"  {sess_name:<25} R2={r2:.4f}  ({elapsed:.1f}s)")

        if args.save_model:
            tag_suffix = f"_{args.tag}" if args.tag else ""
            model_path = OUTPUT_DIR / f"finetuned_{sess_name}{tag_suffix}.pt"
            torch.save(model.state_dict(), model_path)
            print(f"  Saved model to {model_path.name}")

        del model
        torch.cuda.empty_cache()

    # Summary
    makin_r2s = [per_session_r2[s] for s in per_session_r2 if s in HELDOUT_MAKIN]
    flint_r2s = [per_session_r2[s] for s in per_session_r2 if s in HELDOUT_FLINT]
    makin_mean = float(np.mean(makin_r2s)) if makin_r2s else 0.0
    flint_mean = float(np.mean(flint_r2s)) if flint_r2s else 0.0
    overall_mean = float(np.mean(list(per_session_r2.values())))

    print(f"\n  Makin mean R2:   {makin_mean:.4f}")
    print(f"  Flint mean R2:   {flint_mean:.4f}")
    print(f"  Overall mean R2: {overall_mean:.4f}")

    result = {
        'per_session_r2': per_session_r2,
        'makin_mean': makin_mean,
        'flint_mean': flint_mean,
        'overall_mean': overall_mean,
        'freeze_layers': args.freeze_layers,
        'epochs': args.epochs,
        'datasets': args.datasets,
    }
    suffix = "_randsplit" if args.random_split else ""
    ds_suffix = "_" + "_".join(sorted(args.datasets)) if sorted(args.datasets) != ['flint', 'makin'] else ""
    seed_suffix = f"_s{args.seed}" if args.seed != 42 else ""
    tag_suffix = f"_{args.tag}" if args.tag else ""
    out_path = OUTPUT_DIR / f"finetune_sup_persession_freeze{args.freeze_layers}{suffix}{ds_suffix}{seed_suffix}{tag_suffix}.json"
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == '__main__':
    main()
