"""Make a 3-panel animated GIF: LFP channels | vx,vy | 2D trace.

Usage:
    python scripts/make_gif.py \
        --npz data/makin/preprocessed/indy_20160622_01_rawlfp.npz \
        --start-sec 540 --duration-sec 15 \
        --channels 10 25 40 60 80 \
        --out lfp_velocity_trace.gif
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


FS = 100  # Hz, all preprocessed LFP


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--npz', type=str,
                   default='data/makin/preprocessed/indy_20160622_01_rawlfp.npz')
    p.add_argument('--start-sec', type=float, default=540.0)
    p.add_argument('--duration-sec', type=float, default=15.0)
    p.add_argument('--window-sec', type=float, default=5.0,
                   help='rolling window size for LFP / velocity panels')
    p.add_argument('--channels', type=int, nargs='+',
                   default=[10, 25, 40, 60, 80],
                   help='which LFP channels to plot (4-6 recommended)')
    p.add_argument('--fps', type=int, default=20)
    p.add_argument('--stride', type=int, default=5,
                   help='advance N samples per frame (5 @ 100Hz = 50ms/frame)')
    p.add_argument('--out', type=str, default='lfp_velocity_trace.gif')
    p.add_argument('--dpi', type=int, default=80)
    p.add_argument('--figsize', type=float, nargs=2, default=[12.0, 3.6])
    return p.parse_args()


def main():
    args = parse_args()
    npz_path = Path(args.npz)
    data = np.load(npz_path)
    lfp = data['lfp_data']
    vel = data['targets']
    if lfp.ndim == 3:
        lfp = lfp[:, 0, :]

    s0 = int(args.start_sec * FS)
    s1 = s0 + int(args.duration_sec * FS)
    win = int(args.window_sec * FS)
    chans = args.channels
    assert 4 <= len(chans) <= 6, 'pick 4-6 channels'

    seg_lfp = lfp[chans, max(0, s0 - win):s1]
    seg_vel = vel[max(0, s0 - win):s1]
    pad = s0 - max(0, s0 - win)

    dt = 1.0 / FS
    trace = np.cumsum(vel[s0:s1] * dt, axis=0)

    fig, (ax_lfp, ax_vel, ax_trace) = plt.subplots(
        1, 3, figsize=tuple(args.figsize), dpi=args.dpi,
        gridspec_kw={'width_ratios': [1.2, 1.2, 1.0]}
    )

    lfp_std = np.std(seg_lfp)
    offset = 6 * lfp_std
    lfp_lines = []
    for i, ch in enumerate(chans):
        line, = ax_lfp.plot([], [], lw=0.9, color=plt.cm.viridis(i / len(chans)))
        lfp_lines.append(line)
    ax_lfp.set_xlim(0, args.window_sec)
    ax_lfp.set_ylim(-offset, offset * len(chans))
    ax_lfp.set_yticks([i * offset for i in range(len(chans))])
    ax_lfp.set_yticklabels([f'ch{c}' for c in chans])
    ax_lfp.set_xlabel('time (s)')
    ax_lfp.set_title('LFP')
    ax_lfp.spines[['top', 'right']].set_visible(False)

    vx_line, = ax_vel.plot([], [], lw=1.2, color='tab:blue', label='vx')
    vy_line, = ax_vel.plot([], [], lw=1.2, color='tab:orange', label='vy')
    vmax = float(np.max(np.abs(seg_vel))) * 1.05
    ax_vel.set_xlim(0, args.window_sec)
    ax_vel.set_ylim(-vmax, vmax)
    ax_vel.axhline(0, color='gray', lw=0.5, alpha=0.5)
    ax_vel.set_xlabel('time (s)')
    ax_vel.set_ylabel('velocity (a.u.)')
    ax_vel.set_title('vx, vy')
    ax_vel.legend(loc='upper right', frameon=False)
    ax_vel.spines[['top', 'right']].set_visible(False)

    pad_xy = max(np.ptp(trace[:, 0]), np.ptp(trace[:, 1])) * 0.1 + 1e-3
    xlim = (trace[:, 0].min() - pad_xy, trace[:, 0].max() + pad_xy)
    ylim = (trace[:, 1].min() - pad_xy, trace[:, 1].max() + pad_xy)
    trace_line, = ax_trace.plot([], [], lw=1.4, color='tab:red')
    trace_dot, = ax_trace.plot([], [], 'o', color='tab:red', ms=6)
    ax_trace.set_xlim(*xlim)
    ax_trace.set_ylim(*ylim)
    ax_trace.set_aspect('equal', adjustable='box')
    ax_trace.set_xlabel('x')
    ax_trace.set_ylabel('y')
    ax_trace.set_title('integrated trace')
    ax_trace.spines[['top', 'right']].set_visible(False)

    fig.suptitle(f'{npz_path.stem}   t = {args.start_sec:.1f}s + ...', y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    duration_n = s1 - s0
    frame_idx = list(range(0, duration_n, args.stride))
    t_axis = np.arange(win) / FS

    def slice_or_pad(arr, start, end):
        n = end - start
        if start >= 0:
            return arr[..., start:end]
        out_shape = list(arr.shape)
        out_shape[-1] = n
        out = np.full(out_shape, np.nan, dtype=arr.dtype)
        valid = arr[..., 0:end]
        out[..., -valid.shape[-1]:] = valid
        return out

    def update(k):
        end = k + pad
        start = end - win
        lfp_chunk = slice_or_pad(seg_lfp, start, end)
        vel_chunk = slice_or_pad(seg_vel.T, start, end).T
        for i, line in enumerate(lfp_lines):
            line.set_data(t_axis, lfp_chunk[i] + i * offset)
        vx_line.set_data(t_axis, vel_chunk[:, 0])
        vy_line.set_data(t_axis, vel_chunk[:, 1])
        kk = max(1, k + 1)
        trace_line.set_data(trace[:kk, 0], trace[:kk, 1])
        trace_dot.set_data([trace[kk - 1, 0]], [trace[kk - 1, 1]])
        return (*lfp_lines, vx_line, vy_line, trace_line, trace_dot)

    anim = FuncAnimation(fig, update, frames=frame_idx,
                         interval=1000 / args.fps, blit=True)
    out_path = Path(args.out)
    anim.save(out_path, writer=PillowWriter(fps=args.fps), dpi=args.dpi)
    print(f'wrote {out_path}  ({len(frame_idx)} frames @ {args.fps} fps)')


if __name__ == '__main__':
    main()
