"""
convolution.py, the convolution operation written out, and what it assumes.

The Evolution of Machine Learning and AI, Week 4 Lab
South East Technological University

Week 3 ended with a claim: choosing a model is choosing an assumption. This
file is the first half of the proof. A convolution is not a clever trick for
images. It is a pair of assumptions about the world, written as arithmetic:

    LOCALITY        what a pixel means is decided by its neighbours,
                    not by pixels on the far side of the picture.
    STATIONARITY    a feature worth detecting in the top left corner is
                    worth detecting in the bottom right corner too,
                    so use the same detector everywhere.

Everything else in this file follows from those two sentences. The parameter
count follows. The translation behaviour follows. The failure modes follow.

Booklet cross-references:
    Chapter 6   convolution, the operation
    Chapter 7   what a convolution layer assumes
    Chapter 8   feature maps, channels, and what filters learn
    Chapter 9   pooling, stride and the receptive field

Nothing here imports a deep learning library.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import sys as _sys
import matplotlib
if "ipykernel" not in _sys.modules:      # keep inline plots working in Jupyter
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

SLATE = "#47546B"
AMBER = "#A96E08"
RED = "#9E3B2C"
GREEN = "#2E7D4F"
BLUE = "#4E7291"
GREY = "#8A93A5"


# ---------------------------------------------------------------------------
# 1. The operation itself, in eight lines
# ---------------------------------------------------------------------------

def convolve2d(image, kernel, stride=1, padding=0):
    """Slide `kernel` over `image`, taking a weighted sum at every position.

    This is the whole operation. Note what is NOT here: there is no index
    into a table of weights per position. The same `kernel` is used at every
    position, which is the stationarity assumption made concrete. That single
    fact is why a convolution layer has so few parameters.

    Strictly this is a cross-correlation, not a convolution: a true
    convolution flips the kernel first. Every deep learning library does the
    unflipped version and calls it convolution, because when the kernel is
    learned the distinction makes no difference at all. The learned kernel is
    simply the flip of whatever it would otherwise have been.
    """
    if padding:
        image = np.pad(image, padding, mode="constant")
    kh, kw = kernel.shape
    h = (image.shape[0] - kh) // stride + 1
    w = (image.shape[1] - kw) // stride + 1
    out = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            patch = image[i * stride:i * stride + kh, j * stride:j * stride + kw]
            out[i, j] = np.sum(patch * kernel)
    return out


def output_size(n, k, stride=1, padding=0):
    """The size arithmetic, which you will do by hand more often than you expect."""
    return (n + 2 * padding - k) // stride + 1


# ---------------------------------------------------------------------------
# 2. Kernels a human can read
# ---------------------------------------------------------------------------
# Every one of these is three by three. Nine numbers. Before 2012 people
# designed these by hand and published them; a CNN learns them, and the ones
# it learns in its first layer look remarkably like these.

KERNELS = {
    "identity": np.array([[0, 0, 0],
                          [0, 1, 0],
                          [0, 0, 0]], float),

    "vertical edge": np.array([[-1, 0, 1],
                               [-2, 0, 2],
                               [-1, 0, 1]], float),

    "horizontal edge": np.array([[-1, -2, -1],
                                 [0, 0, 0],
                                 [1, 2, 1]], float),

    "blur": np.ones((3, 3)) / 9.0,

    "sharpen": np.array([[0, -1, 0],
                         [-1, 5, -1],
                         [0, -1, 0]], float),

    "diagonal": np.array([[-2, -1, 0],
                          [-1, 0, 1],
                          [0, 1, 2]], float),
}


def synthetic_image(n=32):
    """A picture with a horizontal edge, a vertical edge and a corner in it.

    Hand-made rather than downloaded, so the lab needs no network access and
    so you can see exactly which structure each filter is responding to.
    """
    img = np.zeros((n, n))
    img[:, n // 2:] = 1.0                     # a vertical edge down the middle
    img[n // 3:2 * n // 3, :] += 0.5          # a horizontal band
    img[2 * n // 3:, 2 * n // 3:] = 0.25      # a dimmer square, bottom right
    rng = np.random.default_rng(4)
    img += rng.normal(0, 0.02, img.shape)     # a little sensor noise
    return np.clip(img, 0, 1.5)


def kernel_table():
    """What each hand-designed kernel responds to, as a number."""
    img = synthetic_image(32)
    rows = []
    for name, k in KERNELS.items():
        out = convolve2d(img, k, padding=1)
        rows.append({
            "kernel": name,
            "weights": int(k.size),
            "mean response": round(float(out.mean()), 4),
            "max response": round(float(np.abs(out).max()), 3),
            "fraction active": round(float((np.abs(out) > 0.3).mean()), 3),
        })
    return pd.DataFrame(rows)


def fig_feature_maps():
    """One image, six filters, six feature maps. The lab's opening figure."""
    img = synthetic_image(32)
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.6))

    axes[0, 0].imshow(img, cmap="gray")
    axes[0, 0].set_title("the input\n32 by 32, one channel", fontsize=10)
    axes[0, 1].axis("off")
    axes[0, 1].text(0.5, 0.5,
                    "Six filters.\nNine numbers each.\n\nEach one asks a\ndifferent question\nof every position\nin the image.",
                    ha="center", va="center", fontsize=10, color=SLATE)

    slots = [(0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3)]
    for (r, c), (name, k) in zip(slots, KERNELS.items()):
        out = convolve2d(img, k, padding=1)
        axes[r, c].imshow(out, cmap="RdBu_r", vmin=-2, vmax=2)
        axes[r, c].set_title(f"{name}\n{out.shape[0]} by {out.shape[1]}", fontsize=10)

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("One image, six filters, six feature maps",
                 fontsize=13, color=SLATE, weight="bold")
    fig.tight_layout()
    return fig


def fig_stride_padding():
    """Why the size arithmetic matters, drawn."""
    img = synthetic_image(32)
    settings = [
        ("stride 1, pad 1", 1, 1),
        ("stride 1, pad 0", 1, 0),
        ("stride 2, pad 0", 2, 0),
        ("stride 4, pad 0", 4, 0),
    ]
    k = KERNELS["vertical edge"]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.9))
    for ax, (label, s, p) in zip(axes, settings):
        out = convolve2d(img, k, stride=s, padding=p)
        ax.imshow(out, cmap="RdBu_r", vmin=-2, vmax=2)
        ax.set_title(f"{label}\n{out.shape[0]} by {out.shape[1]}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Same filter, same image, four settings. Stride throws information away deliberately.",
                 fontsize=12, color=SLATE)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 3. The parameter count, which is the argument in one table
# ---------------------------------------------------------------------------

def parameter_comparison(sizes=(28, 64, 224)):
    """A fully connected layer against a convolution layer, on the same input.

    This table is the single most persuasive object in the week. The fully
    connected layer is not being unfair to itself: it is doing exactly what
    you asked, which is to consider every possible relationship between every
    pair of pixels. The problem is that you did not want that.
    """
    rows = []
    for n in sizes:
        pixels = n * n
        fc_hidden = 128
        fc = pixels * fc_hidden + fc_hidden

        # 32 filters, 3 by 3, one input channel
        conv_filters, k = 32, 3
        conv = conv_filters * (k * k * 1) + conv_filters

        rows.append({
            "input": f"{n} by {n}",
            "pixels": pixels,
            "dense (128 units)": fc,
            "conv (32 filters, 3x3)": conv,
            "ratio": f"{fc / conv:,.0f} to 1",
        })
    return pd.DataFrame(rows)


def receptive_field_table():
    """How far back into the image can one unit at layer L actually see?

    The recurrence, which is worth knowing by heart:

        r_1 = k_1
        r_L = r_(L-1) + (k_L - 1) * (product of all strides before L)

    A single 3 by 3 convolution sees nine pixels. Stack a few, with stride,
    and the field grows quickly. This is why depth buys global understanding
    out of local operations: nobody ever wrote a rule about the whole image.
    """
    layers = [
        ("conv 3x3", 3, 1),
        ("conv 3x3", 3, 1),
        ("pool 2x2", 2, 2),
        ("conv 3x3", 3, 1),
        ("conv 3x3", 3, 1),
        ("pool 2x2", 2, 2),
        ("conv 3x3", 3, 1),
        ("conv 3x3", 3, 1),
    ]
    rf, jump, rows = 1, 1, []
    for i, (name, k, s) in enumerate(layers, start=1):
        rf = rf + (k - 1) * jump
        jump = jump * s
        rows.append({
            "layer": i,
            "operation": name,
            "receptive field": f"{rf} by {rf}",
            "pixels seen": rf * rf,
            "stride so far": jump,
        })
    return pd.DataFrame(rows)


def fig_receptive_field():
    """The receptive field growing, drawn on a 40 by 40 grid."""
    df = receptive_field_table()
    fields = [int(s.split(" ")[0]) for s in df["receptive field"]]
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    ax.imshow(synthetic_image(40), cmap="gray", alpha=0.55)
    centre = 20
    cols = plt.cm.viridis(np.linspace(0.15, 0.9, len(fields)))
    for f, c, layer in zip(fields, cols, df["layer"]):
        half = f / 2
        ax.add_patch(plt.Rectangle((centre - half, centre - half), f, f,
                                   fill=False, edgecolor=c, lw=2.1))
        ax.text(centre + half + 0.4, centre - half + 0.9, f"L{layer}: {f}px",
                color=c, fontsize=9, weight="bold")
    ax.plot([centre], [centre], "o", color=RED, ms=6)
    ax.set_title("One output unit, and how much of the image it can see\n"
                 "after each layer. Nobody wrote a global rule.",
                 fontsize=11, color=SLATE)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 4. Translation equivariance, tested rather than asserted
# ---------------------------------------------------------------------------

def equivariance_test(shift=4):
    """Convolve, then shift. Shift, then convolve. Compare.

    If the two orders give the same answer, the operation is EQUIVARIANT to
    translation: moving the input moves the output by the same amount and
    changes nothing else. That is a mathematical property of the operation,
    not something the network has to learn from examples, and it is the
    reason a CNN needs less data than a dense network on images.

    A dense layer has no such property. To it, the shifted image is simply a
    different vector, and every weight has to learn about it separately.
    """
    img = synthetic_image(32)
    k = KERNELS["vertical edge"]

    conv_then_shift = np.roll(convolve2d(img, k, padding=1), shift, axis=1)
    shift_then_conv = convolve2d(np.roll(img, shift, axis=1), k, padding=1)

    interior = (slice(2, -2), slice(shift + 2, -shift - 2))
    conv_err = np.abs(conv_then_shift[interior] - shift_then_conv[interior]).max()

    # The same test for a dense layer, using a fixed random weight matrix.
    rng = np.random.default_rng(0)
    W = rng.normal(0, 0.05, (32 * 32, 64))
    dense_a = img.ravel() @ W
    dense_b = np.roll(img, shift, axis=1).ravel() @ W
    dense_err = np.abs(dense_a - dense_b).max()
    dense_rel = dense_err / (np.abs(dense_a).max() + 1e-12)

    return pd.DataFrame([
        {"operation": "convolution", "quantity": "max |conv(shift(x)) - shift(conv(x))|",
         "value": f"{conv_err:.2e}", "verdict": "equivariant, by construction"},
        {"operation": "dense layer", "quantity": "max |W shift(x) - W x|",
         "value": f"{dense_err:.3f}", "verdict": f"unrelated output ({dense_rel:.0%} of scale)"},
    ])


def fig_equivariance(shift=6):
    """The same claim, as a picture."""
    img = synthetic_image(32)
    k = KERNELS["diagonal"]
    fig, axes = plt.subplots(2, 3, figsize=(11, 7))

    panels = [
        (0, img, "the image"),
        (1, convolve2d(img, k, padding=1), "convolved"),
        (2, np.roll(convolve2d(img, k, padding=1), shift, axis=1), "convolved, then shifted"),
    ]
    for c, data, title in panels:
        axes[0, c].imshow(data, cmap="gray" if c == 0 else "RdBu_r",
                          vmin=None if c == 0 else -2, vmax=None if c == 0 else 2)
        axes[0, c].set_title(title, fontsize=10)

    shifted = np.roll(img, shift, axis=1)
    panels2 = [
        (0, shifted, "the image, shifted"),
        (1, convolve2d(shifted, k, padding=1), "shifted, then convolved"),
    ]
    for c, data, title in panels2:
        axes[1, c].imshow(data, cmap="gray" if c == 0 else "RdBu_r",
                          vmin=None if c == 0 else -2, vmax=None if c == 0 else 2)
        axes[1, c].set_title(title, fontsize=10)

    diff = np.abs(np.roll(convolve2d(img, k, padding=1), shift, axis=1)
                  - convolve2d(shifted, k, padding=1))
    axes[1, 2].imshow(diff[2:-2, shift + 2:-shift - 2], cmap="Reds", vmin=0, vmax=2)
    axes[1, 2].set_title(f"the difference\nmax = {diff[2:-2, shift+2:-shift-2].max():.1e}",
                         fontsize=10)

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Convolution commutes with translation. The bottom right panel is empty, and that is the point.",
                 fontsize=12, color=SLATE, weight="bold")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 5. What the first layer learns, without being told
# ---------------------------------------------------------------------------

def learn_first_layer(n_filters=8, epochs=400, lr=0.5, seed=0):
    """Train 3x3 filters to reconstruct image patches, with no labels at all.

    This is a sparse-coding style objective: compress each patch to
    `n_filters` numbers and reconstruct it. Nobody tells the filters to look
    for edges. Edges are what you get, because edges are what natural image
    patches are actually made of. The same thing happens in the first layer
    of a supervised CNN, which is why first-layer filters from AlexNet,
    ResNet and a random undergraduate project all look alike.
    """
    rng = np.random.default_rng(seed)
    imgs = [synthetic_image(32) for _ in range(6)]
    for i, im in enumerate(imgs):                       # rotate for variety
        imgs[i] = np.rot90(im, i % 4)

    patches = []
    for im in imgs:
        for _ in range(400):
            r, c = rng.integers(0, 29), rng.integers(0, 29)
            p = im[r:r + 3, c:c + 3].ravel()
            patches.append(p - p.mean())
    P = np.array(patches)
    P /= (np.linalg.norm(P, axis=1, keepdims=True) + 1e-8)

    W = rng.normal(0, 0.3, (9, n_filters))
    for _ in range(epochs):
        H = P @ W                                       # encode
        R = H @ W.T                                     # decode
        E = R - P
        grad = (E.T @ H + P.T @ (E @ W)) / len(P)
        W -= lr * grad
        W /= (np.linalg.norm(W, axis=0, keepdims=True) + 1e-8)

    recon = np.abs((P @ W) @ W.T - P).mean()
    return W.T.reshape(n_filters, 3, 3), float(recon)


def fig_learned_filters():
    """The learned filters, next to the hand-designed ones."""
    learned, err = learn_first_layer()
    fig, axes = plt.subplots(2, 8, figsize=(13, 4.0))
    for i, f in enumerate(learned):
        axes[0, i].imshow(f, cmap="RdBu_r", vmin=-0.8, vmax=0.8)
        axes[0, i].set_title(f"learned {i+1}", fontsize=9)
    names = list(KERNELS)
    for i in range(8):
        if i < len(names):
            k = KERNELS[names[i]]
            axes[1, i].imshow(k / (np.abs(k).max()), cmap="RdBu_r", vmin=-1, vmax=1)
            axes[1, i].set_title(names[i], fontsize=9)
        else:
            axes[1, i].axis("off")
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"Top: filters learned from patches with no labels (reconstruction error {err:.4f}).  "
                 "Bottom: filters humans designed by hand.",
                 fontsize=11.5, color=SLATE)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------

def main():
    print("\nSix hand-designed kernels, on one synthetic image")
    print(kernel_table().to_string(index=False))

    print("\nDense against convolutional, same input")
    print(parameter_comparison().to_string(index=False))

    print("\nThe receptive field, growing")
    print(receptive_field_table().to_string(index=False))

    print("\nTranslation equivariance, tested")
    print(equivariance_test().to_string(index=False))

    print("\nSize arithmetic spot checks")
    for n, k, s, p in [(32, 3, 1, 1), (32, 3, 1, 0), (32, 3, 2, 0), (224, 7, 2, 3)]:
        print(f"  n={n:3d} k={k} stride={s} pad={p}  ->  {output_size(n, k, s, p)}")


if __name__ == "__main__":
    main()
