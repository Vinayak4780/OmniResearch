"""Generate the synthetic benchmark corpus.

Every document states on its first page that it is SYNTHETIC: all numbers are invented to exercise
the software (table parsing, OCR, contradiction detection) and are NOT real research results.

Usage:  python backend/tests/data/make_corpus.py [output_dir]
"""

from __future__ import annotations

import sys
from pathlib import Path

import pymupdf

DISCLAIMER = (
    "SYNTHETIC TEST DOCUMENT. All values in this document were invented to test the OmniResearch software "
    "and do not report real experiments."
)


def _page(doc: pymupdf.Document) -> pymupdf.Page:
    return doc.new_page(width=595, height=842)


class Writer:
    def __init__(self, doc: pymupdf.Document):
        self.doc = doc
        self.page = _page(doc)
        self.y = 60

    def need(self, h: float) -> None:
        if self.y + h > 790:
            self.page = _page(self.doc)
            self.y = 60

    def text(self, s: str, size: float = 10.5, bold: bool = False, gap: float = 6) -> None:
        font = "hebo" if bold else "helv"
        for lines in range(1, 60):
            h = lines * size * 1.45 + 4
            self.need(h)
            rc = self.page.insert_textbox(pymupdf.Rect(60, self.y, 535, self.y + h), s, fontsize=size, fontname=font)
            if rc >= 0:  # insert_textbox writes nothing when the text does not fit
                self.y += h - rc + gap
                return
        raise ValueError("text block too long")

    def heading(self, s: str, level: int = 1) -> None:
        self.text(s, size=15 if level == 1 else 12.5, bold=True, gap=4)

    def table(self, rows: list[list[str]], widths: list[float]) -> None:
        h = 18 * len(rows) + 6
        self.need(h)
        x0 = 60
        for r, row in enumerate(rows):
            x = x0
            for c, cell in enumerate(row):
                rect = pymupdf.Rect(x, self.y, x + widths[c], self.y + 18)
                self.page.draw_rect(rect, color=(0, 0, 0), width=0.6)
                self.page.insert_text((x + 4, self.y + 12.5), cell, fontsize=9, fontname="hebo" if r == 0 else "helv")
                x += widths[c]
            self.y += 18
        self.y += 10

    def bar_chart(self, labels: list[str], values: list[float], unit: str) -> None:
        h = 170
        self.need(h + 20)
        top, left, height, bw = self.y + 10, 90, 130, 60
        vmax = max(values)
        self.page.draw_line((left - 5, top), (left - 5, top + height), color=(0, 0, 0))
        self.page.draw_line((left - 5, top + height), (left + len(values) * (bw + 30), top + height), color=(0, 0, 0))
        for i, (lab, v) in enumerate(zip(labels, values)):
            x = left + i * (bw + 30)
            bh = height * v / vmax
            self.page.draw_rect(pymupdf.Rect(x, top + height - bh, x + bw, top + height), color=None, fill=(0.25, 0.45, 0.75))
            self.page.insert_text((x, top + height + 14), lab, fontsize=8)
            self.page.insert_text((x + 8, top + height - bh - 4), f"{v:g} {unit}", fontsize=8)
        self.y += h + 10


def doc_quantization(out: Path) -> None:
    doc = pymupdf.open()
    w = Writer(doc)
    doc.set_metadata({"title": "Low-Bit Weight Quantization for Memory-Constrained LLM Serving (Synthetic)", "author": "OmniResearch Test Suite"})
    w.heading("Low-Bit Weight Quantization for Memory-Constrained LLM Serving")
    w.text(DISCLAIMER, size=9)
    w.heading("Abstract", 2)
    w.text(
        "We study post-training weight quantization for serving a 13B-parameter decoder on a single GPU with 24 GB of VRAM. "
        "Four-bit group-wise quantization (group size 128) reduces weight memory from 26.0 GB to 7.4 GB while increasing "
        "WikiText-2 perplexity from 5.09 to 5.31. Three-bit quantization saves further memory but degrades perplexity to 6.12."
    )
    w.heading("1 Introduction", 2)
    w.text(
        "Serving large language models on GPUs with limited VRAM is constrained primarily by weight memory and KV-cache memory. "
        "Weight-only quantization targets the first constraint. Activation outliers make 4-bit activation quantization difficult, "
        "so we quantize weights only and keep activations in FP16."
    )
    w.heading("2 Method", 2)
    w.text(
        "Weights are quantized per output channel in groups of 128 using a calibration set of 128 sequences of 2048 tokens. "
        "Scales are stored in FP16. Dequantization is fused into the matrix-multiplication kernel."
    )
    w.heading("3 Results", 2)
    w.text("Table 1 reports memory and quality for the 13B model on an RTX 4090 (24 GB), batch size 1, sequence length 2048.")
    w.table(
        [
            ["Precision", "Weight memory (GB)", "WikiText-2 PPL", "Decode tokens/s"],
            ["FP16", "26.0", "5.09", "OOM"],
            ["INT8 (W8A16)", "13.5", "5.11", "38.2"],
            ["INT4 g128 (W4A16)", "7.4", "5.31", "61.7"],
            ["INT3 g128 (W3A16)", "5.8", "6.12", "64.0"],
        ],
        [120, 120, 110, 110],
    )
    w.text("Table 1: Memory, perplexity and decode throughput of the synthetic 13B model under weight quantization.", size=9)
    w.heading("4 Discussion and Limitations", 2)
    w.text(
        "Four-bit quantization is the best trade-off in our setting: 3.5x lower weight memory for a 0.22 perplexity increase. "
        "Quantization does not reduce KV-cache memory, which dominates at long context lengths; KV-cache compression is complementary. "
        "We did not evaluate downstream task accuracy, so perplexity may understate quality loss."
    )
    doc.save(out / "synthetic_quantization.pdf")


def doc_kvcache(out: Path) -> None:
    doc = pymupdf.open()
    w = Writer(doc)
    doc.set_metadata({"title": "Paged KV-Cache Management and Continuous Batching (Synthetic)", "author": "OmniResearch Test Suite"})
    w.heading("Paged KV-Cache Management and Continuous Batching")
    w.text(DISCLAIMER, size=9)
    w.heading("Abstract", 2)
    w.text(
        "Naive contiguous KV-cache allocation wastes 60% to 80% of reserved KV memory through fragmentation and over-reservation. "
        "Paging the KV cache into fixed blocks of 16 tokens reduces waste to under 4%. Combined with continuous (iteration-level) "
        "batching, serving throughput increases by 23x relative to static batching on the same A100 40 GB GPU."
    )
    w.heading("1 Background", 2)
    w.text(
        "During autoregressive decoding each token's keys and values are cached. For a 13B model with 40 layers and hidden size 5120, "
        "the KV cache needs about 0.8 MB per token in FP16, so a 2048-token sequence needs about 1.6 GB."
    )
    w.heading("2 Paged Attention", 2)
    w.text(
        "The cache is split into blocks; a block table maps logical to physical blocks, as in virtual memory. Blocks can be shared "
        "between sequences for parallel sampling, with copy-on-write."
    )
    w.heading("3 Continuous Batching", 2)
    w.text(
        "Static batching waits for every sequence in a batch to finish. Continuous batching schedules at every decoding iteration, "
        "admitting new requests as soon as others complete, which keeps the GPU saturated."
    )
    # Force the chart onto page 2 so citations to it carry a distinct page number.
    w.page = _page(doc)
    w.y = 60
    w.heading("4 Results", 2)
    w.text("Figure 1 shows throughput in requests per second for three serving configurations on an A100 40 GB.")
    w.bar_chart(["Static", "Continuous", "Cont.+Paged"], [1.0, 8.6, 23.0], "req/s")
    w.text("Figure 1: Serving throughput (requests/s) of static batching, continuous batching, and continuous batching with paged KV cache.", size=9)
    w.table(
        [
            ["Configuration", "KV memory waste", "Throughput (req/s)", "Max batch"],
            ["Static batching", "71%", "1.0", "8"],
            ["Continuous batching", "71%", "8.6", "8"],
            ["Continuous + paged KV", "3.8%", "23.0", "96"],
        ],
        [140, 110, 120, 90],
    )
    w.text("Table 2: KV-cache waste and throughput by configuration.", size=9)
    w.heading("5 Limitations", 2)
    w.text("Paging adds an indirection in the attention kernel, costing roughly 3% to 5% kernel latency. Results are for a single GPU.")
    doc.save(out / "synthetic_kvcache.pdf")


def doc_scanned(out: Path) -> None:
    """A two-page report whose second page is an image only (no text layer), to exercise OCR."""
    doc = pymupdf.open()
    w = Writer(doc)
    doc.set_metadata({"title": "Attention Kernel Optimization Notes (Synthetic, partially scanned)", "author": "OmniResearch Test Suite"})
    w.heading("Attention Kernel Optimization Notes")
    w.text(DISCLAIMER, size=9)
    w.text(
        "This report compares fused attention kernels. Page 2 is a scanned memo reproduced as an image. "
        "A separate benchmark found continuous batching improves throughput by only 8x over static batching, "
        "because the evaluated workload had uniform output lengths."
    )
    # Render a text page to an image, then place only the image on page 2.
    src = pymupdf.open()
    sp = src.new_page(width=595, height=842)
    sp.insert_textbox(
        pymupdf.Rect(60, 60, 535, 800),
        "Scanned memo: fused attention kernels\n\n"
        "IO-aware tiling keeps attention blocks in on-chip SRAM and avoids writing the full attention matrix to HBM. "
        "On our A100 test the fused kernel reached a 2.4x speedup over the unfused baseline at sequence length 4096, "
        "and attention memory became linear in sequence length.\n\n"
        "Recommendation: combine fused attention with 4-bit weights and a paged KV cache for 24 GB GPUs.",
        fontsize=13,
        fontname="helv",
    )
    pix = sp.get_pixmap(dpi=150)
    page2 = doc.new_page(width=595, height=842)
    page2.insert_image(page2.rect, stream=pix.tobytes("png"))
    doc.save(out / "synthetic_attention_scanned.pdf")


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "corpus"
    out.mkdir(parents=True, exist_ok=True)
    doc_quantization(out)
    doc_kvcache(out)
    doc_scanned(out)
    print(f"wrote corpus to {out}")


if __name__ == "__main__":
    main()
