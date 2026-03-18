#!/usr/bin/env python3
"""Generate PTO-AS / MLIR / C++ stubs for binop tests (simple or ST-driven)."""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from typing import Iterable, List


@dataclass(frozen=True)
class Case:
    dtype_token: str
    dst_h: int
    dst_w: int
    src0_h: int
    src0_w: int
    src1_h: int
    src1_w: int
    v_rows: int
    v_cols: int


DTYPE_MAP = {
    "np.float32": {"pto": "f32", "cpp": "float", "case": "float"},
    "np.float16": {"pto": "f16", "cpp": "half", "case": "half"},
    "np.int32": {"pto": "i32", "cpp": "int32_t", "case": "int32"},
    "np.int16": {"pto": "i16", "cpp": "int16_t", "case": "int16"},
    "np.int8": {"pto": "i8", "cpp": "int8_t", "case": "int8"},
}

DTYPE_MLIR = {
    "np.float32": "f32",
    "np.float16": "f16",
    "np.int32": "i32",
    "np.int16": "i16",
    "np.int8": "i8",
}

# Common binop-style ops (not enforced unless --allow-unknown is false).
KNOWN_OPS = {
    "tadd",
    "taddc",
    "tadds",
    "taddsc",
    "tsub",
    "tsubc",
    "tsubs",
    "tsubsc",
    "tmul",
    "tmuls",
    "tdiv",
    "tdivs",
    "tmin",
    "tmins",
    "tmax",
    "tmaxs",
    "tand",
    "tands",
    "tor",
    "tors",
    "txor",
    "txors",
    "tshl",
    "tshls",
    "tshr",
    "tshrs",
    "trem",
    "trems",
    "tfmod",
    "tfmods",
    "tprelu",
}


def _split_ops(values: Iterable[str]) -> List[str]:
    ops: List[str] = []
    for v in values:
        if not v:
            continue
        for part in v.split(","):
            op = part.strip()
            if op:
                ops.append(op)
    return ops


def _parse_shape(shape: str) -> tuple[int, int]:
    m = re.fullmatch(r"(\d+)[xX](\d+)", shape.strip())
    if not m:
        raise ValueError("shape must be like 16x16")
    rows = int(m.group(1))
    cols = int(m.group(2))
    if rows <= 0 or cols <= 0:
        raise ValueError("rows/cols must be positive")
    return rows, cols


def _validate_dtype(dtype: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_]+", dtype):
        raise ValueError("dtype must be an MLIR-style token like f16, f32, bf16, i32")
    return dtype


def _emit_program(ops: List[str], rows: int, cols: int, dtype: str, with_types: bool, chain: bool) -> str:
    tile = f"!pto.tile<{rows}x{cols}x{dtype}>"
    lines: List[str] = []

    lines.append(f".arg %a : {tile};")
    lines.append(f".arg %b : {tile};")

    prev = "%a"
    for i, op in enumerate(ops):
        dst = f"%r{i}"
        if chain and i > 0:
            src0 = prev
            src1 = "%b"
        else:
            src0 = "%a"
            src1 = "%b"

        if with_types:
            sig = f": ({tile}, {tile}) -> {tile};"
        else:
            sig = ";"
        lines.append(f"{dst} = {op} {src0}, {src1}{sig}")
        prev = dst

    return "\n".join(lines) + "\n"


def emit_simple_pto(ops: List[str], rows: int, cols: int, dtype: str, with_types: bool, chain: bool) -> str:
    return _emit_program(ops, rows, cols, dtype, with_types, chain)


def parse_cases(gen_data_path: str, class_name: str) -> List[Case]:
    text = open(gen_data_path, "r", encoding="utf-8").read()
    pattern = re.compile(rf"{re.escape(class_name)}\(([^)]*)\)")
    cases: List[Case] = []
    for m in pattern.finditer(text):
        args = [a.strip() for a in m.group(1).split(",") if a.strip()]
        if len(args) != 9:
            raise ValueError(f"Unexpected arg count in {class_name}: {m.group(0)}")
        dtype = args[0]
        if dtype not in DTYPE_MAP:
            raise ValueError(f"Unsupported dtype token: {dtype}")
        nums = [int(a) for a in args[1:]]
        cases.append(Case(dtype, *nums))
    if not cases:
        raise ValueError(f"No cases found for {class_name} in {gen_data_path}")
    return cases


def case_name(prefix: str, c: Case) -> str:
    dtype_str = DTYPE_MAP[c.dtype_token]["case"]
    return (
        f"{prefix}Test.case_{dtype_str}_"
        f"{c.dst_h}x{c.dst_w}_"
        f"{c.src0_h}x{c.src0_w}_"
        f"{c.src1_h}x{c.src1_w}_"
        f"{c.v_rows}x{c.v_cols}"
    )


def emit_pto(op: str, prefix: str, cases: List[Case]) -> str:
    lines: List[str] = []
    for c in cases:
        pto_dtype = DTYPE_MAP[c.dtype_token]["pto"]
        name = case_name(prefix, c)
        lines.append(f"; {name}")
        lines.append(f"; valid_region = {c.v_rows}x{c.v_cols}")
        lines.append(".const %c0 = 0 : index;")
        lines.append(
            f".arg %src0_gm : !pto.memref<1x1x1x{c.v_rows}x{c.v_cols}x{pto_dtype}>;"
        )
        lines.append(
            f".arg %src1_gm : !pto.memref<1x1x1x{c.v_rows}x{c.v_cols}x{pto_dtype}>;"
        )
        lines.append(
            f".arg %dst_gm : !pto.memref<1x1x1x{c.v_rows}x{c.v_cols}x{pto_dtype}>;"
        )
        lines.append(
            f"%t0 = tload %src0_gm[%c0, %c0] : (!pto.memref<1x1x1x{c.v_rows}x{c.v_cols}x{pto_dtype}>, index, index) -> "
            f"!pto.tile<{c.src0_h}x{c.src0_w}x{pto_dtype}>;"
        )
        lines.append(
            f"%t1 = tload %src1_gm[%c0, %c0] : (!pto.memref<1x1x1x{c.v_rows}x{c.v_cols}x{pto_dtype}>, index, index) -> "
            f"!pto.tile<{c.src1_h}x{c.src1_w}x{pto_dtype}>;"
        )
        lines.append(
            f"%dst = {op} %t0, %t1 : (!pto.tile<{c.src0_h}x{c.src0_w}x{pto_dtype}>, "
            f"!pto.tile<{c.src1_h}x{c.src1_w}x{pto_dtype}>) -> "
            f"!pto.tile<{c.dst_h}x{c.dst_w}x{pto_dtype}>;"
        )
        lines.append("tstore %dst, %dst_gm[%c0, %c0];")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _tile_buf_type(dtype: str, rows: int, cols: int) -> str:
    return (
        "!pto.tile_buf<loc=vec, dtype="
        + dtype
        + f", rows={rows}, cols={cols}, v_row=?, v_col=?, "
        + "blayout=row_major, slayout=none_box, fractal=512, pad=0>"
    )


def emit_mlir(op: str, prefix: str, cases: List[Case]) -> str:
    lines: List[str] = []
    lines.append("module {")
    for idx, c in enumerate(cases):
        mlir_dtype = DTYPE_MLIR[c.dtype_token]
        name = case_name(prefix, c)
        func = f"case_{idx}_{prefix.lower()}"

        ptr_ty = f"!pto.ptr<{mlir_dtype}>"
        tv_ty = f"!pto.tensor_view<{c.v_rows}x{c.v_cols}x{mlir_dtype}>"
        pv_ty = f"!pto.partition_tensor_view<{c.v_rows}x{c.v_cols}x{mlir_dtype}>"
        tb_dst = _tile_buf_type(mlir_dtype, c.dst_h, c.dst_w)
        tb_src0 = _tile_buf_type(mlir_dtype, c.src0_h, c.src0_w)
        tb_src1 = _tile_buf_type(mlir_dtype, c.src1_h, c.src1_w)

        lines.append(f"  // {name}")
        lines.append(f"  func.func @{func}(%out: {ptr_ty}, %src0: {ptr_ty}, %src1: {ptr_ty}) {{")
        lines.append("    %c0 = arith.constant 0 : index")
        lines.append("    %c1 = arith.constant 1 : index")
        lines.append("    %c4 = arith.constant 4 : index")
        lines.append(f"    %vrows = arith.constant {c.v_rows} : index")
        lines.append(f"    %vcols = arith.constant {c.v_cols} : index")
        if _use_block_offset(op, c):
            kblockcols = c.v_cols // 4
            tile_stride = c.v_rows * kblockcols
            lines.append(f"    %kblockcols = arith.constant {kblockcols} : index")
            lines.append(f"    %tile_stride = arith.constant {tile_stride} : index")
            lines.append("    %bid_i64 = pto.get_block_idx")
            lines.append("    %bid = arith.index_cast %bid_i64 : i64 to index")
            lines.append("    %bid_div = arith.divsi %bid, %c4 : index")
            lines.append("    %bid_mod = arith.remsi %bid, %c4 : index")
            lines.append("    %off0 = arith.muli %bid_div, %tile_stride : index")
            lines.append("    %off1 = arith.muli %bid_mod, %kblockcols : index")
            lines.append("    %offset = arith.addi %off0, %off1 : index")
            lines.append(f"    %src0_off = pto.addptr %src0, %offset : {ptr_ty} -> {ptr_ty}")
            lines.append(f"    %src1_off = pto.addptr %src1, %offset : {ptr_ty} -> {ptr_ty}")
            lines.append(f"    %out_off = pto.addptr %out, %offset : {ptr_ty} -> {ptr_ty}")
            src0_ptr = "%src0_off"
            src1_ptr = "%src1_off"
            out_ptr = "%out_off"
        else:
            src0_ptr = "%src0"
            src1_ptr = "%src1"
            out_ptr = "%out"

        lines.append(
            f"    %tv0 = pto.make_tensor_view {src0_ptr}, shape = [%vrows, %vcols], strides = [%vcols, %c1] : {tv_ty}"
        )
        lines.append(
            f"    %tv1 = pto.make_tensor_view {src1_ptr}, shape = [%vrows, %vcols], strides = [%vcols, %c1] : {tv_ty}"
        )
        lines.append(
            f"    %tv2 = pto.make_tensor_view {out_ptr}, shape = [%vrows, %vcols], strides = [%vcols, %c1] : {tv_ty}"
        )
        lines.append(
            f"    %pv0 = pto.partition_view %tv0, offsets = [%c0, %c0], sizes = [%vrows, %vcols] : {tv_ty} -> {pv_ty}"
        )
        lines.append(
            f"    %pv1 = pto.partition_view %tv1, offsets = [%c0, %c0], sizes = [%vrows, %vcols] : {tv_ty} -> {pv_ty}"
        )
        lines.append(
            f"    %pv2 = pto.partition_view %tv2, offsets = [%c0, %c0], sizes = [%vrows, %vcols] : {tv_ty} -> {pv_ty}"
        )
        lines.append(f"    %tb0 = pto.alloc_tile valid_row = %vrows valid_col = %vcols : {tb_src0}")
        lines.append(f"    %tb1 = pto.alloc_tile valid_row = %vrows valid_col = %vcols : {tb_src1}")
        lines.append(f"    %tb2 = pto.alloc_tile valid_row = %vrows valid_col = %vcols : {tb_dst}")
        lines.append(
            f"    pto.tload ins(%pv0 : {pv_ty}) outs(%tb0 : {tb_src0})"
        )
        lines.append(
            f"    pto.tload ins(%pv1 : {pv_ty}) outs(%tb1 : {tb_src1})"
        )
        use_src0 = "%tb0"
        use_src1 = "%tb1"
        if c.src0_h != c.dst_h or c.src0_w != c.dst_w:
            lines.append(
                f"    %tb0s = pto.subset %tb0[%c0, %c0] sizes [{c.dst_h}, {c.dst_w}] : {tb_src0}"
            )
            use_src0 = "%tb0s"
        if c.src1_h != c.dst_h or c.src1_w != c.dst_w:
            lines.append(
                f"    %tb1s = pto.subset %tb1[%c0, %c0] sizes [{c.dst_h}, {c.dst_w}] : {tb_src1}"
            )
            use_src1 = "%tb1s"
        lines.append(
            "    pto.set_flag [#pto.pipe<PIPE_MTE2>, #pto.pipe<PIPE_V>, #pto.event<EVENT_ID0>]"
        )
        lines.append(
            "    pto.wait_flag [#pto.pipe<PIPE_MTE2>, #pto.pipe<PIPE_V>, #pto.event<EVENT_ID0>]"
        )
        lines.append(
            f"    pto.{op} ins({use_src0}, {use_src1} : {tb_dst}, {tb_dst}) outs(%tb2 : {tb_dst})"
        )
        lines.append(
            "    pto.set_flag [#pto.pipe<PIPE_V>, #pto.pipe<PIPE_MTE3>, #pto.event<EVENT_ID0>]"
        )
        lines.append(
            "    pto.wait_flag [#pto.pipe<PIPE_V>, #pto.pipe<PIPE_MTE3>, #pto.event<EVENT_ID0>]"
        )
        lines.append(
            f"    pto.tstore ins(%tb2 : {tb_dst}) outs(%pv2 : {pv_ty})"
        )
        lines.append("    return")
        lines.append("  }")
        lines.append("")

    lines.append("}")
    return "\n".join(lines).rstrip() + "\n"


def _use_block_offset(op: str, c: Case) -> bool:
    # Only TMIN uses block_idx GM offset in the original ST kernels.
    if op != "tmin":
        return False
    same_tile = (
        c.dst_h == c.src0_h == c.src1_h == c.v_rows
        and c.dst_w == c.src0_w == c.src1_w == c.v_cols
    )
    return same_tile and (c.v_cols % 4 == 0)


def emit_cpp(op: str, prefix: str, cases: List[Case]) -> str:
    lines: List[str] = []
    lines.append("#include <cstdint>")
    lines.append("#include <pto/pto-inst.hpp>")
    lines.append("#include \"acl/acl.h\"")
    lines.append("")
    lines.append("using namespace pto;")
    lines.append("")
    for idx, c in enumerate(cases):
        cpp_dtype = DTYPE_MAP[c.dtype_token]["cpp"]
        name = case_name(prefix, c)
        func = f"case_{idx}_{prefix.lower()}"
        lines.append(f"// {name}")
        lines.append(f"// valid_region = {c.v_rows}x{c.v_cols}")
        lines.append(
            f"__global__ AICORE void {func}(__gm__ {cpp_dtype}* out, __gm__ {cpp_dtype}* src0, __gm__ {cpp_dtype}* src1)"
        )
        lines.append("{")
        lines.append("    using DynShape = pto::Shape<-1, -1, -1, -1, -1>;")
        lines.append("    using DynStride = pto::Stride<-1, -1, -1, -1, -1>;")
        lines.append(f"    using GlobalData = GlobalTensor<{cpp_dtype}, DynShape, DynStride>;")
        if _use_block_offset(op, c):
            lines.append(f"    constexpr int kBlockCols = {c.v_cols} / 4;")
            lines.append("    int offset = (block_idx / 4) * (" f"{c.v_rows} * kBlockCols) + (block_idx % 4) * kBlockCols;")
            lines.append(
                f"    GlobalData dstGlobal(out + offset, pto::Shape(1, 1, 1, {c.v_rows}, {c.v_cols}),"
            )
            lines.append(
                f"                         pto::Stride({c.dst_h} * {c.dst_w}, {c.dst_h} * {c.dst_w}, {c.dst_h} * {c.dst_w}, {c.dst_w}, 1));"
            )
            lines.append(
                f"    GlobalData src0Global(src0 + offset, pto::Shape(1, 1, 1, {c.v_rows}, {c.v_cols}),"
            )
            lines.append(
                f"                         pto::Stride({c.src0_h} * {c.src0_w}, {c.src0_h} * {c.src0_w}, {c.src0_h} * {c.src0_w}, {c.src0_w}, 1));"
            )
            lines.append(
                f"    GlobalData src1Global(src1 + offset, pto::Shape(1, 1, 1, {c.v_rows}, {c.v_cols}),"
            )
            lines.append(
                f"                         pto::Stride({c.src1_h} * {c.src1_w}, {c.src1_h} * {c.src1_w}, {c.src1_h} * {c.src1_w}, {c.src1_w}, 1));"
            )
        else:
            lines.append(
                f"    GlobalData dstGlobal(out, pto::Shape(1, 1, 1, {c.v_rows}, {c.v_cols}),"
            )
            lines.append(
                f"                         pto::Stride({c.dst_h} * {c.dst_w}, {c.dst_h} * {c.dst_w}, {c.dst_h} * {c.dst_w}, {c.dst_w}, 1));"
            )
            lines.append(
                f"    GlobalData src0Global(src0, pto::Shape(1, 1, 1, {c.v_rows}, {c.v_cols}),"
            )
            lines.append(
                f"                         pto::Stride({c.src0_h} * {c.src0_w}, {c.src0_h} * {c.src0_w}, {c.src0_h} * {c.src0_w}, {c.src0_w}, 1));"
            )
            lines.append(
                f"    GlobalData src1Global(src1, pto::Shape(1, 1, 1, {c.v_rows}, {c.v_cols}),"
            )
            lines.append(
                f"                         pto::Stride({c.src1_h} * {c.src1_w}, {c.src1_h} * {c.src1_w}, {c.src1_h} * {c.src1_w}, {c.src1_w}, 1));"
            )
        lines.append("")
        lines.append(
            f"    using TileDst = Tile<TileType::Vec, {cpp_dtype}, {c.dst_h}, {c.dst_w}, BLayout::RowMajor, -1, -1>;"
        )
        lines.append(
            f"    using TileSrc0 = Tile<TileType::Vec, {cpp_dtype}, {c.src0_h}, {c.src0_w}, BLayout::RowMajor, -1, -1>;"
        )
        lines.append(
            f"    using TileSrc1 = Tile<TileType::Vec, {cpp_dtype}, {c.src1_h}, {c.src1_w}, BLayout::RowMajor, -1, -1>;"
        )
        lines.append(f"    TileDst dstTile({c.v_rows}, {c.v_cols});")
        lines.append(f"    TileSrc0 src0Tile({c.v_rows}, {c.v_cols});")
        lines.append(f"    TileSrc1 src1Tile({c.v_rows}, {c.v_cols});")
        if _use_block_offset(op, c):
            lines.append("    TASSIGN(src0Tile, 0x0 + 0x400 * block_idx);")
            lines.append("    TASSIGN(src1Tile, 0x4000 + 0x400 * block_idx);")
            lines.append("    TASSIGN(dstTile, 0x8000 + 0x400 * block_idx);")
        else:
            lines.append("    TASSIGN(src0Tile, 0x0);")
            lines.append("    TASSIGN(src1Tile, 0x10000);")
            lines.append("    TASSIGN(dstTile, 0x20000);")
        lines.append("")
        lines.append("    TLOAD(src0Tile, src0Global);")
        lines.append("    TLOAD(src1Tile, src1Global);")
        lines.append("    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);")
        lines.append("    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);")
        lines.append(f"    {op.upper()}<TileDst, TileSrc0, TileSrc1>(dstTile, src0Tile, src1Tile);")
        lines.append("    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);")
        lines.append("    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);")
        lines.append("    TSTORE(dstGlobal, dstTile);")
        lines.append("}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"

def _build_root_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate binop PTO-AS/MLIR/C++ either from ST cases or manual params."
    )
    sub = parser.add_subparsers(dest="mode", metavar="MODE")
    sub.add_parser("simple", help="Generate a simple PTO-AS program from manual params.")
    sub.add_parser("st", help="Generate MLIR/PTO + C++ stubs from ST gen_data.py.")
    return parser


def _build_simple_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a simple PTO-AS program for binop-style ops."
    )
    parser.add_argument(
        "--op",
        action="append",
        default=[],
        help="Operation name (repeatable or comma-separated), e.g., --op tadd --op tmin",
    )
    parser.add_argument(
        "--shape",
        default="16x16",
        help="Tile shape as ROWSxCOLS (default: 16x16)",
    )
    parser.add_argument(
        "--dtype",
        default="f16",
        help="Element type token (default: f16)",
    )
    parser.add_argument(
        "--out",
        default="-",
        help="Output file path (default: stdout)",
    )
    parser.add_argument(
        "--no-types",
        action="store_true",
        help="Omit explicit type signatures on instructions",
    )
    parser.add_argument(
        "--chain",
        action="store_true",
        help="Feed each op result into the next op as src0",
    )
    parser.add_argument(
        "--allow-unknown",
        action="store_true",
        help="Allow ops outside the known binop list",
    )
    return parser


def _build_st_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate PTO-AS/MLIR and C++ stubs from ST gen_data.py binop cases."
    )
    parser.add_argument("--op", required=True, help="Operation name, e.g., tmin")
    parser.add_argument("--prefix", required=True, help="Testcase prefix, e.g., TMIN")
    parser.add_argument("--class-name", required=True, help="Params class name, e.g., TMinParams")
    parser.add_argument(
        "--gen-data",
        required=True,
        help="Path to gen_data.py (e.g., tests/npu/a5/src/st/testcase/tmin/gen_data.py)",
    )
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument(
        "--emit",
        choices=["pto", "mlir"],
        default="mlir",
        help="Output format for the IR file (default: mlir)",
    )
    return parser


def _run_simple(args: argparse.Namespace) -> int:
    ops = _split_ops(args.op)
    if not ops:
        ops = ["tadd"]

    unknown = [op for op in ops if op not in KNOWN_OPS]
    if unknown and not args.allow_unknown:
        sys.stderr.write(
            "Unknown op(s): "
            + ", ".join(unknown)
            + "\nUse --allow-unknown to force.\n"
        )
        return 2

    try:
        rows, cols = _parse_shape(args.shape)
        dtype = _validate_dtype(args.dtype)
    except ValueError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 2

    text = emit_simple_pto(
        ops=ops,
        rows=rows,
        cols=cols,
        dtype=dtype,
        with_types=not args.no_types,
        chain=args.chain,
    )

    if args.out == "-":
        sys.stdout.write(text)
        return 0

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"wrote {args.out}")
    return 0


def _run_st(args: argparse.Namespace) -> int:
    cases = parse_cases(args.gen_data, args.class_name)

    os.makedirs(args.out_dir, exist_ok=True)
    if args.emit == "mlir":
        ir_path = os.path.join(args.out_dir, f"{args.op}_cases.mlir")
    else:
        ir_path = os.path.join(args.out_dir, f"{args.op}_cases.pto")
    cpp_path = os.path.join(args.out_dir, f"{args.op}_cases.cpp")

    with open(ir_path, "w", encoding="utf-8") as f:
        if args.emit == "mlir":
            f.write(emit_mlir(args.op, args.prefix, cases))
        else:
            f.write(emit_pto(args.op, args.prefix, cases))

    with open(cpp_path, "w", encoding="utf-8") as f:
        f.write(emit_cpp(args.op, args.prefix, cases))

    print(f"wrote {ir_path}")
    print(f"wrote {cpp_path}")
    return 0


def main(argv: List[str]) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        _build_root_parser().print_help()
        return 0

    if argv[0] == "simple":
        args = _build_simple_parser().parse_args(argv[1:])
        return _run_simple(args)
    if argv[0] == "st":
        args = _build_st_parser().parse_args(argv[1:])
        return _run_st(args)

    # Legacy mode auto-detection
    if any(flag in argv for flag in ("--gen-data", "--class-name", "--out-dir", "--emit", "--prefix")):
        args = _build_st_parser().parse_args(argv)
        return _run_st(args)

    args = _build_simple_parser().parse_args(argv)
    return _run_simple(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
