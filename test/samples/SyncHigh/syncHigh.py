#!/usr/bin/env python3
from mlir.ir import Context, Location, Module, InsertionPoint, IndexType
from mlir.dialects import func, pto, arith

def cidx(v):
    return arith.ConstantOp(IndexType.get(), v).result

def main():
    with Context() as ctx, Location.unknown():
        pto.register_dialect(ctx)
        module = Module.create()
        with InsertionPoint(module.body):
            f = func.FuncOp("run_sync_high", func.FunctionType.get([], []))
        entry = f.add_entry_block()
        with InsertionPoint(entry):
            # Unrolled coverage for each SyncOpType (record + wait)
            pto.record_event(pto.SyncOpType.TLOAD,       pto.SyncOpType.TLOAD,       pto.EVENT.EVENT_ID0)
            pto.wait_event  (pto.SyncOpType.TLOAD,       pto.SyncOpType.TLOAD,       pto.EVENT.EVENT_ID0)

            pto.record_event(pto.SyncOpType.TSTORE_ACC,  pto.SyncOpType.TSTORE_ACC,  pto.EVENT.EVENT_ID1)
            pto.wait_event  (pto.SyncOpType.TSTORE_ACC,  pto.SyncOpType.TSTORE_ACC,  pto.EVENT.EVENT_ID1)

            pto.record_event(pto.SyncOpType.TSTORE_VEC,  pto.SyncOpType.TSTORE_VEC,  pto.EVENT.EVENT_ID2)
            pto.wait_event  (pto.SyncOpType.TSTORE_VEC,  pto.SyncOpType.TSTORE_VEC,  pto.EVENT.EVENT_ID2)

            pto.record_event(pto.SyncOpType.TMOV_M2L,    pto.SyncOpType.TMOV_M2L,    pto.EVENT.EVENT_ID3)
            pto.wait_event  (pto.SyncOpType.TMOV_M2L,    pto.SyncOpType.TMOV_M2L,    pto.EVENT.EVENT_ID3)

            pto.record_event(pto.SyncOpType.TMOV_M2S,    pto.SyncOpType.TMOV_M2S,    pto.EVENT.EVENT_ID4)
            pto.wait_event  (pto.SyncOpType.TMOV_M2S,    pto.SyncOpType.TMOV_M2S,    pto.EVENT.EVENT_ID4)

            pto.record_event(pto.SyncOpType.TMOV_M2B,    pto.SyncOpType.TMOV_M2B,    pto.EVENT.EVENT_ID5)
            pto.wait_event  (pto.SyncOpType.TMOV_M2B,    pto.SyncOpType.TMOV_M2B,    pto.EVENT.EVENT_ID5)

            pto.record_event(pto.SyncOpType.TMOV_M2V,    pto.SyncOpType.TMOV_M2V,    pto.EVENT.EVENT_ID6)
            pto.wait_event  (pto.SyncOpType.TMOV_M2V,    pto.SyncOpType.TMOV_M2V,    pto.EVENT.EVENT_ID6)

            pto.record_event(pto.SyncOpType.TMOV_V2M,    pto.SyncOpType.TMOV_V2M,    pto.EVENT.EVENT_ID7)
            pto.wait_event  (pto.SyncOpType.TMOV_V2M,    pto.SyncOpType.TMOV_V2M,    pto.EVENT.EVENT_ID7)

            pto.record_event(pto.SyncOpType.TMATMUL,     pto.SyncOpType.TMATMUL,     pto.EVENT.EVENT_ID0)
            pto.wait_event  (pto.SyncOpType.TMATMUL,     pto.SyncOpType.TMATMUL,     pto.EVENT.EVENT_ID0)

            pto.record_event(pto.SyncOpType.TVEC,        pto.SyncOpType.TVEC,        pto.EVENT.EVENT_ID1)
            pto.wait_event  (pto.SyncOpType.TVEC,        pto.SyncOpType.TVEC,        pto.EVENT.EVENT_ID1)

            pto.record_event(pto.SyncOpType.TVECWAIT_EVENT, pto.SyncOpType.TVECWAIT_EVENT, pto.EVENT.EVENT_ID2)
            pto.wait_event  (pto.SyncOpType.TVECWAIT_EVENT, pto.SyncOpType.TVECWAIT_EVENT, pto.EVENT.EVENT_ID2)

            # Barrier coverage for TMATMUL and TVEC
            pto.barrier(pto.SyncOpType.TMATMUL)
            pto.barrier(pto.SyncOpType.TVEC)
            func.ReturnOp([])
        print(module)

if __name__ == "__main__":
    main()
