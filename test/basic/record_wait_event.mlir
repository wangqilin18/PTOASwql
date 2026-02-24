// RUN: ptoas %s | FileCheck %s

module {
  func.func @sync_ops() {
    pto.record_event [#pto.sync_op_type<TLOAD>, #pto.sync_op_type<TVEC>, #pto.event<EVENT_ID0>]
    pto.wait_event [#pto.sync_op_type<TLOAD>, #pto.sync_op_type<TVEC>, #pto.event<EVENT_ID0>]
    return
  }
}

// CHECK: Success
