function token_value(key,    i, n, fields) {
    n = split($0, fields, /[[:space:]]+/)
    for (i = 1; i <= n; i++) {
        if (index(fields[i], key "=") == 1) {
            return substr(fields[i], length(key) + 2)
        }
    }
    return ""
}

function emit_progress(percent, component, detail) {
    if (ui == 1) {
        printf("%d|%s|%s\n", percent, component, detail)
    } else {
        printf("%d|%s\n", percent, (component == "FAIL") ? detail : component)
    }
}

function weighted_frontier(completed_count, weight) {
    if (completed_count <= 0) {
        return 0
    }
    if (completed_count >= 16) {
        return weight
    }
    return int(completed_count * weight / 16)
}

function frontier_detail(qk, softmax, pv, dma) {
    return sprintf("QK %d/16 | Softmax %d/16 | PV %d/16 | DMA %d/16",
                   qk, softmax, pv, dma)
}

/\[ATTENTION_MILESTONE\]/ {
    stage = token_value("stage")
    status = token_value("status")
    core = token_value("core")
    if (stage != "" && status != "") {
        seen = 1
        if (status == "fail") {
            failed = stage
        } else if (status == "done" && core != "") {
            key = stage SUBSEP core
            if (!(key in completed_by_core)) {
                completed_by_core[key] = 1
                completed[stage] += 1
            }
        }
    }
}

END {
    if (seen != 1) {
        print "none"
        exit
    }
    if (failed != "") {
        emit_progress(98, "FAIL", "Attention failed: " failed)
        exit
    }
    if (completed["root_tensor_complete"] >= 1) {
        emit_progress(99, "FINALIZE", "Attention tensor complete")
        exit
    }
    if (completed["manager_local_complete"] > 0) {
        managers = completed["manager_local_complete"]
        if (managers > 4) managers = 4
        emit_progress(94 + managers, "FINALIZE",
                      sprintf("Manager completion frontier %d/4", managers))
        exit
    }

    qk = completed["final_qk_tile_complete"] + 0
    softmax = completed["final_softmax_tile_complete"] + 0
    pv = completed["final_pv_tile_complete"] + 0
    dma = completed["final_output_dma_ack"] + 0
    if (qk > 16) qk = 16
    if (softmax > 16) softmax = 16
    if (pv > 16) pv = 16
    if (dma > 16) dma = 16
    if (completed["worker_dispatch_accept"] >= 16 ||
        qk > 0 || softmax > 0 || pv > 0 || dma > 0) {
        progress = 16 + weighted_frontier(qk, 54) + \
                   weighted_frontier(softmax, 4) + \
                   weighted_frontier(pv, 15) + \
                   weighted_frontier(dma, 5)
        if (qk < 16) component = "QK"
        else if (softmax < 16) component = "SOFTMAX"
        else if (pv < 16) component = "PV"
        else if (dma < 16) component = "DMA"
        else component = "FINALIZE"
        emit_progress(progress, component, frontier_detail(qk, softmax, pv, dma))
        exit
    }
    if (completed["worker_dispatch_accept"] > 0) {
        workers = completed["worker_dispatch_accept"]
        if (workers > 16) workers = 16
        emit_progress(8 + weighted_frontier(workers, 8), "DISPATCH",
                      sprintf("Worker dispatch frontier %d/16", workers))
        exit
    }
    if (completed["manager_dispatch_complete"] > 0) {
        managers = completed["manager_dispatch_complete"]
        if (managers > 4) managers = 4
        emit_progress(4 + managers, "DISPATCH",
                      sprintf("Manager dispatch frontier %d/4", managers))
        exit
    }
    if (completed["root_descriptor_accept"] >= 1) {
        emit_progress(3, "SETUP", "Attention descriptor accepted")
        exit
    }
    if (seen == 1) {
        emit_progress(2, "SETUP", "Attention runtime starting")
        exit
    }
}
