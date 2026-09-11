// -*- mode: c++ -*-

// Copyright 2009-2025 NTESS. Under the terms
// of Contract DE-NA0003525 with NTESS, the U.S.
// Government retains certain rights in this software.
//
// Copyright (c) 2009-2025, NTESS
// All rights reserved.
//
// Portions are copyright of other developers:
// See the file CONTRIBUTORS.TXT in the top level directory
// of the distribution for more information.
//
// This file is part of the SST software package. For license
// information, see the LICENSE file in the top level directory of the
// distribution.


#ifndef COMPONENTS_MERLIN_ARBITRATION_OUTPUT_ARB_BASIC_H
#define COMPONENTS_MERLIN_ARBITRATION_OUTPUT_ARB_BASIC_H

#include <sst/elements/merlin/router.h>
#include <sst/elements/merlin/arbitration/single_arb.h>

#include <algorithm>
#include <vector>

namespace SST {
namespace Merlin {

class output_arb_basic : public PortInterface::OutputArbitration {

public:

    SST_ELI_REGISTER_SUBCOMPONENT(
        output_arb_basic,
        "merlin",
        "arb.output.basic",
        SST_ELI_ELEMENT_VERSION(1,0,0),
        "Basic output arbitration for PortControl",
        SST::Merlin::PortInterface::OutputArbitration
    )

    SST_ELI_DOCUMENT_PARAMS(
        {"arb",    "Tyoe of arbitration to use","merlin.arb.base.single.roundrobin"},
        {"vn_priority_order", "Comma-separated VNs in descending priority order. Empty preserves round-robin.", ""},
        {"vn_starvation_vn", "VN allowed to bypass priority after max_starvation_cycles.", "-1"},
        {"max_starvation_cycles", "Maximum network age before vn_starvation_vn is serviced. 0 disables bypass.", "0"},
    )

    SST_ELI_DOCUMENT_STATISTICS(
        {"vn_priority_high_grants", "Packets granted from VNs above the starvation VN.", "packets", 1},
        {"vn_priority_low_grants", "Packets granted from the starvation VN.", "packets", 1},
        {"vn_priority_starvation_grants", "Starvation-VN packets promoted after the wait bound.", "packets", 1},
    )


private:


    int num_vcs;
    int num_vns;
    std::string arb_name;
    SingleArbitration* arb;
    std::vector<int> priority_vns;
    std::vector<int> vcs_per_vn_;
    std::vector<int> vn_offsets_;
    std::vector<int> vn_rr_vcs_;
    int starvation_vn;
    uint64_t max_starvation_cycles;
    Statistic<uint64_t>* priority_high_grants;
    Statistic<uint64_t>* priority_low_grants;
    Statistic<uint64_t>* priority_starvation_grants;

public:

    output_arb_basic(ComponentId_t cid, Params& params) :
        OutputArbitration(cid),
        num_vcs(0),
        num_vns(0),
        arb(nullptr),
        starvation_vn(params.find<int>("vn_starvation_vn", -1)),
        max_starvation_cycles(params.find<uint64_t>("max_starvation_cycles", 0))
    {
        arb_name = params.find<std::string>("arb","merlin.arb.base.single.roundrobin");
        const std::string priority_order =
            params.find<std::string>("vn_priority_order", "");
        if ( priority_order.find_first_not_of(" \t\r\n") != std::string::npos ) {
            params.find_array<int>("vn_priority_order", priority_vns);
        }
        priority_high_grants = registerStatistic<uint64_t>("vn_priority_high_grants");
        priority_low_grants = registerStatistic<uint64_t>("vn_priority_low_grants");
        priority_starvation_grants = registerStatistic<uint64_t>("vn_priority_starvation_grants");
    }

    ~output_arb_basic() {
        delete arb;
    }

    void setVCs(int n_vns, int* vcs_per_vn) {
        num_vns = n_vns;
        num_vcs = 0;
        vcs_per_vn_.assign(n_vns, 0);
        vn_offsets_.assign(n_vns, 0);
        vn_rr_vcs_.assign(n_vns, 0);
        for ( int i = 0; i < n_vns; ++i ) {
            vcs_per_vn_[i] = vcs_per_vn[i];
            vn_offsets_[i] = num_vcs;
            num_vcs += vcs_per_vn[i];
        }

        if ( !priority_vns.empty() ) {
            std::vector<int> normalized;
            std::vector<bool> seen(n_vns, false);
            for ( int vn : priority_vns ) {
                if ( vn < 0 || vn >= n_vns ) {
                    merlin_abort.fatal(CALL_INFO, -1,
                        "output_arb_basic: VN %d in vn_priority_order is outside [0,%d).\n",
                        vn, n_vns);
                }
                if ( !seen[vn] ) {
                    normalized.push_back(vn);
                    seen[vn] = true;
                }
            }
            for ( int vn = 0; vn < n_vns; ++vn ) {
                if ( !seen[vn] ) normalized.push_back(vn);
            }
            priority_vns.swap(normalized);
            if ( starvation_vn >= n_vns ) {
                merlin_abort.fatal(CALL_INFO, -1,
                    "output_arb_basic: vn_starvation_vn=%d is outside [0,%d).\n",
                    starvation_vn, n_vns);
            }
        }
        Params empty;
        arb = loadModule<SingleArbitration>(arb_name,empty,num_vcs);

    }

    int arbitrate(Cycle_t UNUSED(cycle), PortInterface::port_queue_t* out_q, int* port_out_credits, bool isHostPort, bool& have_packets) {
        if ( !priority_vns.empty() ) {
            have_packets = false;
            for ( int vc = 0; vc < num_vcs; ++vc ) {
                if ( !out_q[vc].empty() ) have_packets = true;
            }

            auto eligible = [&](int vc) {
                if ( out_q[vc].empty() ) return false;
                internal_router_event* event = out_q[vc].front();
                const int credit_vc = isHostPort ? event->getVN() : vc;
                return port_out_credits[credit_vc] >= event->getFlitCount();
            };

            auto choose_from_vn = [&](int vn, bool oldest) {
                if ( vn < 0 || vn >= num_vns ) return -1;
                const int count = vcs_per_vn_[vn];
                const int offset = vn_offsets_[vn];
                int chosen = -1;
                SimTime_t oldest_injection = 0;
                for ( int i = 0; i < count; ++i ) {
                    const int local_vc = (vn_rr_vcs_[vn] + i) % count;
                    const int vc = offset + local_vc;
                    if ( !eligible(vc) ) continue;
                    if ( !oldest ) {
                        chosen = vc;
                        break;
                    }
                    const SimTime_t injection =
                        out_q[vc].front()->getEncapsulatedEvent()->getInjectionTime();
                    if ( chosen == -1 || injection < oldest_injection ) {
                        chosen = vc;
                        oldest_injection = injection;
                    }
                }
                return chosen;
            };

            int chosen_vc = -1;
            bool starvation_bypass = false;
            if ( max_starvation_cycles != 0 && starvation_vn >= 0 ) {
                const int candidate = choose_from_vn(starvation_vn, true);
                if ( candidate >= 0 ) {
                    const SimTime_t injection =
                        out_q[candidate].front()->getEncapsulatedEvent()->getInjectionTime();
                    const SimTime_t now = getCurrentSimTimeNano();
                    if ( now >= injection && now - injection >= max_starvation_cycles ) {
                        chosen_vc = candidate;
                        starvation_bypass = true;
                    }
                }
            }

            if ( chosen_vc < 0 ) {
                for ( int vn : priority_vns ) {
                    chosen_vc = choose_from_vn(vn, false);
                    if ( chosen_vc >= 0 ) break;
                }
            }

            if ( chosen_vc >= 0 ) {
                const int vn = out_q[chosen_vc].front()->getVN();
                if ( vn >= 0 && vn < num_vns ) {
                    vn_rr_vcs_[vn] =
                        (chosen_vc - vn_offsets_[vn] + 1) % vcs_per_vn_[vn];
                }
                if ( vn == starvation_vn ) {
                    priority_low_grants->addData(1);
                    if ( starvation_bypass ) priority_starvation_grants->addData(1);
                } else {
                    priority_high_grants->addData(1);
                }
            }
            return chosen_vc;
        }

        int vc_to_send = -1;
        bool found = false;
        internal_router_event* send_event = NULL;
        have_packets = false;

        for ( int i = 0; i < num_vcs; ++i ) {
            int vc = arb->next();
            if ( out_q[vc].empty() ) continue;
            have_packets = true;
            send_event = out_q[vc].front();
            if ( port_out_credits[isHostPort ? send_event->getVN() : vc] < send_event->getFlitCount() ) continue;
            vc_to_send = vc;
            found = true;
            arb->satisfied();
            break;
        }
        return vc_to_send;
    }

    void dumpState(std::ostream& stream) {
    }

};

}
}

#endif // COMPONENTS_MERLIN_ROUTER_H
