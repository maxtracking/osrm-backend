#ifndef OSRM_CONTRACTOR_CONTRACTOR_HEAP_HPP_
#define OSRM_CONTRACTOR_CONTRACTOR_HEAP_HPP_

#include "util/search_heap.hpp"
#include "util/typedefs.hpp"
#include "util/xor_fast_hash_storage.hpp"

namespace osrm
{
namespace contractor
{
struct ContractorHeapData
{
    ContractorHeapData() {}
    ContractorHeapData(short hop_, bool target_) : hop(hop_), target(target_) {}

    short hop = 0;
    bool target = false;
};

using ContractorHeap = util::SearchHeap<NodeID, EdgeWeight, ContractorHeapData>;

} // namespace contractor
} // namespace osrm

#endif // OSRM_CONTRACTOR_CONTRACTOR_HEAP_HPP_
