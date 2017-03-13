#ifndef OSRM_UTIL_SEARCH_HEAP_HPP
#define OSRM_UTIL_SEARCH_HEAP_HPP

#include <boost/heap/d_ary_heap.hpp>

#include <deque>
#include <type_traits>
#include <unordered_map>

namespace osrm
{
namespace util
{

template <typename NodeID, typename Weight, typename Data = void> struct SearchHeap
{

    using WeightType = Weight;
    using DataType = Data;

    struct NodeData;
    using DataContainer = std::deque<NodeData>;
    using DataReference = typename std::add_pointer<typename DataContainer::reference>::type;
    using DataIndex = std::unordered_map<NodeID, DataReference>;
    using HeapData = std::pair<WeightType, DataReference>;
    using HeapContainer = boost::heap::d_ary_heap<HeapData,
                                                  boost::heap::arity<4>,
                                                  boost::heap::mutable_<true>,
                                                  boost::heap::compare<std::greater<HeapData>>>;
    using HeapHandle = typename HeapContainer::handle_type;

    struct NodeData
    {
        NodeID node;
        WeightType weight;
        DataType data;
        HeapHandle handle;
    };

    DataContainer data;
    DataIndex index;
    HeapContainer heap;

    SearchHeap(size_t) { index.rehash(1000); }

    void Clear()
    {
        index.clear();
        heap.clear();
        data.clear();
    }

    std::size_t Size() const { return heap.size(); }

    bool Empty() const { return heap.empty(); }

    void Insert(NodeID node, WeightType weight, DataType node_data)
    {
        data.emplace_back(NodeData{node, weight, std::move(node_data), HeapHandle()});
        auto reference = &data.back();
        reference->handle = heap.push(std::make_pair(weight, reference));
        index.insert(std::make_pair(node, reference));
    }

    DataType const &GetData(NodeID node) const
    {
        BOOST_ASSERT(WasInserted(node));
        return index.find(node)->second->data;
    }

    DataType &GetData(NodeID node)
    {
        BOOST_ASSERT(WasInserted(node));
        return index.find(node)->second->data;
    }

    WeightType const &GetKey(NodeID node) const
    {
        BOOST_ASSERT(WasInserted(node));
        return index.find(node)->second->weight;
    }

    bool WasRemoved(const NodeID node) const
    {
        BOOST_ASSERT(WasInserted(node));
        return index.find(node)->second->handle == HeapHandle();
    }

    bool WasInserted(const NodeID node) const { return index.find(node) != index.end(); }

    WeightType MinKey() const
    {
        BOOST_ASSERT(!heap.empty());
        return heap.top().first;
    }

    NodeID Min() const
    {
        BOOST_ASSERT(!heap.empty());
        return heap.top().second->node;
    }

    NodeID DeleteMin()
    {
        BOOST_ASSERT(!heap.empty());
        BOOST_ASSERT(heap.top().first == heap.top().second->weight);
        auto node_data = heap.top().second;
        heap.pop();

        node_data->handle = HeapHandle();
        return node_data->node;
    }

    void DeleteAll()
    {
        std::for_each(data.begin(), data.end(), [](auto &node) { node.handle = HeapHandle(); });
        heap.clear();
    }

    void DecreaseKey(NodeID node, WeightType weight)
    {
        BOOST_ASSERT(WasInserted(node));
        auto reference = index.find(node)->second;
        reference->weight = weight;
        heap.increase(reference->handle, std::make_pair(weight, reference));
    }
};
}
}

#endif // OSRM_UTIL_SEARCH_HEAP_HPP
