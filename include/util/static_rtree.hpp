#ifndef STATIC_RTREE_HPP
#define STATIC_RTREE_HPP

#include "storage/io.hpp"
#include "util/bearing.hpp"
#include "util/coordinate_calculation.hpp"
#include "util/deallocating_vector.hpp"
#include "util/exception.hpp"
#include "util/integer_range.hpp"
#include "util/rectangle.hpp"
#include "util/typedefs.hpp"
#include "util/vector_view.hpp"
#include "util/web_mercator.hpp"

#include "osrm/coordinate.hpp"

#include "storage/shared_memory_ownership.hpp"

#include <boost/assert.hpp>
#include <boost/filesystem.hpp>
#include <boost/filesystem/fstream.hpp>
#include <boost/format.hpp>
#include <boost/iostreams/device/mapped_file.hpp>

#include <tbb/parallel_for.h>
#include <tbb/parallel_sort.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <queue>
#include <string>
#include <vector>

// An extended alignment is implementation-defined, so use compiler attributes
// until alignas(LEAF_PAGE_SIZE) is compiler-independent.
#if defined(_MSC_VER)
#define ALIGNED(x) __declspec(align(x))
#elif defined(__GNUC__)
#define ALIGNED(x) __attribute__((aligned(x)))
#else
#define ALIGNED(x)
#endif

namespace osrm
{
namespace util
{

// from https://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2
constexpr std::uint64_t round_up_power_2(std::uint64_t v)
{
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v |= v >> 32;
    v++;
    return v;
}

// Static RTree for serving nearest neighbour queries
// All coordinates are pojected first to Web Mercator before the bounding boxes
// are computed, this means the internal distance metric doesn not represent meters!
template <class EdgeDataT,
          storage::Ownership Ownership = storage::Ownership::Container,
          std::uint32_t BRANCHING_FACTOR = 64,
          std::uint32_t UNUSED = 1>
class StaticRTree
{
    template <typename T> using Vector = ViewOrVector<T, Ownership>;

  public:
    using Rectangle = RectangleInt2D;
    using EdgeData = EdgeDataT;
    using CoordinateList = Vector<util::Coordinate>;

    static constexpr auto UNALIGNED_LEAF_PAGE_SIZE =
        BRANCHING_FACTOR * sizeof(EdgeDataT) + sizeof(Rectangle) + sizeof(std::uint32_t);

    // Find the next power of 2 bigger than UNALIGNED_LEAF_PAGE_SIZE
    static constexpr auto LEAF_PAGE_SIZE = round_up_power_2(UNALIGNED_LEAF_PAGE_SIZE);

    static_assert(LEAF_PAGE_SIZE >= sizeof(uint32_t) + sizeof(EdgeDataT), "page size is too small");
    static_assert(((LEAF_PAGE_SIZE - 1) & LEAF_PAGE_SIZE) == 0, "page size is not a power of 2");
    static constexpr std::uint32_t LEAF_NODE_SIZE =
        (LEAF_PAGE_SIZE - sizeof(uint32_t) - sizeof(Rectangle)) / sizeof(EdgeDataT);

    static_assert(LEAF_NODE_SIZE >= BRANCHING_FACTOR,
                  "Leaf node count needs to be equal to the branching factor");

    struct CandidateSegment
    {
        Coordinate fixed_projected_coordinate;
        EdgeDataT data;
    };

    struct TreeIndex
    {
        TreeIndex() : index(0), is_leaf(false) {}
        TreeIndex(std::size_t index, bool is_leaf) : index(index), is_leaf(is_leaf) {}
        std::uint32_t index : 31;
        std::uint32_t is_leaf : 1;
    };
    struct TreeNode
    {
        TreeNode() : child_count(0) {}
        std::uint32_t child_count : 31;
        bool contains_leaves : 1;
        std::uint32_t first_child_index;
        Rectangle minimum_bounding_rectangle;
    };

    struct ALIGNED(LEAF_PAGE_SIZE) LeafNode
    {
        LeafNode() : object_count(0), objects() {}
        std::uint32_t object_count;
        Rectangle minimum_bounding_rectangle;
        std::array<EdgeDataT, BRANCHING_FACTOR> objects;
    };
    static_assert(sizeof(LeafNode) == LEAF_PAGE_SIZE, "LeafNode size does not fit the page size");

  private:
    struct QueryCandidate
    {
        QueryCandidate(std::uint64_t squared_min_dist, TreeIndex tree_index)
            : squared_min_dist(squared_min_dist), tree_index(tree_index),
              segment_index(std::numeric_limits<std::uint32_t>::max())
        {
        }

        QueryCandidate(std::uint64_t squared_min_dist,
                       TreeIndex tree_index,
                       std::uint32_t segment_index,
                       const Coordinate &coordinate)
            : squared_min_dist(squared_min_dist), tree_index(tree_index),
              segment_index(segment_index), fixed_projected_coordinate(coordinate)
        {
        }

        inline bool is_segment() const
        {
            return segment_index != std::numeric_limits<std::uint32_t>::max();
        }

        inline bool operator<(const QueryCandidate &other) const
        {
            // Attn: this is reversed order. std::pq is a max pq!
            return other.squared_min_dist < squared_min_dist;
        }

        std::uint64_t squared_min_dist;
        TreeIndex tree_index;
        std::uint32_t segment_index;
        Coordinate fixed_projected_coordinate;
    };

    Vector<TreeNode> m_search_tree;
    const Vector<Coordinate> &m_coordinate_list;

    boost::iostreams::mapped_file_source m_leaves_region;
    // read-only view of leaves
    util::vector_view<const LeafNode> m_leaves;

  public:
    StaticRTree(const StaticRTree &) = delete;
    StaticRTree &operator=(const StaticRTree &) = delete;

    // In-place partial sort-by-group
    template <typename Iterator, typename Compare>
    void grouped_partial_sort(Iterator left, Iterator right, std::size_t n, Compare compare)
    {
        std::stack<Iterator> stack;
        stack.push(left);
        stack.push(right);
        Iterator mid;

        while (!stack.empty())
        {
            right = stack.top();
            stack.pop();
            left = stack.top();
            stack.pop();

            if (std::distance(left, right) <= n)
                continue;

            // Important: note the use of static_cast<double>" here - we need this to be floating
            // point math, because we depend on the "ceil" behiaviour rounding up in some
            // circumstances.  If we left it as integer math, we'll sometimes round down, which
            // will put us into an infinite loop.  TODO: could this be achieved with integer math
            // and a +1 ?
            mid = left + std::ceil(static_cast<double>(std::distance(left, right)) / n / 2.) * n;

            std::partial_sort(left, mid, right, compare);

            stack.push(left);
            stack.push(mid);
            stack.push(mid);
            stack.push(right);
        }
    }

    // Constructs a packed RTree with the Lee-Lee OMT approach
    // This should minimize leaf-node overlap, which works well for the typical
    // layout of road network geometries
    explicit StaticRTree(const std::vector<EdgeDataT> &input_data_vector,
                         const std::string &tree_node_filename,
                         const std::string &leaf_node_filename,
                         const Vector<Coordinate> &coordinate_list)
        : m_coordinate_list(coordinate_list)
    {
        auto leaves = const_cast<std::vector<EdgeDataT> &>(input_data_vector);

        struct Range
        {
            Range(std::size_t parent_, std::size_t left_, std::size_t right_, std::size_t height_)
                : parent{parent_}, left{left_}, right{right_}, height{height_}
            {
            }
            std::size_t parent;
            std::size_t left;
            std::size_t right;
            std::size_t height;
        };

        // We use a queue here so that we can do a breadth-first
        std::queue<Range> queue;
        queue.emplace(0, 0, leaves.size() - 1, 0);

        // TODO: we do a lot of sorting - it would make sense to only calculate centroids once
        auto longitude_compare = [this](const EdgeDataT &a, const EdgeDataT &b) {
            auto a_centroid =
                coordinate_calculation::centroid(m_coordinate_list[a.u], m_coordinate_list[a.v]);
            auto b_centroid =
                coordinate_calculation::centroid(m_coordinate_list[b.u], m_coordinate_list[b.v]);
            return a_centroid.lon < b_centroid.lon;
        };

        auto latitude_compare = [this](const EdgeDataT &a, const EdgeDataT &b) {
            auto a_centroid =
                coordinate_calculation::centroid(m_coordinate_list[a.u], m_coordinate_list[a.v]);
            auto b_centroid =
                coordinate_calculation::centroid(m_coordinate_list[b.u], m_coordinate_list[b.v]);
            return a_centroid.lat < b_centroid.lat;
        };

        // util::Log() << "LEAF_NODE_SIZE " << LEAF_NODE_SIZE;

        boost::filesystem::ofstream leaf_node_file(leaf_node_filename, std::ios::binary);

        // position of the last leaf node written to diskcountindex
        std::size_t leaf_node_count = 0;

        while (!queue.empty())
        {
            auto r = queue.front();
            queue.pop();

            /*
                        std::clog << "Dequing h=" << r.height << " from  " << r.left << " to  " <<
               r.right
                                  << std::endl;
                                  */

            auto N = r.right - r.left + 1;
            auto M = BRANCHING_FACTOR;

            // We're processing a leaf
            if (N <= M)
            {
                LeafNode current_leaf;
                current_leaf.object_count = N;
                // This copies the actual EdgeDataT objects into our leaf node struct
                BOOST_ASSERT(N - 1 <= LEAF_NODE_SIZE);
                std::copy(leaves.begin() + r.left,
                          leaves.begin() + r.right + 1,
                          current_leaf.objects.begin());
                // Now calculate the bounding-box
                std::for_each(current_leaf.objects.begin(),
                              current_leaf.objects.begin() + current_leaf.object_count,
                              [this, &current_leaf](const EdgeDataT &edge) {
                                  Coordinate projected_u{web_mercator::fromWGS84(
                                      Coordinate{m_coordinate_list[edge.u]})};
                                  Coordinate projected_v{web_mercator::fromWGS84(
                                      Coordinate{m_coordinate_list[edge.v]})};
                                  current_leaf.minimum_bounding_rectangle.Extend(projected_u.lon,
                                                                                 projected_u.lat);
                                  current_leaf.minimum_bounding_rectangle.Extend(projected_v.lon,
                                                                                 projected_v.lat);
                                  /*
std::cout
<< "{ "
"\"type\":\"Feature\",\"properties\":{},\"geometry\":{\"type\":"
"\"LineString\",\"coordinates\":[";
std::cout << "[" << toFloating(projected_u.lon) << ","
<< toFloating(projected_u.lat) << "],";
std::cout << "[" << toFloating(projected_v.lon) << ","
<< toFloating(projected_v.lat) << "],";
std::cout << "]}}," << std::endl;
*/
                              });

                leaf_node_file.write((char *)&current_leaf, sizeof(current_leaf));

                /*
                                std::cout << "{ "
                                             "\"type\":\"Feature\",\"properties\":{},\"geometry\":{\"type\":"
                                             "\"Polygon\",\"coordinates\":[[";
                                std::cout << "[" <<
                   toFloating(current_leaf.minimum_bounding_rectangle.min_lon)
                                          << "," <<
                   toFloating(current_leaf.minimum_bounding_rectangle.min_lat)
                                          << "],";
                                std::cout << "[" <<
                   toFloating(current_leaf.minimum_bounding_rectangle.min_lon)
                                          << "," <<
                   toFloating(current_leaf.minimum_bounding_rectangle.max_lat)
                                          << "],";
                                std::cout << "[" <<
                   toFloating(current_leaf.minimum_bounding_rectangle.max_lon)
                                          << "," <<
                   toFloating(current_leaf.minimum_bounding_rectangle.max_lat)
                                          << "],";
                                std::cout << "[" <<
                   toFloating(current_leaf.minimum_bounding_rectangle.max_lon)
                                          << "," <<
                   toFloating(current_leaf.minimum_bounding_rectangle.min_lat)
                                          << "],";
                                std::cout << "[" <<
                   toFloating(current_leaf.minimum_bounding_rectangle.min_lon)
                                          << "," <<
                   toFloating(current_leaf.minimum_bounding_rectangle.min_lat)
                                          << "]";
                                std::cout << "]]}}," << std::endl;

                                std::clog << r.parent << std::endl;
                                */

                // Special case for tiny trees:
                if (m_search_tree.empty())
                {
                    TreeNode current_node;
                    m_search_tree.push_back(current_node);
                }
                BOOST_ASSERT(m_search_tree[r.parent].child_count <= BRANCHING_FACTOR);

                // If this is the first node for the TreeNode, set the offset
                // We should be packing in breadth-first-search order, so we only
                // need to set the first index, then track the count
                if (m_search_tree[r.parent].child_count == 0)
                {
                    m_search_tree[r.parent].first_child_index = leaf_node_count;
                    m_search_tree[r.parent].contains_leaves = true;
                }
                // Grow the parent node's bounding box'
                m_search_tree[r.parent].minimum_bounding_rectangle.MergeBoundingBoxes(
                    current_leaf.minimum_bounding_rectangle);
                ++leaf_node_count;
                ++m_search_tree[r.parent].child_count;
                BOOST_ASSERT(m_search_tree[r.parent].child_count <= BRANCHING_FACTOR);
                continue;
            }

            TreeNode current_node;
            m_search_tree.push_back(current_node);

            // Special case for the first item (height = 0),
            if (r.height == 0)
            {
                /*
                std::clog << "Processing the first node" << std::endl;
                std::clog << "There are " << N
                          << " nodes that need to be indexed and our factor is " << M << std::endl;
                          */
                r.height = std::ceil(std::log(N) / std::log(M));
                // std::clog << "Max height is " << r.height << std::endl;
                M = std::ceil(N / std::pow(M, r.height - 1));
                // std::clog << "M for the root node is " << M << std::endl;
            }
            else
            {
                // std::clog << "processing a tree node with parent " << r.parent << std::endl;
                BOOST_ASSERT(m_search_tree[r.parent].child_count < BRANCHING_FACTOR);
                if (m_search_tree[r.parent].child_count == 0)
                {
                    /*
                    std::clog << r.parent << " has child count of 0, setting to "
                              << m_search_tree.size() - 1 << std::endl;
                              */
                    m_search_tree[r.parent].first_child_index = m_search_tree.size() - 1;
                    m_search_tree[r.parent].contains_leaves = false;
                }
                ++m_search_tree[r.parent].child_count;
            }

            std::size_t N2 = std::ceil(static_cast<double>(N) / M);
            std::size_t N1 = N2 * std::ceil(std::sqrt(M));
            // std::clog << "N2 is " << N2 << " and N1 is " << N1 << std::endl;

            /*
                        grouped_partial_sort(
                            leaves.begin() + r.left, leaves.begin() + r.right, N,
               longitude_compare);
                            */

            std::sort(leaves.begin() + r.left, leaves.begin() + r.right, longitude_compare);

            // Now, for each column (there are S columns)
            for (auto i = r.left; i <= r.right; i += N1)
            {
                auto right2 = std::min(i + N1 - 1, r.right);
                /*
                grouped_partial_sort(
                    leaves.begin() + i, leaves.begin() + right2, N2, latitude_compare);
                    */
                std::sort(leaves.begin() + i, leaves.begin() + right2, latitude_compare);
                for (auto j = i; j <= right2; j += N2)
                {
                    auto right3 = std::min(j + N2 - 1, right2);
                    /*
                    std::clog << "Queueing up h=" << r.height - 1 << " from " << j << " to "
                              << right3 << std::endl;
                              */
                    queue.emplace(m_search_tree.size() - 1, j, right3, r.height - 1);
                }
                // std::clog << "-" << std::endl;
            }
            // std::clog << "#############" << std::endl;
        }

        // Because we used a queue above, the m_search_tree vector is sorted in
        // the same order as a breadth-first-search of the tree.  We can iterate
        // over this in reverse and propogate node recangle sizes up the tree
        // The leaf nodes already have their bounding box set, so we just need to
        // propogate those up the tree

        std::stack<std::size_t> n;
        for (auto n = m_search_tree.rbegin(); n != m_search_tree.rend(); n++)
        {
            // Skip the bottom of the tree, the sizes are already set
            if (n->child_count == 0 || n->contains_leaves)
            {
                /*
                std::cout << "{ "
                             "\"type\":\"Feature\",\"properties\":{},\"geometry\":{\"type\":"
                             "\"Polygon\",\"coordinates\":[[";
                std::cout << "[" << toFloating(n->minimum_bounding_rectangle.min_lon) << ","
                          << toFloating(n->minimum_bounding_rectangle.min_lat) << "],";
                std::cout << "[" << toFloating(n->minimum_bounding_rectangle.min_lon) << ","
                          << toFloating(n->minimum_bounding_rectangle.max_lat) << "],";
                std::cout << "[" << toFloating(n->minimum_bounding_rectangle.max_lon) << ","
                          << toFloating(n->minimum_bounding_rectangle.max_lat) << "],";
                std::cout << "[" << toFloating(n->minimum_bounding_rectangle.max_lon) << ","
                          << toFloating(n->minimum_bounding_rectangle.min_lat) << "],";
                std::cout << "[" << toFloating(n->minimum_bounding_rectangle.min_lon) << ","
                          << toFloating(n->minimum_bounding_rectangle.min_lat) << "]";
                std::cout << "]]}}," << std::endl;
                */
                continue;
            }
            for (auto i = 0u; i < n->child_count; i++)
            {
                n->minimum_bounding_rectangle.MergeBoundingBoxes(
                    m_search_tree[n->first_child_index + i].minimum_bounding_rectangle);
            }
            /*
            if (n != m_search_tree.rbegin())
                std::cout << "," << std::endl;
            std::cout << "{ "
                         "\"type\":\"Feature\",\"properties\":{},\"geometry\":{\"type\":"
                         "\"Polygon\",\"coordinates\":[[";
            std::cout << "[" << toFloating(n->minimum_bounding_rectangle.min_lon) << ","
                      << toFloating(n->minimum_bounding_rectangle.min_lat) << "],";
            std::cout << "[" << toFloating(n->minimum_bounding_rectangle.min_lon) << ","
                      << toFloating(n->minimum_bounding_rectangle.max_lat) << "],";
            std::cout << "[" << toFloating(n->minimum_bounding_rectangle.max_lon) << ","
                      << toFloating(n->minimum_bounding_rectangle.max_lat) << "],";
            std::cout << "[" << toFloating(n->minimum_bounding_rectangle.max_lon) << ","
                      << toFloating(n->minimum_bounding_rectangle.min_lat) << "],";
            std::cout << "[" << toFloating(n->minimum_bounding_rectangle.min_lon) << ","
                      << toFloating(n->minimum_bounding_rectangle.min_lat) << "]";
            std::cout << "]]}}," << std::endl;
            */
        }
        // std::cout << std::endl;

        /*
                util::Log() << "There are now " << leaf_node_count << " leaf nodes and "
                            << m_search_tree.size() << " tree nodes";
                            */

        // open tree file
        storage::io::FileWriter tree_node_file(tree_node_filename,
                                               storage::io::FileWriter::GenerateFingerprint);

        std::uint64_t size_of_tree = m_search_tree.size();
        BOOST_ASSERT_MSG(0 < size_of_tree, "tree empty");

        tree_node_file.WriteOne(size_of_tree);
        tree_node_file.WriteFrom(&m_search_tree[0], size_of_tree);

        MapLeafNodesFile(leaf_node_filename);
    }

    explicit StaticRTree(const boost::filesystem::path &node_file,
                         const boost::filesystem::path &leaf_file,
                         const Vector<Coordinate> &coordinate_list)
        : m_coordinate_list(coordinate_list)
    {
        storage::io::FileReader tree_node_file(node_file,
                                               storage::io::FileReader::VerifyFingerprint);

        const auto tree_size = tree_node_file.ReadElementCount64();

        m_search_tree.resize(tree_size);
        tree_node_file.ReadInto(&m_search_tree[0], tree_size);

        MapLeafNodesFile(leaf_file);
    }

    explicit StaticRTree(TreeNode *tree_node_ptr,
                         const uint64_t number_of_nodes,
                         const boost::filesystem::path &leaf_file,
                         const Vector<Coordinate> &coordinate_list)
        : m_search_tree(tree_node_ptr, number_of_nodes), m_coordinate_list(coordinate_list)
    {
        MapLeafNodesFile(leaf_file);
    }

    void MapLeafNodesFile(const boost::filesystem::path &leaf_file)
    {
        // open leaf node file and return a pointer to the mapped leaves data
        try
        {
            m_leaves_region.open(leaf_file);
            std::size_t num_leaves = m_leaves_region.size() / sizeof(LeafNode);
            auto data_ptr = m_leaves_region.data();
            BOOST_ASSERT(reinterpret_cast<uintptr_t>(data_ptr) % alignof(LeafNode) == 0);
            m_leaves.reset(reinterpret_cast<const LeafNode *>(data_ptr), num_leaves);
        }
        catch (const std::exception &exc)
        {
            throw exception(boost::str(boost::format("Leaf file %1% mapping failed: %2%") %
                                       leaf_file % exc.what()) +
                            SOURCE_REF);
        }
    }

    /* Returns all features inside the bounding box.
       Rectangle needs to be projected!*/
    std::vector<EdgeDataT> SearchInBox(const Rectangle &search_rectangle) const
    {
        const Rectangle projected_rectangle{
            search_rectangle.min_lon,
            search_rectangle.max_lon,
            toFixed(FloatLatitude{
                web_mercator::latToY(toFloating(FixedLatitude(search_rectangle.min_lat)))}),
            toFixed(FloatLatitude{
                web_mercator::latToY(toFloating(FixedLatitude(search_rectangle.max_lat)))})};
        std::vector<EdgeDataT> results;

        std::queue<TreeIndex> traversal_queue;
        traversal_queue.push(TreeIndex{});

        while (!traversal_queue.empty())
        {
            auto const current_tree_index = traversal_queue.front();
            traversal_queue.pop();

            if (current_tree_index.is_leaf)
            {
                const LeafNode &current_leaf_node = m_leaves[current_tree_index.index];

                for (const auto i : irange(0u, current_leaf_node.object_count))
                {
                    const auto &current_edge = current_leaf_node.objects[i];

                    // we don't need to project the coordinates here,
                    // because we use the unprojected rectangle to test against
                    const Rectangle bbox{std::min(m_coordinate_list[current_edge.u].lon,
                                                  m_coordinate_list[current_edge.v].lon),
                                         std::max(m_coordinate_list[current_edge.u].lon,
                                                  m_coordinate_list[current_edge.v].lon),
                                         std::min(m_coordinate_list[current_edge.u].lat,
                                                  m_coordinate_list[current_edge.v].lat),
                                         std::max(m_coordinate_list[current_edge.u].lat,
                                                  m_coordinate_list[current_edge.v].lat)};

                    // use the _unprojected_ input rectangle here
                    if (bbox.Intersects(search_rectangle))
                    {
                        results.push_back(current_edge);
                    }
                }
            }
            else
            {
                const TreeNode &current_tree_node = m_search_tree[current_tree_index.index];

                // If it's a tree node, look at all children and add them
                // to the search queue if their bounding boxes intersect
                for (std::uint32_t i = 0; i < current_tree_node.child_count; ++i)
                {
                    const auto child_id = current_tree_node.first_child_index + i;
                    const auto &child_rectangle =
                        current_tree_node.contains_leaves
                            ? m_leaves[child_id].minimum_bounding_rectangle
                            : m_search_tree[child_id].minimum_bounding_rectangle;

                    if (child_rectangle.Intersects(projected_rectangle))
                    {
                        if (current_tree_node.contains_leaves)
                        {
                            traversal_queue.push(TreeIndex{child_id, true});
                        }
                        else
                        {
                            traversal_queue.push(TreeIndex{child_id, false});
                        }
                    }
                }
            }
        }
        return results;
    }

    // Override filter and terminator for the desired behaviour.
    std::vector<EdgeDataT> Nearest(const Coordinate input_coordinate,
                                   const std::size_t max_results) const
    {
        return Nearest(input_coordinate,
                       [](const CandidateSegment &) { return std::make_pair(true, true); },
                       [max_results](const std::size_t num_results, const CandidateSegment &) {
                           return num_results >= max_results;
                       });
    }

    // Override filter and terminator for the desired behaviour.
    template <typename FilterT, typename TerminationT>
    std::vector<EdgeDataT> Nearest(const Coordinate input_coordinate,
                                   const FilterT filter,
                                   const TerminationT terminate) const
    {
        std::vector<EdgeDataT> results;
        auto projected_coordinate = web_mercator::fromWGS84(input_coordinate);
        Coordinate fixed_projected_coordinate{projected_coordinate};

        // initialize queue with root element
        std::priority_queue<QueryCandidate> traversal_queue;
        traversal_queue.push(QueryCandidate{0, TreeIndex{}});

        while (!traversal_queue.empty())
        {
            QueryCandidate current_query_node = traversal_queue.top();
            traversal_queue.pop();

            const TreeIndex &current_tree_index = current_query_node.tree_index;
            if (!current_query_node.is_segment())
            { // current object is a tree node
                if (current_tree_index.is_leaf)
                {
                    ExploreLeafNode(current_tree_index,
                                    fixed_projected_coordinate,
                                    projected_coordinate,
                                    traversal_queue);
                }
                else
                {
                    ExploreTreeNode(
                        current_tree_index, fixed_projected_coordinate, traversal_queue);
                }
            }
            else
            { // current candidate is an actual road segment
                auto edge_data =
                    m_leaves[current_tree_index.index].objects[current_query_node.segment_index];
                const auto &current_candidate =
                    CandidateSegment{current_query_node.fixed_projected_coordinate, edge_data};

                /*
                                std::clog << "Selecting item at " <<
                   current_query_node.squared_min_dist
                                          << std::endl;
                                          */
                // to allow returns of no-results if too restrictive filtering, this needs to be
                // done here even though performance would indicate that we want to stop after
                // adding the first candidate
                if (terminate(results.size(), current_candidate))
                {
                    // std::clog << "Terminating" << std::endl;
                    break;
                }

                auto use_segment = filter(current_candidate);
                if (!use_segment.first && !use_segment.second)
                {
                    // std::clog << "Can't use" << std::endl;
                    continue;
                }
                edge_data.forward_segment_id.enabled &= use_segment.first;
                edge_data.reverse_segment_id.enabled &= use_segment.second;

                // std::clog << "Pushing result" << std::endl;
                // store phantom node in result vector
                results.push_back(std::move(edge_data));
            }
        }

        return results;
    }

  private:
    template <typename QueueT>
    void ExploreLeafNode(const TreeIndex &leaf_id,
                         const Coordinate &projected_input_coordinate_fixed,
                         const FloatCoordinate &projected_input_coordinate,
                         QueueT &traversal_queue) const
    {
        const LeafNode &current_leaf_node = m_leaves[leaf_id.index];

        // current object represents a block on disk
        for (const auto i : irange(0u, current_leaf_node.object_count))
        {
            const auto &current_edge = current_leaf_node.objects[i];
            const auto projected_u = web_mercator::fromWGS84(m_coordinate_list[current_edge.u]);
            const auto projected_v = web_mercator::fromWGS84(m_coordinate_list[current_edge.v]);

            FloatCoordinate projected_nearest;
            std::tie(std::ignore, projected_nearest) =
                coordinate_calculation::projectPointOnSegment(
                    projected_u, projected_v, projected_input_coordinate);

            const auto squared_distance = coordinate_calculation::squaredEuclideanDistance(
                projected_input_coordinate_fixed, projected_nearest);

            /*
                        std::cout << "{ "
                                     "\"type\":\"Feature\",\"properties\":{"
                                     "  \"stroke\": \"#ff0f00\","
                                     "  \"stroke-width\": 2,"
                                     " \"stroke-opacity\": 1"
                                     " \"squared-distance\": "
                                  << squared_distance << "},\"geometry\":{\"type\":"
                                                         "\"LineString\",\"coordinates\":[";
                        std::cout << "[" << toFloating(Coordinate{projected_u}.lon) << ","
                                  << toFloating(Coordinate{projected_u}.lat) << "],";
                        std::cout << "[" << toFloating(Coordinate{projected_v}.lon) << ","
                                  << toFloating(Coordinate{projected_v}.lat) << "],";
                        std::cout << "]}}," << std::endl;
                        */

            // distance must be non-negative
            BOOST_ASSERT(0. <= squared_distance);
            traversal_queue.push(
                QueryCandidate{squared_distance, leaf_id, i, Coordinate{projected_nearest}});
        }
        // std::clog << "---" << std::endl;
    }

    template <class QueueT>
    void ExploreTreeNode(const TreeIndex &parent_id,
                         const Coordinate &fixed_projected_input_coordinate,
                         QueueT &traversal_queue) const
    {
        const TreeNode &parent = m_search_tree[parent_id.index];
        for (std::uint32_t i = 0; i < parent.child_count; ++i)
        {
            const std::uint32_t child_id = parent.first_child_index + i;
            const auto &child_rectangle = parent.contains_leaves
                                              ? m_leaves[child_id].minimum_bounding_rectangle
                                              : m_search_tree[child_id].minimum_bounding_rectangle;
            const auto squared_lower_bound_to_element =
                child_rectangle.GetMinSquaredDist(fixed_projected_input_coordinate);
            traversal_queue.push(QueryCandidate{squared_lower_bound_to_element,
                                                TreeIndex{child_id, parent.contains_leaves}});
        }
    }
};

//[1] "On Packing R-Trees"; I. Kamel, C. Faloutsos; 1993; DOI: 10.1145/170088.170403
//[2] "Nearest Neighbor Queries", N. Roussopulos et al; 1995; DOI: 10.1145/223784.223794
//[3] "Distance Browsing in Spatial Databases"; G. Hjaltason, H. Samet; 1999; ACM Trans. DB Sys
// Vol.24 No.2, pp.265-318
}
}

#endif // STATIC_RTREE_HPP
