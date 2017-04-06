#ifndef ALTERNATIVE_PATH_ROUTING_HPP
#define ALTERNATIVE_PATH_ROUTING_HPP

#include "engine/datafacade/contiguous_internalmem_datafacade.hpp"
#include "engine/internal_route_result.hpp"

#include "engine/algorithm.hpp"
#include "engine/search_engine_data.hpp"

#include "util/exception.hpp"

namespace osrm
{
namespace engine
{
namespace routing_algorithms
{

InternalRouteResult
alternativePathSearch(SearchEngineData<ch::Algorithm> &search_engine_data,
                      const datafacade::ContiguousInternalMemoryDataFacade<ch::Algorithm> &facade,
                      const PhantomNodes &phantom_node_pair);

InternalRouteResult
alternativePathSearch(SearchEngineData<mld::Algorithm> &search_engine_data,
                      const datafacade::ContiguousInternalMemoryDataFacade<mld::Algorithm> &facade,
                      const PhantomNodes &phantom_node_pair);

} // namespace routing_algorithms
} // namespace engine
} // namespace osrm

#endif
