/*  @@@@@@@@  @@           @@@@@@   @@@@@@@@ @@
   /@@/////  /@@          @@////@@ @@////// /@@
   /@@       /@@  @@@@@  @@    // /@@       /@@
   /@@@@@@@  /@@ @@///@@/@@       /@@@@@@@@@/@@
   /@@////   /@@/@@@@@@@/@@       ////////@@/@@
   /@@       /@@/@@//// //@@    @@       /@@/@@
   /@@       @@@//@@@@@@ //@@@@@@  @@@@@@@@ /@@
   //       ///  //////   //////  ////////  //

   Copyright (c) 2016, Los Alamos National Security, LLC
   All rights reserved.
                                                                              */

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <mpi.h>

#include "legion.h"
#include "mappers/default_mapper.h"
#include "mapper.h"

using namespace Legion;
using namespace Legion::Mapping;

enum TaskIDs {
  TOP_LEVEL_TASK_ID,
  FILL_PART_TASK_ID,
  FILL_TASK_ID,
  CHECK_TASK_ID,
};

enum FieldIDs {
  FID_VAL,
  FID_RECT,
};


// Globals
MPILegionHandshake handshake;
int num_cpus = 0;

//-------------------------------------------------------------------------
// top level task
//-------------------------------------------------------------------------
void
top_level_task(const Task * task,
  const std::vector<PhysicalRegion> & regions,
  Context ctx,
  HighLevelRuntime * runtime) {

  // set-up handshake

  const std::map<int, Legion::AddressSpace> & forward_mapping =
    runtime->find_forward_MPI_mapping();

  for(std::map<int, Legion::AddressSpace>::const_iterator it =
        forward_mapping.begin();
      it != forward_mapping.end(); it++)
    printf(
      "MPI Rank %d maps to Legion Address Space %d\n", it->first, it->second);

  handshake.legion_wait_on_mpi();

  const size_t num_elmts = 64;
  const size_t num_ghosts = 2;
  const size_t num_colors = num_cpus;

  Rect<1> color_bounds(0, num_colors - 1);
  IndexSpaceT<1> color_is = runtime->create_index_space(ctx, color_bounds);

  size_t max_size = size_t(-1) / (num_colors * sizeof(int));

  ///////////////////////////////////////////////////////////////////////////
  // Creating Logical regions and partitions of these regions for src and dst
  ///////////////////////////////////////////////////////////////////////////

  Rect<2> rect_blis(
    Legion::Point<2>(0, 0), Legion::Point<2>(num_colors - 1, max_size - 1));

  IndexSpace is_blis = runtime->create_index_space(ctx, rect_blis);

  FieldSpace fs_blis = runtime->create_field_space(ctx);
  {
    FieldAllocator allocator = runtime->create_field_allocator(ctx, fs_blis);
    allocator.allocate_field(sizeof(size_t), FID_VAL);
  }

  LogicalRegion lr_blis = runtime->create_logical_region(ctx, is_blis, fs_blis);

 //-------------------------------------------------
 //create used/unused disjoint complete partitioning
 //-------------------------------------------------
 
 //1 create colors x 2 index space that will store infomration about rects
 
  Rect<2> rect_part(
    Legion::Point<2>(0, 0), Legion::Point<2>(num_colors - 1, 1));
  IndexSpaceT<2> is_part = runtime->create_index_space(ctx, rect_part); 
  FieldSpace fs_part = runtime->create_field_space(ctx);
  {
    FieldAllocator allocator = runtime->create_field_allocator(ctx, fs_part);
    allocator.allocate_field(sizeof(Legion::Rect<2>), FID_RECT);
  }
  LogicalRegion lr_part = runtime->create_logical_region(ctx, is_part, fs_part);  
  // partition lr_part by rows;

  Legion::Transform<2, 1> idx_tsfm;
  idx_tsfm[0][0] = 1;
  idx_tsfm[1][0] = 0;
  IndexPartition ip_part = runtime->create_partition_by_restriction
    (ctx,
    is_part,
    color_is,
    idx_tsfm,
    {{0,0}, {0,1}});
  LogicalPartition lp_part = runtime->get_logical_partition(lr_part, ip_part);

  // fill logical region that stores information about partitioning
  printf("Filling partition information topology sizes...\n");
  ArgumentMap idx_arg_map;
  IndexLauncher part_init_launcher(FILL_PART_TASK_ID, color_bounds,
                              TaskArgument(NULL, 0), idx_arg_map);
  part_init_launcher.add_region_requirement(
      RegionRequirement(lp_part, 0, WRITE_DISCARD, EXCLUSIVE, lr_part));
  part_init_launcher.region_requirements[0].add_field(FID_RECT);
  runtime->execute_index_space(ctx, part_init_launcher);

  // Create partition of the extended index space based on the regions
  // This is used/unused DISJOINT COMPLETE PARTITION
  IndexPartition ip_dc = runtime->create_partition_by_image_range(
      ctx,
      is_blis,
      lp_part,
      lr_part,//runtime->get_parent_logical_region(idx_lp),
      FID_RECT,
      color_is,
      DISJOINT_COMPLETE_KIND);
  LogicalPartition lp_dc = runtime->get_logical_partition(lr_blis, ip_dc);

  //----------------
  //check the pertition
  //-----------------

  IndexLauncher check_init_launcher(CHECK_TASK_ID, color_bounds,
                              TaskArgument(NULL, 0), idx_arg_map);
  check_init_launcher.add_region_requirement(
      RegionRequirement(lp_dc, 0, NO_ACCESS, EXCLUSIVE, lr_blis));
  check_init_launcher.region_requirements[0].add_field(FID_VAL);
  runtime->execute_index_space(ctx, check_init_launcher);

#if 0
  // Create primary partitioning for is_blis
  DomainColoring pr_partitioning_blis;
  for(size_t color = 0; color < num_colors; ++color) {
    Rect<2> subrect(Point<2>(color, 0), Point<2>(color, num_elmts - 1));

    pr_partitioning_blis[color] = subrect;
  }

  IndexPartition ip_pr = runtime->create_index_partition(
    ctx, is_blis, color_bounds, pr_partitioning_blis, true /*disjoint*/);

  LogicalPartition lp_pr =
    runtime->get_logical_partition(ctx, lr_blis, ip_pr);

  // Create ghost partitioning for is_blis
  DomainColoring gh_partitioning_blis;
  for(size_t color = 0; color < num_colors; ++color) {
    Rect<2> subrect(Point<2>(color, num_elmts), Point<2>(color, num_elmts +num_ghosts - 1));

    gh_partitioning_blis[color] = subrect;
  }

  IndexPartition ip_gh = runtime->create_index_partition(
    ctx, is_blis, color_bounds, gh_partitioning_blis, true /*disjoint*/);

  LogicalPartition lp_gh =
    runtime->get_logical_partition(ctx, lr_blis, ip_gh); 


  ArgumentMap arg_map;
  ////////////////////////////////////////////////////////////////////////
  // fill task
  ////////////////////////////////////////////////////////////////////////
  IndexLauncher fill_launcher(
    FILL_TASK_ID, color_bounds, TaskArgument(NULL, 0), arg_map);

  {
    Legion::RegionRequirement rr1(lp_pr, 0, WRITE_DISCARD,
      EXCLUSIVE, lr_blis);
    rr1.add_field(FID_VAL);
    Legion::RegionRequirement rr2(lp_gh, 0, WRITE_DISCARD,
      EXCLUSIVE, lr_blis);
    rr2.add_field(FID_VAL);
    fill_launcher.add_region_requirement(rr1);
    fill_launcher.add_region_requirement(rr2);
  } // scope

  auto future_fill = runtime->execute_index_space(ctx, fill_launcher);

  ////////////////////////////////////////////////////////////////////////
  // check task
  ////////////////////////////////////////////////////////////////////////
  IndexLauncher check_launcher(
    CHECK_TASK_ID, color_bounds, TaskArgument(NULL, 0), arg_map);

  {
    Legion::RegionRequirement rr1(lp_pr, 0, READ_WRITE,
      EXCLUSIVE, lr_blis);
    rr1.add_field(FID_VAL);
    Legion::RegionRequirement rr2(lp_gh, 0, READ_ONLY,
      EXCLUSIVE, lr_blis);
    rr2.add_field(FID_VAL);
    check_launcher.add_region_requirement(rr1);
    check_launcher.add_region_requirement(rr2);
  } // scope

  auto future_check = runtime->execute_index_space(ctx, check_launcher);


#endif

  ///////////////////////////////////////////////////////////////////////
  // clean up our region, index space, and field space
  ///////////////////////////////////////////////////////////////////////

  runtime->destroy_logical_region(ctx, lr_blis);
  runtime->destroy_field_space(ctx, fs_blis);
  runtime->destroy_logical_region(ctx, lr_part);
  runtime->destroy_field_space(ctx, fs_part);

  handshake.legion_handoff_to_mpi();
}

void 
fill_part_task(const Task * task,
  const std::vector<PhysicalRegion> & regions,
  Context ctx,
  HighLevelRuntime * runtime) {
  assert(regions.size() == 1);
  Legion::Domain dom = runtime->get_index_space_domain(
    ctx, task->regions[0].region.get_index_space());

  const FieldAccessor<WRITE_DISCARD,Rect<2>,2> acc(regions[0], FID_RECT);
  PointInRectIterator<2> pir(dom);
  std::size_t c = task->index_point.point_data[0];

  acc[*pir]=Rect<2>(Point<2>(c,0),Point<2>(c,c+5));
  pir++;
  size_t max_size = size_t(-1) / (2 * sizeof(int));
  acc[*pir]=Rect<2>(Point<2>(c,c+5),Point<2>(c, max_size));
}

void
check_task(const Task * task,
  const std::vector<PhysicalRegion> & regions,
  Context ctx,
  HighLevelRuntime * runtime) {
  assert(regions.size() == 1);

  Legion::Domain dom = runtime->get_index_space_domain(
    ctx, task->regions[0].region.get_index_space());

  std::cout <<"IRINA DEBUG rect = "<<dom<<std::endl;

} //resize_task

#if 0

void
fill_task(const Task * task,
  const std::vector<PhysicalRegion> & regions,
  Context ctx,
  HighLevelRuntime * runtime) {
  assert(regions.size() == 2);

  std::vector<PhysicalRegion> comb_regions;
  comb_regions.push_back(regions[0]);
  comb_regions.push_back(regions[1]);

  Legion::Domain pr_dom = runtime->get_index_space_domain(
    ctx, task->regions[0].region.get_index_space());
  Legion::Domain gh_dom = runtime->get_index_space_domain(
    ctx, task->regions[1].region.get_index_space());

  const Legion::MultiRegionAccessor<size_t, 2, Legion::coord_t,
    Realm::AffineAccessor<size_t, 2, Legion::coord_t>>
    mrac(comb_regions, FID_VAL, sizeof(size_t));

  size_t i = 0;
  for(PointInDomainIterator<2, coord_t> itr(pr_dom); itr(); itr++, i++) {
    // writing to primary:
    mrac[*itr] = i;
    i++;
  }

  for(PointInDomainIterator<2, coord_t> itr(gh_dom); itr(); itr++, i++) {
    //reading ghost
    size_t tmp2 = mrac[*itr];
    // writing to ghost
    mrac[*itr] = i;
    i++;
  }
 

} // fill_task

#endif
int
main(int argc, char ** argv) {
#if defined(GASNET_CONDUIT_MPI) || defined(REALM_USE_MPI)
  // The GASNet MPI conduit and/or the Realm MPI network layer
  // require that MPI be initialized for multiple threads
  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
  // If you fail this assertion, then your version of MPI
  // does not support calls from multiple threads and you
  // cannot use the GASNet MPI conduit
  if(provided < MPI_THREAD_MULTIPLE)
    printf("ERROR: Your implementation of MPI does not support "
           "MPI_THREAD_MULTIPLE which is required for use of the "
           "GASNet MPI conduit or the Realm MPI network layer "
           "with the Legion-MPI Interop!\n");
  assert(provided == MPI_THREAD_MULTIPLE);
#else
  // Perform MPI start-up like normal for most GASNet conduits
  MPI_Init(&argc, &argv);
#endif

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_cpus);

  Runtime::configure_MPI_interoperability(rank);

  // register tasks
  Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);

  {
    Legion::TaskVariantRegistrar registrar(TOP_LEVEL_TASK_ID, "top_level_task");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_inner();
    registrar.set_replicable();
    Runtime::preregister_task_variant<top_level_task>(
      registrar, "top_level_task");
  }

  {
    Legion::TaskVariantRegistrar registrar(CHECK_TASK_ID, "initial_check_task");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Legion::Runtime::preregister_task_variant<check_task>(
      registrar, "check_task");
  } // scope 

#if 0
  {
    Legion::TaskVariantRegistrar registrar(FILL_TASK_ID, "fill_task");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Legion::Runtime::preregister_task_variant<fill_task>(
      registrar, "fill_task");
  } // scope
#endif
  {
    Legion::TaskVariantRegistrar registrar(FILL_PART_TASK_ID, "fill_part");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Legion::Runtime::preregister_task_variant<fill_part_task>(
      registrar, "fill_part");
  } // scope


  handshake = Runtime::create_handshake(true /*MPI initial control*/,
    1 /*MPI participants*/, 1 /*Legion participants*/);

   Runtime::add_registration_callback(mapper_registration);
  Runtime::start(argc, argv, true);

  MPI_Barrier(MPI_COMM_WORLD);
  handshake.mpi_handoff_to_legion();
  handshake.mpi_wait_on_legion();
  MPI_Barrier(MPI_COMM_WORLD);

  Runtime::wait_for_shutdown();

  printf("SUCCESS!\n");

#ifndef GASNET_CONDUIT_MPI
  // Then finalize MPI like normal
  // Exception for the MPI conduit which does its own finalization
  MPI_Finalize();
#endif

  return 0;
}
