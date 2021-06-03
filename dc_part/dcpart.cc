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
  FILL_META_TASK_ID,
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
size_t max_size =0;

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

  std::cout <<"NUM COLORS = "<<num_colors<<std::endl;

  Rect<1> color_bounds(0, num_colors - 1);
  IndexSpaceT<1> color_is = runtime->create_index_space(ctx, color_bounds);
  max_size = size_t(-1) / (num_colors * sizeof(int));

  ///////////////////////////////////////////////////////////////////////////
  // Creating Logical regions and partitions of these regions for src and dst
  ///////////////////////////////////////////////////////////////////////////

  Rect<2> rect_blis(
    Legion::Point<2>(0, 0), Legion::Point<2>(num_colors - 1, max_size - 1));

  IndexSpace is_blis = runtime->create_index_space(ctx, rect_blis);
  runtime->attach_name(is_blis, "blis_is");

  FieldSpace fs_blis = runtime->create_field_space(ctx);
  {
    FieldAllocator allocator = runtime->create_field_allocator(ctx, fs_blis);
    allocator.allocate_field(sizeof(size_t), FID_VAL);
  }

  LogicalRegion lr_blis = runtime->create_logical_region(ctx, is_blis, fs_blis);
  runtime->attach_name(lr_blis, "lr_blis");

 //-------------------------------------------------
 //create used/unused disjoint complete partitioning
 //-------------------------------------------------
 
 //1 create colors x 2 index space that will store infomration about rects
 
  Rect<2> rect_part(
    Legion::Point<2>(0, 0), Legion::Point<2>(num_colors - 1, 1));
  IndexSpaceT<2> is_part = runtime->create_index_space(ctx, rect_part); 
  runtime->attach_name(is_part, "is_part");
  FieldSpace fs_part = runtime->create_field_space(ctx);
  {
    FieldAllocator allocator = runtime->create_field_allocator(ctx, fs_part);
    allocator.allocate_field(sizeof(Legion::Rect<2>), FID_RECT);
  }
  LogicalRegion lr_part = runtime->create_logical_region(ctx, is_part, fs_part);  
  runtime->attach_name(lr_part, "lr_part");
  // partition lr_part by rows;

  Legion::Transform<2, 1> idx_tsfm;
  idx_tsfm[0][0] = 1;
  idx_tsfm[1][0] = 0;
  IndexPartition ip_part = runtime->create_partition_by_restriction
    (ctx,
    is_part,
    color_is,
    idx_tsfm,
    {{0,0}, {0,1}}, DISJOINT_COMPLETE_KIND);
  runtime->attach_name(ip_part, "ip_part");
  LogicalPartition lp_part = runtime->get_logical_partition(lr_part, ip_part);
  runtime->attach_name(lp_part, "lp_part");

  // fill logical region that stores information about partitioning
  printf("Filling partition information...\n");
  ArgumentMap idx_arg_map;
  IndexLauncher part_init_launcher(FILL_PART_TASK_ID, color_bounds,
                              TaskArgument(NULL, 0), idx_arg_map);
  part_init_launcher.add_region_requirement(
      RegionRequirement(lp_part, 0, WRITE_DISCARD, EXCLUSIVE, lr_part));
  part_init_launcher.region_requirements[0].add_field(FID_RECT);
  runtime->execute_index_space(ctx, part_init_launcher);

 //----------------------------------------------------------------------
 // create and fill 2xcolors meta region
 // ---------------------------------------------------------------------
  Rect<2> rect_meta(
    Legion::Point<2>(0, 0), Legion::Point<2>(1, num_colors-1));
  IndexSpaceT<2> is_meta = runtime->create_index_space(ctx, rect_meta);
  runtime->attach_name(is_meta, "is_meta");
  FieldSpace fs_meta = runtime->create_field_space(ctx);
  {
    FieldAllocator allocator = runtime->create_field_allocator(ctx, fs_meta);
    allocator.allocate_field(sizeof(Legion::Rect<2>), FID_RECT);
  }
  LogicalRegion lr_meta = runtime->create_logical_region(ctx, is_meta, fs_meta);
  runtime->attach_name(lr_meta, "lr_meta");

  Rect<1> color_meta_bounds(0, 1);
  IndexSpaceT<1> color_meta_is = runtime->create_index_space(ctx,
		color_meta_bounds);


  IndexPartition ip_meta = runtime->create_partition_by_restriction
    (ctx,
    is_meta,
    color_meta_is,
    idx_tsfm,
    {{0,0}, {0,num_colors-1}}, DISJOINT_COMPLETE_KIND);
  runtime->attach_name(ip_meta, "ip_meta");
  LogicalPartition lp_meta = runtime->get_logical_partition(lr_meta, ip_meta);
  runtime->attach_name(lp_meta, "lp_part");


  // fill logical meta region that stores information about partitioning
  printf("Filling meta information ...\n");
  IndexLauncher part_meta_launcher(FILL_META_TASK_ID, color_meta_bounds,
                              TaskArgument(NULL, 0), idx_arg_map);
  part_meta_launcher.add_region_requirement(
      RegionRequirement(lp_meta, 0, WRITE_DISCARD, EXCLUSIVE, lr_meta));
  part_meta_launcher.region_requirements[0].add_field(FID_RECT);
  runtime->execute_index_space(ctx, part_meta_launcher);

  //partition lp_part by used and unused
  IndexPartition ip_part_dc = runtime->create_partition_by_image_range(
      ctx,
      is_part,
      lp_meta,
      lr_meta,//runtime->get_parent_logical_region(idx_lp),
      FID_RECT,
      color_meta_is,
      DISJOINT_COMPLETE_KIND);
  runtime->attach_name(ip_part_dc, "ip_part_dc");
  LogicalPartition lp_part_dc = runtime->get_logical_partition(lr_part, ip_part_dc);
  runtime->attach_name(lp_part_dc, "lp_part_dc");



  // Create partition of the extended index space based on the regions
  // This is used/unused DISJOINT COMPLETE PARTITION
  IndexPartition ip_dc = runtime->create_partition_by_image_range(
      ctx,
      is_blis,
      lp_part_dc,
      lr_part,//runtime->get_parent_logical_region(idx_lp),
      FID_RECT,
      color_meta_is,
      DISJOINT_COMPLETE_KIND);
      
  runtime->attach_name(ip_dc, "ip_dc");
  LogicalPartition lp_dc = runtime->get_logical_partition(lr_blis, ip_dc);
  runtime->attach_name(lp_dc, "lp_dc");


  //---------------------------------------
  // get "used" subregion for blis and part
  // ---------------------------------------
  LogicalRegion lr_blis_used = runtime->get_logical_subregion_by_color(ctx, lp_dc, 0);
  IndexSpace is_blis_used=lr_blis_used.get_index_space();

  // ---------------------------------------
  // partition lr_blis_used and lr_part_blis by colors
  // ---------------------------------------

  IndexPartition ip_blis_used_dc = runtime->create_partition_by_restriction
    (ctx,
    is_blis_used,
    color_is,
    idx_tsfm,
    {{0,0}, {0,max_size}}, DISJOINT_COMPLETE_KIND);
  runtime->attach_name(ip_blis_used_dc, "ip_blis_used_dc");
  LogicalPartition lp_blis_used_dc = runtime->get_logical_partition(lr_blis_used, ip_blis_used_dc);
  runtime->attach_name(lp_blis_used_dc, "lp_blis_used_dc");
 

#if 0
  LogicalRegion lr_part_used = runtime->get_logical_subregion_by_color(ctx, lp_part_dc, 0);
  IndexSpace is_part_used=lr_part_used.get_index_space();

  // ---------------------------------------
  // partition lr_blis_used and lr_part_blis by colors
  // ---------------------------------------

  IndexPartition ip_part_used = runtime->create_partition_by_restriction
    (ctx,
    is_part_used,
    color_is,
    idx_tsfm,
    {{0,0}, {0,0}}, DISJOINT_COMPLETE_KIND);
  runtime->attach_name(ip_part_used, "ip_part_used");
  LogicalPartition lp_part_used = runtime->get_logical_partition(lr_part_used, ip_part_used);
  runtime->attach_name(lp_part_used, "lp_part_used");

  IndexPartition ip_blis_used_dc = runtime->create_partition_by_image_range(
      ctx,
      is_blis_used,
      lp_part_used,
      lr_part_used,//runtime->get_parent_logical_region(idx_lp),
      FID_RECT,
      color_is,
      DISJOINT_COMPLETE_KIND);

  runtime->attach_name(ip_blis_used, "ip_blis_used");
  LogicalPartition lp_blis_used = runtime->get_logical_partition(lr_blis_used, ip__blis_used_dc);
  runtime->attach_name(lp_blis_used, "lp_blis_used");
#endif

   

  //----------------
  //check the partition
  //-----------------
#if 1

  IndexLauncher check_init_launcher(CHECK_TASK_ID, color_bounds,
                              TaskArgument(NULL, 0), idx_arg_map);
  check_init_launcher.add_region_requirement(
      RegionRequirement(lp_blis_used_dc, 0, NO_ACCESS, EXCLUSIVE, lr_blis_used));
  check_init_launcher.region_requirements[0].add_field(FID_VAL);
  runtime->execute_index_space(ctx, check_init_launcher);
#endif

  ///////////////////////////////////////////////////////////////////////
  // clean up our region, index space, and field space
  ///////////////////////////////////////////////////////////////////////

  runtime->destroy_logical_region(ctx, lr_blis);
  runtime->destroy_field_space(ctx, fs_blis);
  runtime->destroy_logical_region(ctx, lr_part);
  runtime->destroy_field_space(ctx, fs_part);
  runtime->destroy_logical_region(ctx, lr_meta);
  runtime->destroy_field_space(ctx, fs_meta);

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
  acc[*pir]=Rect<2>(Point<2>(c,c+5+1),Point<2>(c, max_size));
}

void
fill_meta_task(const Task * task,
  const std::vector<PhysicalRegion> & regions,
  Context ctx,
  HighLevelRuntime * runtime) {
  assert(regions.size() == 1);
  Legion::Domain dom = runtime->get_index_space_domain(
    ctx, task->regions[0].region.get_index_space());

  const FieldAccessor<WRITE_DISCARD,Rect<2>,2> acc(regions[0], FID_RECT);
  std::size_t c = task->index_point.point_data[0];

  size_t i = 0;
  for(PointInDomainIterator<2, coord_t> itr(dom); itr(); itr++, i++) {
     acc[*itr]=Rect<2>(Point<2>(i,c),Point<2>(i,c));
std::cout <<"IRNA DEBUG META RECT = "<<acc.read(*itr)<<std::endl;
  } 
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

  {
    Legion::TaskVariantRegistrar registrar(FILL_META_TASK_ID, "fill_meta_task");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Legion::Runtime::preregister_task_variant<fill_meta_task>(
      registrar, "fill_meta_task");
  } // scope
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
