/*
    @@@@@@@@  @@           @@@@@@   @@@@@@@@ @@
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
#pragma once

/*! @file */

#include "default_mapper.h"
#include "legion.h"
/*!
 Mapper ID

 @ingroup legion-execution
 */

size_t part_color=0;

enum {
  MPI_MAPPER_ID = 1,
};

enum {
  // Use the first 8 bits for storing the rhsf index
  MAPPER_FORCE_RANK_MATCH = 0x00001000,
  MAPPER_COMPACTED_STORAGE = 0x00002000,
  MAPPER_SUBRANK_LAUNCH = 0x00003000,
  PREFER_GPU = 0x11000001,
  PREFER_OMP = 0x11000002,
  ATTACH = 0x00004000,
  EXCLUSIVE_LR = 0x00005000,
  FILL_INDIRECT_FIELD_TASK = 0x00006000,
};


/*
 The mpi_mapper_t - is a custom mapper that handles mpi-legion
 interoperability in FLeCSI

 @ingroup legion-execution
*/

class mpi_mapper_t : public Legion::Mapping::DefaultMapper
{
public:
  /*!
   Contructor. Derives from the Legion's Default Mapper

   @param machine Machine type for Legion's Realm
   @param _runtime Legion runtime
   @param local processor type: currently supports only
           LOC_PROC and TOC_PROC
   */

  mpi_mapper_t(Legion::Machine machine,
    Legion::Runtime * _runtime,
    Legion::Processor local)
    : Legion::Mapping::DefaultMapper(_runtime->get_mapper_runtime(),
        machine,
        local,
        "default"),
      machine(machine) {
    using namespace Legion;
    using namespace Legion::Mapping;
    using legion_machine = Legion::Machine;
    using legion_proc = Legion::Processor;

    legion_machine::ProcessorQuery pq =
      legion_machine::ProcessorQuery(machine).same_address_space_as(local);
    for(legion_machine::ProcessorQuery::iterator pqi = pq.begin();
        pqi != pq.end(); ++pqi) {
      legion_proc p = *pqi;
      if(p.kind() == legion_proc::LOC_PROC)
        local_cpus.push_back(p);
      else if(p.kind() == legion_proc::TOC_PROC)
        local_gpus.push_back(p);
      else
        continue;

      std::map<Realm::Memory::Kind, Realm::Memory> & mem_map = proc_mem_map[p];

      legion_machine::MemoryQuery mq =
        legion_machine::MemoryQuery(machine).has_affinity_to(p);
      for(legion_machine::MemoryQuery::iterator mqi = mq.begin();
          mqi != mq.end(); ++mqi) {
        Realm::Memory m = *mqi;
        mem_map[m.kind()] = m;

        if(m.kind() == Realm::Memory::SYSTEM_MEM)
          local_sysmem = m;
      } // end for
    } // end for

    // Get our local memories
    {
      Machine::MemoryQuery sysmem_query(machine);
      sysmem_query.local_address_space();
      sysmem_query.only_kind(Memory::SYSTEM_MEM);
      local_sysmem = sysmem_query.first();
      assert(local_sysmem.exists());
    }
    if(!local_gpus.empty()) {
      Machine::MemoryQuery zc_query(machine);
      zc_query.local_address_space();
      zc_query.only_kind(Memory::Z_COPY_MEM);
      local_zerocopy = zc_query.first();
      assert(local_zerocopy.exists());
    }
    else {
      local_zerocopy = Memory::NO_MEMORY;
    }
    if(local_kind == Processor::TOC_PROC) {
      Machine::MemoryQuery fb_query(machine);
      fb_query.local_address_space();
      fb_query.only_kind(Memory::GPU_FB_MEM);
      fb_query.best_affinity_to(local_proc);
      local_framebuffer = fb_query.first();
      assert(local_framebuffer.exists());
    }
    else {
      local_framebuffer = Memory::NO_MEMORY;
    }

  } // end mpi_mapper_t

  /*!
    Destructor
   */
  virtual ~mpi_mapper_t(){};

  Legion::LayoutConstraintID default_policy_select_layout_constraints(
    Legion::Mapping::MapperContext ctx,
    Realm::Memory target_memory,
    const Legion::RegionRequirement & req,
    Legion::Mapping::DefaultMapper::MappingKind mapping_kind,
    bool needs_field_constraint_check,
    bool & force_new_instances) {
    // We always set force_new_instances to false since we are
    // deciding to optimize for minimizing memory usage instead
    // of avoiding Write-After-Read (WAR) dependences
    force_new_instances = false;
    std::vector<Legion::DimensionKind> ordering;
    ordering.push_back(Legion::DimensionKind::DIM_Y);
    ordering.push_back(Legion::DimensionKind::DIM_X);
    ordering.push_back(Legion::DimensionKind::DIM_F); // SOA
    Legion::OrderingConstraint ordering_constraint(
      ordering, true /*contiguous*/);
    Legion::LayoutConstraintSet layout_constraint;
    layout_constraint.add_constraint(ordering_constraint);

    // Do the registration
    Legion::LayoutConstraintID result =
      runtime->register_layout(ctx, layout_constraint);
    return result;
  }

  /*!
   Specialization of the default_policy_select_instance_region methid for FleCSI

   @param ctx Mapper Context
   @param target_memory target memory for the instance to be allocated
   @param req Reqion requirement for witch instance is going to be allocated
   @layout_constraints Layout constraints
  */
  virtual Legion::LogicalRegion default_policy_select_instance_region(
    Legion::Mapping::MapperContext ctx,
    Realm::Memory target_memory,
    const Legion::RegionRequirement & req,
    const Legion::LayoutConstraintSet & layout_constraints,
    bool force_new_instances,
    bool meets_constraints) {
    // If it is not something we are making a big region for just
    // return the region that is actually needed
    Legion::LogicalRegion result = req.region;
    if(!meets_constraints || (req.privilege == REDUCE))
      return result;

    return result;
  } // default_policy_select_instance_region

  /*!
   THis function will find a CPU variat for the task
  */
  Legion::VariantID find_cpu_variant(const Legion::Mapping::MapperContext ctx,
    Legion::TaskID task_id) {
    std::map<Legion::TaskID, Legion::VariantID>::const_iterator finder =
      cpu_variants.find(task_id);
    if(finder != cpu_variants.end())
      return finder->second;
    std::vector<Legion::VariantID> variants;
    runtime->find_valid_variants(
      ctx, task_id, variants, Legion::Processor::LOC_PROC);
    cpu_variants[task_id] = variants[0];
    return variants[0];
  }

  /*!
   THis function will find a OpenMP variat for the task
  */
  Legion::VariantID find_omp_variant(const Legion::Mapping::MapperContext ctx,
    Legion::TaskID task_id) {
    using namespace Legion;
    std::map<TaskID, VariantID>::const_iterator finder =
      omp_variants.find(task_id);
    if(finder != omp_variants.end())
      return finder->second;
    std::vector<VariantID> variants;
    runtime->find_valid_variants(ctx, task_id, variants, Processor::OMP_PROC);
    omp_variants[task_id] = variants[0];
    return variants[0];
  }

  /*!
   THis function will find a GPU variat for the task
  */
  Legion::VariantID find_gpu_variant(const Legion::Mapping::MapperContext ctx,
    Legion::TaskID task_id) {
    using namespace Legion;
    std::map<TaskID, VariantID>::const_iterator finder =
      gpu_variants.find(task_id);
    if(finder != gpu_variants.end())
      return finder->second;
    std::vector<VariantID> variants;
    runtime->find_valid_variants(ctx, task_id, variants, Processor::TOC_PROC);
    gpu_variants[task_id] = variants[0];
    return variants[0];
  }

  /*!
   THis function will create PhysicalInstance for Reduction task
  */
  void creade_reduction_instance(const Legion::Mapping::MapperContext ctx,
    const Legion::Task & task,
    Legion::Mapping::Mapper::MapTaskOutput & output,
    const Legion::Memory & target_mem,
    const size_t & indx) {
    // using dummy constraints for REDUCTION
    std::set<Legion::FieldID> dummy_fields;
    Legion::TaskLayoutConstraintSet dummy_constraints;

    size_t instance_size = 0;
    assert(default_create_custom_instances(ctx, task.target_proc, target_mem,
      task.regions[indx], indx, dummy_fields, dummy_constraints,
      false /*need check*/, output.chosen_instances[indx], &instance_size));

    std::cout << "task " << task.get_task_name()
              << " allocates physical instance with size " << instance_size
              << " for the region requirement #" << indx << std::endl;

    if(instance_size > 1000000000) {
      std::cout << "task " << task.get_task_name()
                << " is trying to allocate physical instance with \
           the size > than 1 Gb("
                << instance_size << " )"
                << " for the region requirement # " << indx << std::endl;
    } // if
  } // create reduction instance

  /*!
   THis function will create PhysicalInstance Unstructured mesh data havdle (
    compacted Exclusive, SHared and Ghost)
  */
  void create_compacted_instance(const Legion::Mapping::MapperContext ctx,
    const Legion::Task & task,
    Legion::Mapping::Mapper::MapTaskOutput & output,
    const Legion::Memory & target_mem,
    const Legion::LayoutConstraintSet & layout_constraints,
    const size_t & indx) {
    using namespace Legion;
    using namespace Legion::Mapping;

    // check if instance was already created and stored in the
    // local_instamces_ map
    const std::pair<Legion::LogicalRegion, Legion::Memory> key1(
      task.regions[indx].region, target_mem);
    auto & key2 = task.regions[indx].privilege_fields;
    instance_map_t::const_iterator finder1 = local_instances_.find(key1);
    if(finder1 != local_instances_.end()) {
      const field_instance_map_t & innerMap = finder1->second;
      field_instance_map_t::const_iterator finder2 = innerMap.find(key2);
      if(finder2 != innerMap.end()) {
        for(size_t j = 0; j < 3; j++) {
          output.chosen_instances[indx + j].clear();
          output.chosen_instances[indx + j].push_back(finder2->second);
        } // for
        return;
      } // if
    } // if

    Legion::Mapping::PhysicalInstance result;
    std::vector<Legion::LogicalRegion> regions;
    bool created;

    // creating physical instance for the compacted storaged

    assert((task.regions.size() >= (indx + 2)));

    assert((task.regions[indx].region.exists()));

    // compacting region requirements for exclusive, shared and ghost into one
    // instance
    regions.push_back(task.regions[indx].region);
    regions.push_back(task.regions[indx + 1].region);
    regions.push_back(task.regions[indx + 2].region);

    size_t instance_size = 0;
    assert(runtime->find_or_create_physical_instance(ctx, target_mem,
      layout_constraints, regions, result, created, true /*acquire*/,
      GC_NEVER_PRIORITY, true, &instance_size));

    std::cout << "task " << task.get_task_name()
              << " allocates physical instance with size " << instance_size
              << " for the region requirement #" << indx << std::endl;

    if(instance_size > 1000000000) {
      std::cout << "task " << task.get_task_name()
                << " is trying to allocate physical compacted instance with \
                the size > than 1 Gb("
                << instance_size << " )"
                << " for the region requirement # " << indx << std::endl;
    }

    for(size_t j = 0; j < 3; j++) {
      output.chosen_instances[indx + j].clear();
      output.chosen_instances[indx + j].push_back(result);
    } // for
    local_instances_[key1][key2] = result;
  } // create_compacted_instance

  /*!
   THis function will create PhysicalInstance for a task
  */
  void create_instance(const Legion::Mapping::MapperContext ctx,
    const Legion::Task & task,
    Legion::Mapping::Mapper::MapTaskOutput & output,
    const Legion::Memory & target_mem,
    const Legion::LayoutConstraintSet & layout_constraints,
    const size_t & indx) {
    using namespace Legion;
    using namespace Legion::Mapping;

    // check if instance was already created and stored in the
    // local_instamces_ map
    const std::pair<Legion::LogicalRegion, Legion::Memory> key1(
      task.regions[indx].region, target_mem);
    auto key2 = task.regions[indx].privilege_fields;
    instance_map_t::const_iterator finder1 = local_instances_.find(key1);
    if(finder1 != local_instances_.end()) {
      const field_instance_map_t & innerMap = finder1->second;
      field_instance_map_t::const_iterator finder2 = innerMap.find(key2);
      if(finder2 != innerMap.end()) {
        output.chosen_instances[indx].clear();
        output.chosen_instances[indx].push_back(finder2->second);
        return;
      } // if
    } // if

    Legion::Mapping::PhysicalInstance result;
    std::vector<Legion::LogicalRegion> regions;
    bool created;

    regions.push_back(task.regions[indx].region);

    size_t instance_size = 0;
    assert(runtime->find_or_create_physical_instance(ctx, target_mem,
      layout_constraints, regions, result, created, true, 0, false,
      &instance_size));
    // FIXME                  true /*acquire*/, GC_NEVER_PRIORITY, true,
    // &instance_size));

    std::cout << "task " << task.get_task_name()
              << " allocates physical instance with size " << instance_size
              << " for the region requirement #" << indx << std::endl;

    if(!(task.regions[indx].tag == ATTACH) && (instance_size > 1000000000)) {
      std::cout
        << "task " << task.get_task_name()
        << " is trying to allocate physical instance with the size > than 1 Gb("
        << instance_size << " )"
        << " for the region requirement # " << indx << std::endl;
    } // if

    output.chosen_instances[indx].push_back(result);
    local_instances_[key1][key2] = result;
  } // create_instance

  /*!
   Specialization of the map_task funtion for FLeCSI
   By default, map_task will execute Legions map_task from DefaultMapper.
   In the case the launcher has been tagged with the
     "MAPPER_COMPACTED_STORAGE" tag, mapper will create single physical
   instance for exclusive, shared and ghost partitions for each data handle

    @param ctx Mapper Context
    @param task Legion's task
    @param input Input information about task mapping
    @param output Output information about task mapping
   */

  virtual void map_task(const Legion::Mapping::MapperContext ctx,
    const Legion::Task & task,
    const Legion::Mapping::Mapper::MapTaskInput & input,
    Legion::Mapping::Mapper::MapTaskOutput & output) {

    using namespace Legion;
    using namespace Legion::Mapping;

    if (task.tag & FILL_INDIRECT_FIELD_TASK)
    {
std::cout <<"IRINA DEBUG inside if for task "<< task.get_task_name()<<std::endl;
      indirect_lps.push_back( task.regions[0].partition);
    };

    Processor::Kind target_kind = task.target_proc.kind();
    // Get the variant that we are going to use to map this task
    VariantInfo chosen = default_find_preferred_variant(
      task, ctx, true /*needs tight bound*/, true /*cache*/, target_kind);
    output.chosen_variant = chosen.variant;
    output.task_priority = default_policy_select_task_priority(ctx, task);
    output.postmap_task = false;
    default_policy_select_target_processors(ctx, task, output.target_procs);

    output.chosen_instances.resize(task.regions.size());

    if(task.regions.size() > 0) {

      for(size_t indx = 0; indx < task.regions.size(); indx++) {

        Legion::LayoutConstraintSet layout_constraints;

        Memory target_mem;

        if((task.tag & PREFER_GPU) && !local_gpus.empty())
          target_mem = local_framebuffer;
        else
          target_mem = local_sysmem;

        if(task.regions[indx].tag == ATTACH) {
          layout_constraints.add_constraint(Legion::OrderingConstraint());
          if((task.tag & PREFER_GPU) && !local_gpus.empty()) {
            layout_constraints.add_constraint(
              Legion::MemoryConstraint(Memory::GPU_FB_MEM));
          }
          else {
            layout_constraints.add_constraint(
              Legion::MemoryConstraint(Memory::SYSTEM_MEM));
          }
        }
        else {
          // creating ordering constraint (SOA )
          std::vector<Legion::DimensionKind> ordering;
          ordering.push_back(Legion::DimensionKind::DIM_Y);
          ordering.push_back(Legion::DimensionKind::DIM_X);
          ordering.push_back(Legion::DimensionKind::DIM_F); // SOA
          Legion::OrderingConstraint ordering_constraint(
            ordering, true /*contiguous*/);
          layout_constraints.add_constraint(ordering_constraint);
          layout_constraints.add_constraint(
            Legion::MemoryConstraint(target_mem.kind()));
        }
        // No specialization
        layout_constraints.add_constraint(Legion::SpecializedConstraint());
        // Have all the field for the instance available
        std::vector<Legion::FieldID> all_fields;
        for(auto fid : task.regions[indx].privilege_fields) {
          all_fields.push_back(fid);
        } // for
        layout_constraints.add_constraint(
          Legion::FieldConstraint(all_fields, true));

        // creating physical instance for the reduction task
        if(task.regions[indx].privilege == REDUCE) {
          creade_reduction_instance(ctx, task, output, target_mem, indx);
        }
        else if(task.regions[indx].tag == EXCLUSIVE_LR) {

          create_compacted_instance(
            ctx, task, output, target_mem, layout_constraints, indx);
          indx = indx + 2;
        }
        else {
          create_instance(
            ctx, task, output, target_mem, layout_constraints, indx);
        } // end if
        //        } // end if
      } // end for

    } // end if

    runtime->acquire_instances(ctx, output.chosen_instances);

  } // map_task

  virtual void slice_task(const Legion::Mapping::MapperContext ctx,
    const Legion::Task & task,
    const Legion::Mapping::Mapper::SliceTaskInput & input,
    Legion::Mapping::Mapper::SliceTaskOutput & output) {

    using legion_proc = Legion::Processor;

    using namespace Legion;

    if(task.tag == MAPPER_SUBRANK_LAUNCH) {
      // expect a 1-D index domain
      assert(input.domain.get_dim() == 1);
      // send the whole domain to our local processor
      output.slices.resize(1);
      output.slices[0].domain = input.domain;
      output.slices[0].proc = task.target_proc;
      return;
    } // end if MAPPER_SUBRANK_LAUNCH

    if((task.tag == MAPPER_FORCE_RANK_MATCH) ||
       (task.tag == MAPPER_COMPACTED_STORAGE)) {
      // expect a 1-D index domain - each point goes to the corresponding node
      assert(input.domain.get_dim() == 1);
      LegionRuntime::Arrays::Rect<1> r = input.domain.get_rect<1>();

      // go through all the CPU processors and find a representative for each
      //  node (i.e. address space)
      std::map<int, Legion::Processor> targets;

      Legion::Machine::ProcessorQuery pq =
        Legion::Machine::ProcessorQuery(machine).only_kind(
          Legion::Processor::LOC_PROC);
      for(Legion::Machine::ProcessorQuery::iterator it = pq.begin();
          it != pq.end(); ++it) {
        Legion::Processor p = *it;
        int a = p.address_space();
        if(targets.count(a) == 0)
          targets[a] = p;
      }

      output.slices.resize(1);
      for(int a = r.lo[0]; a <= r.hi[0]; a++) {
        assert(targets.count(a) > 0);
        output.slices[0].domain = // Legion::Domain::from_rect<1>(
          Legion::Rect<1>(a, a);
        output.slices[0].proc = targets[a];
      }
      return;
    } // MAPPER_FORCE_RANK_MATCH

    // We've already been control replicated, so just divide our points
    // over the local processors, depending on which kind we prefer
    if((task.tag == PREFER_GPU) && !local_gpus.empty()) {
      unsigned local_gpu_index = 0;
      for(Domain::DomainPointIterator itr(input.domain); itr; itr++) {
        TaskSlice slice;
        slice.domain = Domain(itr.p, itr.p);
        slice.proc = local_gpus[local_gpu_index++];
        if(local_gpu_index == local_gpus.size())
          local_gpu_index = 0;
        slice.recurse = false;
        slice.stealable = false;
        output.slices.push_back(slice);
      }
    }
    else if((task.tag == PREFER_OMP) && !local_omps.empty()) {
      unsigned local_omp_index = 0;
      for(Domain::DomainPointIterator itr(input.domain); itr; itr++) {
        TaskSlice slice;
        slice.domain = Domain(itr.p, itr.p);
        slice.proc = local_omps[local_omp_index++];
        if(local_omp_index == local_omps.size())
          local_omp_index = 0;
        slice.recurse = false;
        slice.stealable = false;
        output.slices.push_back(slice);
      }
    }
    else {
      // Opt for our cpus instead of our openmap processors
      unsigned local_cpu_index = 0;
      for(Domain::DomainPointIterator itr(input.domain); itr; itr++) {
        TaskSlice slice;
        slice.domain = Domain(itr.p, itr.p);
        slice.proc = local_cpus[local_cpu_index++];
        if(local_cpu_index == local_cpus.size())
          local_cpu_index = 0;
        slice.recurse = false;
        slice.stealable = false;
        output.slices.push_back(slice);
      }
    }

  } // slice_task

struct SelectPartitionProjectionInput {
 std::vector<Legion::LogicalPartition> open_complete_partitions;
};
struct SelectPartitionProjectionOutput {
  Legion::LogicalPartition chosen_partition;
};


void select_partition_projection(const Legion::Mapping::MapperContext  ctx,
                                 const Legion::Partition& partition,
                                 const SelectPartitionProjectionInput& input,
                                 SelectPartitionProjectionOutput& output)
{
   std::cout <<"IRINA DEBUG indirect_lps.size = " <<indirect_lps.size()<<std::endl;

  //if there is complete partition, choose it
  if (!input.open_complete_partitions.empty()) {
    output.chosen_partition = input.open_complete_partitions.front();
    return;
  }

  // Otherwise see if we can find another partition
#if 0
  if (!indirect_lps.empty()){
    output.chosen_partition = indirect_lps[0];
    return;
  }
#endif

  if(runtime->has_index_partition(ctx,
        partition.requirement.region.get_index_space(),  part_color /*color*/))
   {
      Legion::IndexPartition ip = runtime->get_index_partition(ctx,
       partition.requirement.region.get_index_space(),part_color /*color*/);
      
      Legion::LogicalPartition lp_temp = runtime->get_parent_logical_partition(ctx,
        partition.requirement.region);

      Legion::LogicalRegion lr = runtime->get_parent_logical_region(ctx, lp_temp);
 
     output.chosen_partition = runtime->get_logical_partition(ctx,
        lr, ip);
     return;
   } 


  output.chosen_partition = Legion::LogicalPartition::NO_PART;
  return;
}


private:
  std::map<Legion::Processor, std::map<Realm::Memory::Kind, Realm::Memory>>
    proc_mem_map;
  Realm::Machine machine;

  // the map of the locac intances that have been already created
  // the first key is the pair of Logical region and Memory that is
  // used as an identifier for the instance, second key is fid
  typedef std::map<std::set<Legion::FieldID>, Legion::Mapping::PhysicalInstance>
    field_instance_map_t;

  typedef std::map<std::pair<Legion::LogicalRegion, Legion::Memory>,
    field_instance_map_t>
    instance_map_t;

  instance_map_t local_instances_;

protected:
  std::map<Legion::TaskID, Legion::VariantID> cpu_variants;
  std::map<Legion::TaskID, Legion::VariantID> gpu_variants;
  std::map<Legion::TaskID, Legion::VariantID> omp_variants;

  std::vector<Legion::LogicalPartition> indirect_lps;

  Legion::Memory local_sysmem, local_zerocopy, local_framebuffer;
};

/*!
 mapper_registration is used to replace DefaultMapper with mpi_mapper_t in
 FLeCSI

 @ingroup legion-execution
 */

inline void
mapper_registration(Legion::Machine machine,
  Legion::HighLevelRuntime * rt,
  const std::set<Legion::Processor> & local_procs) {
  for(std::set<Legion::Processor>::const_iterator it = local_procs.begin();
      it != local_procs.end(); it++) {
    mpi_mapper_t * mapper = new mpi_mapper_t(machine, rt, *it);
    rt->replace_default_mapper(mapper, *it);
  }
} // mapper registration
