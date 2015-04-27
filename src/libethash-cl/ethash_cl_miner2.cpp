/*
  This file is part of c-ethash.

  c-ethash is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  c-ethash is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with cpp-ethereum.  If not, see <http://www.gnu.org/licenses/>.
*/
/** @file ethash_cl_miner.cpp
* @author Tim Hughes <tim@twistedfury.com>
* @date 2015
*/

#define _CRT_SECURE_NO_WARNINGS

#include <assert.h>
#include <queue>
#include "ethash_cl_miner2.h"
#include "ethash_cl_miner2_kernel.h"
#include "../libethash/util.h"

#undef min
#undef max

#define ETHASH_BYTES 32
#define INIT_BYTES 64
#define MIX_BYTES 128


static void add_definition(std::string& source, char const* id, unsigned value)
{
	char buf[256];
	sprintf(buf, "#define %s %uu\n", id, value);
	source.insert(source.begin(), buf, buf + strlen(buf));
}

ethash_cl_miner2::ethash_cl_miner2()
{
}

void ethash_cl_miner2::finish()
{
	if (m_hash_queue())
	{
		m_hash_queue.finish();
	}
	if (m_mem_queue())
	{
		m_mem_queue.finish();
	}
}

bool ethash_cl_miner2::init(ethash_params const& params, const uint8_t seed[32], unsigned workgroup_size)
{
	// store params
	m_params = params;

	// use requested workgroup size
	m_workgroup_size = workgroup_size;

	// get all platforms
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
	if (platforms.empty())
	{
		debugf("No OpenCL platforms found.\n");
		return false;
	}

	// use default platform
	debugf("Using platform: %s\n", platforms[0].getInfo<CL_PLATFORM_NAME>().c_str());

    // get GPU device of the default platform
    std::vector<cl::Device> devices;
    platforms[0].getDevices(CL_DEVICE_TYPE_ALL, &devices);
    if (devices.empty())
	{
		debugf("No OpenCL devices found.\n");
		return false;
	}

	// use default device
	cl::Device& device = devices[0];
	debugf("Using device: %s\n", device.getInfo<CL_DEVICE_NAME>().c_str());

	// create context
	m_context = cl::Context(std::vector<cl::Device>(&device, &device+1));
	m_hash_queue = cl::CommandQueue(m_context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE);
	m_mem_queue = cl::CommandQueue(m_context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE);

	// patch source code
	std::string code(ETHASH_CL_MINER2_KERNEL, ETHASH_CL_MINER2_KERNEL + ETHASH_CL_MINER2_KERNEL_SIZE);
	add_definition(code, "GROUP_SIZE", workgroup_size);
	add_definition(code, "DAG_SIZE", (unsigned)(params.full_size / MIX_BYTES));
	add_definition(code, "MAX_OUTPUTS", c_max_search_results);
	//debugf("%s", code.c_str());

	// create miner OpenCL program
	cl::Program::Sources sources;
	sources.push_back({code.c_str(), code.size()});

	cl::Program program(m_context, sources);
	try
	{
		program.build({device});
	}
	catch (cl::Error err)
	{
		debugf("%s\n", program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device).c_str());
		return false;
	}
	m_init_kernel = cl::Kernel(program, "ethash_init");
	m_inner_kernel = cl::Kernel(program, "ethash_inner");
	m_hash_kernel = cl::Kernel(program, "ethash_hash");
	m_search_kernel = cl::Kernel(program, "ethash_search");

	// create buffer for dag
	m_dag = cl::Buffer(m_context, CL_MEM_READ_ONLY, params.full_size);
	
	// create buffers for header and target
	m_header = cl::Buffer(m_context, CL_MEM_READ_ONLY, 32);

	// compute dag on CPU
	{
		void* cache_mem = malloc(params.cache_size + 63);
		ethash_cache cache;
		cache.mem = (void*)(((uintptr_t)cache_mem + 63) & ~63);
		ethash_mkcache(&cache, &params, seed);

		void* dag_ptr = m_mem_queue.enqueueMapBuffer(m_dag, true, CL_MAP_WRITE_INVALIDATE_REGION, 0, params.full_size);
		ethash_compute_full_data(dag_ptr, &params, &cache);
		m_mem_queue.enqueueUnmapMemObject(m_dag, dag_ptr);
		m_mem_queue.enqueueBarrierWithWaitList(NULL);

		free(cache_mem);
	}

	// create mining buffers
	for (unsigned i = 0; i != c_num_buffers; ++i)
	{
		m_search_buf[i] = cl::Buffer(m_context, CL_MEM_WRITE_ONLY, 4 * (1 + c_max_search_results));
		m_hash_buf[i] = cl::Buffer(m_context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, ETHASH_BYTES*c_hash_batch_size);
		m_mix_buf[i] = cl::Buffer(m_context, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, ETHASH_BYTES*std::max(c_search_batch_size, c_hash_batch_size));
		m_init_buf[i] = cl::Buffer(m_context, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, INIT_BYTES*std::max(c_search_batch_size, c_hash_batch_size));
	}
	return true;
}

void ethash_cl_miner2::hash(uint8_t* ret, uint8_t const* header, uint64_t start_nonce, unsigned count)
{
	struct pending_batch
	{
		unsigned base;
		unsigned count;
		unsigned buf;
	};
	std::queue<pending_batch> pending;
	
	// update header constant buffer
	m_hash_queue.enqueueWriteBuffer(m_header, false, 0, 32, header);
	m_hash_queue.enqueueBarrierWithWaitList(NULL);

	/*__kernel void ethash_init(
		__global ulong* restrict g_init,
		__constant ulong const* g_header,
		ulong start_nonce,
		uint isolate
		)*/
	m_init_kernel.setArg(1, m_header);
	m_init_kernel.setArg(2, start_nonce);
	m_init_kernel.setArg(3, ~0u);

	/*__kernel void ethash_inner(
		__global uint* restrict g_mix,
		__global uint const* g_init,
		__global uint const* g_dag,
		uint accesses,
		uint isolate
		)*/
	m_inner_kernel.setArg(2, m_dag);
	m_inner_kernel.setArg(3, ACCESSES);
	m_inner_kernel.setArg(4, ~0u);

	/*__kernel  void ethash_hash(
		__global ulong* restrict g_hashes,
		__global ulong const* g_init,
		__global ulong const* g_mix,
		uint isolate
		)*/
	m_hash_kernel.setArg(3, ~0u);

	unsigned buf = 0;
	for (unsigned i = 0; i < count || !pending.empty(); )
	{
		// how many this batch
		if (i < count)
		{
			unsigned const c_min_batch_size = MIX_BYTES / 16;
			unsigned const this_count = std::min(count - i, c_hash_batch_size);
			unsigned const batch_count = std::max(this_count, c_min_batch_size);

			// supply args to kernels
			m_init_kernel.setArg(0, m_init_buf[buf]);
			m_inner_kernel.setArg(0, m_mix_buf[buf]);
			m_inner_kernel.setArg(1, m_init_buf[buf]);
			m_hash_kernel.setArg(0, m_hash_buf[buf]);
			m_hash_kernel.setArg(1, m_init_buf[buf]);
			m_hash_kernel.setArg(2, m_mix_buf[buf]);

			cl::Event init_event, inner_event;
			m_hash_queue.enqueueNDRangeKernel(m_init_kernel, cl::NullRange, batch_count, m_workgroup_size, NULL, &init_event);

			std::vector<cl::Event> init_event_vec = std::vector<cl::Event>{init_event};
			m_mem_queue.enqueueNDRangeKernel(m_inner_kernel, cl::NullRange, batch_count, m_workgroup_size, &init_event_vec, &inner_event);

			std::vector<cl::Event> inner_event_vec = std::vector<cl::Event>{inner_event};
			m_hash_queue.enqueueNDRangeKernel(m_hash_kernel, cl::NullRange, batch_count, m_workgroup_size, &inner_event_vec);

			pending.push({i, this_count, buf});
			i += this_count;
			buf = (buf + 1) % c_num_buffers;
		}

		// read results
		if (i == count || pending.size() == c_num_buffers)
		{
			pending_batch const& batch = pending.front();

			// could use pinned host pointer instead, but this path isn't that important.
			uint8_t* hashes = (uint8_t*)m_hash_queue.enqueueMapBuffer(m_hash_buf[batch.buf], true, CL_MAP_READ, 0, batch.count * ETHASH_BYTES);
			memcpy(ret + batch.base*ETHASH_BYTES, hashes, batch.count*ETHASH_BYTES);
			m_hash_queue.enqueueUnmapMemObject(m_hash_buf[batch.buf], hashes);

			pending.pop();
		}
	}

	m_mem_queue.finish();
	m_hash_queue.finish();
}

void ethash_cl_miner2::search(uint8_t const* header, uint64_t target, search_hook& hook)
{
	struct pending_batch
	{
		uint64_t start_nonce;
		unsigned buf;
	};
	std::queue<pending_batch> pending;

	static uint32_t const c_zero = 0;

	// update header constant buffer
	m_hash_queue.enqueueWriteBuffer(m_header, false, 0, 32, header);
	for (unsigned i = 0; i != c_num_buffers; ++i)
	{
		m_hash_queue.enqueueWriteBuffer(m_search_buf[i], false, 0, 4, &c_zero);
	}
	cl::Event pre_return_event;
	m_hash_queue.enqueueBarrierWithWaitList(NULL, &pre_return_event);

	/* __kernel  void ethash_init(
		__global hash64_t* g_init,
		__constant hash32_t const* g_header,
		ulong start_nonce,
		uint isolate
		)*/
	m_init_kernel.setArg(1, m_header);
	m_init_kernel.setArg(3, ~0u);

	/* __kernel  void ethash_inner(
		__global hash32_t* g_mix,
		__global hash64_t const* g_init,
		__global hash128_t const* g_dag,
		uint accesses,
		uint isolate
		)*/
	m_inner_kernel.setArg(2, m_dag);
	m_inner_kernel.setArg(3, ACCESSES);
	m_inner_kernel.setArg(4, ~0u);

	/*__kernel void ethash_search(
		__global uint* restrict g_output,
		__global hash64_t const* g_init,
		__global hash32_t const* g_mix,
		ulong target
		uint isolate
		)*/
	m_search_kernel.setArg(3, target);
	m_search_kernel.setArg(4, ~0u);

	unsigned buf = 0;
	for (uint64_t start_nonce = 0; ; start_nonce += c_search_batch_size)
	{
		// supply args to kernels
		m_init_kernel.setArg(0, m_init_buf[buf]);
		m_init_kernel.setArg(2, start_nonce);
		m_inner_kernel.setArg(0, m_mix_buf[buf]);
		m_inner_kernel.setArg(1, m_init_buf[buf]);
		m_search_kernel.setArg(0, m_search_buf[buf]);
		m_search_kernel.setArg(1, m_init_buf[buf]);
		m_search_kernel.setArg(2, m_mix_buf[buf]);

		// execute it!
		cl::Event init_event, inner_event;
		m_hash_queue.enqueueNDRangeKernel(m_init_kernel, cl::NullRange, c_search_batch_size, m_workgroup_size, NULL, &init_event);

		std::vector<cl::Event> init_event_vec = std::vector<cl::Event>{init_event};
		m_mem_queue.enqueueNDRangeKernel(m_inner_kernel, cl::NullRange, c_search_batch_size, m_workgroup_size, &init_event_vec, &inner_event);

		std::vector<cl::Event> inner_event_vec = std::vector<cl::Event>{inner_event};
		m_hash_queue.enqueueNDRangeKernel(m_search_kernel, cl::NullRange, c_search_batch_size, m_workgroup_size, &inner_event_vec);

		pending.push({start_nonce, buf});
		buf = (buf + 1) % c_num_buffers;

		// read results
		if (pending.size() == c_num_buffers)
		{
			pending_batch const& batch = pending.front();

			// could use pinned host pointer instead
			uint32_t* results = (uint32_t*)m_hash_queue.enqueueMapBuffer(m_search_buf[batch.buf], true, CL_MAP_READ, 0, (1+c_max_search_results) * sizeof(uint32_t));
			unsigned num_found = std::min(results[0], c_max_search_results);

			uint64_t nonces[c_max_search_results];
			for (unsigned i = 0; i != num_found; ++i)
			{
				nonces[i] = batch.start_nonce + results[i+1];
			}
			
			m_hash_queue.enqueueUnmapMemObject(m_search_buf[batch.buf], results);
			
			bool exit = num_found && hook.found(nonces, num_found);
			exit |= hook.searched(batch.start_nonce, c_search_batch_size); // always report searched before exit
			if (exit)
				break;

			// reset search buffer if we're still going
			if (num_found)
				m_mem_queue.enqueueWriteBuffer(m_search_buf[batch.buf], true, 0, 4, &c_zero);

			pending.pop();
		}
	}

	// not safe to return until this is ready
	pre_return_event.wait();
}
