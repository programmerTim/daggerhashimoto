#pragma once

#define __CL_ENABLE_EXCEPTIONS 
#define CL_USE_DEPRECATED_OPENCL_2_0_APIS
#include <CL/cl.hpp>
#include <time.h>
#include "../libethash/ethash.h"
#include "ethash_cl_miner.h"

class ethash_cl_miner2 : public ethash_cl_miner_base
{
public:
	ethash_cl_miner2();

	bool init(ethash_params const& params, const uint8_t seed[32], unsigned workgroup_size = 64);

	void finish();
	void hash(uint8_t* ret, uint8_t const* header, uint64_t start_nonce, unsigned count);
	void search(uint8_t const* header, uint64_t target, search_hook& hook);

private:
	static unsigned const c_num_buffers = 2;
	static unsigned const c_max_search_results = 63;
	static unsigned const c_hash_batch_size = 1024*256;
	static unsigned const c_search_batch_size = 1024*256;

	unsigned m_workgroup_size;
	ethash_params m_params;
	cl::Context m_context;
	cl::CommandQueue m_hash_queue;
	cl::CommandQueue m_mem_queue;
	cl::Kernel m_init_kernel;
	cl::Kernel m_inner_kernel;
	cl::Kernel m_hash_kernel;
	cl::Kernel m_search_kernel;
	cl::Buffer m_dag;
	cl::Buffer m_header;

	cl::Buffer m_init_buf[c_num_buffers];
	cl::Buffer m_mix_buf[c_num_buffers];
	cl::Buffer m_hash_buf[c_num_buffers];
	cl::Buffer m_search_buf[c_num_buffers];
};