// Copyright (c) 2009-2011, Tor M. Aamodt, Tayler Hetherington
// The University of British Columbia
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice, this
// list of conditions and the following disclaimer in the documentation and/or
// other materials provided with the distribution.
// Neither the name of The University of British Columbia nor the names of its
// contributors may be used to endorse or promote products derived from this
// software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "gpu-cache.h"
#include "stat-tool.h"
#include <assert.h>
#include <iostream>
#define MAX_DEFAULT_CACHE_SIZE_MULTIBLIER 4

long g_mshr_changed=0; // global vars. define/init/use only in this file. 

// used to allocate memory that is large enough to adapt the changes in cache size across kernels

const char * cache_request_status_str(enum cache_request_status status) 
{
   static const char * static_cache_request_status_str[] = {
      "HIT",
      "HIT_RESERVED",
      "MISS",
      "RESERVATION_FAIL"
   }; 

   assert(sizeof(static_cache_request_status_str) / sizeof(const char*) == NUM_CACHE_REQUEST_STATUS); 
   assert(status < NUM_CACHE_REQUEST_STATUS); 

   return static_cache_request_status_str[status]; 
}

void l2_cache_config::init(linear_to_raw_address_translation *address_mapping){
	cache_config::init(m_config_string,FuncCachePreferNone);
	m_address_mapping = address_mapping;
}

unsigned l2_cache_config::set_index(new_addr_type addr) const{
	if(!m_address_mapping){
		return(addr >> m_line_sz_log2) & (m_nset-1);
	}else{
		// Calculate set index without memory partition bits to reduce set camping
		new_addr_type part_addr = m_address_mapping->partition_address(addr);
		return(part_addr >> m_line_sz_log2) & (m_nset -1);
	}
}

tag_array::~tag_array() 
{
    delete[] m_lines;
}

tag_array::tag_array( cache_config &config,
                      int core_id,
                      int type_id,
                      cache_block_t* new_lines)
    : m_config( config ),
      m_lines( new_lines )
{
    init( core_id, type_id );
}

void tag_array::update_cache_parameters(cache_config &config)
{
	m_config=config;
}

tag_array::tag_array( cache_config &config,
                      int core_id,
                      int type_id )
    : m_config( config )
{
    //assert( m_config.m_write_policy == READ_ONLY ); Old assert
    m_lines = new cache_block_t[MAX_DEFAULT_CACHE_SIZE_MULTIBLIER*config.get_num_lines()];//- new array of cache_blocks
    init( core_id, type_id );
}

void tag_array::init( int core_id, int type_id )
{
    m_access = 0;
    m_miss = 0;
    m_pending_hit = 0;
    m_res_fail = 0;
    // initialize snapshot counters for visualizer
    m_prev_snapshot_access = 0;
    m_prev_snapshot_miss = 0;
    m_prev_snapshot_pending_hit = 0;
    m_core_id = core_id; 
    m_type_id = type_id;
}
// search in cache,return 4 type: HIT/PENDING_HIT/MISS/RESERVATION_FAIL. if miss ,idx is the evict line.
enum cache_request_status tag_array::probe( new_addr_type addr, unsigned &idx ) const {
    //assert( m_config.m_write_policy == READ_ONLY );
    unsigned set_index = m_config.set_index(addr);//-get the set No this addr belong to. low bits of addr.
    new_addr_type tag = m_config.tag(addr);// tag of this line of this addr, high bits of addr.

    unsigned invalid_line = (unsigned)-1;
    unsigned valid_line = (unsigned)-1;
    unsigned valid_timestamp = (unsigned)-1;

    bool all_reserved = true;

    // check for hit or pending hit
    for (unsigned way=0; way<m_config.m_assoc; way++) { //-ways/sets is simulated with a array. set No main order.
        unsigned index = set_index*m_config.m_assoc+way;//-visit every way of  corresponding set
        cache_block_t *line = &m_lines[index];
        if (line->m_tag == tag) { //-the tag is the same;
            if ( line->m_status == RESERVED ) {
                idx = index;
                return HIT_RESERVED;//-data is not return yet, pending hit.
            } else if ( line->m_status == VALID ) {
                idx = index;
                return HIT;// hit
            } else if ( line->m_status == MODIFIED ) {
                idx = index;
                return HIT;// hit
            } else {
                assert( line->m_status == INVALID );//-if not the 3 status above, must be INVALID, empty.
            }
        }//-if
        //-the tag of current way != quest.
        if (line->m_status != RESERVED) {//-at least one is not RESERVED, all_reserved = false;means one can be evict.
            all_reserved = false;
            if (line->m_status == INVALID) {
                invalid_line = index; //- the cache has a blank line, can be allocte to the quest if miss.
            } else {
                //- valid line : keep track of most appropriate replacement candidate. select a cadidate line if miss.
                if ( m_config.m_replacement_policy == LRU ) {
                    if ( line->m_last_access_time < valid_timestamp ) {//-find a line with earlyest access time 
                        valid_timestamp = line->m_last_access_time; 
                        valid_line = index;
                    }
                } else if ( m_config.m_replacement_policy == FIFO ) {
                    if ( line->m_alloc_time < valid_timestamp ) { //-find a line with earlyest alloc time
                        valid_timestamp = line->m_alloc_time;
                        valid_line = index;
                    }
                }
            }
        }
    }//-end of for
    //-if run to here,must not be (pending)hit.
    if ( all_reserved ) {//- miss and all lines have been allocated ,not free line for this miss(reason 1)
        assert( m_config.m_alloc_policy == ON_MISS ); 
        return RESERVATION_FAIL; 
    }
    //-if has a invalid candidate, return with it ; or return with a line selected by LRU/FIFO for evict.
    if ( invalid_line != (unsigned)-1 ) {
        idx = invalid_line;
    } else if ( valid_line != (unsigned)-1) {
        idx = valid_line;
    } else abort(); //- if an unreserved block exists, it is either invalid or replaceable 
    
    return MISS; //- MISS means var idx is the replace line.
}
//- 3 params
enum cache_request_status tag_array::access( new_addr_type addr, unsigned time, unsigned &idx )
{
    bool wb=false;
    cache_block_t evicted;
    enum cache_request_status result = access(addr,time,idx,wb,evicted); //-call access() with 5 params.
    assert(!wb);
    return result;
}
//- 5 params. return 1 in 4 type.
enum cache_request_status tag_array::access( new_addr_type addr, unsigned time, unsigned &idx, bool &wb, cache_block_t &evicted ) 
{
    m_access++;
    shader_cache_access_log(m_core_id, m_type_id, 0); // log accesses to cache
    enum cache_request_status status = probe(addr,idx); //-probe again (can optimize by save and transfer 'status'&'idx'to this function?)
    switch (status) {
    case HIT_RESERVED: 
        m_pending_hit++;  //HIT_RESERVED ->  pending hit 
    case HIT: 
        m_lines[idx].m_last_access_time=time;  //-modify access time of the hit line. this var decide LRU
        break;
    case MISS:
        m_miss++;
        shader_cache_access_log(m_core_id, m_type_id, 1); // log cache misses
        if ( m_config.m_alloc_policy == ON_MISS )  {
            if( m_lines[idx].m_status == MODIFIED ) {
                wb = true;
                evicted = m_lines[idx]; //- save the replaced modified line to var:evicted if MISS & ON_MISS
            }
            m_lines[idx].allocate( m_config.tag(addr), m_config.block_addr(addr), time );
        }    //-simulate this line is allocate again , mark this line reserved. 
        break;
    case RESERVATION_FAIL:
        m_res_fail++;
        shader_cache_access_log(m_core_id, m_type_id, 1); // log cache misses
        break;
    default:
        fprintf( stderr, "tag_array::access - Error: Unknown"
            "cache_request_status %d\n", status );
        abort();
    }
    return status;
}

void tag_array::fill( new_addr_type addr, unsigned time )//-cache block : L2 -> icnt -> L1.tag_array[]
{
    assert( m_config.m_alloc_policy == ON_FILL );
    unsigned idx;
    enum cache_request_status status = probe(addr,idx); //-get position for this addr;
    assert(status==MISS); // MSHR should have prevented redundant memory request
    m_lines[idx].allocate( m_config.tag(addr), m_config.block_addr(addr), time ); //-not allocate line before. reserved
    m_lines[idx].fill(time);//-valid
}

void tag_array::fill( unsigned index, unsigned time ) 
{
    assert( m_config.m_alloc_policy == ON_MISS );
    m_lines[index].fill(time);//- has been allocate a line before, so fill directly.
}

void tag_array::flush() 
{
    for (unsigned i=0; i < m_config.get_num_lines(); i++)
        m_lines[i].m_status = INVALID;
}

float tag_array::windowed_miss_rate( ) const
{
    unsigned n_access    = m_access - m_prev_snapshot_access;
    unsigned n_miss      = m_miss - m_prev_snapshot_miss;
    // unsigned n_pending_hit = m_pending_hit - m_prev_snapshot_pending_hit;

    float missrate = 0.0f;
    if (n_access != 0)
        missrate = (float) n_miss / n_access;
    return missrate;
}

void tag_array::new_window()
{
    m_prev_snapshot_access = m_access;
    m_prev_snapshot_miss = m_miss;
    m_prev_snapshot_pending_hit = m_pending_hit;
}

void tag_array::print( FILE *stream, unsigned &total_access, unsigned &total_misses ) const
{
    m_config.print(stream);
    fprintf( stream, "\t\tAccess = %d, Miss = %d (%.3g), PendingHit = %d (%.3g)\n", 
             m_access, m_miss, (float) m_miss / m_access, 
             m_pending_hit, (float) m_pending_hit / m_access);
    total_misses+=m_miss;
    total_access+=m_access;
}

void tag_array::get_stats(unsigned &total_access, unsigned &total_misses, unsigned &total_hit_res, unsigned &total_res_fail) const{
    // Update statistics from the tag array
    total_access    = m_access;
    total_misses    = m_miss;
    total_hit_res   = m_pending_hit;
    total_res_fail  = m_res_fail;
}

// events is a list, can have write/read/writeback in the same time.
bool was_write_sent( const std::list<cache_event> &events )
{
    for( std::list<cache_event>::const_iterator e=events.begin(); e!=events.end(); e++ ) {
        if( *e == WRITE_REQUEST_SENT ) 
            return true;
    }
    return false;
}

bool was_writeback_sent( const std::list<cache_event> &events )
{
    for( std::list<cache_event>::const_iterator e=events.begin(); e!=events.end(); e++ ) {
        if( *e == WRITE_BACK_REQUEST_SENT ) 
            return true;
    }
    return false;
}

bool was_read_sent( const std::list<cache_event> &events )
{
    for( std::list<cache_event>::const_iterator e=events.begin(); e!=events.end(); e++ ) {
        if( *e == READ_REQUEST_SENT ) 
            return true;
    }
    return false;
}
/****************************************************************** MSHR ******************************************************************/

/// Checks if there is a pending request to the lower memory level already, (check if this addr is already in mshr.)
bool mshr_table::probe( new_addr_type block_addr ) const{
    table::const_iterator a = m_data.find(block_addr); // m_data is a map ( addr , list<mf> )
    return a != m_data.end();
}

/// Checks if there is space for tracking a new memory access, (check if mshr can accept this addr.)
bool mshr_table::full( new_addr_type block_addr ) const{
    table::const_iterator i=m_data.find(block_addr);
    if ( i != m_data.end() ) //-found in map.
        return i->second.m_list.size() >= m_max_merged;// this addr is full.
    else
        return m_data.size() >= m_num_entries;// all line is occupyed by other address.
}

/// Add or merge this access(mf)
void mshr_table::add( new_addr_type block_addr, mem_fetch *mf ){ // add to tail.
	m_data[block_addr].m_list.push_back(mf);// advantage of map,easy to add a new line.user needn't to know where to place
    m_mf_num++;
    g_mshr_changed =g_mshr_changed +1  ;
	assert( m_data.size() <= m_num_entries );
	assert( m_data[block_addr].m_list.size() <= m_max_merged );
	// indicate that this MSHR entry contains an atomic operation
	if ( mf->isatomic() ) {
		m_data[block_addr].m_has_atomic = true;
	}
}

/// Accept a new cache fill response: mark entry ready for processing
void mshr_table::mark_ready( new_addr_type block_addr, bool &has_atomic ){
    assert( !busy() );
    table::iterator a = m_data.find(block_addr);
    assert( a != m_data.end() ); // don't remove same request twice, //-must can be find
    m_current_response.push_back( block_addr );//-add ADDR(not mf) to response list, when the data back to cache
    m_resplist_len ++;  
    has_atomic = a->second.m_has_atomic;
    assert( m_current_response.size() <= m_data.size() );// response list is the finished part of m_data.
}

/// Returns next ready access( once a mf )
mem_fetch *mshr_table::next_access(){//-mshr_entry is a fifo. the oldest mf is first seviced.
    assert( access_ready() );
    new_addr_type block_addr = m_current_response.front(); //-get a addr, one addr map to a list of mf.
    assert( !m_data[block_addr].m_list.empty() );
    mem_fetch *result = m_data[block_addr].m_list.front();  //-get the oldest mf of this addr
    m_data[block_addr].m_list.pop_front();//-pop the oldest mf of this addr from mshr 
    m_mf_num-- ;
    g_mshr_changed = g_mshr_changed +1000 ;//-global var for debug
    if ( m_data[block_addr].m_list.empty() ) { //-if all mf of this mshr_entrye is poped , the mshr_entry be deleted.
        // release entry , and pop m_current_response.front();
        m_data.erase(block_addr);
        m_current_response.pop_front(); //- all mf in the correspongding m_list is poped, this addr is poped.
        m_resplist_len --;  
    }
    return result; //-the returned mf will be del by L1I/L1D/L2
}

void mshr_table::display( FILE *fp )const {
    fprintf(fp,"MSHR contents\n");
    fprintf(fp,"m_mf_num=%d, m_resplist_len=%d \n",m_mf_num, m_resplist_len);
    for ( table::const_iterator e=m_data.begin(); e!=m_data.end(); ++e ) {
        unsigned block_addr = e->first;
        fprintf(fp,"MSHR: tag=0x%06x, atomic=%d ,%zu entries :->{ ", block_addr, e->second.m_has_atomic, e->second.m_list.size());
        if ( !e->second.m_list.empty() ) {
            mem_fetch *mf = e->second.m_list.front();//-only the first mf be print.not all.
            fprintf(fp,"%p :",mf); //- %p :print a pointer
            mf->print(fp);
            fprintf(fp," }<- ");
        } else {
            fprintf(fp," no memory requests???\n");
        }
    }// end of for to print m_data.
    fprintf(fp,"   response list:");
    std::list<new_addr_type>::const_iterator e;
    for(  e= m_current_response.begin(); e!=m_current_response.end(); e++ ) 
        fprintf(fp,"[ %llX ]", *e); // unsigned long long 
}
/***************************************************************** Caches *****************************************************************/
cache_stats::cache_stats(){
    m_stats.resize(NUM_MEM_ACCESS_TYPE); //- 11:(0-10)
    for(unsigned i=0; i<NUM_MEM_ACCESS_TYPE; ++i){
        m_stats[i].resize(NUM_CACHE_REQUEST_STATUS, 0); //-4:(0-3)
    }
    m_cache_port_available_cycles = 0; 
    m_cache_data_port_busy_cycles = 0; 
    m_cache_fill_port_busy_cycles = 0; 
}

void cache_stats::clear(){
    ///
    /// Zero out all current cache statistics
    ///
    for(unsigned i=0; i<NUM_MEM_ACCESS_TYPE; ++i){
        std::fill(m_stats[i].begin(), m_stats[i].end(), 0);
    }
    m_cache_port_available_cycles = 0; 
    m_cache_data_port_busy_cycles = 0; 
    m_cache_fill_port_busy_cycles = 0; 
}

void cache_stats::inc_stats(int access_type, int access_outcome){
    ///
    /// Increment the stat corresponding to (access_type, access_outcome) by 1.
    ///
    if(!check_valid(access_type, access_outcome))
        assert(0 && "Unknown cache access type or access outcome");
    //         0-10         0-3
    m_stats[access_type][access_outcome]++;
}

enum cache_request_status cache_stats::select_stats_status(enum cache_request_status probe, enum cache_request_status access) const {
	///
	/// This function selects how the cache access outcome should be counted. HIT_RESERVED is considered as a MISS
	/// in the cores, however, it should be counted as a HIT_RESERVED in the caches.
	///
	if(probe == HIT_RESERVED && access != RESERVATION_FAIL)
		return probe;
	else
		return access;
}

unsigned &cache_stats::operator()(int access_type, int access_outcome){
    ///
    /// Simple method to read/modify the stat corresponding to (access_type, access_outcome)
    /// Used overloaded () to avoid the need for separate read/write member functions
    ///
    if(!check_valid(access_type, access_outcome))
        assert(0 && "Unknown cache access type or access outcome");

    return m_stats[access_type][access_outcome];
}

unsigned cache_stats::operator()(int access_type, int access_outcome) const{
    ///
    /// Const accessor into m_stats.
    ///
    if(!check_valid(access_type, access_outcome))
        assert(0 && "Unknown cache access type or access outcome");

    return m_stats[access_type][access_outcome];
}

cache_stats cache_stats::operator+(const cache_stats &cs){
    ///
    /// Overloaded + operator to allow for simple stat accumulation
    ///
    cache_stats ret;
    for(unsigned type=0; type<NUM_MEM_ACCESS_TYPE; ++type){ //-mf type: 
        for(unsigned status=0; status<NUM_CACHE_REQUEST_STATUS; ++status){
            ret(type, status) = m_stats[type][status] + cs(type, status);
        }
    }
    ret.m_cache_port_available_cycles = m_cache_port_available_cycles + cs.m_cache_port_available_cycles; 
    ret.m_cache_data_port_busy_cycles = m_cache_data_port_busy_cycles + cs.m_cache_data_port_busy_cycles; 
    ret.m_cache_fill_port_busy_cycles = m_cache_fill_port_busy_cycles + cs.m_cache_fill_port_busy_cycles; 
    return ret;
}

cache_stats &cache_stats::operator+=(const cache_stats &cs){
    ///
    /// Overloaded += operator to allow for simple stat accumulation
    ///
    for(unsigned type=0; type<NUM_MEM_ACCESS_TYPE; ++type){
        for(unsigned status=0; status<NUM_CACHE_REQUEST_STATUS; ++status){
            m_stats[type][status] += cs(type, status);
        }
    }
    m_cache_port_available_cycles += cs.m_cache_port_available_cycles; 
    m_cache_data_port_busy_cycles += cs.m_cache_data_port_busy_cycles; 
    m_cache_fill_port_busy_cycles += cs.m_cache_fill_port_busy_cycles; 
    return *this;
}

void cache_stats::print_stats(FILE *fout, const char *cache_name) const{
    ///
    /// Print out each non-zero cache statistic for every memory access type and status
    /// "cache_name" defaults to "Cache_stats" when no argument is provided, otherwise
    /// the provided name is used.
    /// The printed format is "<cache_name>[<request_type>][<request_status>] = <stat_value>"
    ///
    std::string m_cache_name = cache_name;
    for (unsigned type = 0; type < NUM_MEM_ACCESS_TYPE; ++type) {
        for (unsigned status = 0; status < NUM_CACHE_REQUEST_STATUS; ++status) {
            if(m_stats[type][status] > 0){
                fprintf(fout, "\t%s[%12s][%16s] = %6u\n",
                    m_cache_name.c_str(),
                    mem_access_type_str((enum mem_access_type)type),
                    cache_request_status_str((enum cache_request_status)status),
                    m_stats[type][status]);
            }
        }
    }
}

void cache_sub_stats::print_port_stats(FILE *fout, const char *cache_name) const
{
    float data_port_util = 0.0f; 
    if (port_available_cycles > 0) {
        data_port_util = (float) data_port_busy_cycles / port_available_cycles; 
    }
    fprintf(fout, "%s_data_port_util = %.3f\n", cache_name, data_port_util); 
    float fill_port_util = 0.0f; 
    if (port_available_cycles > 0) {
        fill_port_util = (float) fill_port_busy_cycles / port_available_cycles; 
    }
    fprintf(fout, "%s_fill_port_util = %.3f\n", cache_name, fill_port_util); 
}

unsigned cache_stats::get_stats(enum mem_access_type *access_type, unsigned num_access_type, enum cache_request_status *access_status, unsigned num_access_status) const{
    ///
    /// Returns a sum of the stats corresponding to each "access_type" and "access_status" pair.
    /// "access_type" is an array of "num_access_type" mem_access_types.
    /// "access_status" is an array of "num_access_status" cache_request_statuses.
    ///
    unsigned total=0;
    for(unsigned type =0; type < num_access_type; ++type){
        for(unsigned status=0; status < num_access_status; ++status){
            if(!check_valid((int)access_type[type], (int)access_status[status]))
                assert(0 && "Unknown cache access type or access outcome");
            total += m_stats[access_type[type]][access_status[status]];
        }
    }
    return total;
}
void cache_stats::get_sub_stats(struct cache_sub_stats &css) const{
    ///
    /// Overwrites "css" with the appropriate statistics from this cache.
    ///
    struct cache_sub_stats t_css;
    t_css.clear();

    for (unsigned type = 0; type < NUM_MEM_ACCESS_TYPE; ++type) { //-all 11 types, for example: INST_ACC_R /GLBAL_ACC_W
        for (unsigned status = 0; status < NUM_CACHE_REQUEST_STATUS; ++status) {// -all 4 types.
            if(status == HIT || status == MISS || status == HIT_RESERVED)//- not include "reservation fail"
                t_css.accesses += m_stats[type][status];

            if(status == MISS)
                t_css.misses += m_stats[type][status];

            if(status == HIT_RESERVED)
                t_css.pending_hits += m_stats[type][status];

            if(status == RESERVATION_FAIL)
                t_css.res_fails += m_stats[type][status];
        }
    }

    t_css.port_available_cycles = m_cache_port_available_cycles; 
    t_css.data_port_busy_cycles = m_cache_data_port_busy_cycles; 
    t_css.fill_port_busy_cycles = m_cache_fill_port_busy_cycles; 

    css = t_css;
}

bool cache_stats::check_valid(int type, int status) const{
    ///
    /// Verify a valid access_type/access_status
    ///
    if((type >= 0) && (type < NUM_MEM_ACCESS_TYPE) && (status >= 0) && (status < NUM_CACHE_REQUEST_STATUS))
        return true;
    else
        return false;
}

void cache_stats::sample_cache_port_utility(bool data_port_busy, bool fill_port_busy) 
{
    m_cache_port_available_cycles += 1; 
    if (data_port_busy) {
        m_cache_data_port_busy_cycles += 1; 
    } 
    if (fill_port_busy) {
        m_cache_fill_port_busy_cycles += 1; 
    } 
}

baseline_cache::bandwidth_management::bandwidth_management(cache_config &config) 
: m_config(config)
{
    m_data_port_occupied_cycles = 0; 
    m_fill_port_occupied_cycles = 0; 
}

/// use the data port based on the outcome and events generated by the mem_fetch request 
void baseline_cache::bandwidth_management::use_data_port(mem_fetch *mf, enum cache_request_status outcome, const std::list<cache_event> &events)
{
    unsigned data_size = mf->get_data_size(); 
    unsigned port_width = m_config.m_data_port_width; 
    switch (outcome) {
    case HIT: {
        unsigned data_cycles = data_size / port_width + ((data_size % port_width > 0)? 1 : 0); 
        m_data_port_occupied_cycles += data_cycles; 
        } break; 
    case HIT_RESERVED: 
    case MISS: {
        // the data array is accessed to read out the entire line for write-back 
        if (was_writeback_sent(events)) {
            unsigned data_cycles = m_config.m_line_sz / port_width; 
            m_data_port_occupied_cycles += data_cycles; 
        }
        } break; 
    case RESERVATION_FAIL: 
        // Does not consume any port bandwidth 
        break; 
    default: 
        assert(0); 
        break; 
    } 
}

/// use the fill port 
void baseline_cache::bandwidth_management::use_fill_port(mem_fetch *mf)
{
    // assume filling the entire line with the returned request 
    unsigned fill_cycles = m_config.m_line_sz / m_config.m_data_port_width; 
    m_fill_port_occupied_cycles += fill_cycles; 
}

/// called every cache cycle to free up the ports 
void baseline_cache::bandwidth_management::replenish_port_bandwidth()
{
    if (m_data_port_occupied_cycles > 0) {
        m_data_port_occupied_cycles -= 1; 
    }
    assert(m_data_port_occupied_cycles >= 0); 

    if (m_fill_port_occupied_cycles > 0) {
        m_fill_port_occupied_cycles -= 1; 
    }
    assert(m_fill_port_occupied_cycles >= 0); 
}

/// query for data port availability 
bool baseline_cache::bandwidth_management::data_port_free() const
{
    return (m_data_port_occupied_cycles == 0); 
}

/// query for fill port availability 
bool baseline_cache::bandwidth_management::fill_port_free() const
{
    return (m_fill_port_occupied_cycles == 0); 
}

/// Sends next request to lower level of memory
void baseline_cache::cycle(){
    if ( !m_miss_queue.empty() ) {
        mem_fetch *mf = m_miss_queue.front();
        if ( !m_memport->full(mf->size(),mf->get_is_write()) ) {
            m_miss_queue.pop_front();//-move mf from miss Q 
            m_memport->push(mf); //-into ICNT(for L1), or into Dram(for L2);
        }
    }
    bool data_port_busy = !m_bandwidth_management.data_port_free(); //-occupied this port for N cycles;
    bool fill_port_busy = !m_bandwidth_management.fill_port_free(); 
    m_stats.sample_cache_port_utility(data_port_busy, fill_port_busy); 
    m_bandwidth_management.replenish_port_bandwidth(); 
}

/// Interface for response from lower memory level (model bandwidth restictions in caller)
void baseline_cache::fill(mem_fetch *mf, unsigned time){//- mf insert to cache
    extra_mf_fields_lookup::iterator e = m_extra_mf_fields.find(mf);//- a map<*mf , struct > var
    assert( e != m_extra_mf_fields.end() ); //- mf must can be find in 'm_extra_mf_fields';
    assert( e->second.m_valid );
    mf->set_data_size( e->second.m_data_size );
    if ( m_config.m_alloc_policy == ON_MISS )
        m_tag_array->fill(e->second.m_cache_index,time);//- call fill() only;
    else if ( m_config.m_alloc_policy == ON_FILL )
        m_tag_array->fill(e->second.m_block_addr,time);//- call probe(), allocate(), and fill();
    else abort();
    bool has_atomic = false;
    m_mshrs.mark_ready(e->second.m_block_addr, has_atomic); //-fill this 'addr' to mshr's m_current_response
    if (has_atomic) {
        assert(m_config.m_alloc_policy == ON_MISS);
        cache_block_t &block = m_tag_array->get_block(e->second.m_cache_index);
        block.m_status = MODIFIED; // mark line as dirty for atomic operation
    }
    m_extra_mf_fields.erase(mf); //-remove this pointer form map;
    m_bandwidth_management.use_fill_port(mf); 
}

/// Checks if mf is waiting to be filled by lower memory level
bool baseline_cache::waiting_for_fill( mem_fetch *mf ){
    extra_mf_fields_lookup::iterator e = m_extra_mf_fields.find(mf);
    return e != m_extra_mf_fields.end();
}

void baseline_cache::print(FILE *fp, unsigned &accesses, unsigned &misses) const{
    fprintf( fp, "Cache %s:\t", m_name.c_str() );
    m_tag_array->print(fp,accesses,misses);
}

void baseline_cache::display_state( FILE *fp ) const{
    fprintf(fp,"Cache %s:\n", m_name.c_str() );
    m_mshrs.display(fp);
    fprintf(fp,"\n");
}

/// Read miss handler without writeback. 9 params
void baseline_cache::send_read_request(new_addr_type addr, new_addr_type block_addr, unsigned cache_index, mem_fetch *mf,
		unsigned time, bool &do_miss, std::list<cache_event> &events, bool read_only, bool wa){

	bool wb=false;  //-add 1
	cache_block_t e;//-add 2
	send_read_request(addr, block_addr, cache_index, mf, time, do_miss, wb, e, events, read_only, wa); //-11 params
}

//- 2016.04.05 add to show mf status string 
#define MF_TUP_BEGIN(X) static const char* Status_str[] = {
#define MF_TUP(X) #X
#define MF_TUP_END(X) };
#include "mem_fetch_status.tup"
#undef MF_TUP_BEGIN
#undef MF_TUP
#undef MF_TUP_END

extern int g_show_mf_travel;
/// Read miss handler. Check MSHR hit or MSHR available, ( bool &wb, cache_block_t &evicted) added. 11 params
void baseline_cache::send_read_request( new_addr_type addr,
                                        new_addr_type block_addr,
                                        unsigned cache_index,
                                        mem_fetch *mf,
		                                unsigned time,
                                        bool &do_miss,
                                        bool &wb,
                                        cache_block_t &evicted,
                                        std::list<cache_event> &events,
                                        bool read_only,
                                        bool wa){

    bool mshr_hit = m_mshrs.probe(block_addr);
    bool mshr_avail = !m_mshrs.full(block_addr);
    if ( mshr_hit && mshr_avail ) { //- hit in mshr( HIT_RESERVED, pending hit), need not add mf to miss_queue.
    	if(read_only) //-read only cache (.e.g L1I), need not wb;
    		m_tag_array->access(block_addr,time,cache_index); //-retrun 4 cache_status, but no save & use here.
    	else         //-L1D/L1C/L1T
    		m_tag_array->access(block_addr,time,cache_index,wb,evicted);

        m_mshrs.add(block_addr,mf);
        do_miss = true;
    } else if ( !mshr_hit && mshr_avail && (m_miss_queue.size() < m_config.m_miss_queue_size) ) {//- miss in mshr
    	if(read_only)
    		m_tag_array->access(block_addr,time,cache_index); //-mark the right line as RESERVED.
    	else
    		m_tag_array->access(block_addr,time,cache_index,wb,evicted); //-wb & evicted is return value.

        m_mshrs.add(block_addr,mf); //-mf put in m_mshrs & m_miss_queue
        m_extra_mf_fields[mf] = extra_mf_fields(block_addr,cache_index, mf->get_data_size());
        mf->set_data_size( m_config.get_line_sz() );//-data_size is set to line_size(before here, data_size =? )
        m_miss_queue.push_back(mf); // miss, send to miss queue, wait next level to return;** m_miss_queue.push_back(mf);
        mf->set_status(m_miss_queue_status,time);
        //- 2016.04.05
        if(g_show_mf_travel == 1)
            printf("@%5u,s%2d,w%2d,id%4d@  %22s, address=%8X, is_pre=%d, status=%2d, %30s, %10s\n",time,mf->get_sid(), mf->get_wid(),mf->get_request_uid(),"send_read_request()",mf->get_addr() , mf->m_is_pre, m_miss_queue_status, Status_str[m_miss_queue_status], this->m_name.c_str() );
        if(!wa)
        	events.push_back(READ_REQUEST_SENT);
        do_miss = true;
    }
    // else means no mshr space or no miss_queque space for this mf, do_miss is false,reservation_false
}//- [do_miss, wb & evict] return to upper call. do_miss means this mf is processed. 


/// Sends write request to lower level memory (write or writeback)
void data_cache::send_write_request(    mem_fetch *mf, 
                                        cache_event request, 
                                        unsigned time, 
                                        std::list<cache_event> &events){
    events.push_back(request);
    m_miss_queue.push_back(mf); //--------------------- m_miss_queue.push_back();
    mf->set_status(m_miss_queue_status,time);
    //- 2016.04.05
    if(g_show_mf_travel == 1)
        printf("@%5u,s%2d,w%2d,id%4d@  %22s, address=%8X, is_pre=%d, status=%2d, %30s, %10s\n",time,mf->get_sid(), mf->get_wid(),mf->get_request_uid(),"send_write_request()",mf->get_addr() , mf->m_is_pre, m_miss_queue_status, Status_str[m_miss_queue_status], this->m_name.c_str() );
}


/****** Write-hit functions (Set by config file) ******/

/// Write-back hit: Mark block as modified , //-return: retrun HIT
cache_request_status 
        data_cache::wr_hit_wb(  new_addr_type addr, 
                                unsigned cache_index, 
                                mem_fetch *mf,
                                unsigned time,
                                std::list<cache_event> &events,
                                enum cache_request_status status ){
	new_addr_type block_addr = m_config.block_addr(addr);
	m_tag_array->access(block_addr,time,cache_index); // update LRU state , 3 param 
	cache_block_t &block = m_tag_array->get_block(cache_index);
	block.m_status = MODIFIED; //-mark dirty data.

	return HIT;
}

/// Write-through hit: Directly send request to lower level memory ; //-return RES_FAIL/ HIT
cache_request_status 
    data_cache::wr_hit_wt(  new_addr_type addr, 
                            unsigned cache_index, 
                            mem_fetch *mf, 
                            unsigned time, 
                            std::list<cache_event> &events, 
                            enum cache_request_status status ){
	if(miss_queue_full(0))
		return RESERVATION_FAIL; // cannot handle request this cycle

	new_addr_type block_addr = m_config.block_addr(addr);
	m_tag_array->access(block_addr,time,cache_index); // update LRU state
	cache_block_t &block = m_tag_array->get_block(cache_index);
	block.m_status = MODIFIED;

	// generate a write-through
	send_write_request(mf, WRITE_REQUEST_SENT, time, events);

	return HIT;
}

/// Write-evict hit: Send request to lower level memory and invalidate corresponding block
cache_request_status                                        //-return HIT,RESRERVATION_FAIL
    data_cache::wr_hit_we(  new_addr_type addr, 
                            unsigned cache_index, 
                            mem_fetch *mf, 
                            unsigned time, 
                            std::list<cache_event> &events, 
                            enum cache_request_status status ){
	if(miss_queue_full(0))
		return RESERVATION_FAIL; // cannot handle request this cycle

	// generate a write-through/evict
	cache_block_t &block = m_tag_array->get_block(cache_index);
	send_write_request(mf, WRITE_REQUEST_SENT, time, events);

	// Invalidate block
	block.m_status = INVALID; //-this line be evicted to next level memory.

	return HIT;
}

/// Global write-evict, local write-back: Useful for private caches
enum cache_request_status 
        data_cache::wr_hit_global_we_local_wb(  new_addr_type addr, 
                                                unsigned cache_index, 
                                                mem_fetch *mf, 
                                                unsigned time, 
                                                std::list<cache_event> &events, 
                                                enum cache_request_status status ){
	bool evict = (mf->get_access_type() == GLOBAL_ACC_W); // evict a line that hits on global memory write
	if(evict)
		return wr_hit_we(addr, cache_index, mf, time, events, status); // Write-evict
	else
		return wr_hit_wb(addr, cache_index, mf, time, events, status); // Write-back
}

/****** Write-miss functions (Set by config file) ******/

/// Write-allocate miss: Send write request to lower level memory
// and send a read request for the same block
enum cache_request_status  //-return MISS,RESERVATION_FAIL
data_cache::wr_miss_wa( new_addr_type addr,
                        unsigned cache_index, mem_fetch *mf,
                        unsigned time, std::list<cache_event> &events,
                        enum cache_request_status status )
{
    new_addr_type block_addr = m_config.block_addr(addr);

    // Write allocate, maximum 3 requests (write miss, read request, write back request)
    // Conservatively ensure the worst-case request can be handled this cycle
    bool mshr_hit = m_mshrs.probe(block_addr);
    bool mshr_avail = !m_mshrs.full(block_addr);
    if(miss_queue_full(2) 
            || (!(mshr_hit && mshr_avail) 
            && !(!mshr_hit && mshr_avail 
            && (m_miss_queue.size() < m_config.m_miss_queue_size))))
        return RESERVATION_FAIL;

    send_write_request(mf, WRITE_REQUEST_SENT, time, events); //-<send 1, write >

    // Tries to send write allocate request, returns true on success and false on failure
    //if(!send_write_allocate(mf, addr, block_addr, cache_index, time, events))
    //    return RESERVATION_FAIL;

    const mem_access_t *ma = new  mem_access_t( m_wr_alloc_type,    //-new mem_access_t
                        mf->get_addr(),
                        mf->get_data_size(),
                        false, // Now performing a read
                        mf->get_access_warp_mask(),
                        mf->get_access_byte_mask() );

    mem_fetch *n_mf = new mem_fetch( *ma,               //- new mem_fetch
                    NULL,
                    mf->get_ctrl_size(),
                    mf->get_wid(),
                    mf->get_sid(),
                    mf->get_tpc(),
                    mf->get_mem_config());

    bool do_miss = false;
    bool wb = false;
    cache_block_t evicted;

    //--------------- < Send 2> read request resulting from write miss
    send_read_request(addr, block_addr, cache_index, n_mf, time, do_miss, wb,
        evicted, events, false, true);

    if( do_miss ){
        // If evicted block is modified and not a write-through
        // (already modified lower level)
        if( wb && (m_config.m_write_policy != WRITE_THROUGH) ) { //- 2016.04.05 ,modify 'wb' to 'wb_mf'
            mem_fetch *wb_mf = m_memfetch_creator->alloc(evicted.m_block_addr, 
                m_wrbk_type,m_config.get_line_sz(),true);
            //----------------- < send 3, push_back directly >
            m_miss_queue.push_back(wb_mf);
            wb_mf->set_status(m_miss_queue_status,time);
            //- 2016.04.05
            if(g_show_mf_travel == 1)
                printf("@%5u,s%2d,w%2d,id%4d@  %22s, address=%8X, is_pre=%d, status=%2d, %30s, %10s\n",time,wb_mf->get_sid(), wb_mf->get_wid(),wb_mf->get_request_uid(), "wr_miss_wa():wb_mf",wb_mf->get_addr() , wb_mf->m_is_pre, m_miss_queue_status, Status_str[m_miss_queue_status], this->m_name.c_str() );
        }
        return MISS;
    }

    return RESERVATION_FAIL;
}

/// No write-allocate miss: Simply send write request to lower level memory
enum cache_request_status       //-retrun MISS/RESERVATION_FAIL
data_cache::wr_miss_no_wa( new_addr_type addr,
                           unsigned cache_index,
                           mem_fetch *mf,
                           unsigned time,
                           std::list<cache_event> &events,
                           enum cache_request_status status )
{
    if(miss_queue_full(0))
        return RESERVATION_FAIL; // cannot handle request this cycle

    // <send 1, write > on miss, generate write through (no write buffering -- too many threads for that)
    send_write_request(mf, WRITE_REQUEST_SENT, time, events); 

    return MISS;
}

/****** Read hit functions (Set by config file) ******/

/// Baseline read hit: Update LRU status of block.
// Special case for atomic instructions -> Mark block as modified
enum cache_request_status                    //retrun HIT
data_cache::rd_hit_base( new_addr_type addr,
                         unsigned cache_index,
                         mem_fetch *mf,
                         unsigned time,
                         std::list<cache_event> &events,
                         enum cache_request_status status )
{
    new_addr_type block_addr = m_config.block_addr(addr);
    m_tag_array->access(block_addr,time,cache_index); //-update hit line access tiem,

    // Atomics treated as global read/write requests - Perform read, mark line as
    // MODIFIED
    if(mf->isatomic()){ 
        assert(mf->get_access_type() == GLOBAL_ACC_R);
        cache_block_t &block = m_tag_array->get_block(cache_index);
        block.m_status = MODIFIED;  // mark line as dirty
    }

    //=========================================BEGIN: prefetch in L1D ===================================================
    //m_miss_queue_status="IN_L1D_MISS_QUEUE", only profetch in L1D.
//    if(m_miss_queue_status == 2 &&  mf->get_access_type() == GLOBAL_ACC_R ) {
//        //-generate a pre_mf= mf+128;
//        new_addr_type pre_addr= mf->get_addr() +128; 
//        new_addr_type pre_block_addr = m_config.block_addr( pre_addr );
//        mem_access_t pre_acc(GLOBAL_ACC_R, pre_addr , mf->get_ctrl_size() ,false);
//        //-new a access object( nbytes=16/8 ) .in heap
//        mem_fetch *pre_mf = new mem_fetch(pre_acc,
//                 NULL, //&pre_inst_copy, 
//                 mf->get_ctrl_size(),
//                 mf->get_wid(),
//                 mf->get_sid(),
//                 mf->get_tpc(),
//                 mf->get_mem_config() );//-new pre_mf. in heap
//        pre_mf->m_is_pre=true;
//        //-get if pre_mf is already in L1D, get replace line idx.
//        unsigned pre_cache_index = (unsigned)-1;
//        enum cache_request_status pre_probe_status
//            = m_tag_array->probe( pre_block_addr, pre_cache_index ); 
//        if(pre_probe_status == MISS){
//            //.................................copy from rd_miss_base.........................    
//             bool pre_do_miss = false;
//             bool pre_wb = false;
//             cache_block_t pre_evicted; //- this block is for replaced line.
//             std::list<cache_event> pre_events;
//             //---------------- <send 1, read> ,return weather need writeback & evicted line.
//             send_read_request( pre_addr,
//                                pre_block_addr,
//                                pre_cache_index,
//                                pre_mf, time, pre_do_miss, pre_wb, pre_evicted, pre_events, false, false);
//
//             if( pre_do_miss ) { //-mf is processed.
//                 // If evicted block is modified and not a write-through
//                 if(pre_wb && (m_config.m_write_policy != WRITE_THROUGH) ){ 
//                     mem_fetch *wb_mf = m_memfetch_creator->alloc(pre_evicted.m_block_addr, m_wrbk_type,m_config.get_line_sz(),true);
//                     //- m_wrbk_type = L1_WRBK_ACC or  L2_WRBK_ACC
//                     //------------<send 2, write>    
//                     send_write_request(wb_mf, WRITE_BACK_REQUEST_SENT, time, pre_events); 
//                     //-read op can invoke a write op; wrtie_back_request_sent only here
//                 }
//                 //return MISS;
//             }
//             //return RESERVATION_FAIL;
//             //.................................copy from rd_miss_base.........................    
//        }//-if MISS
//    }//- if in L1D 

    //===========================================END: prefetch in L1D ===================================================
    return HIT;
}

/****** Read miss functions (Set by config file) ******/

/// Baseline read miss: Send read request to lower level memory,
// perform write-back as necessary
enum cache_request_status //-return 2 type: MISS/RESERVATION_FAIL
data_cache::rd_miss_base( new_addr_type addr,
                          unsigned cache_index,
                          mem_fetch *mf,
                          unsigned time,
                          std::list<cache_event> &events,
                          enum cache_request_status status ){
    if(miss_queue_full(1)) //-if unable to add 1 element, retrun RES_FAIL(reason 2).??? HIT_RES need not put into missQ. 
        return RESERVATION_FAIL; 

    new_addr_type block_addr = m_config.block_addr(addr);
    bool do_miss = false;
    bool wb = false;
    cache_block_t evicted; //- this block is for replaced line.
    //---------------- <send 1, read> ,return weather need writeback & evicted line.
    send_read_request( addr,
                       block_addr,
                       cache_index,
                       mf, time, do_miss, wb, evicted, events, false, false);

    if( do_miss ) { //-mf is processed.
        // If evicted block is modified and not a write-through
        if(wb && (m_config.m_write_policy != WRITE_THROUGH) ){ 
            mem_fetch *wb_mf = m_memfetch_creator->alloc(evicted.m_block_addr, m_wrbk_type,m_config.get_line_sz(),true);
            //- m_wrbk_type = L1_WRBK_ACC or  L2_WRBK_ACC
            //------------<send 2, write>    
            send_write_request(wb_mf, WRITE_BACK_REQUEST_SENT, time, events); 
            //-read op can invoke a write op; wrtie_back_request_sent only here
        }
        return MISS; //-HIT_RES --> MISS;
    }
    return RESERVATION_FAIL; //-do_miss==fail, means no MSHR(reason 3) or missQ room.
}

/// Access cache for read_only_cache: returns RESERVATION_FAIL if
// request could not be accepted (for any reason)
enum cache_request_status
read_only_cache::access( new_addr_type addr,
                         mem_fetch *mf,
                         unsigned time,
                         std::list<cache_event> &events )
{
    assert( mf->get_data_size() <= m_config.get_line_sz());
    assert(m_config.m_write_policy == READ_ONLY);
    assert(!mf->get_is_write());
    new_addr_type block_addr = m_config.block_addr(addr);// from byte address to block address
    unsigned cache_index = (unsigned)-1;    // return 4 type: HIT  MISS  RESERVATION_HIT RESERVATION_FAIL
    enum cache_request_status status = m_tag_array->probe(block_addr,cache_index);// visit m_tag_array,get status.
    enum cache_request_status cache_status = RESERVATION_FAIL; 

    if ( status == HIT ) {
        cache_status = m_tag_array->access(block_addr,time,cache_index); // update LRU state,return 4 type.
    }else if ( status != RESERVATION_FAIL ) { // pending hit or miss
        if(!miss_queue_full(0)){
            bool do_miss=false; //-if send mf to next level.decide by next line call.
            send_read_request(addr, block_addr, cache_index, mf, time, do_miss, events, true, false);//-9 params
            if(do_miss)//- do_miss is a  & para .
                cache_status = MISS;
            else
                cache_status = RESERVATION_FAIL;
        }else{
            cache_status = RESERVATION_FAIL;
        }
    }
    //-this access record in m_stats.  m_stats[access_type][access_outcome]++;
    m_stats.inc_stats(mf->get_access_type(), m_stats.select_stats_status(status, cache_status));
    return cache_status;
}

//! A general function that takes the result of a tag_array probe
//  and performs the correspding functions based on the cache configuration
//  The access fucntion calls this function
enum cache_request_status                                       //-retrun HIT/MISS/ RES_FAIL  (no HIT_RES)
data_cache::process_tag_probe( bool wr,
                               enum cache_request_status probe_status,// get from probe()
                               new_addr_type addr,
                               unsigned cache_index, // get from probe()
                               mem_fetch* mf,
                               unsigned time,
                               std::list<cache_event>& events )
{
    // Each function pointer ( m_[rd/wr]_[hit/miss] ) is set in the
    // data_cache constructor to reflect the corresponding cache configuration
    // options. Function pointers were used to avoid many long conditional
    // branches resulting from many cache configuration options.
    cache_request_status access_status = probe_status; //-process Hit/Miss/Hit_reserved(pending), exclude RES_FAIL.
    if(wr){ // Write
        if(probe_status == HIT){                        //-on(HIT)           return (HIT)
            access_status = (this->*m_wr_hit)( addr,
                                      cache_index,
                                      mf, time, events, probe_status );
        }else if ( probe_status != RESERVATION_FAIL ) { //-on(MISS, HIT_RES) return (MISS .RES_FAIL.)
            access_status = (this->*m_wr_miss)( addr,
                                       cache_index,
                                       mf, time, events, probe_status );
        }
    }else{ // Read
        if(probe_status == HIT){                        //-on(HIT),          return HIT, RESVATION_FAIL.
            access_status = (this->*m_rd_hit)( addr,
                                      cache_index,
                                      mf, time, events, probe_status );
        }else if ( probe_status != RESERVATION_FAIL ) { //-on(MISS, HIT_RES), return MISS, RESERVATION_FAIL
            access_status = (this->*m_rd_miss)( addr,
                                       cache_index,
                                       mf, time, events, probe_status );
        }
    }

    m_bandwidth_management.use_data_port(mf, access_status, events); //- add a prefetch, then the port need time;
    return access_status; //-return 
}

// Both the L1 and L2 currently use the same access function.
// Differentiation between the two caches is done through configuration
// of caching policies.
// Both the L1 and L2 override this function to provide a means of
// performing actions specific to each cache when such actions are implemnted.

extern int g_d_prefetch_interval; //-define in 
extern int g_d_prefetch_num;      //-define in 
extern int g_d_prefetch_open ;    //- 0=close, 1=open

unsigned   g_old_addr=0;          //-record previous_addr;
unsigned   g_oldold_addr=0;          //-record previous_addr;
unsigned   last_show_time=0;       //-contral m_mkv->show();


enum cache_request_status
data_cache::access( new_addr_type addr,
                    mem_fetch *mf,
                    unsigned time,
                    std::list<cache_event> &events )
{
    assert( mf->get_data_size() <= m_config.get_line_sz());
    bool wr = mf->get_is_write();
    new_addr_type block_addr = m_config.block_addr(addr);
    unsigned cache_index = (unsigned)-1;

    enum cache_request_status probe_status //-RES_FAIL reason:(1)no cache line (2)no missQ room (3)no MSHR; RES_FAIL lead this mf be deleted. and try next cycle.
        = m_tag_array->probe( block_addr, cache_index ); // search in cache tag[], return 1 in 4 status,and a cache_index

    //-only print in L1D
    int tp=mf->get_access_type(); 
    if(g_show_mf_travel == 2 && m_miss_queue_status == 2 && (tp == GLOBAL_ACC_R || tp == GLOBAL_ACC_W) ) {
       new_addr_type block_addr = mf->get_addr() & ~(128-1); //-copy function block_addr()
       char wr=mf->get_is_write() ==1 ?'w':'r';
       printf("ldst_unit@l1_cache %2u \t%llx \t%llx \t%3d \t%c \t%d \t%8u\n",
           mf->get_sid() ,mf->get_addr(), block_addr,  mf->get_data_size(), wr, probe_status, mf->get_timestamp() );
    }

    enum cache_request_status access_status //-return 1 in 3 status. no Hit_reserved.
        = process_tag_probe( wr, probe_status, addr, cache_index, mf, time, events );//-modify cache tag array. add mf to cache's miss queue if miss.
    //=========================================BEGIN: prefetch in L1D ===================================================
    // IN_PARTITION_L2_MISS_QUEUE= 12
    //if(m_miss_queue_status == 12 &&  mf->get_access_type() == GLOBAL_ACC_R ) {
    //m_miss_queue_status="IN_L1D_MISS_QUEUEi=2", only profetch in L1D.
    if( g_d_prefetch_open == 1 && m_miss_queue_status == 2 &&  mf->get_access_type() == GLOBAL_ACC_R ) {
        //_________________________________next line_____________________________________________
        //-generate a pre_mf= mf+128;
        // input: block_addr
        // output: pre_addr/pre_block_addr;
        //new_addr_type pre_addr= mf->get_addr() +128 + g_d_prefetch_interval * 128 ; 
        //new_addr_type pre_block_addr = m_config.block_addr( pre_addr );
        //_________________________________next line_____________________________________________
        
        //_________________________________Markov_____________________________________________
        // input: g_old_addr, block_addr
        // output: pre_addr, pre_block_addr;
        m_mkv->update_table( g_oldold_addr, block_addr,time);//-now can replace by itself 
       // printf("markov:update  %X--->%X\n", g_old_addr, block_addr);
        g_oldold_addr=g_old_addr;
        g_old_addr=block_addr;
        new_addr_type pre_addr= m_mkv->find_max(block_addr, time );
        new_addr_type pre_block_addr = pre_addr;
        //m_mkv->cut_table_batch(time); //-cut the all lists which not used in recent 1500 cycles.
        //m_mkv->cut_table_LRU(time);   //-cut one list every 500 cycles;
        //m_mkv->cut_entry_LRU(time);   //-cut one candidate in all lists every 500 cycles;
        if(time-last_show_time>500){
            m_mkv->show(m_name);
            last_show_time=time;
        }
        //_________________________________Markov_____________________________________________
        // if (pre_block_addr % 256 != 0) //-only used in prefetch L2 to avoid access other L2 bank.
        float my_rand= (float)rand() / (float)(RAND_MAX + 1);
        if ( my_rand<0.15 && pre_addr > 0 )//-get valid addr from m_mkv;
        {
            printf("markov:%s:  %X--->%X\n", m_name.c_str(), block_addr, pre_addr);
            unsigned pre_cache_index = (unsigned)-1;
            std::list<cache_event> pre_events;

            mem_access_t pre_acc(GLOBAL_ACC_R, pre_addr , mf->get_ctrl_size() ,false);
            //-new a access object( nbytes=16/8 ) .in heap
            mem_fetch *pre_mf = new mem_fetch(pre_acc,
                     NULL, //&pre_inst_copy, 
                     mf->get_ctrl_size(),
                     mf->get_wid(),
                     mf->get_sid(),
                     mf->get_tpc(),
                     mf->get_mem_config() );//-new pre_mf. in heap
            pre_mf->m_is_pre=true;
            //-get if pre_mf is already in L1D, get replace line idx.

            enum cache_request_status pre_probe_status
                = m_tag_array->probe( pre_block_addr, pre_cache_index ); //-check if mf+N*128   in the tag[];
            //-only if MISS, sed read mf+ N*128 ;(Donn't process when hit/res_fail/hit_res)
            if(pre_probe_status == MISS){
                g_d_prefetch_num++;
                enum cache_request_status pre_access_status 
                    = process_tag_probe( false , pre_probe_status, pre_addr, pre_cache_index, pre_mf, time, pre_events );
            }
        }//-block in the same L2 bank
    } //- in specify cache.
    //=======================================add prefetch=====================================

    m_stats.inc_stats(mf->get_access_type(),
        m_stats.select_stats_status(probe_status, access_status));
    return access_status;
}

/// This is meant to model the first level data cache in Fermi.
/// It is write-evict (global) or write-back (local) at the
/// granularity of individual blocks (Set by GPGPU-Sim configuration file)
/// (the policy used in fermi according to the CUDA manual)
int show_addr_num=0;  //-my custom global var.
enum cache_request_status
l1_cache::access( new_addr_type addr,
                  mem_fetch *mf,
                  unsigned time,
                  std::list<cache_event> &events )
{   //cjllean. to show L1 access.
	/*if(show_addr_num<30){
		//system("echo -e \"\\033[1;36m * l1_cache::access() *\\033[0m\" ");
	    printf("\t\t@L1D no=[%d] mf:{%s} addr=%llx addr=%llx  \n",show_addr_num, 
    		mem_access_type_str(mf->get_access_type()), mf->get_addr(), addr);		
		show_addr_num++;
	}*/
    enum cache_request_status status= data_cache::access( addr, mf, time, events ); //- call function of base class
    //bool my_wr = mf->get_is_write();
	// addr len isw l1_status
    //printf("\t%u \t%lld \t%llx \t%d \t %d \t %d \t@@@\n",
			//mf->get_sid(), m_config.block_addr(addr),m_config.block_addr(addr), mf->get_data_size(),my_wr,status);	
	return status;
}

// The l2 cache access function calls the base data_cache access
// implementation.  When the L2 needs to diverge from L1, L2 specific
// changes should be made here.
enum cache_request_status
l2_cache::access( new_addr_type addr,
                  mem_fetch *mf,
                  unsigned time,
                  std::list<cache_event> &events )
{
    return data_cache::access( addr, mf, time, events );
}

/// Access function for tex_cache
/// return values: RESERVATION_FAIL if request could not be accepted
/// otherwise returns HIT_RESERVED or MISS; NOTE: *never* returns HIT
/// since unlike a normal CPU cache, a "HIT" in texture cache does not
/// mean the data is ready (still need to get through fragment fifo)
enum cache_request_status tex_cache::access( new_addr_type addr, mem_fetch *mf,
    unsigned time, std::list<cache_event> &events )
{
    if ( m_fragment_fifo.full() || m_request_fifo.full() || m_rob.full() )
        return RESERVATION_FAIL;

    assert( mf->get_data_size() <= m_config.get_line_sz());

    // at this point, we will accept the request : access tags and immediately allocate line
    new_addr_type block_addr = m_config.block_addr(addr);
    unsigned cache_index = (unsigned)-1;
    enum cache_request_status status = m_tags.access(block_addr,time,cache_index);
    enum cache_request_status cache_status = RESERVATION_FAIL;
    assert( status != RESERVATION_FAIL );
    assert( status != HIT_RESERVED ); // as far as tags are concerned: HIT or MISS
    m_fragment_fifo.push( fragment_entry(mf,cache_index,status==MISS,mf->get_data_size()) );
    if ( status == MISS ) {
        // we need to send a memory request...
        unsigned rob_index = m_rob.push( rob_entry(cache_index, mf, block_addr) );
        m_extra_mf_fields[mf] = extra_mf_fields(rob_index);
        mf->set_data_size(m_config.get_line_sz());
        m_tags.fill(cache_index,time); // mark block as valid
        m_request_fifo.push(mf);
        mf->set_status(m_request_queue_status,time);
        events.push_back(READ_REQUEST_SENT);
        cache_status = MISS;
    } else {
        // the value *will* *be* in the cache already
        cache_status = HIT_RESERVED;
    }
    m_stats.inc_stats(mf->get_access_type(), m_stats.select_stats_status(status, cache_status));
    return cache_status;
}

void tex_cache::cycle(){
    // send next request to lower level of memory
    if ( !m_request_fifo.empty() ) {
        mem_fetch *mf = m_request_fifo.peek();
        if ( !m_memport->full(mf->get_ctrl_size(),false) ) {
            m_request_fifo.pop();
            m_memport->push(mf);
        }
    }
    // read ready lines from cache
    if ( !m_fragment_fifo.empty() && !m_result_fifo.full() ) {
        const fragment_entry &e = m_fragment_fifo.peek();
        if ( e.m_miss ) {
            // check head of reorder buffer to see if data is back from memory
            unsigned rob_index = m_rob.next_pop_index();
            const rob_entry &r = m_rob.peek(rob_index);
            assert( r.m_request == e.m_request );
            assert( r.m_block_addr == m_config.block_addr(e.m_request->get_addr()) );
            if ( r.m_ready ) {
                assert( r.m_index == e.m_cache_index );
                m_cache[r.m_index].m_valid = true;
                m_cache[r.m_index].m_block_addr = r.m_block_addr;
                m_result_fifo.push(e.m_request);
                m_rob.pop();
                m_fragment_fifo.pop();
            }
        } else {
            // hit:
            assert( m_cache[e.m_cache_index].m_valid );
            assert( m_cache[e.m_cache_index].m_block_addr
                == m_config.block_addr(e.m_request->get_addr()) );
            m_result_fifo.push( e.m_request );
            m_fragment_fifo.pop();
        }
    }
}

/// Place returning cache block into reorder buffer
void tex_cache::fill( mem_fetch *mf, unsigned time )
{
    extra_mf_fields_lookup::iterator e = m_extra_mf_fields.find(mf);
    assert( e != m_extra_mf_fields.end() );
    assert( e->second.m_valid );
    assert( !m_rob.empty() );
    mf->set_status(m_rob_status,time);

    unsigned rob_index = e->second.m_rob_index;
    rob_entry &r = m_rob.peek(rob_index);
    assert( !r.m_ready );
    r.m_ready = true;
    r.m_time = time;
    assert( r.m_block_addr == m_config.block_addr(mf->get_addr()) );
}

void tex_cache::display_state( FILE *fp ) const
{
    fprintf(fp,"%s (texture cache) state:\n", m_name.c_str() );
    fprintf(fp,"fragment fifo entries  = %u / %u\n",
        m_fragment_fifo.size(), m_fragment_fifo.capacity() );
    fprintf(fp,"reorder buffer entries = %u / %u\n",
        m_rob.size(), m_rob.capacity() );
    fprintf(fp,"request fifo entries   = %u / %u\n",
        m_request_fifo.size(), m_request_fifo.capacity() );
    if ( !m_rob.empty() )
        fprintf(fp,"reorder buffer contents:\n");
    for ( int n=m_rob.size()-1; n>=0; n-- ) {
        unsigned index = (m_rob.next_pop_index() + n)%m_rob.capacity();
        const rob_entry &r = m_rob.peek(index);
        fprintf(fp, "tex rob[%3d] : %s ",
            index, (r.m_ready?"ready  ":"pending") );
        if ( r.m_ready )
            fprintf(fp,"@%6u", r.m_time );
        else
            fprintf(fp,"       ");
        fprintf(fp,"[idx=%4u]",r.m_index);
        r.m_request->print(fp,false);
    }
    if ( !m_fragment_fifo.empty() ) {
        fprintf(fp,"fragment fifo (oldest) :");
        fragment_entry &f = m_fragment_fifo.peek();
        fprintf(fp,"%s:          ", f.m_miss?"miss":"hit ");
        f.m_request->print(fp,false);
    }
}
/**************************************** end of file *******************************************************/

