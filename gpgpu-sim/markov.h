//-version 5
//-2016.06.03 
//-pure LRU algorithm.

#include <stdio.h>  
#include <list>
#include <map>
#include <string>
#include <limits.h>
#include <algorithm>
#ifndef MARKOV_H
#define MARKOV_H

struct MKVCandidate{
        unsigned int  m_addr;
        unsigned int  m_count;
        unsigned int  m_create_time;
        unsigned int  m_update_time;

        MKVCandidate( unsigned int addr, unsigned int count, unsigned time){
            m_addr  = addr ;
            m_count = count ;
            m_create_time  = time;
            m_update_time  = time;
        }
};
struct MKVEntry{
    std::list<MKVCandidate> m_list;
    unsigned int  m_create_time;    
    unsigned int  m_update_time;
    MKVEntry(unsigned time){
        m_create_time  = time;
        m_update_time  = time;
    }
};

class MKVTable{
    private:
        std::map<  unsigned int, MKVEntry> m_table;
        int m_table_maxsize;
        int m_entry_maxlen;
        int m_table_last_cut_time ;
        int m_entry_last_cut_time ;
        int m_table_cut_interval;
        int m_entry_cut_interval;
        int m_table_batch_cut_interval;
    public:

        unsigned int m_process_count;
        //- 4 params;
        MKVTable(int tsize, int esize, int table_cut_inter, int entry_cut_inter, int table_batch_cut_inter){
            m_table_maxsize = tsize;
            m_entry_maxlen = esize;
            m_process_count = 0;
            m_table_last_cut_time = 0;
            m_entry_last_cut_time = 0;
            m_table_cut_interval = table_cut_inter;
            m_entry_cut_interval = entry_cut_inter;
            m_table_batch_cut_interval = table_batch_cut_inter;
        }
        ~MKVTable ()
        {
            printf("destroy class MKVTable obj[%X]\n ", this);
            std::map<  unsigned int, MKVEntry > ::iterator   mapit;

            for(mapit = m_table.begin(); mapit != m_table.end(); mapit++) 
            { 
                mapit->second.m_list.clear();
            }
            m_table.clear();
            printf("all list/map cleared \n");

        }
        //- use update_time to del entry;
        void evict_candi_LRU(MKVEntry &en,int time){
            m_entry_last_cut_time= time;

            std::list<MKVCandidate>::iterator lit;
            std::list<MKVCandidate>::iterator lit_oldest = en.m_list.end();
            
            unsigned oldest=UINT_MAX;
            for( lit = en.m_list.begin(); lit != en.m_list.end() ;lit++)
                if(lit->m_update_time < oldest ){
                    oldest = lit->m_update_time;
                    lit_oldest =lit;
                }
            //-travel this list.
            if(lit_oldest != en.m_list.end() )
                en.m_list.erase(lit_oldest);                //-cut a record with oldest time in every list;
        }
        //-use update time;
        void evict_entry_LRU(unsigned time)
        {
            m_table_last_cut_time= time;

            std::map<  unsigned int, MKVEntry > ::iterator   mapit, map_oldest;
            unsigned oldest=UINT_MAX;
            map_oldest= m_table.end();
            for (mapit = m_table.begin() ; mapit != m_table.end(); mapit ++ ){
                  if( oldest > mapit->second.m_update_time  ){          
                      oldest = mapit->second.m_update_time;    //-find the oldest , smallest;
                      map_oldest = mapit;  
                  }
            }
            if(map_oldest != m_table.end())
                m_table.erase(map_oldest);
        }
        //-update this list with second addr.(count or add in)
        //- 3 result: (1) hit,and count++; (2) miss & free_room, insert a candi; (3) miss & no room. do nothing;
        int update_entry( MKVEntry &en,  unsigned int second_addr, unsigned time){
            std::list<MKVCandidate>::iterator it ;
            for( it=en.m_list.begin(); it != en.m_list.end() ; it++ )
                if(it->m_addr == second_addr )
                    break;
            if(it != en.m_list.end()){ //- second_addr hit 
                it->m_count++;    
                it->m_update_time = time;                  //-update candidate's count & m_update_time.
                return 1;
            }
            else {      //- second_addr miss, add the candi
                if(en.m_list.size() >= m_entry_maxlen){  
                    //-replace the LRU candi
                    evict_candi_LRU(en, time);
                }
                MKVCandidate temp(second_addr, 1, time);   //<-----------------create a Candidate obj with now time;
                en.m_list.push_back( temp );
                return 1;
            }

            return 0;

        }

        //- 3 result: (1) hit,call update_entry(); (2) miss & free_room, insert a entry; (3) miss & no room. do nothing;
        int update_table( unsigned int first_addr,  unsigned int second_addr, unsigned time){
            m_process_count ++; 
            if( first_addr == second_addr){
                return 0 ;
            }
            if(first_addr<=0 || second_addr <=0) 
                return 0;//- not valid addr. or duplicate addr. return 0

            std::map<  unsigned int, MKVEntry > ::iterator   mapit;
            mapit=m_table.find(first_addr);
            if(mapit != m_table.end() ){ //- hit in map;
                mapit->second.m_update_time = time;           //-update this entry m_update_time when hit a entry;
                return update_entry( mapit->second , second_addr, time);
            }
            else {//-if miss, add in
                if(m_table.size() >= m_table_maxsize){ //-if no room, replace the oldest entry; 
                    //-LRU, cut a entry;
                    evict_entry_LRU(time);
                }
                MKVCandidate tmp_candi(second_addr,1,time);   //<-------------------------- create a Candidate obj
                MKVEntry tmp_entry(time);                     //<-------------------------- create a Entry obj
                tmp_entry.m_list.push_back( tmp_candi );

                m_table.insert(std::make_pair(first_addr ,tmp_entry )); 
                mapit->second.m_update_time = time;           //-update this entry m_update_time when insert a new entry;
                return 1;
            }
            return 0; //-no room for store, rerurn 0;
        }

        void show(std::string mess){
            std::map<  unsigned int, MKVEntry > ::iterator   mapit;
            int entry_count=0;              //-count lists
            int all_candidate_count=0;      //-count all candis
            int cumulate_candidate_fre=0;   //- all candis * frequency
            unsigned entry_smallest_updatetime=UINT_MAX;
            unsigned entry_bigest_updatetime=0;
            //printf("-----------------------------------------------\n");
            //printf("<........ ,cre, upd> :........count  cre  upd   :   %\n");
            for(mapit = m_table.begin(); mapit != m_table.end(); mapit++) 
            { 
               // printf("<%8X ,%3u, %3u> :",mapit->first,
               //         mapit->second.m_create_time, mapit->second.m_update_time) ;
                entry_count++;
                if(entry_smallest_updatetime > mapit->second.m_update_time)
                    entry_smallest_updatetime = mapit->second.m_update_time;
                if(entry_bigest_updatetime < mapit->second.m_update_time)
                    entry_bigest_updatetime = mapit->second.m_update_time;

                std::list<MKVCandidate>::iterator lit;
                int each_candidate_count=0;
                for( lit = (mapit->second).m_list.begin(); lit != (mapit->second).m_list.end() ;lit++){
                    //printf("%8X{%3d}[%3d][%3d]  ", lit->m_addr, lit->m_count, lit->m_create_time, lit->m_update_time);
                    cumulate_candidate_fre += lit->m_count;
                    each_candidate_count++;
                    all_candidate_count++;
                }
               // printf(" :{%5.2f}\n",(float)each_candidate_count/m_entry_maxlen );
            }
            printf("%s table%= %5.2f  aver_entry%= %5.2f  exictEntry_upda[ %8u %8u ] EntryCut[%8u] CandiCut[%8u] exict_fre= %d\n",
                mess.c_str(),(float)entry_count/m_table_maxsize, (float)all_candidate_count/(entry_count * m_entry_maxlen),
                entry_smallest_updatetime, entry_bigest_updatetime, 
                m_table_last_cut_time, m_entry_last_cut_time, cumulate_candidate_fre); 
           // printf("-----------------------------------------------\n");
        }
        unsigned int  find_max(unsigned int in, unsigned int time ){
            unsigned int max_next_addr=0;
            unsigned int max_count=0;
            std::map<  unsigned int, MKVEntry > ::iterator   mapit;
            mapit=m_table.find(in);
            if(mapit != m_table.end() ){
                std::list<MKVCandidate> find_list=mapit->second.m_list ;
                std::list<MKVCandidate>::iterator lit;
                for( lit = find_list.begin(); lit != find_list.end() ;lit++){
                    if(lit->m_count > max_count ){
                        max_count=lit->m_count;
                        max_next_addr= lit->m_addr;
                    }
                }
            }
            return max_next_addr;
        }

}; //-class
#endif
