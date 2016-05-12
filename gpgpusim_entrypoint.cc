// Copyright (c) 2009-2011, Tor M. Aamodt, Wilson W.L. Fung, Ivan Sham,
// Andrew Turner, Ali Bakhoda, The University of British Columbia
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

#include "gpgpusim_entrypoint.h"
#include <stdio.h>

#include "option_parser.h"
#include "cuda-sim/cuda-sim.h"
#include "cuda-sim/ptx_ir.h"
#include "cuda-sim/ptx_parser.h"
#include "gpgpu-sim/gpu-sim.h"
#include "gpgpu-sim/icnt_wrapper.h"
#include "stream_manager.h"

#include <pthread.h>
#include <semaphore.h>

#define MAX(a,b) (((a)>(b))?(a):(b))



struct gpgpu_ptx_sim_arg *grid_params;

sem_t g_sim_signal_start;
sem_t g_sim_signal_finish;
sem_t g_sim_signal_exit;
time_t g_simulation_starttime;
pthread_t g_simulation_thread;

gpgpu_sim_config g_the_gpu_config;
gpgpu_sim *g_the_gpu;
stream_manager *g_stream_manager;



static int sg_argc = 3;
static const char *sg_argv[] = {"", "-config","gpgpusim.config"};



static void print_simulation_time();

void *gpgpu_sim_thread_sequential(void*)
{
   // at most one kernel running at a time
   bool done;
   do {
      sem_wait(&g_sim_signal_start);
      done = true;
      if( g_the_gpu->get_more_cta_left() ) {
          done = false;
          g_the_gpu->init();
          while( g_the_gpu->active() ) {
              g_the_gpu->cycle();
              g_the_gpu->deadlock_check();
          }
          g_the_gpu->print_stats();
          g_the_gpu->update_stats();
          print_simulation_time();
      }
      sem_post(&g_sim_signal_finish);
   } while(!done);
   sem_post(&g_sim_signal_exit);
   return NULL;
}

pthread_mutex_t g_sim_lock = PTHREAD_MUTEX_INITIALIZER;
bool g_sim_active = false;
bool g_sim_done = true; // outer while condition, means program finished,and over.

void *gpgpu_sim_thread_concurrent(void*)
{
	system("echo -e \"\\033[1;33m *** gpgpusim_entrypoint.cc:gpgpu_sim_thread_concurrent() \\033[0m\" ");
	bool gpu_should_exit=false;
	bool gpu_max_limit_set=g_the_gpu->has_max_limit();
	int inwhile=0,outwhile=0;
    // concurrent kernel execution simulation thread
    do { // outer while
       if(g_debug_execution >= 3) {
          printf("GPGPU-Sim: *** simulation thread starting and spinning waiting for work ***\n");
          fflush(stdout);
       }
        while( g_stream_manager->empty_protected() && !g_sim_done )
            ;
        if(g_debug_execution >= 3) {
           printf("GPGPU-Sim: ** START simulation thread (detected work) **\n");
           g_stream_manager->print(stdout);
           fflush(stdout);
        }
        pthread_mutex_lock(&g_sim_lock);
        g_sim_active = true; //  is assinged to 'false' near outer while()
        pthread_mutex_unlock(&g_sim_lock);
        bool active = false; // inner loop conditon ,be set to ture in 136,near the  inner while()
        bool sim_cycles = false;
        g_the_gpu->init(); // init before every inner while.
        do {/* inner while*/
            // check if a kernel has completed
            // launch operation on device if one is pending and can be run

            // Need to break this loop when a kernel completes. This was a
            // source of non-deterministic behaviour in GPGPU-Sim (bug 147).
            // If another stream operation is available, g_the_gpu remains active,
            // causing this loop to not break. If the next operation happens to be
            // another kernel, the gpu is not re-initialized and the inter-kernel
            // behaviour may be incorrect. Check that a kernel has finished and
            // no other kernel is currently running.
            if(g_stream_manager->operation(&sim_cycles) && !g_the_gpu->active())//-call m_gpu->print_stats(),output result
                break; // go out of inner while

            if( g_the_gpu->active() ) {
                g_the_gpu->cycle();
                sim_cycles = true;
                g_the_gpu->deadlock_check();
            }
            active=g_the_gpu->active() || !g_stream_manager->empty_protected();
            inwhile++;
        } while( active );// inner while;

        if(g_debug_execution >= 3) {
           printf("GPGPU-Sim: ** STOP simulation thread (no work) **\n");
           fflush(stdout);
        }
        if(sim_cycles) { //- if gpu pass a cycle ,update the status.
            g_the_gpu->update_stats();
            print_simulation_time();
        }
        pthread_mutex_lock(&g_sim_lock);// the lock can stop the program to continue;
        g_sim_active = false;
        pthread_mutex_unlock(&g_sim_lock);
        outwhile++;
        fprintf(stderr,"[@@@]out while=%d, in while=%d \n",outwhile,inwhile); //-grep only dule with stdout.this mess can show without disturbed by grep.
    } while( !g_sim_done ); // outer while

	system("echo -e \"\\033[1;33m ******* end of outer while ********\\033[0m\" ");

    if(g_debug_execution >= 3) {
       printf("GPGPU-Sim: *** simulation thread exiting ***\n");
       fflush(stdout);
    }
    sem_post(&g_sim_signal_exit);
	system("echo -e \"\\033[1;33m ******** end of all gpusim? **********\\033[0m\" ");
    return NULL;
}

void synchronize()
{
    printf("GPGPU-Sim: synchronize waiting for inactive GPU simulation\n");
    g_stream_manager->print(stdout);
    fflush(stdout);
//    sem_wait(&g_sim_signal_finish);
    bool done = false;
    do {
        pthread_mutex_lock(&g_sim_lock);
        done = g_stream_manager->empty() && !g_sim_active;
        pthread_mutex_unlock(&g_sim_lock);
    } while (!done);
    printf("GPGPU-Sim: detected inactive GPU simulation thread\n");
    fflush(stdout);
//    sem_post(&g_sim_signal_start);
}

void exit_simulation()
{
    g_sim_done=true;
    printf("GPGPU-Sim: exit_simulation called\n");
    fflush(stdout);
    sem_wait(&g_sim_signal_exit);
    printf("GPGPU-Sim: simulation thread signaled exit\n");
    fflush(stdout);
}

extern bool g_cuda_launch_blocking;

int  g_prefetch_rec_max_size=0; //-defined and inited here, used in "shader.cc:604"
int  g_prefetch_interval    =0; //-defined and inited here, used in "shader.cc:605"
int  g_prefetch_length      =0; //-defined and inited here, used in "shader.cc:606"
int  g_prefetch_mode        =1; //-0x01= PRE_ON_HIT; 0x10=PRE_ON_MISS; 0x11=PRE_ON_ALL (ON_XXX is used in cahce mode)

long g_prefetch_mem         =0; //-defined and inited here, used in "shader.cc:607","gpu-sim.cc":883
long g_prefetch             =0; //-defined and inited here, used in "shader.cc:607","gpu-sim.cc":883
long g_fetch                =0; //-defined and inited here, used in "shader.cc:607","gpu-sim.cc":883
long g_fetch_stage_cycles   =0; //-

#include <fstream> 
#include <cstdlib>
using namespace std;
void read_my_config(){ //-read my own global vars.
    ifstream myfile("my_options.config"); //- open file.
    if(!myfile){
        printf("Unable to open myfile\n");
        exit(1);// -quit sim ,return to OS.
    }
    else 
        printf("Open my_options.config OK!\n");
    char buffer[256];
    char option_name[20];
    int option_value;
    // get rec len
    myfile.getline (buffer,256); 
    sscanf(buffer,"%s %d" , &option_name , &option_value);
    g_prefetch_rec_max_size = option_value; //- init global var 'g_prefetch_rec_max_size'
    printf("my_options.config :{%s   %d} g_prefetch_rec_max_size=%d\n",option_name,option_value,g_prefetch_rec_max_size);
    // get begin pos
    myfile.getline (buffer,256); 
    sscanf(buffer,"%s %d" , &option_name , &option_value);
    g_prefetch_interval = option_value; //- init global var 'g_prefetch_interval'
    printf("my_options.config : {%s   %d}      g_prefetch_interval =%d\n", option_name , option_value,g_prefetch_interval);
    // get prefetch length
    myfile.getline (buffer,256); 
    sscanf(buffer,"%s %d" , &option_name , &option_value);
    g_prefetch_length = option_value; //- init global var 'g_prefetch_length'
    printf("my_options.config : {%s   %d}      g_prefetch_length=%d\n", option_name , option_value,g_prefetch_length);
    // get prefetch mode
    myfile.getline (buffer,256); 
    sscanf(buffer,"%s %d" , &option_name , &option_value);
    g_prefetch_mode = option_value; //- init global var 'g_prefetch_mode'
    printf("my_options.config : {%s   %d}      g_prefetch_mode=%d\n", option_name , option_value,g_prefetch_mode);
    myfile.close();
    printf("rec max size = %d; prefetch interval = %d; prefetch length=%d; prefetch mode=%d\n",g_prefetch_rec_max_size,g_prefetch_interval,g_prefetch_length, g_prefetch_mode);
}
gpgpu_sim *gpgpu_ptx_sim_init_perf()
{
	system("echo -e \"\\033[1;33m ******** gpgpu_ptx_sim_init_pref() begin **********\\033[0m\" ");
   read_my_config();//-read my own vars from file 'my_options.config'
   srand(1);
   print_splash();
   read_sim_environment_variables(); //-read shell vars
   read_parser_environment_variables();
   option_parser_t opp = option_parser_create();

   icnt_reg_options(opp);
   g_the_gpu_config.reg_options(opp); // register GPU microrachitecture options
   ptx_reg_options(opp);
   ptx_opcocde_latency_options(opp);
   option_parser_cmdline(opp, sg_argc, sg_argv); // parse configuration options
   fprintf(stdout, "GPGPU-Sim: Configuration options:\n\n");
   option_parser_print(opp, stdout); // output large txt message of config.
   printf("******************************  option_parser_print() over**********\n\n");//cjllean
   // Set the Numeric locale to a standard locale where a decimal point is a "dot" not a "comma"
   // so it does the parsing correctly independent of the system environment variables
   assert(setlocale(LC_NUMERIC,"C"));
   g_the_gpu_config.init(); // output large txt 
   printf("***************************gpu_config.init() over *********\n\n");//cjllean   
   g_the_gpu = new gpgpu_sim(g_the_gpu_config); // new many thread ......[Thread ox7ffff35c3700 (LWP 15690)]  [... exited ]
   g_stream_manager = new stream_manager(g_the_gpu,g_cuda_launch_blocking);

   printf("*************************** g_the_gpu/ g_stream_manager newed *********\n\n");//cjllean   
   g_simulation_starttime = time((time_t *)NULL);//-global vars, inited when g_the_gpu new().

   sem_init(&g_sim_signal_start,0,0);
   sem_init(&g_sim_signal_finish,0,0);
   sem_init(&g_sim_signal_exit,0,0);
   printf("***************************************** sim_init() over *********\n\n");//cjllean
   system("echo -e \"\\033[1;33m ******** gpgpu_ptx_sim_init_pref() over **********\\033[0m\" ");

   return g_the_gpu;
}

void start_sim_thread(int api)
{
    if( g_sim_done ) {
        g_sim_done = false;// not finish 
        if( api == 1 ) {
   system("echo -e \"\\033[1;33m ******** New Thread created ! **********\\033[0m\" ");
           pthread_create(&g_simulation_thread,NULL,gpgpu_sim_thread_concurrent,NULL);

        } else {
           pthread_create(&g_simulation_thread,NULL,gpgpu_sim_thread_sequential,NULL);
        }
    }
}

void print_simulation_time()
{
   time_t current_time, difference, d, h, m, s;
   current_time = time((time_t *)NULL);
   difference   = MAX(current_time - g_simulation_starttime, 1);//-all kernels accmulated time.

   d = difference/(3600*24);
   h = difference/3600 - 24*d;
   m = difference/60 - 60*(h + 24*d);
   s = difference - 60*(m + 60*(h + 24*d));

   fflush(stderr);
   printf("\ngpgpu_simulation_time     = %u days, %u hrs, %u min, %u sec (total %u sec)\n",
          (unsigned)d, (unsigned)h, (unsigned)m, (unsigned)s, (unsigned)difference );
   printf("gpgpu_simulation_rate     = %u (inst/sec)\n", (unsigned)(g_the_gpu->gpu_tot_sim_insn / difference) );
   printf("gpgpu_simulation_rate     = %u (cycle/sec)\n\n", (unsigned)(gpu_tot_sim_cycle / difference) );
   fflush(stdout); //-performence simulate output.
}

int gpgpu_opencl_ptx_sim_main_perf( kernel_info_t *grid )
{
   g_the_gpu->launch(grid);
   sem_post(&g_sim_signal_start);
   sem_wait(&g_sim_signal_finish);
   return 0;
}

//! Functional simulation of OpenCL
/*!
 * This function call the CUDA PTX functional simulator
 */
int gpgpu_opencl_ptx_sim_main_func( kernel_info_t *grid )
{
    //calling the CUDA PTX simulator, sending the kernel by reference and a flag set to true,
    //the flag used by the function to distinguish OpenCL calls from the CUDA simulation calls which
    //it is needed by the called function to not register the exit the exit of OpenCL kernel as it doesn't register entering in the first place as the CUDA kernels does
   gpgpu_cuda_ptx_sim_main_func( *grid, true );
   return 0;
}
