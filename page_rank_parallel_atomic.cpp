#include "core/graph.h"
#include "core/utils.h"
#include <iomanip>
#include <iostream>
#include <stdlib.h>
#include <thread>
#include <cmath>

#ifdef USE_INT
#define INIT_PAGE_RANK 100000
#define EPSILON 1000
#define PAGE_RANK(x) (15000 + (5 * x) / 6)
#define CHANGE_IN_PAGE_RANK(x, y) std::abs(x - y)
typedef int64_t PageRankType;
#else
#define INIT_PAGE_RANK 1.0
#define EPSILON 0.01
#define DAMPING 0.85
#define PAGE_RANK(x) (1 - DAMPING + DAMPING * x)
#define CHANGE_IN_PAGE_RANK(x, y) std::fabs(x - y)
typedef float PageRankType;
#endif

static int numberOfThreads;
static CustomBarrier* my_barrier;

void pageRankParallel(uintV k, uintV n, int max_iters, std::atomic<PageRankType> *pr_curr, std::atomic<PageRankType> *pr_next, Graph &g, double& time_taken)
{
  timer t1;
  time_taken = 0.0;
  PageRankType tempRank;
  std::thread::id this_id = std::this_thread::get_id();

  t1.start();
  for (int iter = 0; iter < max_iters; iter++){
    for (uintV u = k; u < n; u++){
      uintE out_degree = g.vertices_[u].getOutDegree();
      for (uintE i = 0; i < out_degree; i++)
      {
        uintV v = g.vertices_[u].getOutNeighbor(i);
        tempRank = pr_next[v];
        while (!pr_next[v].compare_exchange_weak(tempRank, pr_next[v] + (pr_curr[u] / out_degree))){}
      }
    }
    my_barrier->wait();
    for (uintV v = k; v < n; v++){
      pr_next[v] = PAGE_RANK(pr_next[v]);
      pr_curr[v].store(pr_next[v]);
      pr_next[v] = 0.0;
    }
    my_barrier->wait();
  }
  time_taken = t1.stop();
}

void pageRankSerial(Graph &g, int max_iters)
{
  uintV n = g.n_;

  std::atomic<PageRankType> *pr_curr = new std::atomic<PageRankType>[n];
  std::atomic<PageRankType> *pr_next = new std::atomic<PageRankType>[n];
  my_barrier = new CustomBarrier(numberOfThreads);
  std::thread threads[numberOfThreads];

  for (uintV i = 0; i < n; i++)
  {
    pr_curr[i] = INIT_PAGE_RANK;
    pr_next[i] = 0.0;
  }

  double times[numberOfThreads];
  double time_taken = 0;

  double numThreads = (double) numberOfThreads;
  double numVertexes = (double) n;
  double start = 0;
  double finish = 0;
  uintV startCaster = 0;
  uintV finishCaster = 0;

  timer t2;
  t2.start();
  for (int i = 0; i < numberOfThreads; i++){
    if(((start + numVertexes) / numThreads) < n){
      finish = start + (numVertexes / numThreads);
    }
    else {
      finish = numVertexes;
    }
    startCaster = (uintV) trunc(start);
    finishCaster = (uintV) trunc(finish);
    threads[i] = std::thread(pageRankParallel, startCaster, finishCaster, max_iters, std::ref(pr_curr), std::ref(pr_next), std::ref(g), std::ref(times[i]));
    start = finish;
  }
  for(int i = 0; i < numberOfThreads; i++){
    threads[i].join();
  }
  time_taken = t2.stop();

  // -------------------------------------------------------------------
  // std::cout << "thread_id, time_taken\n";
  // Print the above statistics for each thread
  // Example output for 2 threads:
  // thread_id, time_taken
  // 0, 0.12
  // 1, 0.12
  for (int i = 0; i < numberOfThreads; i++)
  {
    std::cout << i << " " << times[i] << std::endl;
  }
//
  PageRankType sum_of_page_ranks = 0;
  for (uintV u = 0; u < n; u++)
  {
    sum_of_page_ranks += pr_curr[u];
  }
  std::cout << "Sum of page rank : " << sum_of_page_ranks << "\n";
  std::cout << "Time taken (in seconds) : " << time_taken << "\n";
  delete[] pr_curr;
  delete[] pr_next;
}

int main(int argc, char *argv[])
{
  cxxopts::Options options(
      "page_rank_push",
      "Calculate page_rank using serial and parallel execution");
  options.add_options(
      "",
      {
          {"nWorkers", "Number of workers",
           cxxopts::value<uint>()->default_value(DEFAULT_NUMBER_OF_WORKERS)},
          {"nIterations", "Maximum number of iterations",
           cxxopts::value<uint>()->default_value(DEFAULT_MAX_ITER)},
          {"inputFile", "Input graph file path",
           cxxopts::value<std::string>()->default_value(
               "/scratch/input_graphs/roadNet-CA")},
      });

  auto cl_options = options.parse(argc, argv);
  uint n_workers = cl_options["nWorkers"].as<uint>();
  uint max_iterations = cl_options["nIterations"].as<uint>();
  std::string input_file_path = cl_options["inputFile"].as<std::string>();
  numberOfThreads = n_workers;

#ifdef USE_INT
  std::cout << "Using INT\n";
#else
  std::cout << "Using FLOAT\n";
#endif
  std::cout << std::fixed;
  std::cout << "Number of workers : " << n_workers << "\n";

  Graph g;
  std::cout << "Reading graph\n";
  g.readGraphFromBinary<int>(input_file_path);
  std::cout << "Created graph\n";
  pageRankSerial(g, max_iterations);

  return 0;
}
