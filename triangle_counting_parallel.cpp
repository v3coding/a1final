#include "core/graph.h"
#include "core/utils.h"
#include <future>
#include <iomanip>
#include <iostream>
#include <stdlib.h>
#include <thread>
#include <cmath>

static int numberOfThreads;

long countTriangles(uintV *array1, uintE len1, uintV *array2, uintE len2,
                     uintV u, uintV v) {

  uintE i = 0, j = 0; // indexes for array1 and array2
  long count = 0;

  if (u == v)
    return count;

  while ((i < len1) && (j < len2)) {
    if (array1[i] == array2[j]) {
      if ((array1[i] != u) && (array1[i] != v)) {
        count++;
      }
      i++;
      j++;
    } else if (array1[i] < array2[j]) {
      i++;
    } else {
      j++;
    }
  }
  return count;
}

void triangleCountParallel(uintV start, uintV finish, Graph &g, uintV n, uintV& triangle_count,double& time_taken){
  timer t2;
  t2.start();
    for (uintV u = start; u < finish; u++) {
    // For each outNeighbor v, find the intersection of inNeighbor(u) and
    // outNeighbor(v)
    uintE out_degree = g.vertices_[u].getOutDegree();
    for (uintE i = 0; i < out_degree; i++) {
      uintV v = g.vertices_[u].getOutNeighbor(i);
      triangle_count += countTriangles(g.vertices_[u].getInNeighbors(),
                                       g.vertices_[u].getInDegree(),
                                       g.vertices_[v].getOutNeighbors(),
                                       g.vertices_[v].getOutDegree(), u, v);
    }
  }
  time_taken = t2.stop();
}

void triangleCountSerial(Graph &g) {
  uintV n = g.n_;
  long triangle_count = 0;
  double time_taken = 0.0;
  timer t1;
  uintV Counter[numberOfThreads];
  for(int i = 0; i < numberOfThreads; i++){
    Counter[i] = 0;
  }
  // The outNghs and inNghs for a given vertex are already sorted

  // Create threads and distribute the work across T threads
  // -------------------------------------------------------------------
  double numThreads = (double) numberOfThreads;
  double numVertexes = (double) n;
  double start = 0;
  double finish = 0;
  uintV startCaster = 0;
  uintV finishCaster = 0;
  std::thread threads[numberOfThreads];
  double timers[numberOfThreads];
  for(int i = 0; i < numberOfThreads; i++){
    timers[i] = 0;
  }

  t1.start();
  // Process each edge <u,v>
    for (int i = 0; i < numberOfThreads; i++){
    if(((start + numVertexes) / numThreads) < n){
      finish = start + (numVertexes / numThreads);
    }
    else {
      finish = numVertexes;
    }
    startCaster = (uintV) trunc(start);
    finishCaster = (uintV) trunc(finish);
    threads[i] = std::thread(triangleCountParallel,startCaster,finishCaster,std::ref(g),n,std::ref(Counter[i]),std::ref(timers[i]));
    start = finish;
  }
  for(int i = 0; i < numberOfThreads; i++){
    threads[i].join();
  }

  time_taken = t1.stop();

  triangle_count = 0;
  for(int i = 0; i < numberOfThreads; i++){
    triangle_count += Counter[i];
  }
  // -------------------------------------------------------------------
  // Here, you can just print the number of non-unique triangles counted by each
  // thread std::cout << "thread_id, triangle_count, time_taken\n"; Print the
  // above statistics for each thread Example output for 2 threads: thread_id,
  // triangle_count, time_taken 1, 102, 0.12 0, 100, 0.12
  for(int i = 0; i < numberOfThreads; i++){
    std::cout << i << ", " << Counter[i] << ", " << timers[i] << " " << std::endl; 
  }

  // Print the overall statistics
  std::cout << "Number of triangles : " << triangle_count << "\n";
  std::cout << "Number of unique triangles : " << triangle_count / 3 << "\n";
  std::cout << "Time taken (in seconds) : " << std::setprecision(TIME_PRECISION)
            << time_taken << "\n";
}

int main(int argc, char *argv[]) {
  cxxopts::Options options(
      "triangle_counting_serial",
      "Count the number of triangles using serial and parallel execution");
  options.add_options(
      "custom",
      {
          {"nWorkers", "Number of workers",
           cxxopts::value<uint>()->default_value(DEFAULT_NUMBER_OF_WORKERS)},
          {"inputFile", "Input graph file path",
           cxxopts::value<std::string>()->default_value(
               "/scratch/input_graphs/roadNet-CA")},
      });

  auto cl_options = options.parse(argc, argv);
  uint n_workers = cl_options["nWorkers"].as<uint>();
  std::string input_file_path = cl_options["inputFile"].as<std::string>();
  std::cout << std::fixed;
  std::cout << "Number of workers : " << n_workers << "\n";
  numberOfThreads = n_workers;

  Graph g;
  std::cout << "Reading graph\n";
  g.readGraphFromBinary<int>(input_file_path);
  std::cout << "Created graph\n";

  triangleCountSerial(g);

  return 0;
}
