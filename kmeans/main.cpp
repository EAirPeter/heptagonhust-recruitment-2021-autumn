// ==================================
// IMPORTANT: DO NOT MODIFY THIS FILE
// ==================================
#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>

#include "kmeans.hpp"

using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::microseconds;

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cout << "Usage: ./kmeans <output_file.txt>";
    return 1;
  }
  int numPoints, numCenters;
  std::cin >> numPoints >> numCenters;

  std::vector<Point> points(numPoints);
  std::vector<Point> centers(numCenters);
  for (int i = 0; i < numCenters; i++)
    std::cin >> centers[i];
  for (int i = 0; i < numPoints; i++)
    std::cin >> points[i];

  std::shared_ptr<Kmeans> problem(new Kmeans(points, centers));

  auto start = high_resolution_clock::now();
  auto result = problem->Run();
  auto end = high_resolution_clock::now();
  std::cout
    << "Time cost: "
    << (double) duration_cast<microseconds>(end - start).count() / 1000000.0
    << "s" << std::endl;

  std::cout << "Writing output files..." << std::endl;
  std::ofstream ofs;
  ofs.open(argv[1]);
  for (int i = 0; i < numPoints; i++)
    ofs << points[i] << " " << result[i] << std::endl;
  ofs.close();
  return 0;
}