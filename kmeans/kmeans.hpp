#ifndef __KMEANS_HPP_
#define __KMEANS_HPP_

#include <iosfwd>
#include <memory>
#include <vector>

using index_t = int;

struct Point
{
	double x, y;

	Point()
		: x(0), y(0)
	{
	}

	Point(int InX, int InY)
		: x(InX), y(InY)
	{
	}

	Point(const Point&) = default;
	~Point() = default;

	[[nodiscard]]
	double Distance(const Point& Other) const noexcept;
};

std::istream& operator>>(std::istream& LHS, Point& RHS);
std::ostream& operator<<(std::ostream& LHS, const Point& RHS);

struct FKMeans;

struct FKMeansDeleter : std::default_delete<FKMeans>
{
	void operator()(FKMeans* Obj) const noexcept;
};

// Public interface
class Kmeans
{
public:
	Kmeans(const std::vector<Point>& InPoints, const std::vector<Point>& InInitCenters);
	std::vector<index_t> Run(int MaxIterations = 1000);

private:
	std::unique_ptr<FKMeans, FKMeansDeleter> Impl;
};

#endif // __KMEANS_HPP_
