#ifndef __KMEANS_HPP_
#define __KMEANS_HPP_

#include <iostream>
#include <vector>

#ifdef _MSC_VER
#define FORCEINLINE __forceinline
#else
#define FORCEINLINE __attribute__((always_inline))
#endif

using index_t = int;

struct Point
{
	union
	{
		double x;
		double X;
	};

	union
	{
		double y;
		double Y;
	};

	FORCEINLINE Point() noexcept = default;
	Point(int InX, int InY) : X(InX), Y(InY) {}

	[[nodiscard]]
	double Distance(const Point& Other) const noexcept;
};

std::istream& operator>>(std::istream& LHS, Point& RHS);
std::ostream& operator<<(std::ostream& LHS, const Point& RHS);

namespace Solution
{
	using FIndex = ::index_t;
	using FPoint = ::Point;

	template<class RElement>
	using TVector = ::std::vector<RElement>;

	struct FKMeans
	{
		TVector<FPoint> Points;
		TVector<FPoint> Centers;
		FIndex NumPoint;
		FIndex NumCenter;

		FKMeans(const TVector<FPoint>& InPoints, const TVector<FPoint>& InInitCenters);

		TVector<FIndex> Run(int MaxIterations = 1000);
	};
}

// Public interface
class Kmeans : private ::Solution::FKMeans
{
public:
	using FKMeans::FKMeans;
	using FKMeans::Run;
};

#endif // __KMEANS_HPP_
