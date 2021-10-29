#include "kmeans.hpp"

#include <algorithm>
#include <execution>
#include <queue>
#include <limits>

double Point::Distance(const Point& Other) const noexcept
{
	const double A = X - Other.X;
	const double B = Y - Other.Y;
	return A * A + B * B;
}

std::istream& operator>>(std::istream& LHS, Point& RHS)
{
	return LHS >> RHS.X >> RHS.Y;
}

std::ostream& operator<<(std::ostream& LHS, const Point& RHS)
{
	return LHS << RHS.X << " " << RHS.Y;
}

namespace Solution
{
	FKMeans::FKMeans(const TVector<FPoint>& InPoints, const TVector<FPoint>& InInitCenters)
		: Points(InPoints)
		, Centers(InInitCenters)
		, NumPoint(static_cast<FIndex>(InPoints.size()))
		, NumCenter(static_cast<FIndex>(InInitCenters.size()))
	{}

	TVector<FIndex> FKMeans::Run(int MaxIterations)
	{
		using std::swap;

		// The return vector
		TVector<FIndex> Assignment(NumPoint);
		TVector<FIndex> AssignmentBak(NumPoint);

		int CurrIteration = 0;
		std::cout << "Running kmeans with num points = " << NumPoint
			<< ", num centers = " << NumCenter
			<< ", max iterations = " << MaxIterations << "...\n";

		TVector<FIndex> PointCount(NumCenter);
		while (MaxIterations--)
		{
			++CurrIteration;

			for (FIndex I = 0; I < NumPoint; ++I)
			{
				const FPoint& PointI = Points[I];
				double MinDis = std::numeric_limits<double>::max();
				FIndex AssignmentI = -1;
				for (int K = 0; K < NumCenter; ++K)
				{
					const FPoint& CenterK = Centers[K];
					const double Dis = PointI.Distance(CenterK);
					if (Dis < MinDis)
					{
						MinDis = Dis;
						AssignmentI = K;
					}
				}
				Assignment[I] = AssignmentI;
			}

			if (!__builtin_memcmp(Assignment.data(), AssignmentBak.data(), sizeof(FIndex) * NumPoint))
			{
				goto JConverge;
			}

			__builtin_memset(Centers.data(), 0, sizeof(FPoint) * NumCenter);
			__builtin_memset(PointCount.data(), 0, sizeof(FIndex) * NumCenter);

			for (FIndex I = 0; I < NumPoint; ++I)
			{
				const FIndex ClusterI = Assignment[I];
				Centers[ClusterI].X += Points[I].X;
				Centers[ClusterI].Y += Points[I].Y;
				PointCount[ClusterI]++;
			}

			for (FIndex J = 0; J < NumCenter; ++J)
			{
				Centers[J].X /= PointCount[J];
				Centers[J].Y /= PointCount[J];
			}

			swap(Assignment, AssignmentBak);
		}

	JConverge:
		std::cout << "Finished in " << CurrIteration << " iterations.\n";
		return Assignment;
	}

}
