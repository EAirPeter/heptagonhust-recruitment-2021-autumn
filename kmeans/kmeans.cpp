#include "kmeans.hpp"

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

#ifdef _MSC_VER
#define __builtin_memset memset
#endif

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
		TVector<FIndex> Assignment(NumPoint, 0);
		TVector<FIndex> AssignmentBak(NumPoint, 0);

		int CurrIteration = 0;
		std::cout << "Running kmeans with num points = " << NumPoint
			<< ", num centers = " << NumCenter
			<< ", max iterations = " << MaxIterations << "...\n";

		TVector<FIndex> PointCount(NumCenter, 0);
		while (MaxIterations--)
		{
			++CurrIteration;

			for (FIndex I = 0; I < NumPoint; ++I)
			{
				FPoint& PointI = Points[I];
				double MinDis = std::numeric_limits<double>::max();
				Assignment[I] = -1;
				for (int K = 0; K < NumCenter; ++K)
				{
					FPoint& CenterK = Centers[K];
					const double Dis = PointI.Distance(CenterK);
					if (Dis < MinDis)
					{
						MinDis = Dis;
						Assignment[I] = K;
					}
				}
			}

			if (Assignment == AssignmentBak)
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
