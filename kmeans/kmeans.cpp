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

std::ostream& operator<<(std::ostream& LHS, Point& RHS)
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
		// The return vector
		TVector<FIndex> Assignment(NumPoint, 0);
		int CurrIteration = 0;
		std::cout << "Running kmeans with num points = " << NumPoint
			<< ", num centers = " << NumCenter
			<< ", max iterations = " << MaxIterations << "...\n";

		TVector<FIndex> PointCount(NumCenter, 0);
		while (MaxIterations--)
		{
			++CurrIteration;
			TVector<index_t> AssignmentBak = Assignment;

			for (FIndex I = 0; I < NumPoint; ++I)
			{
				FPoint& PointI = Points[I];
				double MinDis = std::numeric_limits<double>::max();
				Assignment[I] = -1;
				for (int K = 0; K < NumCenter; ++K)
				{
					FPoint CenterK = Centers[K];
					double Dis = PointI.Distance(CenterK);
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

			Centers.assign(NumCenter, FPoint());
			PointCount.assign(NumPoint, 0);

			for (FIndex I = 0; I < NumPoint; ++I)
			{
				FIndex ClusterI = Assignment[I];
				Centers[ClusterI].X += Points[I].X;
				PointCount[ClusterI]++;
			}

			for (FIndex J = 0; J < NumCenter; ++J)
			{
				Centers[J].X /= PointCount[J];
			}

			PointCount.assign(NumPoint, 0);

			for (FIndex I = 0; I < NumPoint; ++I)
			{
				FIndex ClusterI = Assignment[I];
				Centers[ClusterI].Y += Points[I].Y;
				PointCount[ClusterI]++;
			}

			for (FIndex J = 0; J < NumCenter; ++J)
			{
				Centers[J].Y /= PointCount[J];
			}
		}

	JConverge:
		std::cout << "Finished in " << CurrIteration << " iterations." << std::endl;
		return Assignment;
	}

}
