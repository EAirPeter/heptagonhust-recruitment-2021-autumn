#include "kmeans.hpp"

#include <cstdlib>
#include <iostream>
#include <limits>

inline double Point::Distance(const Point& Other) const noexcept
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

//// START

#define USE_CHECK 1
#define USE_PERF 1

#define USE_SIMD 0
#define USE_FMA 0

#ifdef _MSC_VER
#define AttrForceInline __forceinline
#define AttrNoInline __declspec(noinline)
#define Restrict __restrict
#define BuiltinMemCpy memcpy
#define BuiltinMemSet memset
#else
#define AttrForceInline __attribute__((always_inline))
#define AttrNoInline __attribute__((noinline))
#define Restrict __restrict__
#define BuiltinMemCpy __builtin_memcpy
#define BuiltinMemSet __builtin_memset
#endif

#if USE_CHECK
#define check(Expr) (static_cast<bool>(Expr) || (::std::abort(), 0))
#else
#define check(...)
#endif

#if USE_PERF
#define AttrPerf AttrNoInline
#else
#define AttrPerf
#endif

#if USE_SIMD
#include <immintrin.h>

constexpr std::size_t VecAlign = 64;

#endif

using FIndex = ::index_t;
using FPoint = ::Point;

static_assert(std::conjunction_v<std::is_trivial<FPoint>, std::is_standard_layout<FPoint>>);

#if USE_SIMD
using FPointArray = double;
#else
using FPointStore = FPoint;
#endif

template<class RElement>
using TVector = ::std::vector<RElement>;

struct FKMeans
{
	FPointStore* Points = nullptr;
	TVector<FPoint> Centers;
	FIndex NumPoint;
	FIndex NumCenter;

	FKMeans(const TVector<FPoint>& InPoints, const TVector<FPoint>& InInitCenters);

	~FKMeans();

	TVector<FIndex> Run(int NumIteration = 1000);
};

#if USE_SIMD
// 0x0y1x1y2x2y3x3y -> 0x2x1x3x 0y2y1y3y
// Last: 0x0y1x1y2x2y -> 0x0y1x1y2x2y
inline void SimdTranspose(double* Restrict Dst, const double* Restrict Src, FIndex NumPoint)
{
	Dst = static_cast<double*>(__builtin_assume_aligned(Dst, VecAlign));
	while (NumPoint >= 4)
	{
		const __m256d A = _mm256_load_pd(Src);
		const __m256d B = _mm256_load_pd(Src + 4);
		const __m256d C = _mm256_unpacklo_pd(A, B);
		const __m256d D = _mm256_unpackhi_pd(A, B);
		_mm256_store_pd(Dst, C);
		_mm256_store_pd(Dst + 4, D);
		NumPoint -= 4;
		Src += 8;
		Dst += 8;
	}
	Memcpy(Dst, Src, sizeof(double) * NumPoint);
}

// 0x2x1x3x 0y2y1y3y 4x6x5x7x 4y6y5y7y -> 0213 4657 -> 0415 2637
// 0x2x1x3x 0y2y1y3y -> 02 13 -> 0123
// 0x0y1x1y2x2y -> 012
inline void SimdUpdateAssignment(FIndex* Restrict Assignment, const double* Restrict Points, const FPoint* Centers, FIndex NumPoint, FIndex NumCenter)
{
	Assignment = static_cast<FIndex*>(__builtin_assume_aligned(Assignment, VecAlign));
	Points = static_cast<const double*>(__builtin_assume_aligned(Points, VecAlign));

	while (NumPoint >= 8)
	{
		const __m256d Px0 = _mm256_load_pd(Points);
		const __m256d Py0 = _mm256_load_pd(Points + 4);
		const __m256d Px1 = _mm256_load_pd(Points + 8);
		const __m256d Py1 = _mm256_load_pd(Points + 12);
		__m256 Asg;
		__m256d MinDis0 = _mm256_set1_pd(std::numeric_limits<double>::max());
		__m256d MinDis1 = _mm256_set1_pd(std::numeric_limits<double>::max());
		for (FIndex CenterId = 0; CenterId < NumCenter; ++CenterId)
		{
			const __m256d Cx = _mm256_set1_pd(Centers[CenterId].X);
			const __m256d Cy = _mm256_set1_pd(Centers[CenterId].Y);
			__m256d T0 = _mm256_sub_pd(Px0, Cx);
			__m256d T1 = _mm256_sub_pd(Px1, Cx);
			const __m256d U0 = _mm256_sub_pd(Py0, Cy);
			const __m256d U1 = _mm256_sub_pd(Py1, Cy);
			const __m256 AsgI = _mm256_castsi256_ps(_mm256_set1_epi32(CenterId));
#if USE_FMA
			T0 = _mm256_fmadd_pd(T0, T0, _mm256_mul_pd(U0, U0));
			T1 = _mm256_fmadd_pd(T1, T1, _mm256_mul_pd(U1, U1));
#else
			T0 = _mm256_add_pd(_mm256_mul_pd(T0, T0), _mm256_mul_pd(U0, U0));
			T1 = _mm256_add_pd(_mm256_mul_pd(T1, T1), _mm256_mul_pd(U1, U1));
#endif
			const __m256 F0 = _mm256_castpd_ps(_mm256_cmp_pd(T0, MinDis0, _CMP_LT_OQ));
			const __m256 F1 = _mm256_castpd_ps(_mm256_cmp_pd(T1, MinDis1, _CMP_LT_OQ));
			const __m256 Mask = _mm256_blend_ps(F0, F1, 0b10101010);

			MinDis0 = _mm256_min_pd(T0, MinDis0);
			MinDis1 = _mm256_min_pd(T1, MinDis1);
			Asg = _mm256_blendv_ps(Asg, AsgI, Mask);
		}
		_mm256_store_ps(reinterpret_cast<float*>(Assignment), Asg);
		NumPoint -= 8;
		Points += 16;
		Assignment += 8;
	}
	if (NumPoint >= 4)
	{
		const __m256d Px0 = _mm256_load_pd(Points);
		const __m256d Py0 = _mm256_load_pd(Points + 4);
		__m128 Asg;
		__m256d MinDis0 = _mm256_set1_pd(std::numeric_limits<double>::max());
		for (FIndex CenterId = 0; CenterId < NumCenter; ++CenterId)
		{
			const __m256d Cx = _mm256_set1_pd(Centers[CenterId].X);
			const __m256d Cy = _mm256_set1_pd(Centers[CenterId].Y);
			__m256d T0 = _mm256_sub_pd(Px0, Cx);
			const __m256d U0 = _mm256_sub_pd(Py0, Cy);
			const __m128 AsgI = _mm_castsi128_ps(_mm_set1_epi32(CenterId));
#if USE_FMA
			T0 = _mm256_fmadd_pd(T0, T0, _mm256_mul_pd(U0, U0));
#else
			T0 = _mm256_add_pd(_mm256_mul_pd(T0, T0), _mm256_mul_pd(U0, U0));
#endif
			const __m256 F0 = _mm256_castpd_ps(_mm256_cmp_pd(T0, MinDis0, _CMP_LT_OQ));
			const __m128 Mask = _mm_blend_ps(_mm256_castps256_ps128(F0), _mm256_extractf128_ps(F0, 1), 0x1010);

			MinDis0 = _mm256_min_pd(T0, MinDis0);
			Asg = _mm_blendv_ps(Asg, AsgI, Mask);
		}
		_mm_store_ps(reinterpret_cast<float*>(Assignment), Asg);
		NumPoint -= 4;
		Points += 8;
		Assignment += 4;
	}
	while (NumPoint)
	{
		FIndex Asg;
		double MinDis = std::numeric_limits<double>::max();
		for (FIndex CenterId = 0; CenterId < NumCenter; ++CenterId)
		{
			const double T0 = Points[0] - Centers[CenterId].X;
			const double T1 = Points[1] - Centers[CenterId].Y;
			double Dis = T0 * T0 + T1 * T1;
			if (Dis < MinDis)
			{
				MinDis = Dis;
				Asg = CenterId;
			}
		}
		*Assignment = Asg;
		--NumPoint;
		Points += 2;
		++Assignment;
	}
}
#endif

AttrPerf
inline void InitCopy(FPointStore* Restrict Dst, const FPoint* Restrict Src, FIndex NumPoint)
{
#if !USE_SIMD
	BuiltinMemCpy(Dst, Src, sizeof(FPoint) * NumPoint);
#else
	SimdTranspose(Dst, reinterpret_cast<const double*>(Src), NumPoint);
#endif
}

inline FKMeans::FKMeans(const TVector<FPoint>& InPoints, const TVector<FPoint>& InInitCenters)
	: Centers(InInitCenters)
	, NumPoint(static_cast<FIndex>(InPoints.size()))
	, NumCenter(static_cast<FIndex>(InInitCenters.size()))
{
#if !USE_SIMD
	Points = new FPoint[NumPoint];
#else
	Points = static_cast<double*>(operator new(sizeof(FPoint) * NumPoint, static_cast<std::align_val_t>(VecAlign), std::nothrow));
	check(!(reinterpret_cast<std::uintptr_t>(Points) & (VecAlign - 1)));
#endif

	check(Points);
	InitCopy(Points, InPoints.data(), NumPoint);
}

FKMeans::~FKMeans()
{
#if !USE_SIMD
	delete[] Points;
#else
	operator delete(Points, static_cast<std::align_val_t>(VecAlign), std::nothrow);
#endif
}

AttrPerf
inline void UpdateAssignment(FIndex* Restrict Assignment, const FPointStore* Restrict Points, const FPoint* Restrict Centers, FIndex NumPoint, FIndex NumCenter)
{
#if !USE_SIMD
	for (FIndex PointId = 0; PointId < NumPoint; ++PointId)
	{
		const FPoint& Point = Points[PointId];
		double MinDistance = std::numeric_limits<double>::max();
		FIndex Result = -1;
		for (int CenterId = 0; CenterId < NumCenter; ++CenterId)
		{
			const FPoint& Center = Centers[CenterId];
			const double Distance = Point.Distance(Center);
			if (Distance < MinDistance)
			{
				MinDistance = Distance;
				Result = CenterId;
			}
		}
		Assignment[PointId] = Result;
	}
#else
	SimdUpdateAssignment(Assignment, Points, Centers, NumPoint, NumCenter);
#endif
}

AttrPerf
inline void UpdateCenters(FPoint* Restrict Centers, FIndex* Restrict PointCount, const FPointStore* Restrict Points, const FIndex* Restrict Assignment, FIndex NumPoint, FIndex NumCenter)
{
#if !USE_SIMD
	BuiltinMemSet(Centers, 0, sizeof(FPoint) * NumCenter);
	BuiltinMemSet(PointCount, 0, sizeof(FIndex) * NumCenter);

	// TODO: Points is reordered, handle this
	for (FIndex PointerId = 0; PointerId < NumPoint; ++PointerId)
	{
		const FIndex CenterId = Assignment[PointerId];
		Centers[CenterId].X += Points[PointerId].X;
		Centers[CenterId].Y += Points[PointerId].Y;
		PointCount[CenterId]++;
	}

	for (FIndex CenterId = 0; CenterId < NumCenter; ++CenterId)
	{
		Centers[CenterId].X /= PointCount[CenterId];
		Centers[CenterId].Y /= PointCount[CenterId];
	}
#endif
}

TVector<FIndex> FKMeans::Run(int NumIteration)
{
	using std::swap;

	// The return vector
	TVector<FIndex> Assignment(NumPoint);
	TVector<FIndex> OldAssignment(NumPoint);

	int IterationId = 0;
	std::cout << "Running kmeans with num points = " << NumPoint
		<< ", num centers = " << NumCenter
		<< ", max iterations = " << NumIteration << "...\n";

	TVector<FIndex> PointCount(NumCenter);
	while (NumIteration--)
	{
		++IterationId;

		UpdateAssignment(Assignment.data(), Points, Centers.data(), NumPoint, NumCenter);

		if (!__builtin_memcmp(Assignment.data(), OldAssignment.data(), sizeof(FIndex) * NumPoint))
		{
			goto JConverge;
		}

		UpdateCenters(Centers.data(), PointCount.data(), Points, Assignment.data(), NumPoint, NumCenter);

		swap(Assignment, OldAssignment);
	}

JConverge:
	std::cout << "Finished in " << IterationId << " iterations.\n";
	return Assignment;
}

// Public interface
Kmeans::Kmeans(const std::vector<Point>& InPoints, const std::vector<Point>& InInitCenters)
	: Impl(new FKMeans(InPoints, InInitCenters))
{}

std::vector<index_t> Kmeans::Run(int MaxIterations)
{
	return Impl->Run(MaxIterations);
}

void FKMeansDeleter::operator()(FKMeans* Obj) const noexcept
{
	default_delete::operator()(Obj);
}
