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

#define USE_OMP 1
#define USE_SIMD 1

#define USE_SIMD_OPERATOR           (1 && USE_SIMD)
#define USE_SIMD_DISTANCE           (1 && USE_SIMD)
#define USE_SIMD_UPDATE_ASSIGNMENT  (0 && USE_SIMD)
#define USE_SIMD_UPDATE_CENTER      (0 && USE_SIMD)
#define USE_SIMD_FMA                (0 && USE_SIMD)

#ifdef _MSC_VER
#define AttrForceInline __forceinline
#define AttrNoInline __declspec(noinline)
#define VectorCall __vectorcall
#define Restrict __restrict
#define BuiltinMemCmp __builtin_memcmp
#define BuiltinMemCpy memcpy
#define BuiltinMemSet memset
#define BuiltinAssumeAligned __builtin_assume_aligned
#else
#define AttrForceInline __attribute__((always_inline))
#define AttrNoInline __attribute__((noinline))
#define VectorCall
#define Restrict __restrict__
#define BuiltinMemCmp __builtin_memcmp
#define BuiltinMemCpy __builtin_memcpy
#define BuiltinMemSet __builtin_memset
#define BuiltinAssumeAligned __builtin_assume_aligned
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

constexpr std::size_t VecAlignment = 64;

#define AssumeVecAligned(Ptr) static_cast<decltype(Ptr)>(BuiltinAssumeAligned(Ptr, VecAlignment))
#endif

using FIndex = ::index_t;
using FPoint = ::Point;

static_assert(std::conjunction_v<std::is_trivial<FPoint>, std::is_standard_layout<FPoint>>);

template<class RElement>
using TVector = ::std::vector<RElement>;

#if USE_SIMD

AttrForceInline
inline __m128d VectorCall
Load1(const FPoint& Restrict Point) noexcept
{
	return _mm_load_pd(AssumeVecAligned(reinterpret_cast<const double*>(&Point)));
}

AttrForceInline
inline __m128d VectorCall
Load1(const FPoint* Restrict Point) noexcept
{
	return _mm_load_pd(AssumeVecAligned(reinterpret_cast<const double*>(Point)));
}

AttrForceInline
inline __m256d VectorCall
Load2(const FPoint* Restrict Point) noexcept
{
	return _mm256_load_pd(AssumeVecAligned(reinterpret_cast<const double*>(Point)));
}

AttrForceInline
inline void VectorCall
Store1(FPoint& Restrict Point, __m128d Vector) noexcept
{
	_mm_store_pd(AssumeVecAligned(reinterpret_cast<double*>(&Point)), Vector);
}

AttrForceInline
inline void VectorCall
Store1(FPoint* Restrict Point, __m128d Vector) noexcept
{
	_mm_store_pd(AssumeVecAligned(reinterpret_cast<double*>(Point)), Vector);
}

AttrForceInline
inline void VectorCall
Store2(FPoint* Restrict Point, __m256d Vector) noexcept
{
	_mm256_store_pd(AssumeVecAligned(reinterpret_cast<double*>(Point)), Vector);
}

#endif

#if USE_SIMD_OPERATOR
inline FPoint operator+(const FPoint& LHS, const FPoint& RHS) noexcept
{
	alignas(VecAlignment) FPoint Result;
	Store1(Result, _mm_add_pd(Load1(LHS), Load1(RHS)));
	return Result;
}
inline FPoint operator-(const FPoint& LHS, const FPoint& RHS) noexcept
{
	alignas(VecAlignment) FPoint Result;
	Store1(Result, _mm_sub_pd(Load1(LHS), Load1(RHS)));
	return Result;
}
inline FPoint& operator+=(FPoint& LHS, const FPoint& RHS) noexcept
{
	Store1(LHS, _mm_add_pd(Load1(LHS), Load1(RHS)));
	return LHS;
}
inline FPoint& operator-=(FPoint& LHS, const FPoint& RHS) noexcept
{
	Store1(LHS, _mm_sub_pd(Load1(LHS), Load1(RHS)));
	return LHS;
}
inline FPoint& operator*=(FPoint& LHS, double RHS) noexcept
{
	Store1(LHS, _mm_mul_pd(Load1(LHS), _mm_set1_pd(RHS)));
	return LHS;
}
inline FPoint& operator/=(FPoint& LHS, double RHS) noexcept
{
	Store1(LHS, _mm_div_pd(Load1(LHS), _mm_set1_pd(RHS)));
	return LHS;
}
#else
inline FPoint operator+(const FPoint& LHS, const FPoint& RHS) noexcept
{
	return {LHS.X + RHS.X, LHS.Y + RHS.Y};
}
inline FPoint operator-(const FPoint& LHS, const FPoint& RHS) noexcept
{
	return {LHS.X - RHS.X, LHS.Y - RHS.Y};
}
inline FPoint& operator+=(FPoint& LHS, const FPoint& RHS) noexcept
{
	LHS.X += RHS.X;
	LHS.Y += RHS.Y;
	return LHS;
}
inline FPoint& operator-=(FPoint& LHS, const FPoint& RHS) noexcept
{
	LHS.X -= RHS.X;
	LHS.Y -= RHS.Y;
	return LHS;
}
inline FPoint& operator*=(FPoint& LHS, double RHS) noexcept
{
	LHS.X *= RHS;
	LHS.Y *= RHS;
	return LHS;
}
inline FPoint& operator/=(FPoint& LHS, double RHS) noexcept
{
	LHS.X /= RHS;
	LHS.Y /= RHS;
	return LHS;
}
#endif

#if USE_SIMD_DISTANCE
inline double GetDistance(const FPoint& Restrict LHS, const FPoint& Restrict RHS) noexcept
{
	const __m128d VL = Load1(LHS);
	const __m128d VR = Load1(RHS);
	const __m128d VDiff = _mm_sub_pd(VL, VR);
	const __m128d VMul = _mm_mul_pd(VDiff, VDiff);
	const __m128d VMulY = _mm_unpackhi_pd(VMul, VMul);
	const __m128d VRes = _mm_add_sd(VMul, VMulY);
	return _mm_cvtsd_f64(VRes);
}
#else
inline double GetDistance(const FPoint& Restrict LHS, const FPoint& Restrict RHS) noexcept
{
	const double A = LHS.X - RHS.X;
	const double B = LHS.Y - RHS.Y;
	return A * A + B * B;
}
#endif

struct FKMeans
{
	FPoint* Points = nullptr;
	FPoint* Centers = nullptr;
	FIndex NumPoint;
	FIndex NumCenter;

	FKMeans(const TVector<FPoint>& InPoints, const TVector<FPoint>& InInitCenters);

	~FKMeans();

	TVector<FIndex> Run(int NumIteration = 1000);
};

#if USE_SIMD_UPDATE_ASSIGNMENT
inline void SimdUpdateAssignment(FIndex* Restrict Assignment, const double* Restrict Points, const FPoint* Centers, FIndex NumPoint, FIndex NumCenter)
{
	Assignment = static_cast<FIndex*>(AssumeVecAligned(Assignment, VecAlignment));
	Points = static_cast<const double*>(AssumeVecAligned(Points, VecAlignment));

	while (NumPoint >= 2)
	{
		const __m256d P = _mm256_load_pd(Points);
		for (FIndex CenterId = 0; CenterId < NumCenter; ++CenterId)
		{
		}
		NumPoint -= 2;
		Points += 4;
		Assignment += 2;
	}
}
#endif

AttrPerf
inline void InitCopy(FPoint* Restrict Dst, const FPoint* Restrict Src, FIndex NumPoint)
{
}

inline FKMeans::FKMeans(const TVector<FPoint>& InPoints, const TVector<FPoint>& InInitCenters)
	: NumPoint(static_cast<FIndex>(InPoints.size()))
	, NumCenter(static_cast<FIndex>(InInitCenters.size()))
{
#if !USE_SIMD
	Points = new(std::nothrow) FPoint[NumPoint];
	Centers = new(std::nothrow) FPoint[NumCenter];
#else
	//Points = new(static_cast<std::align_val_t>(VecAlignment), std::nothrow) FPoint[NumPoint];
	//Centers = new(static_cast<std::align_val_t>(VecAlignment), std::nothrow) FPoint[NumCenter];
	Points = static_cast<FPoint*>(operator new(sizeof(FPoint) * NumPoint, static_cast<std::align_val_t>(VecAlignment), std::nothrow));
	Centers = static_cast<FPoint*>(operator new(sizeof(FPoint) * NumCenter, static_cast<std::align_val_t>(VecAlignment), std::nothrow));
	check(!(reinterpret_cast<std::uintptr_t>(Points) & (VecAlignment - 1)));
	check(!(reinterpret_cast<std::uintptr_t>(Centers) & (VecAlignment - 1)));
#endif

	check(Points);
	check(Centers);
	BuiltinMemCpy(Points, InPoints.data(), sizeof(FPoint) * NumPoint);
	BuiltinMemCpy(Centers, InInitCenters.data(), sizeof(FPoint) * NumCenter);
}

FKMeans::~FKMeans()
{
#if !USE_SIMD
	delete[] Points;
	delete[] Centers;
#else
	operator delete(Points, static_cast<std::align_val_t>(VecAlignment), std::nothrow);
	operator delete(Centers, static_cast<std::align_val_t>(VecAlignment), std::nothrow);
#endif
}

AttrPerf
inline void UpdateAssignment(FIndex* Restrict Assignment, const FPoint* Restrict Points, const FPoint* Restrict Centers, FIndex NumPoint, FIndex NumCenter)
{
#if !USE_SIMD_UPDATE_ASSIGNMENT
#if USE_OMP
	#pragma omp parallel for
#endif
	for (FIndex PointId = 0; PointId < NumPoint; ++PointId)
	{
		const FPoint& Point = Points[PointId];
		double MinDistance = std::numeric_limits<double>::max();
		FIndex Result = -1;
		for (int CenterId = 0; CenterId < NumCenter; ++CenterId)
		{
			const FPoint& Center = Centers[CenterId];
			const double Distance = GetDistance(Point, Center);
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
inline void UpdateCenters(FPoint* Restrict Centers, FIndex* Restrict PointCount, const FPoint* Restrict Points, const FIndex* Restrict Assignment, FIndex NumPoint, FIndex NumCenter)
{
#if !USE_SIMD_UPDATE_CENTER
	BuiltinMemSet(Centers, 0, sizeof(FPoint) * NumCenter);
	BuiltinMemSet(PointCount, 0, sizeof(FIndex) * NumCenter);

	// TODO: Points is reordered, handle this
	for (FIndex PointerId = 0; PointerId < NumPoint; ++PointerId)
	{
		const FIndex CenterId = Assignment[PointerId];
		Centers[CenterId] += Points[PointerId];
		PointCount[CenterId]++;
	}

	for (FIndex CenterId = 0; CenterId < NumCenter; ++CenterId)
	{
		Centers[CenterId] /= PointCount[CenterId];
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

		UpdateAssignment(Assignment.data(), Points, Centers, NumPoint, NumCenter);

		if (!BuiltinMemCmp(Assignment.data(), OldAssignment.data(), sizeof(FIndex) * NumPoint))
		{
			goto JConverge;
		}

		UpdateCenters(Centers, PointCount.data(), Points, Assignment.data(), NumPoint, NumCenter);

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
