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

#define USE_OMP 0
#define USE_SIMD 1

#define USE_SIMD_OPERATOR            (1 && USE_SIMD)
#define USE_SIMD_DISTANCE            (1 && USE_SIMD)
#define USE_SIMD_UPDATE_ASSIGNMENT   (1 && USE_SIMD)
#define USE_SIMD_UPDATE_CENTER       (1 && USE_SIMD)
#define USE_SIMD_FINALIZE_ASSIGNMENT (1 && USE_SIMD)
#define USE_SIMD_FMA                 (0 && USE_SIMD)

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

constexpr std::size_t XmmAlignment = 16;
constexpr std::size_t YmmAlignment = 32;
constexpr std::size_t VecAlignment = 64;

#define AssumeXmmAligned(Ptr) static_cast<decltype(Ptr)>(BuiltinAssumeAligned(Ptr, XmmAlignment))
#define AssumeYmmAligned(Ptr) static_cast<decltype(Ptr)>(BuiltinAssumeAligned(Ptr, YmmAlignment))
//#define AssumeVecAligned(Ptr) static_cast<decltype(Ptr)>(BuiltinAssumeAligned(Ptr, VecAlignment))
#endif

using FIndex = ::index_t;
using FPoint = ::Point;

static_assert(std::conjunction_v<std::is_trivial<FPoint>, std::is_standard_layout<FPoint>>);

template<class RElement>
using TVector = ::std::vector<RElement>;

template<class T>
AttrForceInline
inline T* AllocArray(std::size_t Num) noexcept
{
#if USE_SIMD
	T* Ptr = static_cast<T*>(operator new(sizeof(T) * Num, static_cast<std::align_val_t>(VecAlignment), std::nothrow));
	check(!(reinterpret_cast<std::uintptr_t>(Ptr) & (VecAlignment - 1)));
#else
	T* Ptr = new(std::nothrow) T[Num];
#endif
	check(Ptr);
	return Ptr;
}

template<class T>
AttrForceInline
inline void FreeArray(T* Ptr) noexcept
{
#if USE_SIMD
	operator delete(Ptr, static_cast<std::align_val_t>(VecAlignment), std::nothrow);
#else
	delete[] Ptr;
#endif
}

#if USE_SIMD

AttrForceInline
inline __m128d VectorCall
Load1(const FPoint& Restrict Point) noexcept
{
	return _mm_load_pd(AssumeXmmAligned(reinterpret_cast<const double*>(&Point)));
}

AttrForceInline
inline __m128d VectorCall
Load1(const FPoint* Restrict Point) noexcept
{
	return _mm_load_pd(AssumeXmmAligned(reinterpret_cast<const double*>(Point)));
}

AttrForceInline
inline __m256d VectorCall
Load2(const FPoint* Restrict Point) noexcept
{
	return _mm256_load_pd(AssumeYmmAligned(reinterpret_cast<const double*>(Point)));
}

AttrForceInline
inline void VectorCall
Store1(FPoint& Restrict Point, __m128d Vector) noexcept
{
	_mm_store_pd(AssumeXmmAligned(reinterpret_cast<double*>(&Point)), Vector);
}

AttrForceInline
inline void VectorCall
Store1(FPoint* Restrict Point, __m128d Vector) noexcept
{
	_mm_store_pd(AssumeXmmAligned(reinterpret_cast<double*>(Point)), Vector);
}

AttrForceInline
inline void VectorCall
Store2(FPoint* Restrict Point, __m256d Vector) noexcept
{
	_mm256_store_pd(AssumeYmmAligned(reinterpret_cast<double*>(Point)), Vector);
}

#endif

#if USE_SIMD_OPERATOR
inline FPoint operator+(const FPoint& LHS, const FPoint& RHS) noexcept
{
	alignas(XmmAlignment) FPoint Result;
	Store1(Result, _mm_add_pd(Load1(LHS), Load1(RHS)));
	return Result;
}
inline FPoint operator-(const FPoint& LHS, const FPoint& RHS) noexcept
{
	alignas(XmmAlignment) FPoint Result;
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
	FIndex* Assignment = nullptr;
	FIndex* OldAssignment = nullptr;
	FIndex* PointCount = nullptr;
	FIndex NumPoint;
	FIndex NumCenter;

	FKMeans(const TVector<FPoint>& InPoints, const TVector<FPoint>& InInitCenters) noexcept;

	~FKMeans() noexcept;

	TVector<FIndex> Run(int NumIteration = 1000);
};

inline FKMeans::FKMeans(const TVector<FPoint>& InPoints, const TVector<FPoint>& InInitCenters) noexcept
	: NumPoint(static_cast<FIndex>(InPoints.size()))
	, NumCenter(static_cast<FIndex>(InInitCenters.size()))
{
	Points = AllocArray<FPoint>(NumPoint);
	Centers = AllocArray<FPoint>(NumCenter);
	Assignment = AllocArray<FIndex>(NumPoint);
	OldAssignment = AllocArray<FIndex>(NumPoint);
	PointCount = AllocArray<FIndex>(NumCenter);

	BuiltinMemCpy(Points, InPoints.data(), sizeof(FPoint) * NumPoint);
	BuiltinMemCpy(Centers, InInitCenters.data(), sizeof(FPoint) * NumCenter);
}

FKMeans::~FKMeans() noexcept
{
	FreeArray(Points);
	FreeArray(Centers);
	FreeArray(Assignment);
	FreeArray(OldAssignment);
	FreeArray(PointCount);
}

AttrPerf
inline void UpdateAssignment(FIndex* Restrict Assignment, const FPoint* Restrict Points, const FPoint* Restrict Centers, FIndex NumPoint, FIndex NumCenter) noexcept
{
#if USE_SIMD_UPDATE_ASSIGNMENT
	constexpr FIndex BatchSize = 8;

	const FIndex NumPointForBatch = NumPoint & ~(BatchSize - 1);

	for (FIndex PointId = 0; PointId < NumPointForBatch; PointId += BatchSize)
	{
		const __m256d VPoint01 = Load2(Points + PointId);
		const __m256d VPoint23 = Load2(Points + PointId + 2);
		const __m256d VPoint45 = Load2(Points + PointId + 4);
		const __m256d VPoint67 = Load2(Points + PointId + 6);
		__m256d VMinDis0213 = _mm256_set1_pd(std::numeric_limits<double>::max());
		__m256d VMinDis4657 = _mm256_set1_pd(std::numeric_limits<double>::max());
		__m256 VAssign04261537;
		for (FIndex CenterId = 0; CenterId < NumCenter; ++CenterId)
		{
			const __m256d VCenter = _mm256_broadcast_pd(AssumeXmmAligned(reinterpret_cast<const __m128d*>(&Centers[CenterId])));
			const __m256d VDiff01 = _mm256_sub_pd(VPoint01, VCenter);
			const __m256d VDiff23 = _mm256_sub_pd(VPoint23, VCenter);
			const __m256d VDiff45 = _mm256_sub_pd(VPoint45, VCenter);
			const __m256d VDiff67 = _mm256_sub_pd(VPoint67, VCenter);
			const __m256d VMul01 = _mm256_mul_pd(VDiff01, VDiff01);
			const __m256d VMul23 = _mm256_mul_pd(VDiff23, VDiff23);
			const __m256d VMul45 = _mm256_mul_pd(VDiff45, VDiff23);
			const __m256d VMul67 = _mm256_mul_pd(VDiff67, VDiff23);
			const __m256d VMulX0213 = _mm256_unpacklo_pd(VMul01, VMul23);
			const __m256d VMulX4657 = _mm256_unpacklo_pd(VMul45, VMul67);
			const __m256d VMulY0213 = _mm256_unpackhi_pd(VMul01, VMul23);
			const __m256d VMulY4657 = _mm256_unpackhi_pd(VMul45, VMul67);
			const __m256d VDis0213 = _mm256_add_pd(VMulX0213, VMulY0213);
			const __m256d VDis4657 = _mm256_add_pd(VMulX4657, VMulY4657);
			const __m256 VCmp0213 = _mm256_castpd_ps(_mm256_cmp_pd(VDis0213, VMinDis0213, _CMP_LT_OQ));
			const __m256 VCmp4657 = _mm256_castpd_ps(_mm256_cmp_pd(VDis4657, VMinDis4657, _CMP_LT_OQ));
			VMinDis0213 = _mm256_min_pd(VDis0213, VMinDis0213);
			VMinDis4657 = _mm256_min_pd(VDis4657, VMinDis4657);
			const __m256 VCmp04261537 = _mm256_blend_ps(VCmp0213, VCmp4657, 0b10101010);
			const __m256 VCenterId = _mm256_castsi256_ps(_mm256_set1_epi32(CenterId));
			VAssign04261537 = _mm256_blendv_ps(VAssign04261537, VCenterId, VCmp04261537);
		}
		_mm256_store_ps(AssumeYmmAligned(reinterpret_cast<float*>(Assignment + PointId)), VAssign04261537);
	}

	for (FIndex PointId = NumPointForBatch; PointId < NumPoint; ++PointId)
	{
		const FPoint& Point = Points[PointId];
		double MinDistance = std::numeric_limits<double>::max();
		FIndex Result = -1;
		for (FIndex CenterId = 0; CenterId < NumCenter; ++CenterId)
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
#if USE_OMP
	#pragma omp parallel for
#endif
	for (FIndex PointId = 0; PointId < NumPoint; ++PointId)
	{
		const FPoint& Point = Points[PointId];
		double MinDistance = std::numeric_limits<double>::max();
		FIndex Result = -1;
		for (FIndex CenterId = 0; CenterId < NumCenter; ++CenterId)
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
#endif
}

AttrPerf
inline void UpdateCenters(FPoint* Restrict Centers, FIndex* Restrict PointCount, const FPoint* Restrict Points, const FIndex* Restrict Assignment, FIndex NumPoint, FIndex NumCenter) noexcept
{
#if USE_SIMD_UPDATE_CENTER
	BuiltinMemSet(Centers, 0, sizeof(FPoint) * NumCenter);
	BuiltinMemSet(PointCount, 0, sizeof(FIndex) * NumCenter);

	constexpr FIndex BatchSize = 8;

	const FIndex NumPointForBatch = NumPoint & ~(BatchSize - 1);

	for (FIndex PointId = 0; PointId < NumPointForBatch; PointId += BatchSize)
	{
		const FIndex SubAssign[8]
		{
			Assignment[PointId + 0],
			Assignment[PointId + 4],
			Assignment[PointId + 2],
			Assignment[PointId + 6],
			Assignment[PointId + 1],
			Assignment[PointId + 5],
			Assignment[PointId + 3],
			Assignment[PointId + 7],
		};
		for (FIndex SubPointId = 0; SubPointId < BatchSize; ++SubPointId)
		{
			const FIndex CenterId = SubAssign[SubPointId];
			Centers[CenterId] += Points[PointId + SubPointId];
			++PointCount[CenterId];
		}
	}

	for (FIndex PointId = NumPointForBatch; PointId < NumPoint; ++PointId)
	{
		const FIndex CenterId = Assignment[PointId];
		Centers[CenterId] += Points[PointId];
		++PointCount[CenterId];
	}

	for (FIndex CenterId = 0; CenterId < NumCenter; ++CenterId)
	{
		Centers[CenterId] /= PointCount[CenterId];
	}
#else
	BuiltinMemSet(Centers, 0, sizeof(FPoint) * NumCenter);
	BuiltinMemSet(PointCount, 0, sizeof(FIndex) * NumCenter);

	for (FIndex PointId = 0; PointId < NumPoint; ++PointId)
	{
		const FIndex CenterId = Assignment[PointId];
		Centers[CenterId] += Points[PointId];
		++PointCount[CenterId];
	}

	for (FIndex CenterId = 0; CenterId < NumCenter; ++CenterId)
	{
		Centers[CenterId] /= PointCount[CenterId];
	}
#endif
}

AttrPerf
inline TVector<FIndex> FinalizeAssignment(FIndex* Restrict Assignment, FIndex NumPoint) noexcept
{
#if USE_SIMD_FINALIZE_ASSIGNMENT
	constexpr FIndex BatchSize = 8;

	const FIndex NumPointForBatch = NumPoint & ~(BatchSize - 1);

	const __m256i VPerm = _mm256_set_epi32(7, 3, 5, 1, 6, 2, 4, 0);

	for (FIndex PointId = 0; PointId < NumPointForBatch; PointId += BatchSize)
	{
		const __m256 VAssign04261537 = _mm256_load_ps(AssumeYmmAligned(reinterpret_cast<float*>(Assignment + PointId)));
		const __m256 VResult01234567 = _mm256_permutevar8x32_ps(VAssign04261537, VPerm);
		_mm256_store_ps(AssumeYmmAligned(reinterpret_cast<float*>(Assignment + PointId)), VResult01234567);
	}
#endif

	return {Assignment, Assignment + NumPoint};
}

TVector<FIndex> FKMeans::Run(int NumIteration)
{
	using std::swap;

	int IterationId = 0;
	std::cout << "Running kmeans with num points = " << NumPoint
		<< ", num centers = " << NumCenter
		<< ", max iterations = " << NumIteration << "...\n";

	// Give it an invalid value to trigger at least two iteration
	OldAssignment[0] = NumPoint;

	while (NumIteration--)
	{
		++IterationId;

		UpdateAssignment(Assignment, Points, Centers, NumPoint, NumCenter);

		if (!BuiltinMemCmp(Assignment, OldAssignment, sizeof(FIndex) * NumPoint))
		{
			goto JConverge;
		}

		UpdateCenters(Centers, PointCount, Points, Assignment, NumPoint, NumCenter);

		swap(Assignment, OldAssignment);
	}

JConverge:
	std::cout << "Finished in " << IterationId << " iterations.\n";
	return FinalizeAssignment(Assignment, NumPoint);
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
