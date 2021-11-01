#include "kmeans.hpp"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <limits>

inline double Point::Distance(const Point& Other) const noexcept
{
	const double A = x - Other.y;
	const double B = x - Other.y;
	return A * A + B * B;
}

std::istream& operator>>(std::istream& LHS, Point& RHS)
{
	return LHS >> RHS.x >> RHS.y;
}

std::ostream& operator<<(std::ostream& LHS, const Point& RHS)
{
	return LHS << RHS.x << " " << RHS.y;
}

// ^^^^^^ They will not be used in this file

// TODO: Rearrange Centers array
// TODO: Retest AOT swizzling

//// START

#define USE_CHECK 1
#define USE_PERF 1
#define USE_STATS 0

#define USE_OMP 1
#define USE_SIMD 1

#define USE_SIMD_OPERATOR                  (1 && USE_SIMD) // 1
#define USE_SIMD_DISTANCE                  (1 && USE_SIMD_OPERATOR) // 1
#define USE_SIMD_UPDATE_ASSIGNMENT         (1 && USE_SIMD) // 1
#define USE_SIMD_ASSIGNMENT_SWIZZLE        (0 && USE_SIMD) // 0
#define USE_SIMD_CENTER_ID_ADD             (1 && USE_SIMD && !USE_OMP_UPDATE_ASSIGNMENT) // 1
#define USE_SIMD_POINTS_SWIZZLE            (1 && USE_SIMD && !USE_SIMD_ASSIGNMENT_SWIZZLE) // 1
#define USE_SIMD_POINTS_SWIZZLE_AOT        (0 && USE_SIMD_POINTS_SWIZZLE) // 0
#define USE_SIMD_POINTS_SWIZZLE_AOT_COPY   (0 && USE_SIMD_POINTS_SWIZZLE_AOT) // 0
#define USE_SIMD_FMA                       (1 && USE_SIMD_POINTS_SWIZZLE) // 1

#define USE_OMP_NUM_THREAD                 (1 && USE_OMP)
#define USE_OMP_UPDATE_ASSIGNMENT          (1 && USE_OMP) // 1
#define USE_OMP_UPDATE_ASSIGNMENT_FINE     (0 && USE_OMP_UPDATE_ASSIGNMENT && USE_OMP_NUM_THREAD)
#define USE_OMP_UPDATE_CENTERS             (1 && USE_OMP_NUM_THREAD && !USE_SIMD_ASSIGNMENT_SWIZZLE && (!USE_SIMD_POINTS_SWIZZLE_AOT || USE_SIMD_POINTS_SWIZZLE_AOT_COPY))
#define USE_OMP_PARALLEL_MEMSET            (0 && USE_OMP_UPDATE_CENTERS)

#ifdef _MSC_VER
#define AttrNoInline __declspec(noinline)
#define ForceInline __forceinline
#define VectorCall __vectorcall
#define Restrict __restrict
#define BuiltinMemCmp __builtin_memcmp
#define BuiltinMemCpy memcpy
#define BuiltinMemSet memset
#define BuiltinAssumeAligned __builtin_assume_aligned
#else
#define AttrNoInline __attribute__((noinline))
#define ForceInline __attribute__((always_inline)) inline
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

#if USE_OMP
#include <omp.h>
#endif

#if USE_SIMD
#include <immintrin.h>

constexpr std::size_t XmmAlignment = 16;
constexpr std::size_t YmmAlignment = 32;
constexpr std::size_t VecAlignment = 64;

#define AssumeXmmAligned(Ptr) static_cast<decltype(Ptr)>(BuiltinAssumeAligned(Ptr, XmmAlignment))
#define AssumeYmmAligned(Ptr) static_cast<decltype(Ptr)>(BuiltinAssumeAligned(Ptr, YmmAlignment))
#define AssumeVecAligned(Ptr) static_cast<decltype(Ptr)>(BuiltinAssumeAligned(Ptr, VecAlignment))
#endif

#if USE_STATS

#include <atomic>
#include <chrono>

#define ALL_STATS(E) \
	E(FKMeansConstructor) \
	E(FKMeansRun) \
	E(UpdateAssignment) \
	E(UpdateCenters) \
	E(UpdateCenters_Gather) \
	E(UpdateCenters_Division) \
	E(CompareForQuit) \
	E(ZeroCenterAndPointCount) \
	E(FinalizeAssignment)

namespace StatsImpl
{
	using std::chrono::high_resolution_clock;

	enum class EStat
	{
#define STATS_DEF_ENUM(Stat) Stat,
		ALL_STATS(STATS_DEF_ENUM)
		Num
	};

	template<EStat Stat>
	static std::atomic_int64_t GAccumulator;

	template<EStat Stat>
	struct TStatTimer
	{
		high_resolution_clock::time_point TimeStart;
		int64_t TotalNumNanosecond = 0;

		~TStatTimer() noexcept
		{
			if (TotalNumNanosecond)
			{
				GAccumulator<Stat>.fetch_add(TotalNumNanosecond, std::memory_order_relaxed);
			}
		}

		void Start() noexcept
		{
			TimeStart = high_resolution_clock::now();
		}

		void Stop() noexcept
		{
			const high_resolution_clock::time_point End = high_resolution_clock::now();
			TotalNumNanosecond += std::chrono::duration_cast<std::chrono::nanoseconds>(End - TimeStart).count();
		}
	};

	template<EStat Stat>
	struct TScopedStatTimer
	{
		high_resolution_clock::time_point Start;

		TScopedStatTimer() noexcept
			: Start(high_resolution_clock::now())
		{
		}

		~TScopedStatTimer() noexcept
		{
			const high_resolution_clock::time_point End = high_resolution_clock::now();
			const int64_t NumNanosecond = std::chrono::duration_cast<std::chrono::nanoseconds>(End - Start).count();
			GAccumulator<Stat>.fetch_add(NumNanosecond, std::memory_order_relaxed);
		}
	};

	void PrintStat(const char* Name, const std::atomic_int64_t& Counter)
	{
		const int64_t NumNanosecond = Counter.load(std::memory_order_consume);
		const double NumMillisecond = static_cast<double>(NumNanosecond) * 0.000001;
		std::cout << "[STAT][" << Name << "] " << NumMillisecond << " ms\n";
	}

	struct FStatPrinter
	{
		~FStatPrinter()
		{
#define STATS_PRINT(Stat) PrintStat(# Stat, GAccumulator<EStat::Stat>);
			ALL_STATS(STATS_PRINT)
		}
	};

	static FStatPrinter GStatPrinter;
}

#define SCOPED_TIMER(Stat) ::StatsImpl::TScopedStatTimer<::StatsImpl::EStat::Stat> Timer ## __LINE__
#define STAT_TIMER(Stat, Name) ::StatsImpl::TStatTimer<::StatsImpl::EStat::Stat> Name
#define STAT_START(Timer) Timer.Start()
#define STAT_STOP(Timer) Timer.Stop()

#else
#define SCOPED_TIMER(...)
#define STAT_START(...)
#define STAT_STOP(...)
#endif

using FIndex = ::index_t;
using FInPoint = ::Point;

// NOTE: Under O1, cannot count on GCC for inlining vectorized FPoint operators

struct
#if USE_SIMD
alignas(XmmAlignment)
#endif
FPoint
{
	double X;
	double Y;

#if USE_SIMD_OPERATOR
	ForceInline	VectorCall
	operator __m128d() const noexcept
	{
		return _mm_load_pd(AssumeXmmAligned(reinterpret_cast<const double*>(this)));
	}

	ForceInline	FPoint& VectorCall
	operator=(__m128d InVec) noexcept
	{
		_mm_store_pd(AssumeXmmAligned(reinterpret_cast<double*>(this)), InVec);
		return *this;
	}
#endif
};

static_assert(sizeof(FInPoint) == sizeof(FPoint));
static_assert(std::conjunction_v<std::is_trivial<FPoint>, std::is_standard_layout<FPoint>>);

template<class RElement>
using TVector = ::std::vector<RElement>;

#if USE_SIMD
constexpr FIndex BatchSize = 8;
#endif

template<class T>
ForceInline T* AllocArray(std::size_t Num) noexcept
{
#if USE_SIMD
	T* Ptr = AssumeVecAligned(static_cast<T*>(operator new(sizeof(T) * Num, static_cast<std::align_val_t>(VecAlignment), std::nothrow)));
	check(!(reinterpret_cast<std::uintptr_t>(Ptr) & (VecAlignment - 1)));
#else
	T* Ptr = new(std::nothrow) T[Num];
#endif
	check(Ptr);
	return Ptr;
}

template<class T>
ForceInline void FreeArray(T* Ptr) noexcept
{
#if USE_SIMD
	operator delete(AssumeVecAligned(Ptr), static_cast<std::align_val_t>(VecAlignment), std::nothrow);
#else
	delete[] Ptr;
#endif
}

#if USE_SIMD

#ifdef _MSC_VER
ForceInline __m128d VectorCall operator+(__m128d LHS, __m128d RHS) noexcept { return _mm_add_pd(LHS, RHS); }
ForceInline __m128d VectorCall operator-(__m128d LHS, __m128d RHS) noexcept { return _mm_sub_pd(LHS, RHS); }
ForceInline __m128d VectorCall operator*(__m128d LHS, double RHS) noexcept { return _mm_mul_pd(LHS, _mm_set1_pd(RHS)); }
ForceInline __m128d VectorCall operator/(__m128d LHS, double RHS) noexcept { return _mm_div_pd(LHS, _mm_set1_pd(RHS)); }
ForceInline __m128d& VectorCall operator+=(__m128d& LHS, __m128d RHS) noexcept { return LHS = _mm_add_pd(LHS, RHS); }
ForceInline __m128d& VectorCall operator-=(__m128d& LHS, __m128d RHS) noexcept { return LHS = _mm_sub_pd(LHS, RHS); }
ForceInline __m128d& VectorCall operator*=(__m128d& LHS, double RHS) noexcept { return LHS = _mm_mul_pd(LHS, _mm_set1_pd(RHS)); }
ForceInline __m128d& VectorCall operator/=(__m128d& LHS, double RHS) noexcept { return LHS = _mm_div_pd(LHS, _mm_set1_pd(RHS)); }
#endif

ForceInline __m128d VectorCall
Load1(const FPoint* Restrict Point) noexcept
{
	return _mm_load_pd(AssumeXmmAligned(reinterpret_cast<const double*>(Point)));
}

ForceInline __m256d VectorCall
Load2(const FPoint* Restrict Point) noexcept
{
	return _mm256_load_pd(AssumeYmmAligned(reinterpret_cast<const double*>(Point)));
}

ForceInline void VectorCall
Store1(FPoint* Restrict Point, __m128d Vector) noexcept
{
	_mm_store_pd(AssumeXmmAligned(reinterpret_cast<double*>(Point)), Vector);
}

ForceInline void VectorCall
Store2(FPoint* Restrict Point, __m256d Vector) noexcept
{
	_mm256_store_pd(AssumeYmmAligned(reinterpret_cast<double*>(Point)), Vector);
}

#endif

#if USE_SIMD_OPERATOR

using FLocalPoint = __m128d;
using FConstLocalPointRef = const __m128d;

ForceInline __m128d VectorCall
MakeZeroLocalPoint() noexcept
{
	return _mm_setzero_pd();
}

#ifndef _MSC_VER
ForceInline __m128d operator+(const FPoint& LHS, const FPoint& RHS) noexcept { return static_cast<__m128d>(LHS) + static_cast<__m128d>(RHS); }
ForceInline __m128d operator-(const FPoint& LHS, const FPoint& RHS) noexcept { return static_cast<__m128d>(LHS) - static_cast<__m128d>(RHS); }
ForceInline __m128d operator+(const FPoint& LHS, __m128d RHS) noexcept { return static_cast<__m128d>(LHS) + RHS; }
ForceInline __m128d operator-(const FPoint& LHS, __m128d RHS) noexcept { return static_cast<__m128d>(LHS) - RHS; }
ForceInline __m128d operator*(const FPoint& LHS, double RHS) noexcept { return static_cast<__m128d>(LHS) * _mm_set1_pd(RHS); }
ForceInline __m128d operator/(const FPoint& LHS, double RHS) noexcept { return static_cast<__m128d>(LHS) / _mm_set1_pd(RHS); }
ForceInline __m128d operator+(__m128d LHS, const FPoint& RHS) noexcept { return LHS + static_cast<__m128d>(RHS); }
ForceInline __m128d operator-(__m128d LHS, const FPoint& RHS) noexcept { return LHS - static_cast<__m128d>(RHS); }
ForceInline __m128d& operator+=(__m128d& LHS, const FPoint& RHS) noexcept { return LHS += static_cast<__m128d>(RHS); }
ForceInline __m128d& operator-=(__m128d& LHS, const FPoint& RHS) noexcept { return LHS -= static_cast<__m128d>(RHS); }
#endif

ForceInline FPoint& operator+=(FPoint& LHS, __m128d RHS) noexcept { return LHS = static_cast<__m128d>(LHS) + RHS; }
ForceInline FPoint& operator-=(FPoint& LHS, __m128d RHS) noexcept { return LHS = static_cast<__m128d>(LHS) - RHS; }
ForceInline FPoint& operator*=(FPoint& LHS, double RHS) noexcept { return LHS = static_cast<__m128d>(LHS) * RHS; }
ForceInline FPoint& operator/=(FPoint& LHS, double RHS) noexcept { return LHS = static_cast<__m128d>(LHS) / RHS; }

#else

using FLocalPoint = FPoint;
using FConstLocalPointRef = const FPoint&;

ForceInline FPoint MakeZeroLocalPoint() noexcept
{
	return {0.0, 0.0};
}

ForceInline FPoint operator+(const FPoint& LHS, const FPoint& RHS) noexcept { return {LHS.X + RHS.X, LHS.Y + RHS.Y}; }
ForceInline FPoint operator-(const FPoint& LHS, const FPoint& RHS) noexcept { return {LHS.X - RHS.X, LHS.Y - RHS.Y}; }
ForceInline FPoint operator*(const FPoint& LHS, double RHS) noexcept { return {LHS.X * RHS, LHS.Y * RHS}; }
ForceInline FPoint operator/(const FPoint& LHS, double RHS) noexcept { return {LHS.X / RHS, LHS.Y / RHS}; }
ForceInline FPoint& operator+=(FPoint& LHS, const FPoint& RHS) noexcept { return LHS.X += RHS.X, LHS.Y += RHS.Y, LHS; }
ForceInline FPoint& operator-=(FPoint& LHS, const FPoint& RHS) noexcept { return LHS.X -= RHS.X, LHS.Y -= RHS.Y, LHS; }
ForceInline FPoint& operator*=(FPoint& LHS, double RHS) noexcept { return LHS.X *= RHS, LHS.Y *= RHS, LHS; }
ForceInline FPoint& operator/=(FPoint& LHS, double RHS) noexcept { return LHS.X /= RHS, LHS.Y /= RHS, LHS; }

#endif

#if USE_SIMD_DISTANCE
inline double GetDistance(__m128d VL, __m128d VR) noexcept
{
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

#if USE_SIMD_POINTS_SWIZZLE_AOT
AttrPerf
#if USE_SIMD_POINTS_SWIZZLE_AOT_COPY
inline void SwizzlePoints(FPoint* Restrict SwizzledPoints, const FPoint* Restrict Points, FIndex NumPoint) noexcept
#else
inline void SwizzlePoints(FPoint* Restrict Points, FIndex NumPoint) noexcept
#endif
{
	const FIndex NumPointForBatch = NumPoint & ~(BatchSize - 1);

	for (FIndex PointId = 0; PointId < NumPointForBatch; PointId += BatchSize)
	{
		const __m256d VPoint01 = Load2(Points + PointId);
		const __m256d VPoint23 = Load2(Points + PointId + 2);
		const __m256d VPoint45 = Load2(Points + PointId + 4);
		const __m256d VPoint67 = Load2(Points + PointId + 6);
		const __m256d VPointX0213 = _mm256_unpacklo_pd(VPoint01, VPoint23);
		const __m256d VPointY0213 = _mm256_unpackhi_pd(VPoint01, VPoint23);
		const __m256d VPointX4657 = _mm256_unpacklo_pd(VPoint45, VPoint67);
		const __m256d VPointY4657 = _mm256_unpackhi_pd(VPoint45, VPoint67);
#if USE_SIMD_POINTS_SWIZZLE_AOT_COPY
		Store2(SwizzledPoints + PointId, VPointX0213);
		Store2(SwizzledPoints + PointId + 2, VPointY0213);
		Store2(SwizzledPoints + PointId + 4, VPointX4657);
		Store2(SwizzledPoints + PointId + 6, VPointY4657);
#else
		Store2(Points + PointId, VPointX0213);
		Store2(Points + PointId + 2, VPointY0213);
		Store2(Points + PointId + 4, VPointX4657);
		Store2(Points + PointId + 6, VPointY4657);
#endif
	}
#if USE_SIMD_POINTS_SWIZZLE_AOT_COPY
	BuiltinMemCpy(SwizzledPoints + NumPointForBatch, Points + NumPointForBatch, sizeof(FPoint) * (NumPoint - NumPointForBatch));
#endif
}
#endif

#if USE_OMP_NUM_THREAD
constexpr int OmpMaxNumThread = 64;

AttrPerf
inline int GetOmpDefaultNumThread() noexcept
{
	int Result;
	#pragma omp parallel default(none) shared(Result)
	{
		#pragma omp master
		{
			Result = omp_get_num_threads();
		}
	}
	return std::min(OmpMaxNumThread, Result);
}
#endif

struct FKMeans
{
	FPoint* Points = nullptr;
#if USE_SIMD_POINTS_SWIZZLE_AOT_COPY
	FPoint* SwizzledPoints = nullptr;
#endif
	FPoint* Centers = nullptr;
	FIndex* Assignment = nullptr;
	FIndex* OldAssignment = nullptr;
	FIndex* PointCount = nullptr;
	FIndex NumPoint;
	FIndex NumCenter;
#if USE_OMP_NUM_THREAD
	int NumThread;
#endif

	FKMeans(const FInPoint* Restrict InPoints, const FInPoint* Restrict InInitCenters, FIndex InNumPoint, FIndex InNumCenter) noexcept;

	~FKMeans() noexcept;

	TVector<FIndex> Run(int NumIteration = 1000);
};

inline FKMeans::FKMeans(const FInPoint* Restrict InPoints, const FInPoint* Restrict InInitCenters, FIndex InNumPoint, FIndex InNumCenter) noexcept
	: NumPoint(InNumPoint)
	, NumCenter(InNumCenter)
#if USE_OMP_NUM_THREAD
	, NumThread(GetOmpDefaultNumThread())
#endif
{
	SCOPED_TIMER(FKMeansConstructor);

	Points = AllocArray<FPoint>(NumPoint);
#if USE_SIMD_POINTS_SWIZZLE_AOT_COPY
	SwizzledPoints = AllocArray<FPoint>(NumPoint);
#endif
#if USE_OMP_UPDATE_CENTERS
	Centers = AllocArray<FPoint>(NumCenter * NumThread);
#else
	Centers = AllocArray<FPoint>(NumCenter);
#endif
	Assignment = AllocArray<FIndex>(NumPoint);
	OldAssignment = AllocArray<FIndex>(NumPoint);
#if USE_OMP_UPDATE_CENTERS
	PointCount = AllocArray<FIndex>(NumCenter * NumThread);
#else
	PointCount = AllocArray<FIndex>(NumCenter);
#endif
	BuiltinMemCpy(Points, InPoints, sizeof(FPoint) * NumPoint);
#if USE_SIMD_POINTS_SWIZZLE_AOT_COPY
	SwizzlePoints(SwizzledPoints, Points, NumPoint);
#elif USE_SIMD_POINTS_SWIZZLE_AOT
	SwizzlePoints(Points, NumPoint);
#endif
	BuiltinMemCpy(Centers, InInitCenters, sizeof(FPoint) * NumCenter);
}

FKMeans::~FKMeans() noexcept
{
	FreeArray(Points);
#if USE_SIMD_POINTS_SWIZZLE_AOT_COPY
	FreeArray(SwizzledPoints);
#endif
	FreeArray(Centers);
	FreeArray(Assignment);
	FreeArray(OldAssignment);
	FreeArray(PointCount);
}

AttrPerf
inline void UpdateAssignment(FIndex* Restrict Assignment, const FPoint* Restrict Points, const FPoint* Restrict Centers, FIndex NumPoint, FIndex NumCenter) noexcept
{
	SCOPED_TIMER(UpdateAssignment);

#if USE_SIMD_UPDATE_ASSIGNMENT
	const FIndex NumPointForBatch = NumPoint & ~(BatchSize - 1);

#if !USE_SIMD_ASSIGNMENT_SWIZZLE
	const __m256i VPerm = _mm256_set_epi32(7, 3, 5, 1, 6, 2, 4, 0);
#endif

#if USE_SIMD_CENTER_ID_ADD
	const __m256i VOne = _mm256_set1_epi32(1);
#endif

#if USE_OMP_UPDATE_ASSIGNMENT
	#pragma omp parallel for
#endif
	for (FIndex PointId = 0; PointId < NumPointForBatch; PointId += BatchSize)
	{
#if USE_SIMD_POINTS_SWIZZLE_AOT
		const __m256d VPointX0213 = Load2(Points + PointId);
		const __m256d VPointY0213 = Load2(Points + PointId + 2);
		const __m256d VPointX4657 = Load2(Points + PointId + 4);
		const __m256d VPointY4657 = Load2(Points + PointId + 6);
#else
		const __m256d VPoint01 = Load2(Points + PointId);
		const __m256d VPoint23 = Load2(Points + PointId + 2);
		const __m256d VPoint45 = Load2(Points + PointId + 4);
		const __m256d VPoint67 = Load2(Points + PointId + 6);
#if USE_SIMD_POINTS_SWIZZLE
		const __m256d VPointX0213 = _mm256_unpacklo_pd(VPoint01, VPoint23);
		const __m256d VPointY0213 = _mm256_unpackhi_pd(VPoint01, VPoint23);
		const __m256d VPointX4657 = _mm256_unpacklo_pd(VPoint45, VPoint67);
		const __m256d VPointY4657 = _mm256_unpackhi_pd(VPoint45, VPoint67);
#endif
#endif
		__m256d VMinDis0213 = _mm256_set1_pd(std::numeric_limits<double>::max());
		__m256d VMinDis4657 = _mm256_set1_pd(std::numeric_limits<double>::max());
		__m256 VAssign04261537 = _mm256_castsi256_ps(_mm256_set1_epi32(-1));
#if USE_SIMD_CENTER_ID_ADD
		__m256i VCenterId = _mm256_setzero_si256();
#endif
		for (FIndex CenterId = 0; CenterId < NumCenter; ++CenterId)
		{
#if USE_SIMD_POINTS_SWIZZLE
			const __m256d VCenterX = _mm256_broadcast_sd(&Centers[CenterId].X);
			const __m256d VCenterY = _mm256_broadcast_sd(&Centers[CenterId].Y);
			const __m256d VDiffX0213 = _mm256_sub_pd(VPointX0213, VCenterX);
			const __m256d VDiffX4657 = _mm256_sub_pd(VPointX4657, VCenterX);
			const __m256d VDiffY0213 = _mm256_sub_pd(VPointY0213, VCenterY);
			const __m256d VDiffY4657 = _mm256_sub_pd(VPointY4657, VCenterY);
			const __m256d VMulX0213 = _mm256_mul_pd(VDiffX0213, VDiffX0213);
			const __m256d VMulX4657 = _mm256_mul_pd(VDiffX4657, VDiffX4657);
#if USE_SIMD_FMA
			const __m256d VDis0213 = _mm256_fmadd_pd(VDiffY0213, VDiffY0213, VMulX0213);
			const __m256d VDis4657 = _mm256_fmadd_pd(VDiffY4657, VDiffY4657, VMulX4657);
#else
			const __m256d VMulY0213 = _mm256_mul_pd(VDiffY0213, VDiffY0213);
			const __m256d VMulY4657 = _mm256_mul_pd(VDiffY4657, VDiffY4657);
#endif
#else
			const __m256d VCenter = _mm256_broadcast_pd(AssumeXmmAligned(reinterpret_cast<const __m128d*>(&Centers[CenterId])));
			const __m256d VDiff01 = _mm256_sub_pd(VPoint01, VCenter);
			const __m256d VDiff23 = _mm256_sub_pd(VPoint23, VCenter);
			const __m256d VDiff45 = _mm256_sub_pd(VPoint45, VCenter);
			const __m256d VDiff67 = _mm256_sub_pd(VPoint67, VCenter);
			const __m256d VMul01 = _mm256_mul_pd(VDiff01, VDiff01);
			const __m256d VMul23 = _mm256_mul_pd(VDiff23, VDiff23);
			const __m256d VMul45 = _mm256_mul_pd(VDiff45, VDiff45);
			const __m256d VMul67 = _mm256_mul_pd(VDiff67, VDiff67);
			const __m256d VMulX0213 = _mm256_unpacklo_pd(VMul01, VMul23);
			const __m256d VMulX4657 = _mm256_unpacklo_pd(VMul45, VMul67);
			const __m256d VMulY0213 = _mm256_unpackhi_pd(VMul01, VMul23);
			const __m256d VMulY4657 = _mm256_unpackhi_pd(VMul45, VMul67);
#endif
#if !USE_SIMD_FMA
			const __m256d VDis0213 = _mm256_add_pd(VMulX0213, VMulY0213);
			const __m256d VDis4657 = _mm256_add_pd(VMulX4657, VMulY4657);
#endif
			const __m256 VCmp0213 = _mm256_castpd_ps(_mm256_cmp_pd(VDis0213, VMinDis0213, _CMP_LT_OQ));
			const __m256 VCmp4657 = _mm256_castpd_ps(_mm256_cmp_pd(VDis4657, VMinDis4657, _CMP_LT_OQ));
			VMinDis0213 = _mm256_min_pd(VDis0213, VMinDis0213);
			VMinDis4657 = _mm256_min_pd(VDis4657, VMinDis4657);
			const __m256 VCmp04261537 = _mm256_blend_ps(VCmp0213, VCmp4657, 0b10101010);
#if USE_SIMD_CENTER_ID_ADD
			VAssign04261537 = _mm256_blendv_ps(VAssign04261537, _mm256_castsi256_ps(VCenterId), VCmp04261537);
			VCenterId = _mm256_add_epi32(VCenterId, VOne);
#else
			const __m256 VCenterId = _mm256_castsi256_ps(_mm256_set1_epi32(CenterId));
			VAssign04261537 = _mm256_blendv_ps(VAssign04261537, VCenterId, VCmp04261537);
#endif
		}
#if USE_SIMD_ASSIGNMENT_SWIZZLE
		_mm256_store_ps(AssumeYmmAligned(reinterpret_cast<float*>(Assignment + PointId)), VAssign04261537);
#else
		const __m256 VAssign01234567 = _mm256_permutevar8x32_ps(VAssign04261537, VPerm);
		_mm256_store_ps(AssumeYmmAligned(reinterpret_cast<float*>(Assignment + PointId)), VAssign01234567);
#endif
	}

	for (FIndex PointId = NumPointForBatch; PointId < NumPoint; ++PointId)
	{
		FConstLocalPointRef Point = Points[PointId];
		double MinDistance = std::numeric_limits<double>::max();
		FIndex Result = -1;
		for (FIndex CenterId = 0; CenterId < NumCenter; ++CenterId)
		{
			FConstLocalPointRef Center = Centers[CenterId];
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
#if USE_OMP_UPDATE_ASSIGNMENT
	#pragma omp parallel for
#endif
	for (FIndex PointId = 0; PointId < NumPoint; ++PointId)
	{
		FConstLocalPointRef Point = Points[PointId];
		double MinDistance = std::numeric_limits<double>::max();
		FIndex Result = -1;
		for (FIndex CenterId = 0; CenterId < NumCenter; ++CenterId)
		{
			FConstLocalPointRef Center = Centers[CenterId];
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
#if USE_OMP_UPDATE_CENTERS
inline void UpdateCenters(FPoint* Restrict PerThreadCenters, FIndex* Restrict PerThreadPointCount, const FPoint* Restrict Points, const FIndex* Restrict Assignment, FIndex NumPoint, FIndex NumCenter, int NumThread) noexcept
{
	SCOPED_TIMER(UpdateCenters);

	// TODO: Cache these
	const FIndex NumPointDivThread = NumPoint / NumThread;
	const FIndex NumPointModThread = NumPoint % NumThread;
	const FIndex NumCenterDivThread = NumCenter / NumThread;
	const FIndex NumCenterModThread = NumCenter % NumThread;

#if !USE_OMP_PARALLEL_MEMSET
	{
		SCOPED_TIMER(ZeroCenterAndPointCount);
		BuiltinMemSet(PerThreadCenters, 0, sizeof(FPoint) * NumCenter * NumThread);
		BuiltinMemSet(PerThreadPointCount, 0, sizeof(FIndex) * NumCenter * NumThread);
	}
#endif

	#pragma omp parallel num_threads(NumThread) default(none) firstprivate(PerThreadCenters, PerThreadPointCount, Points, Assignment, NumPoint, NumCenter, NumThread, NumPointDivThread, NumPointModThread, NumCenterDivThread, NumCenterModThread)
	{
		const int ThreadId = omp_get_thread_num();

		{
			const FIndex LocalNumPoint = ThreadId < NumPointModThread ? NumPointDivThread + 1 : NumPointDivThread;
			const FIndex PointIdBegin = ThreadId < NumPointModThread ? ThreadId * LocalNumPoint : ThreadId * LocalNumPoint + NumPointModThread;
			const FIndex PointIdEnd = PointIdBegin + LocalNumPoint;
			FPoint* Restrict LocalCenters = PerThreadCenters + NumCenter * ThreadId;
			FIndex* Restrict LocalPointCount = PerThreadPointCount + NumCenter * ThreadId;

#if USE_OMP_PARALLEL_MEMSET
			{
				SCOPED_TIMER(ZeroCenterAndPointCount);
				BuiltinMemSet(LocalCenters, 0, sizeof(FPoint) * NumCenter);
				BuiltinMemSet(LocalPointCount, 0, sizeof(FIndex) * NumCenter);
			}
#endif

			{
				SCOPED_TIMER(UpdateCenters_Gather);
				for (FIndex PointId = PointIdBegin; PointId < PointIdEnd; ++PointId)
				{
					const FIndex CenterId = Assignment[PointId];
					LocalCenters[CenterId] += Points[PointId];
					++LocalPointCount[CenterId];
				}
			}
		}

		#pragma omp barrier

		{
			SCOPED_TIMER(UpdateCenters_Division);

			const FIndex LocalNumCenter = ThreadId < NumCenterModThread ? NumCenterDivThread + 1 : NumCenterDivThread;
			const FIndex CenterIdBegin = ThreadId < NumCenterModThread ? ThreadId * LocalNumCenter : ThreadId * LocalNumCenter + NumCenterModThread;
			const FIndex CenterIdEnd = CenterIdBegin + LocalNumCenter;

			for (FIndex CenterId = CenterIdBegin; CenterId < CenterIdEnd; ++CenterId)
			{
				FLocalPoint Center = MakeZeroLocalPoint();
				FIndex NumPointAssigned = 0;
				for (int LocalThreadId = 0; LocalThreadId < NumThread; ++LocalThreadId)
				{
					Center += PerThreadCenters[LocalThreadId * NumCenter + CenterId];
					NumPointAssigned += PerThreadPointCount[LocalThreadId * NumCenter + CenterId];
				}
				Center /= NumPointAssigned;
				PerThreadCenters[CenterId] = Center;
			}
		}
	}
}
#else
inline void UpdateCenters(FPoint* Restrict Centers, FIndex* Restrict PointCount, const FPoint* Restrict Points, const FIndex* Restrict Assignment, FIndex NumPoint, FIndex NumCenter) noexcept
{
	SCOPED_TIMER(UpdateCenters);

#if USE_SIMD_POINTS_SWIZZLE_AOT && !USE_SIMD_POINTS_SWIZZLE_AOT_COPY
	{
		SCOPED_TIMER(ZeroCenterAndPointCount)
		BuiltinMemSet(Centers, 0, sizeof(FPoint) * NumCenter);
		BuiltinMemSet(PointCount, 0, sizeof(FIndex) * NumCenter);
	}

	const FIndex NumPointForBatch = NumPoint & ~(BatchSize - 1);

	for (FIndex PointId = 0; PointId < NumPointForBatch; PointId += BatchSize)
	{
		const __m256d VPointX0213 = Load2(Points + PointId);
		const __m256d VPointY0213 = Load2(Points + PointId + 2);
		const __m256d VPointX4657 = Load2(Points + PointId + 4);
		const __m256d VPointY4657 = Load2(Points + PointId + 6);
		const __m256d VPoint01 = _mm256_unpacklo_pd(VPointX0213, VPointY0213);
		const __m256d VPoint23 = _mm256_unpackhi_pd(VPointX0213, VPointY0213);
		const __m256d VPoint45 = _mm256_unpacklo_pd(VPointX4657, VPointY4657);
		const __m256d VPoint67 = _mm256_unpackhi_pd(VPointX4657, VPointY4657);
		const __m128d VPoint0 = _mm256_castpd256_pd128(VPoint01);
		const __m128d VPoint1 = _mm256_extractf128_pd(VPoint01, 1);
		const __m128d VPoint2 = _mm256_castpd256_pd128(VPoint23);
		const __m128d VPoint3 = _mm256_extractf128_pd(VPoint23, 1);
		const __m128d VPoint4 = _mm256_castpd256_pd128(VPoint45);
		const __m128d VPoint5 = _mm256_extractf128_pd(VPoint45, 1);
		const __m128d VPoint6 = _mm256_castpd256_pd128(VPoint67);
		const __m128d VPoint7 = _mm256_extractf128_pd(VPoint67, 1);
		const FIndex CenterId0 = Assignment[PointId];
		const FIndex CenterId1 = Assignment[PointId + 1];
		const FIndex CenterId2 = Assignment[PointId + 2];
		const FIndex CenterId3 = Assignment[PointId + 3];
		const FIndex CenterId4 = Assignment[PointId + 4];
		const FIndex CenterId5 = Assignment[PointId + 5];
		const FIndex CenterId6 = Assignment[PointId + 6];
		const FIndex CenterId7 = Assignment[PointId + 7];
		Store1(&Centers[CenterId0], _mm_add_pd(Load1(&Centers[CenterId0]), VPoint0));
		Store1(&Centers[CenterId1], _mm_add_pd(Load1(&Centers[CenterId1]), VPoint1));
		Store1(&Centers[CenterId2], _mm_add_pd(Load1(&Centers[CenterId2]), VPoint2));
		Store1(&Centers[CenterId3], _mm_add_pd(Load1(&Centers[CenterId3]), VPoint3));
		Store1(&Centers[CenterId4], _mm_add_pd(Load1(&Centers[CenterId4]), VPoint4));
		Store1(&Centers[CenterId5], _mm_add_pd(Load1(&Centers[CenterId5]), VPoint5));
		Store1(&Centers[CenterId6], _mm_add_pd(Load1(&Centers[CenterId6]), VPoint6));
		Store1(&Centers[CenterId7], _mm_add_pd(Load1(&Centers[CenterId7]), VPoint7));
		++PointCount[CenterId0];
		++PointCount[CenterId1];
		++PointCount[CenterId2];
		++PointCount[CenterId3];
		++PointCount[CenterId4];
		++PointCount[CenterId5];
		++PointCount[CenterId6];
		++PointCount[CenterId7];
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
#elif USE_SIMD_ASSIGNMENT_SWIZZLE
	{
		SCOPED_TIMER(ZeroCenterAndPointCount)
		BuiltinMemSet(Centers, 0, sizeof(FPoint) * NumCenter);
		BuiltinMemSet(PointCount, 0, sizeof(FIndex) * NumCenter);
	}

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
	{
		SCOPED_TIMER(ZeroCenterAndPointCount);
		BuiltinMemSet(Centers, 0, sizeof(FPoint) * NumCenter);
		BuiltinMemSet(PointCount, 0, sizeof(FIndex) * NumCenter);
	}

	{
		SCOPED_TIMER(UpdateCenters_Gather);
		for (FIndex PointId = 0; PointId < NumPoint; ++PointId)
		{
			const FIndex CenterId = Assignment[PointId];
			Centers[CenterId] += Points[PointId];
			++PointCount[CenterId];
		}
	}

	{
		SCOPED_TIMER(UpdateCenters_Division);
		for (FIndex CenterId = 0; CenterId < NumCenter; ++CenterId)
		{
			Centers[CenterId] /= PointCount[CenterId];
		}
	}
#endif
}
#endif

AttrPerf
inline TVector<FIndex> FinalizeAssignment(FIndex* Restrict Assignment, FIndex NumPoint) noexcept
{
	SCOPED_TIMER(FinalizeAssignment);

#if USE_SIMD_ASSIGNMENT_SWIZZLE
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
	SCOPED_TIMER(FKMeansRun);

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

#if USE_SIMD_POINTS_SWIZZLE_AOT_COPY
		UpdateAssignment(Assignment, SwizzledPoints, Centers, NumPoint, NumCenter);
#else
		UpdateAssignment(Assignment, Points, Centers, NumPoint, NumCenter);
#endif

		{
			SCOPED_TIMER(CompareForQuit);
			if (!BuiltinMemCmp(Assignment, OldAssignment, sizeof(FIndex) * NumPoint))
			{
				goto JConverge;
			}
		}

#if USE_OMP_UPDATE_CENTERS
		UpdateCenters(Centers, PointCount, Points, Assignment, NumPoint, NumCenter, NumThread);
#else
		UpdateCenters(Centers, PointCount, Points, Assignment, NumPoint, NumCenter);
#endif

		swap(Assignment, OldAssignment);
	}

JConverge:
	std::cout << "Finished in " << IterationId << " iterations.\n";
	return FinalizeAssignment(Assignment, NumPoint);
}

// Public interface
Kmeans::Kmeans(const std::vector<Point>& InPoints, const std::vector<Point>& InInitCenters)
	: Impl(new FKMeans(InPoints.data(), InInitCenters.data(), static_cast<FIndex>(InPoints.size()), static_cast<FIndex>(InInitCenters.size())))
{}

std::vector<index_t> Kmeans::Run(int MaxIterations)
{
	return Impl->Run(MaxIterations);
}

void FKMeansDeleter::operator()(FKMeans* Obj) const noexcept
{
	default_delete::operator()(Obj);
}
