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

//// START

#define USE_CHECK 1
#define USE_PERF 1
#define USE_STATS 0

// Set to non-zero to make the program runs longer
#define USE_FIX_NUM_ITERATION 400

#define USE_OMP 1
#define USE_SIMD 1

#define USE_SIMD_OPERATOR                  (1 && USE_SIMD) // 1
#define USE_SIMD_DISTANCE                  (1 && USE_SIMD_OPERATOR) // 1
#define USE_SIMD_UPDATE_ASSIGNMENT         (1 && USE_SIMD) // 1
#define USE_SIMD_CENTER_ID_ADD             (1 && USE_SIMD_UPDATE_ASSIGNMENT) // 1
#define USE_SIMD_POINTS_SWIZZLE            (1 && USE_SIMD) // 1
#define USE_SIMD_FMA                       (1 && USE_SIMD_POINTS_SWIZZLE) // 1

#define USE_OMP_NUM_THREAD                 (1 && USE_OMP)
#define USE_OMP_UPDATE_ASSIGNMENT          (1 && USE_OMP) // 1
#define USE_OMP_UPDATE_CENTERS             (1 && USE_OMP_NUM_THREAD) // 1
#define USE_OMP_PARALLEL_MEMSET            (0 && USE_OMP_UPDATE_CENTERS)

#define USE_PACKED_CENTERS                 1 // 1

// Assume x86-64 always have cache line size 64-byte.
// And align FPackedCenter to cache line boundary to avoid false sharing when using OMP on UpdateCenters,
// otherwise just pack as tightest as possible, i.e.
#define USE_PACKED_CENTERS_ALIGN           (USE_OMP_UPDATE_CENTERS ? 64 : 4) // (64, 4)

static_assert(!USE_PACKED_CENTERS || (USE_PACKED_CENTERS_ALIGN >= 4 && !(USE_PACKED_CENTERS_ALIGN & (USE_PACKED_CENTERS_ALIGN - 1))));

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

struct
#if USE_SIMD
alignas(XmmAlignment)
#endif
FPoint
{
	double X;
	double Y;

// NOTE: Under O1, cannot count on GCC for inlining vectorized FPoint operators
#if USE_SIMD_OPERATOR
	ForceInline VectorCall
	operator __m128d() const noexcept
	{
		return _mm_load_pd(AssumeXmmAligned(reinterpret_cast<const double*>(this)));
	}

	ForceInline FPoint& VectorCall
	operator=(__m128d InVec) noexcept
	{
		_mm_store_pd(AssumeXmmAligned(reinterpret_cast<double*>(this)), InVec);
		return *this;
	}
#endif
};

static_assert(sizeof(FInPoint) == sizeof(FPoint));
static_assert(std::conjunction_v<std::is_trivial<FPoint>, std::is_standard_layout<FPoint>>);

#if USE_PACKED_CENTERS
#if USE_PACKED_CENTERS_ALIGN == 4
#pragma pack(push, 4)
#endif
struct alignas(USE_PACKED_CENTERS_ALIGN) FPackedCenter
{
	double X;
	double Y;
	FIndex Count;

#if USE_SIMD_OPERATOR
	ForceInline VectorCall
	operator __m128d() const noexcept
	{
#if USE_PACKED_CENTERS_ALIGN >= 16
		return _mm_load_pd(AssumeXmmAligned(reinterpret_cast<const double*>(this)));
#else
		return _mm_loadu_pd(reinterpret_cast<const double*>(this));
#endif
	}

	ForceInline FPackedCenter& VectorCall
	operator=(__m128d InVec) noexcept
	{
#if USE_PACKED_CENTERS_ALIGN >= 16
		_mm_store_pd(AssumeXmmAligned(reinterpret_cast<double*>(this)), InVec);
#else
		_mm_storeu_pd(reinterpret_cast<double*>(this), InVec);
#endif
		return *this;
	}
#endif
};
#if USE_PACKED_CENTERS_ALIGN == 4
#pragma pack(pop)
#endif

// Resharper seems not to respect #pragma pack
#ifndef __RESHARPER__
static_assert(sizeof(FPackedCenter) == ((20 + USE_PACKED_CENTERS_ALIGN - 1) & ~(USE_PACKED_CENTERS_ALIGN - 1)));
#endif
static_assert(std::conjunction_v<std::is_trivial<FPackedCenter>, std::is_standard_layout<FPackedCenter>>);

using FCenter = FPackedCenter;
#else
using FCenter = FPoint;
#endif

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

#if USE_PACKED_CENTERS
#ifndef _MSC_VER
ForceInline __m128d& operator+=(__m128d& LHS, const FPackedCenter& RHS) noexcept { return LHS += static_cast<__m128d>(RHS); }
#endif

ForceInline FPackedCenter& operator+=(FPackedCenter& LHS, __m128d RHS) noexcept { return LHS = static_cast<__m128d>(LHS) + RHS; }
ForceInline FPackedCenter& operator/=(FPackedCenter& LHS, double RHS) noexcept { return LHS = static_cast<__m128d>(LHS) / RHS; }
#endif

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

#if USE_PACKED_CENTERS
ForceInline FPackedCenter& operator+=(FPackedCenter& LHS, __m128d RHS) noexcept { return LHS.X += RHS.X, LHS.Y += RHS.Y, LHS; }
ForceInline FPackedCenter& operator/=(FPackedCenter& LHS, double RHS) noexcept { return LHS.X /= RHS, LHS.Y /= RHS, LHS; }
#endif

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
	FCenter* Centers = nullptr;
	FIndex* Assignment = nullptr;
	FIndex* OldAssignment = nullptr;
#if !USE_PACKED_CENTERS
	FIndex* PointCount = nullptr;
#endif
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
#if USE_OMP_UPDATE_CENTERS
	Centers = AllocArray<FCenter>(NumCenter * NumThread);
#else
	Centers = AllocArray<FCenter>(NumCenter);
#endif
	Assignment = AllocArray<FIndex>(NumPoint);
	OldAssignment = AllocArray<FIndex>(NumPoint);
#if !USE_PACKED_CENTERS
#if USE_OMP_UPDATE_CENTERS
	PointCount = AllocArray<FIndex>(NumCenter * NumThread);
#else
	PointCount = AllocArray<FIndex>(NumCenter);
#endif
#endif
	BuiltinMemCpy(Points, InPoints, sizeof(FPoint) * NumPoint);
#if USE_PACKED_CENTERS
	for (FIndex CenterId = 0; CenterId < NumCenter; ++CenterId)
	{
		BuiltinMemCpy(&Centers[CenterId], &InInitCenters[CenterId], sizeof(FPoint));
	}
#else
	BuiltinMemCpy(Centers, InInitCenters, sizeof(FPoint) * NumCenter);
#endif
}

FKMeans::~FKMeans() noexcept
{
	FreeArray(Points);
	FreeArray(Centers);
	FreeArray(Assignment);
	FreeArray(OldAssignment);
#if !USE_PACKED_CENTERS
	FreeArray(PointCount);
#endif
}

AttrPerf
inline void UpdateAssignment(FIndex* Restrict Assignment, const FPoint* Restrict Points, const FCenter* Restrict Centers, FIndex NumPoint, FIndex NumCenter) noexcept
{
	SCOPED_TIMER(UpdateAssignment);

#if USE_SIMD_UPDATE_ASSIGNMENT
	const FIndex NumPointForBatch = NumPoint & ~(BatchSize - 1);

	const __m256i VPerm = _mm256_set_epi32(7, 3, 5, 1, 6, 2, 4, 0);

#if USE_SIMD_CENTER_ID_ADD
	const __m256i VOne = _mm256_set1_epi32(1);
#endif

#if USE_OMP_UPDATE_ASSIGNMENT
	#pragma omp parallel for
#endif
	for (FIndex PointId = 0; PointId < NumPointForBatch; PointId += BatchSize)
	{
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
#if USE_PACKED_CENTERS
			const __m256d VCenter = _mm256_broadcast_pd(reinterpret_cast<const __m128d*>(&Centers[CenterId]));
#else
			const __m256d VCenter = _mm256_broadcast_pd(AssumeXmmAligned(reinterpret_cast<const __m128d*>(&Centers[CenterId])));
#endif
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
		const __m256 VAssign01234567 = _mm256_permutevar8x32_ps(VAssign04261537, VPerm);
		_mm256_store_ps(AssumeYmmAligned(reinterpret_cast<float*>(Assignment + PointId)), VAssign01234567);
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
#if USE_PACKED_CENTERS
inline void UpdateCenters(FPackedCenter* Restrict PerThreadCenters, const FPoint* Restrict Points, const FIndex* Restrict Assignment, FIndex NumPoint, FIndex NumCenter, int NumThread) noexcept
#else
inline void UpdateCenters(FPoint* Restrict PerThreadCenters, FIndex* Restrict PerThreadPointCount, const FPoint* Restrict Points, const FIndex* Restrict Assignment, FIndex NumPoint, FIndex NumCenter, int NumThread) noexcept
#endif
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
		BuiltinMemSet(PerThreadCenters, 0, sizeof(FCenter) * NumCenter * NumThread);
#if !USE_PACKED_CENTERS
		BuiltinMemSet(PerThreadPointCount, 0, sizeof(FIndex) * NumCenter * NumThread);
#endif
	}
#endif

	#pragma omp parallel num_threads(NumThread)
	{
		const int ThreadId = omp_get_thread_num();

		{
			const FIndex LocalNumPoint = ThreadId < NumPointModThread ? NumPointDivThread + 1 : NumPointDivThread;
			const FIndex PointIdBegin = ThreadId < NumPointModThread ? ThreadId * LocalNumPoint : ThreadId * LocalNumPoint + NumPointModThread;
			const FIndex PointIdEnd = PointIdBegin + LocalNumPoint;
			FCenter* Restrict LocalCenters = PerThreadCenters + NumCenter * ThreadId;
#if !USE_PACKED_CENTERS
			FIndex* Restrict LocalPointCount = PerThreadPointCount + NumCenter * ThreadId;
#endif

#if USE_OMP_PARALLEL_MEMSET
			{
				SCOPED_TIMER(ZeroCenterAndPointCount);
				BuiltinMemSet(PerThreadCenters, 0, sizeof(FCenter) * NumCenter * NumThread);
#if !USE_PACKED_CENTERS
				BuiltinMemSet(PerThreadPointCount, 0, sizeof(FIndex) * NumCenter * NumThread);
#endif
			}
#endif

			{
				SCOPED_TIMER(UpdateCenters_Gather);
				for (FIndex PointId = PointIdBegin; PointId < PointIdEnd; ++PointId)
				{
					const FIndex CenterId = Assignment[PointId];
					LocalCenters[CenterId] += Points[PointId];
#if USE_PACKED_CENTERS
					++LocalCenters[CenterId].Count;
#else
					++LocalPointCount[CenterId];
#endif
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
#if USE_PACKED_CENTERS
					NumPointAssigned += PerThreadCenters[LocalThreadId * NumCenter + CenterId].Count;
#else
					NumPointAssigned += PerThreadPointCount[LocalThreadId * NumCenter + CenterId];
#endif
				}
				Center /= NumPointAssigned;
				PerThreadCenters[CenterId] = Center;
			}
		}
	}
}
#else
#if USE_PACKED_CENTERS
inline void UpdateCenters(FPackedCenter* Restrict Centers, const FPoint* Restrict Points, const FIndex* Restrict Assignment, FIndex NumPoint, FIndex NumCenter) noexcept
#else
inline void UpdateCenters(FPoint* Restrict Centers, FIndex* Restrict PointCount, const FPoint* Restrict Points, const FIndex* Restrict Assignment, FIndex NumPoint, FIndex NumCenter) noexcept
#endif
{
	SCOPED_TIMER(UpdateCenters);

	{
		SCOPED_TIMER(ZeroCenterAndPointCount);
		BuiltinMemSet(Centers, 0, sizeof(FCenter) * NumCenter);
#if !USE_PACKED_CENTERS
		BuiltinMemSet(PointCount, 0, sizeof(FIndex) * NumCenter);
#endif
	}

	{
		SCOPED_TIMER(UpdateCenters_Gather);
		for (FIndex PointId = 0; PointId < NumPoint; ++PointId)
		{
			const FIndex CenterId = Assignment[PointId];
			Centers[CenterId] += Points[PointId];
#if USE_PACKED_CENTERS
			++Centers[CenterId].Count;
#else
			++PointCount[CenterId];
#endif
		}
	}

	{
		SCOPED_TIMER(UpdateCenters_Division);
		for (FIndex CenterId = 0; CenterId < NumCenter; ++CenterId)
		{
#if USE_PACKED_CENTERS
			Centers[CenterId] /= Centers[CenterId].Count;
#else
			Centers[CenterId] /= PointCount[CenterId];
#endif
		}
	}
}
#endif

AttrPerf
inline TVector<FIndex> FinalizeAssignment(FIndex* Restrict Assignment, FIndex NumPoint) noexcept
{
	SCOPED_TIMER(FinalizeAssignment);

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

		UpdateAssignment(Assignment, Points, Centers, NumPoint, NumCenter);

#if !USE_FIX_NUM_ITERATION
		{
			SCOPED_TIMER(CompareForQuit);
			if (!BuiltinMemCmp(Assignment, OldAssignment, sizeof(FIndex) * NumPoint))
			{
				goto JConverge;
			}
		}
#endif

#if USE_OMP_UPDATE_CENTERS
#if USE_PACKED_CENTERS
		UpdateCenters(Centers, Points, Assignment, NumPoint, NumCenter, NumThread);
#else
		UpdateCenters(Centers, PointCount, Points, Assignment, NumPoint, NumCenter, NumThread);
#endif
#elif USE_PACKED_CENTERS
		UpdateCenters(Centers, Points, Assignment, NumPoint, NumCenter);
#else
		UpdateCenters(Centers, PointCount, Points, Assignment, NumPoint, NumCenter);
#endif

		swap(Assignment, OldAssignment);
	}

#if !USE_FIX_NUM_ITERATION
JConverge:
#endif
	std::cout << "Finished in " << IterationId << " iterations.\n";
	return FinalizeAssignment(Assignment, NumPoint);
}

// Public interface
Kmeans::Kmeans(const std::vector<Point>& InPoints, const std::vector<Point>& InInitCenters)
	: Impl(new FKMeans(InPoints.data(), InInitCenters.data(), static_cast<FIndex>(InPoints.size()), static_cast<FIndex>(InInitCenters.size())))
{}

std::vector<index_t> Kmeans::Run(int MaxIterations)
{
#if USE_FIX_NUM_ITERATION
	(void) MaxIterations;
	return Impl->Run(USE_FIX_NUM_ITERATION);
#else
	return Impl->Run(MaxIterations);
#endif
}

void FKMeansDeleter::operator()(FKMeans* Obj) const noexcept
{
	default_delete::operator()(Obj);
}
