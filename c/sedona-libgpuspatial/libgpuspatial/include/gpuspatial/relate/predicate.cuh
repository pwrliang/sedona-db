#pragma once

namespace gpuspatial {

enum class Predicate {
  kEquals,
  kDisjoint,
  kTouches,
  kContains,
  kCovers,
  kIntersects,
  kWithin,
  kCoveredBy
};

/**
 * @brief Converts a Predicate enum class value to its string representation.
 *
 * @param predicate The Predicate value to convert.
 * @return const char* A string literal corresponding to the enum value.
 * Returns "Unknown Predicate" if the value is not recognized.
 */
inline const char* PredicateToString(Predicate predicate) {
  switch (predicate) {
    case Predicate::kEquals:
      return "Equals";
    case Predicate::kDisjoint:
      return "Disjoint";
    case Predicate::kTouches:
      return "Touches";
    case Predicate::kContains:
      return "Contains";
    case Predicate::kCovers:
      return "Covers";
    case Predicate::kIntersects:
      return "Intersects";
    case Predicate::kWithin:
      return "Within";
    case Predicate::kCoveredBy:
      return "CoveredBy";
    default:
      // Handle any unexpected values safely
      return "Unknown Predicate";
  }
}
}  // namespace gpuspatial
