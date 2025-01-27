//----------------------------------------------------------------------------------------
/**
 * \file       utils.hpp
 * \author     Matyáš Švadlenka
 * \date       2024/12/09
 * \brief      Header for special functions and constants.
*/
//----------------------------------------------------------------------------------------
#pragma once
#include <chrono>
#include <string>

// ---- Input from GUI ---- 
#define PAUSE 0 ///< Pause the algorithm
#define RUN 1 ///< Un-pause the algorithm
#define NEXT 2 ///< Run/load the next step/state of algorithm
#define PREVIOUS 3 ///< Load back the previous state of algorithm
#define QUIT 4 ///< Exit from the program
const std::string eventNames[5] = { "PAUSE", "RUN", "NEXT", "PREVIOUS", "QUIT" };

typedef std::chrono::duration<float> fsec;
typedef std::chrono::milliseconds ms;

// https://stackoverflow.com/questions/2808398/easily-measure-elapsed-time
// https://stackoverflow.com/questions/14391327/how-to-get-duration-as-int-millis-and-float-seconds-from-chrono
template <
    class clock_t    = std::chrono::steady_clock,
    class duration_t = std::chrono::milliseconds
>
std::chrono::duration<float> since(std::chrono::time_point<clock_t, duration_t> const& start)
{
    fsec total = clock_t::now() - start;
    return total;
}

/**
   Casts timer difference to miliseconds.

   \param[in] duration The timers difference (end - start)
   \returns The duration cast to milseconds
*/
inline long long toMS(fsec const& duration) {
    return std::chrono::duration_cast<ms>(duration).count();
}

/**
   Clamps number between min and max.

   \param[in] n number
   \param[in] lb lower bound
   \param[in] ub upper bound
   \returns Number in range <lb, ub>
*/
inline float clip(const float n, const float lb, const float ub) {
    return std::max(lb, std::min(n, ub));
}

