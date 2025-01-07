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


// https://stackoverflow.com/questions/2808398/easily-measure-elapsed-time
template <
    class result_t   = std::chrono::milliseconds,
    class clock_t    = std::chrono::steady_clock,
    class duration_t = std::chrono::milliseconds
>
auto since(std::chrono::time_point<clock_t, duration_t> const& start)
{
    return std::chrono::duration_cast<result_t>(clock_t::now() - start);
}
