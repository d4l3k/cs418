% some trivial tests for hw1
% These are mostly just the examples from the homework problem statements.
% These are certainly not a complete set of tests.  You should modify this
% and add tests of your own.
-module(hw1_test).

-include_lib("eunit/include/eunit.hrl").
-import(hw1, [allTails/1, closest/2, longestOverlap/2]).

% The function closest(P, PointList) involve floating point computations,
% and that means round-off error.  We need a test that the value from the
% function being tested is "close" enough to the expected answer. That's
% what tolerance() and close_enough provide.  tolerance() sets the relative
% error tolerance and close_enough checks that two values match to within
% the tolerance.  If the Reference value is not a float, then the match
% must be exact.
tolerance() -> 1.0e-12.  % relative error tolerance
close_enough(V, V) -> true;  % close_enough will work with values of any type
close_enough(Reference, TestResult) when is_float(Reference), is_number(TestResult) ->
  abs(TestResult-Reference) < tolerance()*max(abs(Reference), abs(TestResult));
close_enough(_, _) -> false.

% close_match returns true if the result from closest has the expected
% Index and if the Distanc is close_enough.
close_match({Index, ExpectedDistance}, {Index, TestResultDistance}) ->
  close_enough(ExpectedDistance, TestResultDistance);
close_match(_, _) -> false.

closest_test() ->
  ?assertEqual({1, 0.0}, closest([1,2,3], [[1,2,3]])),
  ?assertEqual({0, 0.0}, closest([1,2,3], [])),
  ?assertEqual({2, 0.0}, closest([1,2,3], [[2,3,4], [1,2,3]])),
  ?assertEqual({1, 0.0}, closest([1,2,3], [[1,2,3], [2,3,4]])),
  ?assert(close_match({3,math:sqrt(2.0)}, closest([1,2,3], [[4,5,6], [0,1,4], [1,1,2], [2,3,4]]))),
  ?assert(close_match({2,math:sqrt(3.0)},
                      closest([1,2,3], [[4,5,6], [0,1,4], [1,8,2], [2,3,4]]))),
  ?assert(close_match({4,math:sqrt(3.0)},
                      closest([1,2,3], [[4,5,6], [0,8,4], [1,8,2], [2,3,4]]))).

allTails_test() ->
  ?assertEqual([[], [3], [2,3], [1,2,3]], allTails([1, 2, 3])),
  ?assertEqual([[], [1]], allTails([1])),
  ?assertEqual([[]], allTails([])).

longestOverlap_test() ->
  L1 = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],
  L2 = [0,1,2,3,4,6,7,8,10,11,12,13,9,14,15,16,18,19,20],
  L3 = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],
  ?assertEqual({6, 3}, longestOverlap(L1, L2)),
  ?assertEqual({1, 0}, longestOverlap(L1, L3)),
  ?assertEqual({1, 5}, longestOverlap(L2, L3)),
  ?assertEqual({1, 0}, longestOverlap([1,2,3], [])),
  ?assertEqual({6, 4}, longestOverlap([0,1,2,3,0,4,5,6,7], [1,1,2,3,1,4,5,6,7])),
  ?assertEqual({2, 4}, longestOverlap([0,1,2,3,4,0,4,5,6], [1,1,2,3,4,1,4,5,6])),
  ?assertEqual({2, 4}, longestOverlap([0,1,2,3,4,0,4,5,6,7], [1,1,2,3,4,1,4,5,6,7])),
  ?assertEqual({1, 0}, longestOverlap([], [1,2,3])).
