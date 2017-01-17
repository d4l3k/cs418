-module(hw1).

% functions that are included to make the assignment easier.
-export([allLess/2, distance/2, sloppyZip/2]).

% functions that you need to write to complete the assignment
-export([allTails/1, closest/2, longestOverlap/2]).

% distance(P1, P1) -> Euclidean distance from P1 to P2
%   P1 and P2 must be list of numbers, and the lengths of P1 and P2 must match.
%   Examples,
%     distance([1], [2]) -> 1.0
%     distance([0,0], [3,4]) -> 5.0
%     distance([1, 2, 3], [5, -10, 0]) -> 13.0
%     distance([1, 3.14159, 2.5], [1.732, 8, -4/3]) -> 6.231726580374371
%   The function 'distance' is included in this template because it may be
%   useful when implementing the function 'closest'.
distance(P1, P2) when is_list(P1), is_list(P2) ->
  math:sqrt(lists:sum([(X1-X2)*(X1-X2) || {X1, X2} <- lists:zip(P1, P2)])).

% closest(P, PointList) -> {Index, Distance}
%   Find the point in PointList that is closest to P.
%   Parameters:
%     P must be a list of numbers.
%     PointList must be a non-empty list of lists of numbers.
%     The length of P and the length of each list in PointList must be the same.
%   Return value:
%     Index: the index in PointList of the closest point to P.
%     Distance: the distance from P to the closest point in PointList.
%     If there are ties, then the lowest such Index is returned.
%   Examples:
%     closest([1,2,3], [[4,5,6], [0,1,4], [1,1,2], [2,3,4]]) ->
%       {3,1.4142135623730951}
%     closest([1,2,3], [[4,5,6], [0,1,4], [1,8,2], [2,3,4]]) ->
%       {2,1.7320508075688772}
%     closest([1,2,3], [[4,5,6], [0,8,4], [1,8,2], [2,3,4]]) ->
%       {4,1.7320508075688772}
closest(_P, []) -> {0, 0.0};
closest(P, [H | PointList]) ->
  {I, D} = closest(P, PointList),
  HD = distance(P, H),
  if
    (D >= HD) or (I == 0) -> {1, HD};
    true -> {I+1, D}
  end.


% allTails(L) -> list of all suffixes of L
%   L must be a list.
%   Examples:
%     allTails([1, 2, 3]) -> [[], [3], [2,3], [1,2,3]]
%     allTails([]) -> [[]]
allTails([]) -> [[]];
allTails(L) -> allTails(tl(L)) ++ [L]. % stub

% sloppyZip(L1, L2) -> Z
%   L1 and L2 must be lists.
%   Z is the list of pairs of corresponding elements of L1 and L2.
%   length(Z) = min(length(L1), length(L2)) -- in other words, we stop
%   zipping when we reach the end of either list.  Compare with lists:zip/2
%   that requires the two lists to be of the same length.  That's why this
%   version is 'sloppy'.  I wrote a head recursive version so that it keeps
%   the elements in the original order, just like lists:zip/2.
%   Examples:
%     sloppyZip([1,2,3], [cat, dogs, mice]) -> [{1,cat}, {2,dogs}, {3,mice}]
%     sloppyZip([1,2,3], [cat, dogs]) -> [{1,cat}, {2,dogs}]
sloppyZip([], _) -> [];
sloppyZip(_, []) -> [];
sloppyZip([H1 | T1], [H2 | T2]) -> [{H1, H2} | sloppyZip(T1, T2)].


% longestOverlap(L1, L2) -> {StartIndex, Length}
%   Find the longest overlapping segments of L1 and L2.
%   Two segments are overlapping if they start at the same index
%   and if they are element-by-element identical.
%   Parameters:
%     L1, L2: lists.  It is acceptable for length(L1) /= length(L2).
%                     The length(L1) == length(L2) case is acceptable as well.
%   Return value: {StartIndex, Length}
%     StartIndex and Length are both integers: StartIndex is positive and
%     Length is non-negative.  For StartIndex =< I < StartIndex+Length,
%       lists:nth(I, L1) =:= lists:nth(I, L2)
%     We return StartIndex and Length for the longest such segment.  In the
%     case of a tie, we return the first one (i.e. the one with the smallest
%     value of StartIndex.
%     If L1 and L2 have no overlapping segments, we return {1, 0}.
longestOverlap([], _L2) -> {1, 0};
longestOverlap(_L1, []) -> {1, 0};
longestOverlap(L1, L2) ->
  Len = lengthOfOverlap(L1, L2),
  if
    (length(L1) == Len) or (length(L2) == Len) -> {1, Len};
    true ->
      {I, L} = longestOverlap(lists:nthtail(Len+1, L1),
                              lists:nthtail(Len+1, L2)),
      if
        Len >= L -> {1, Len};
        true -> {I+Len+1, L}
      end
  end.

lengthOfOverlap([H1 | L1], [H2 | L2]) when H1 == H2 -> lengthOfOverlap(L1, L2) + 1;
lengthOfOverlap(_L1, _L2) -> 0.


% The function allLess was described in the homework as an example of
%   a function where adding comprehensive guards can cause an unacceptable
%   loss of performance.  In this case, if the guards for T1 and T2 are
%   added, the time for allLess grows from O(N) to O(N^2).
%   allLess(L1, L2) returns true if L1 and L2 are of the same length, bot
%   are lists of numbers, and if each element of L1 is less than the
%   corresponding element of L2.
%   If L1 and L2 have different length, or if either has a non-numeric
%   element, then the outcome may depend on which guard expression is used.
%   The details are left as an exercise for those who are really into such
%   things.
allLess([], []) -> true;
allLess([H1 | T1], [H2 | T2])
    when is_number(H1), is_number(H2) ->
         % is_list(T1), is_list(T2), length(T1) == length(T2) ->
  (H1 < H2) andalso allLess(T1, T2).

