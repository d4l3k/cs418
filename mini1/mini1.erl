-module(mini1).

-export([who_am_i/0, findCats/1, rpn/1]).

% who_am_it -> StudentTuple
%   StudentTuple has two elements:
%     your name,
%     your student number
who_am_i() ->  {"Tristan Rice", 25886145}. % This is a stub.  You should write the real version.

% findCats(String) -> CatIndices
%   CatIndices is a list of all positions in String that are the start of
%   of the substring "cat".  CatIndices should be in ascending order.
%   For example,
%     findCats("My cat scattered cattle in Sascatchewan.") -> [4,9,18,31]
findCats(String) -> findCats(String,1).  % This is a stub. You should write the real version.
findCats([], _) -> [];
findCats("cat" ++ String, Pos) -> [Pos | findCats(String, Pos+3)];
findCats(String, Pos) -> findCats(tl(String), Pos+1).

% rpn(InputList) -> Stack
%   InputList is a list of numbers (integer or floating point) and
%     operators ('+', '-', '*', '/', 'p', and 'P').
%   Stack is the result of applying an RPN calculator to this sequence,
%     starting from an empty list.  The first element of Stack is the
%     top-of-stack element.
%   Examples:
%     rpn([2, 3, '+']) -> 5
%     rpn([2, 3, '*', 4, 5, '*']) -> [20, 6].
rpn(In) -> rpn(In,[]).
rpn([], Stack) -> Stack;
rpn(['+' | Rest], [X1, X2 | Stack]) -> rpn(Rest, [X2+X1 | Stack]);
rpn(['-' | Rest], [X1, X2 | Stack]) -> rpn(Rest, [X2-X1 | Stack]);
rpn(['*' | Rest], [X1, X2 | Stack]) -> rpn(Rest, [X2*X1 | Stack]);
rpn(['/' | Rest], [X1, X2 | Stack]) -> rpn(Rest, [X2/X1 | Stack]);
rpn(['p' | Rest], [X | Stack]) -> io:format("X = ~w~n", [X]), rpn(Rest, [X | Stack]);
rpn(['P' | Rest],  Stack) -> io:format("Stack = ~w~n", [Stack]), rpn(Rest,  Stack);
rpn([X | Rest],Stack) -> rpn(Rest, [X | Stack]).


