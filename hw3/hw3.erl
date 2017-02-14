-module(hw3).

% the exports required for the homework problems.
-export([primes/3, sum_inv_twin_primes/2]).

% export some functions that I found handy to access while developing my solution.
-export([primes/1, primes/2, sum_inv_twin_primes/1, twin_primes/1, bench/1]).

bench(F) ->
  bench(F, [4,8,16,32,64,128,256]).

bench(_F, []) -> [];
bench(F, [H|T]) ->
  W = wtree:create(H),
  [ {H, time_it:t(fun() -> F(W) end, 10)} | bench(F, T)].


% primes(W, N, DstKey) -> ok
%   Use the workers of W to compute the primes in [2,N] and store them
%   in  list distributed over the workers for W.  DstKey is the key for
%   this distributed list.
primes(W, N, DstKey) ->
  % Here's a sequential version.
  % You need to replace the body of this function with a parallel version.
  workers:update(W, DstKey, fun(_ProcState, Arg) ->
    {Lo, Hi} = Arg,
    primes(Lo, Hi)
  end,
  misc:intervals(2, N, length(W))).


% sum_inv_twin_primes(W, SrcKey) -> Sum
%   Compute the sum of the reciprocals of the twin primes stored as a
%   distributed list for the workers of W.  SrcKey is the key associated
%   with this list of twin primes.
sum_inv_twin_primes(W, SrcKey) ->
  DstKey = brun_guess,
  wtree:scan(W,
	     fun(ProcState) -> % Leaf1
		 Primes = wtree:get(ProcState, SrcKey),
		 if
		   length(Primes) == 0 -> [];
		   true -> [lists:last(Primes)]
		 end
	     end,
	     fun(ProcState, AccIn) -> % Leaf2
		 Primes = wtree:get(ProcState, SrcKey),
		 Twins = twin_primes(AccIn ++ Primes),
		 SumInv = lists:sum([1/TP || TP <- Twins]),
		 wtree:put(ProcState, DstKey, SumInv)
	     end,
	     fun(_Left, Right) -> % Combine
		 Right
	     end,
	     []),
  lists:sum(workers:retrieve(W, DstKey)).


  % Here's a sequential version.
  % You need to replace the body of this function with a parallel version.
  %lists:sum([1/X || X <- twin_primes(lists:append(workers:retrieve(W, SrcKey)))]).

sum_inv_twin_primes(N) when is_integer(N), 0 =< N ->
  lists:sum([1/TP || TP <- twin_primes(N)]).

% twin_primes(N) return all twin primes where the larger twin is at most N.
twin_primes(N) when is_integer(N) -> twin_primes(primes(N));
twin_primes([]) -> [];
twin_primes(Primes=[_ | P_Tail]) ->
  %   It's easiest just to find all such pairs and concatenate them, but this
  %   duplicates 5 because {3,5} and {5,7} are both twin prime pairs.  This
  %   is the only such pair because for any P, one of P, P+2, or P+4 must be
  %   divisible by 3.  So, I wrote a case that fixes the [3,5,5,7 | _] sequence.
  TP_pairs  = [[P1, P2] || {P1, P2} <- lists:zip(Primes, P_Tail++[0]), P2 == P1+2],
  case lists:append(TP_pairs) of
    [3, 5, 5 | TP_Tail] -> [3, 5 | TP_Tail];
    TwinPrimes ->  TwinPrimes
  end.

primes(Lo, Hi) when is_integer(Lo) and is_integer(Hi) and (Lo > Hi) -> [];
primes(Lo, Hi) when is_integer(Lo) and is_integer(Hi) and (Hi < 5) ->
  lists:filter(fun(E) -> (Lo =< E) and (E =< Hi) end, [2,3]);
primes(Lo, Hi) when is_integer(Lo) and is_integer(Hi) and (Lo =< Hi)  ->
  M = trunc(math:sqrt(Hi)),
  SmallPrimes = primes(2, M),
  BigPrimes = do_primes(SmallPrimes, max(Lo, M+1), Hi),
  if
    (Lo =< 2) -> SmallPrimes ++ BigPrimes;
    (Lo =< M) -> lists:filter(fun(E) -> E >= Lo end, SmallPrimes) ++ BigPrimes;
    true -> BigPrimes
  end.
primes(N) -> primes(1,N). % a simple default

% do_primes(SmallPrimes, Lo, Hi) ->  the elements of [Lo, ..., Hi]
%   that are not divisible} by any element of SmallPrimes.
do_primes(SmallPrimes, Lo, Hi) ->
  lists:foldl(fun(P, L) -> lists:filter(fun(E) -> (E rem P) /= 0 end, L) end,
	      lists:seq(Lo, Hi),
	      SmallPrimes).
