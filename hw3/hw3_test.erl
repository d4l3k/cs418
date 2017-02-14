-module(hw3_test).
-include_lib("eunit/include/eunit.hrl").

-import(hw3, [primes/1, primes/2, primes/3, sum_inv_twin_primes/1,
              sum_inv_twin_primes/2]).

-define(errTol, 1.0e-12).  % round-off error tolerance for floating point results

primes_init() ->
  W = wtree:create(8),
  W.

primes_wrapper(W, N) ->
  _X = primes(W, N, primes),
  lists:append(workers:retrieve(W, primes)).

primes_test_cases(W) ->
  [
    ?_assertEqual(primes(1000), primes(1,1000)),
    ?_assertEqual(primes(1000), primes_wrapper(W, 1000))
  ].

primes_test_() ->
  {setup, fun() -> primes_init() end,         % set up
          fun(W) -> wtree:reap(W) end,      % clean up
          fun(W) -> primes_test_cases(W) end  % the test cases
  }.


sum_inv_twin_primes_init() ->
  W = wtree:create(8),
  W.

sum_inv_twin_primes_wrapper(W, N) ->
  primes(W, N, primes),
  sum_inv_twin_primes(W, primes).

sum_inv_twin_primes_test_cases(W) ->
  [
    ?_assert(abs(sum_inv_twin_primes(1000)
                 - sum_inv_twin_primes_wrapper(W, 1000))
             =< errTol),
    ?_assert(abs(sum_inv_twin_primes(0)
                 - sum_inv_twin_primes_wrapper(W, 0))
             =< errTol),
    ?_assert(abs(sum_inv_twin_primes(10)
                 - sum_inv_twin_primes_wrapper(W, 10))
             =< errTol),
    ?_assert(abs(sum_inv_twin_primes(100000)
                 - sum_inv_twin_primes_wrapper(W, 10000))
             =< errTol)
  ].

sum_inv_twin_primes_test_() ->
  {setup, fun() -> sum_inv_twin_primes_init() end,         % set up
          fun(W) -> wtree:reap(W) end,      % clean up
          fun(W) -> sum_inv_twin_primes_test_cases(W) end  % the test cases
  }.
