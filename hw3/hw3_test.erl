-module(hw3_test).
-include_lib("eunit/include/eunit.hrl").

-import(hw3, [primes/1, primes/2, primes/3]).

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
