% some trivial tests for mini1
% These are mostly just the examples from the homework problem statements.
% These are certainly not a complete set of tests.  You should modify this
% and add tests of your own.
-module(mini1_test).

-include_lib("eunit/include/eunit.hrl").
-import(mini1, [who_am_i/0, findCats/1, rpn/1]).

who_am_i_test() ->
  ?assert(is_tuple(who_am_i())),
  ?assertEqual(2, tuple_size(who_am_i())),
  ?assert(is_list(element(1, (who_am_i())))),
  ?assert(is_integer(element(2, (who_am_i())))).

findCats_test() ->
  ?assertEqual([4,9,18,31], findCats("My cat scattered cattle in Sascatchewan.")),
  ?assertEqual([], findCats("")),
  ?assertEqual([], findCats("duck")),
  ?assertEqual([1], findCats("cat")).

rpn_test() ->
  ?assertEqual([5], rpn([2, 3, '+'])),
  ?assertEqual([26], rpn([2, 3, '*', 4, 5, '*', '+'])),
  ?assertEqual([26], rpn([2, 3, '*', 4, 5, 'p', '*', '+'])),
  ?assertEqual([26], rpn([2, 3, '*', 4, 5, 'P', '*', '+'])),
  ?assertEqual([5, 4, 6], rpn([2, 3, '*', 4, 5])),
  ?assertEqual([3], rpn([5, 2, '-'])),
  ?assertEqual([10], rpn([5, 2, '*'])),
  ?assertEqual([2.5], rpn([5, 2, '/'])).



