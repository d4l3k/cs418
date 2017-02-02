-module(hw2_test).

-include_lib("eunit/include/eunit.hrl").

-import(hw2, [mean/2, vec_mean/2, set_nth/3, bank_statement/4,
              sliding_average/5, compact_transactions/1, process_transaction/2,
             process_transactions/2, process_transactions_cum/2]).
-export([vec_close/2, sliding_average_expected/4, sliding_average_run/5]).

-define(errTol, 1.0e-12).  % round-off error tolerance for floating point results

% NOTE:  To write these tests, I had to learn to use EUnit test fixtures.
% If you like, you can read all about them in
%     http://learnyousomeerlang.com/eunit#fixtures
% Or, you can just use my code.  The test fixtures provid set-up and
% tear-down code for the test.  I use these to create the process trees
% and terminate the worker processes at the end of a batch of tests.
% I also usd the init functions to set-up test data lists in the ProcState
% maps.
%   You can use all this.  For each function required in the homework,
% you will find two functions that you can modify to add more tests:
%   function_name_init() -> WorkerTree
%     You can see where I did a workers:update(W, Key, misc:cut(List, W))
%     one or more times in each of these.  This is where I'm setting up
%     data in the ProcState maps to be used in the tests.  You can add
%     data for your test cases here.  Just give each list (or other data
%     value) that you need a different key.
%   function_name_test_cases(W) -> ListofTestCases
%     This function returns a list of tests.  You can add more
%       ?_assert(...)
%       ?_assertEqual(...)
%       ?_assertError(...)
%       ...
%   tests to the list.  Note the '_' in front of 'assert' in these macro
%   names.  This creates a tuple that describes the test to be run.  The
%   I don't think the details are essential -- it's just part of the
%   set-up, run-tests, clean-up pattern.  For writing tests, you just need
%   to remember to put that '_' in front of the name of the assert macro.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                          %
%  Q1: mean(WTree, DataKey)                                                %
%                                                                          %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

mean_init() ->
  W = wtree:create(8),
  % The example from the template file.
  workers:update(W, data, misc:cut(lists:seq(1,1000), W)),

  % by using a list of integers and making the total number of elements
  % in the list a power of two, we shouldn't have to deal with roundoff error
  wtree:rlist(W, 64, 42, random_ints),

  % OK, I'll deal with the round-off to show how it can be done.
  wtree:rlist(W, 42, 1.0, random_floats),

  % If our list is shorter than W, then some of the workers will have
  %   empty lists.  mean(W, Key) should compute the correct answer anyway.
  workers:update(W, shortlist, misc:cut(lists:seq(1, (2*length(W)) div 3), W)),

  % mean should fail for an empty list.
  workers:update(W, emptylist, misc:cut([], W)),

  % You should add at least one more test case: create the data here
  % and test it in the mean_test_cases(W) function below.

  W. % return the worker tree
% end mean_init

mean_test_cases(W) ->
  Mean = fun(List) -> lists:sum(List) / length(List) end, % reference function
  [ ?_assertEqual(Mean(lists:append(workers:retrieve(W, data))),
                  mean(W, data)),
    ?_assertEqual(Mean(lists:append(workers:retrieve(W, random_ints))),
                  mean(W, random_ints)),
    ?_assert(    abs(   Mean(lists:append(workers:retrieve(W, random_floats)))
                      - mean(W, random_floats))
	      =< errTol),
    ?_assert(    abs(   Mean(lists:append(workers:retrieve(W, shortlist)))
                      - mean(W, shortlist))
	      =< errTol),
    ?_assertError(_, mean(W, emptylist))
    % You should add at least one more test case: create the in the
    % mean_init() function above and test it here.
  ].

mean_test_() ->
  {setup, fun() -> mean_init() end,         % set up
          fun(W) -> wtree:reap(W) end,      % clean up
          fun(W) -> mean_test_cases(W) end  % the test cases
  }.


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                          %
%  Q2: vec_mean(WTree, DataKey)                                            %
%                                                                          %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

vec_mean_init() ->
  W = wtree:create(42),
  % The example from the template file.
  VList = [vec_add(N, [1,2,3]) || N <- lists:seq(0,1000)],
  workers:update(W, vdata, misc:cut(VList, W)),
  workers:update(W, vempty, misc:cut([[],[],[]], W)),

  % You should add more test cases.  The examples from mean_init() and
  % mean_test_cases() should provide "inspiration".
  W.

% vec_add (V1, V2)
%   vec_add was provided in the template for hw2.erl.  There was nothing
%   in the assignment that said that it has to remain in the solution.
%   I'm creating a copy in this module so I know that I'll get the
%   function that I intended.
vec_add(S1, S2) when is_number(S1), is_number(S2) -> S1 + S2;
vec_add(S1, V2) when is_number(S1), is_list(V2) -> [ S1 + S2 || S2 <- V2];
vec_add(V1, S2) when is_list(V1), is_number(S2) -> [ S1 + S2 || S1 <- V1];
vec_add(V1, V2) when is_list(V1), is_list(V2) ->
  [ S1 + S2 || {S1, S2} <- lists:zip(V1, V2)].

% vec_sum(VList) -> Sum
%   vec_sum was provided in the template for hw2.erl.  Again, I'm including
%   a copy here to make sure the test cases have the function I want.
vec_sum(VList) -> lists:foldl(fun(E, Acc) -> vec_add(Acc, E) end, 0, VList).

% check if two vectors are "close"
%   If all components are the same to within ?errTol, we return true
%   Otherwise, we return a list where each element is 'ok' if the corresponding
%     elements of V1 and V2 are close enough, and {X1,X2} otherwise, where X1
%     is the element of V1, and x2 is the element of V2.
vec_close(V1, V2) ->
  case length(V1) == length(V2) of
    true ->
      case lists:all(fun({X1,X2}) -> abs(X1-X2) =< ?errTol end, lists:zip(V1, V2)) of
	true -> true;
	_ -> [    case abs(X1-X2) =< ?errTol*min(abs(X1), 1) of
		    true  -> ok;
		    false -> {X1,X2}
		  end
	       || {X1, X2} <- lists:zip(V1, V2)
	     ]
      end;
    false -> {mismatched_vector_lengths, V1, V2}
  end.

vec_mean_test_cases(W) ->
  VecMean = fun(List) ->
    N = length(List),
    [ X/N || X <- vec_sum(List) ]
  end,
  [ ?_assert(vec_close(VecMean(lists:append(workers:retrieve(W, vdata))),
                      vec_mean(W, vdata)))
    % You should add more test cases here.
  ].

vec_mean_test_() ->
  {setup, fun() -> vec_mean_init() end,         % set up
          fun(W) -> wtree:reap(W) end,          % clean up
          fun(W) -> vec_mean_test_cases(W) end  % the test cases
  }.


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                          %
%  Q3: set_nth(N, Fun, List)                                               %
%                                                                          %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


set_nth_test_() ->
  L = [1,4,9,16],
  F = fun(X) -> X*X + 2 end,
  [ ?_assertEqual([1,4,83,16], set_nth(3, F, L)),
    ?_assertEqual([1,4,9,258], set_nth(4, F, L)),
    ?_assertEqual([3,4,9,16], set_nth(1, F, L)),
    ?_assertError(_, set_nth(5, F, L)),
    ?_assertError(_, set_nth(0, F, L)),
    ?_assertError(_, set_nth(1, F, 5)),
    ?_assertError(_, set_nth(1, 5, L))
    % you should add more tests here
  ].


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                                 %
%  Q4: bank_statement(WTree, SrcKey, DstKey, InitialBalance)set_nth(N, Fun, List) %
%                                                                                 %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

compact_transactions_test_() ->
  [ ?_assertEqual([], compact_transactions([])),
    ?_assertEqual([{deposit, 5}], compact_transactions([{withdraw, -5}])),
    ?_assertEqual([{deposit, 11}],
                  compact_transactions([{withdraw, -5}, {deposit, 6}])),
    ?_assertEqual([{interest, 125.0}],
                  compact_transactions([{interest, 50}, {interest, 50}])),
    ?_assertEqual([{interest, 50}, {deposit, 5}, {interest, 50}, {deposit, 10}],
                  compact_transactions([{interest, 50}, {withdraw, -5},
                                        {interest, 50}, {deposit, 10}]))
  ].
process_transaction_test_() ->
  [ ?_assertEqual(10, process_transaction({deposit, 6}, 4)),
    ?_assertEqual(10, process_transaction({withdraw, -6}, 4)),
    ?_assertEqual(15.0, process_transaction({interest, 50}, 10))
  ].

process_transactions_test_() ->
  [
   ?_assertEqual(11.0, process_transactions([{deposit, 6}, {interest, 10}], 4)),
   ?_assertEqual([10, 11.0], process_transactions_cum([{deposit, 6}, {interest, 10}], 4))
  ].

bank_statement_init() ->
  W = wtree:create(8),
  Transactions = [ % example from the hw2.erl template
    {deposit,  100.00},
    {deposit,  500.00},
    {withdraw,  50.00},
    {withdraw,  25.00},
    {withdraw,  14.00},
    {withdraw,  11.00},
    {interest,   5.00},
    {withdraw,  17.00},
    {deposit,   42.00},
    {interest,  -3.00},
    {withdraw, 123.45},
    {deposit,   19.95},
    {interest,   2.33},
    {deposit,   -0.33},
    {withdraw, 192.68},
    {withdraw, 300.00},
    {interest,  20.00},
    {deposit,   10.00},
    {interest,  -5.00},
    {withdraw,  14.00},
    {deposit,  100.00}
  ],
  %workers:update(W, transactions, misc:cut(Transactions, W)),
  workers:update(W, transactions, misc:cut(Transactions, W)),
  workers:update(W, shortlist, misc:cut([{deposit, 10}, {interest, 10},
                                         {withdraw, 10}], W)),
  workers:update(W, emptylist, misc:cut([], W)),
  W.

bank_statement_run(W, SrcKey, DstKey, InitialBalance) ->
  bank_statement(W, SrcKey, DstKey, InitialBalance),
  lists:append(workers:retrieve(W, DstKey)).

bank_statement_test_cases(W) ->
  [
   ?_assert(vec_close(
      [],
      bank_statement_run(W, emptylist, statement, 100.0)
    )),
   ?_assert(vec_close(
      [10.0, 11.0, 1.0],
      bank_statement_run(W, shortlist, statement, 0.0)
    )),
   ?_assert(vec_close(
      [ 200.00, 700.00, 650.0, 625.0, 611.0, 600.0, 630.0, 613.0, 655.0,
        635.35, 511.90, 531.85, 544.242105, 543.912105, 351.232105,
        51.232105, 61.478526, 71.478526, 67.9045997, 53.9045997, 153.9045997
      ],
      bank_statement_run(W, transactions, statement, 100.0)
    ))
    % You should add more test cases here
  ].

bank_statement_test_() ->
  {setup, fun() -> bank_statement_init() end,         % set up
          fun(W) -> wtree:reap(W) end,                % clean up
          fun(W) -> bank_statement_test_cases(W) end  % the test cases
  }.


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                          %
% Q5: sliding average                                                      %
%                                                                          %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


sliding_average_init() ->
  W = wtree:create(8),
  % The example from the template file.
  Data = [ I rem 5 || I <- lists:seq(1, 57) ],
  workers:update(W, data, misc:cut(Data, W)),
  workers:update(W, emptylist, misc:cut([], W)),
  workers:update(W, shortlist, misc:cut(lists:seq(1, (2*length(W)) div 3), W)),
  W.

sliding_average_test_cases(W) ->
  Kernel1 = [0.125, 0.25, 0.625],
  InitialPrefix1 = [0, -20],
  [
    ?_assert(vec_close(
      sliding_average_expected(W, emptylist, Kernel1, InitialPrefix1),
      sliding_average_run(W, emptylist, data_slide, Kernel1, InitialPrefix1))),
    ?_assert(vec_close(
      sliding_average_expected(W, data, Kernel1, InitialPrefix1),
      sliding_average_run(W, data, data_slide, Kernel1, InitialPrefix1))),
    ?_assert(vec_close(
      sliding_average_expected(W, shortlist, Kernel1, InitialPrefix1),
      sliding_average_run(W, shortlist, data_slide, Kernel1, InitialPrefix1)))
  ].

sliding_average_test_() ->
  {setup, fun()  -> sliding_average_init() end,        % set up
          fun(W) -> wtree:reap(W) end,                 % clean up
          fun(W) -> sliding_average_test_cases(W) end  % the test cases
  }.

sliding_average_expected(W, SrcKey, Kernel, InitialPrefix) ->
  sliding_average_seq(lists:append(workers:retrieve(W, SrcKey)),
                      Kernel, InitialPrefix).

sliding_average_run(W, SrcKey, DstKey, Kernel, InitialPrefix) ->
  sliding_average(W, SrcKey, DstKey, Kernel, InitialPrefix),
  lists:append(workers:retrieve(W, DstKey)).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The next four functions: dp, sliding_average_help, sliding_average_seq, %
%   and sliding_average_equiv provide a sequential implementation of      %
%   sliding_average.  I got these from the template file for hw2.erl.     %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% dp(DataVec, KernelVec, Acc) -> DotProduct | done.
%   This is a variant on dotproduct.
%     If length(DataVec) >= length(KernelVec), then
%       we compute the dot product of KernelVec with the first
%       length(KernelVec) elements of DataVec.
%     Otherwise (DataVec is shorter than KernelVec)
%       we return 'done'.
dp(_Data, [], Acc) -> Acc;
dp([], _Kernel, _Acc) -> done;
dp([HData | TData], [HKernel | TKernel], Acc) ->
  dp(TData, TKernel, Acc + HData*HKernel).

% sliding_average_help(Data, Kernel, Acc) -> AvgData
%   Slide the Kernel along Data, computing the weighted averages as we go.
%   When Kernel is longer than the remaining Data, we return.
%     We want AvgData to be in the same order as Data, but we also want to
%   be able to handle large lists for Data.  So, the implementation is
%   tail-recursive with a call to lists:reverse at the end.  This means
%   we make two tail-recursive passes, but we avoid creating millions of
%   stack frames.
sliding_average_help(Data, Kernel, Acc) ->
  case dp(Data, Kernel, 0.0) of
    done -> lists:reverse(Acc);
    X -> sliding_average_help(tl(Data), Kernel, [X | Acc])
  end.

% sliding_average_seq(SrcList, Kernel, InitialPrefix)
%   A sequential implementation of sliding_average.
sliding_average_seq(SrcList, Kernel, InitialPrefix)
    when is_list(Kernel), is_list(InitialPrefix),
         length(InitialPrefix) =:= length(Kernel)-1 ->
  ExtendedSrc = InitialPrefix ++ SrcList,
  sliding_average_help(ExtendedSrc, Kernel, []).
