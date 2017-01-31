-module(hw2).

-export([mean/2, vec_add/2, vec_sum/1, vec_mean/2, set_nth/3]).
-export([bank_statement/4, sample_transactions/0]).
-export([sliding_average/5]).
-export([sliding_average_help/3, sliding_average_seq/3, avg_demo/0]).
-export([compact_transactions/1, process_transaction/2]).

-import(hw1, [closest/2]).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Q1: mean
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% mean(WTree, DataKey) -> Mean
%   WTree is a worker-tree produces by wtree:create.
%   DataKey is the name of a list of numbers that is distributed across the workers.
%   Here's an example of how one could use this:
%       W = wtree:create(8).  % create a tree of eight worker processes
%       L0_1000 = lists:seq(0,1000), ok.  % create the list [0, 1, 2, ..., 1000].
%       % divide L0_1000 into length(W) pieces and send each piece to a worker in W.
%       % each worker associates its piece with the key 'data'.
%       workers:update(W, data, misc:cut(L0_1000, W)).
%       Mean = hw2:mean(W, data).
%   The last command should set Mean to 500.0
mean(WTree, DataKey) ->
  {N, S} = wtree:reduce(WTree,
                        fun(ProcState) ->
                            Nums = wtree:get(ProcState, DataKey),
                            {length(Nums), lists:sum(Nums)}
                        end,
                        fun(Left, Right) ->
                            {LN, LS} = Left,
                            {RN, RS} = Right,
                            {LN+RN, LS+RS}
                        end),
  if
    N == 0 -> error(empty);
    true -> S/N
  end.



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                          %
% Q2: vec_mean                                                             %
%                                                                          %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% vec_mean(WTree, DataKey) -> Mean
%   WTree is a worker-tree produces by wtree:create.
%   DataKey is the name of a list of vectors that is distributed across the workers.
%   Here's an example of how one could use this:
%       W = wtree:create(8).  % create a tree of eight worker processes
%       % create a list
%       VList = [hw2:vec_add(N, [1,2,3]) || N <- lists:seq(0,1000)], ok.
%       % divide VList and send the pieces to the workers
%       workers:update(W, vdata, misc:cut(VList, W)).
%       VMean = hw2:vec_mean(W, vdata).
%   The last command should set VMean to [501.0, 502.0, 503.0]
vec_mean(WTree, DataKey) ->
  {N, S} = wtree:reduce(WTree,
                        fun(ProcState) ->
                            Nums = wtree:get(ProcState, DataKey),
                            {length(Nums), vec_sum(Nums)}
                        end,
                        fun(Left, Right) ->
                            {LN, LS} = Left,
                            {RN, RS} = Right,
                            {LN+RN, vec_add(LS,RS)}
                        end),
  if
    N == 0 -> error(empty);
    true -> [X / N || X <- S]
  end.


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% I'm providing vec_add and vec_sum   %
% to make your life easier.           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% vec_add(X1, X2) -> X1 + X2
%   X1 and X2 can be vectors of the same length, or one or both of X1 or X2
%   may be plain numbers.  A vector is a list of numbers.
%   If both X1 and X2 are vectors, we return the vector sum:
%     vec_add([1,2,3], [1,4,9]) -> [2,6,12]
%   If X1 is a scalar and X2 is a vector, we add X1 to each element of X2.
%   Likewise if X1 is a vector and X2 is a scalar:
%     vec_add(1, [1,4,9]) -> [2,5,10]
%     vec_add([1,2,3], 2) -> [3,4,5]
%   If both X1 and X2 are scalars, then we just return their sum:
%     vec_add(1, 2) -> 3
vec_add(S1, S2) when is_number(S1), is_number(S2) -> S1 + S2;
vec_add(S1, V2) when is_number(S1), is_list(V2) -> [ S1 + S2 || S2 <- V2];
vec_add(V1, S2) when is_list(V1), is_number(S2) -> [ S1 + S2 || S1 <- V1];
vec_add(V1, V2) when is_list(V1), is_list(V2) ->
  [ S1 + S2 || {S1, S2} <- lists:zip(V1, V2)].

% vec_sum(VList) -> Sum
%   Sum is the sum of the elements using vec_add:
%   vec_sum([[1,2,3], [1,4,9], [1,8,27], 42]) -> [45, 56, 81]
%     If you try it, the Erlang shell will print [45, 56, 81] as "-8Q".
vec_sum(VList) -> lists:foldl(fun(E, Acc) -> vec_add(Acc, E) end, 0, VList).



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                          %
% Q3: set_nth                                                              %
%                                                                          %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% set_nth(N, Fun, List) -> replace the Nth element of List with Fun(lists:nth(List))
set_nth(N, Fun, List)
  when is_number(N), is_function(Fun), is_list(List), N =< length(List), N > 0
       -> lists:sublist(List, N-1) ++
          [Fun(lists:nth(N, List)) | lists:sublist(List, N+1, length(List))].



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                          %
% Q4: bank_statement                                                       %
%                                                                          %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

compact_transactions([]) -> [];
compact_transactions([ H1 | T]) ->
  {A1, N1} = H1,
  if
    A1 == withdraw ->
      compact_transactions([{deposit, -N1} | T]);
    length(T) >= 1 ->
      {A2, N2} = hd(T),
      if
        A1 == deposit andalso A2 == deposit ->
          compact_transactions([{deposit, N1 + N2} | tl(T)]);
        A1 == interest andalso A2 == interest ->
          compact_transactions([{interest, ((1 + N1/100)*(1 + N2/100)-1)*100} | tl(T)]);
        true ->
          [H1 | compact_transactions(T)]
      end;
    true -> [H1 | compact_transactions(T)]
  end.

process_transaction({withdraw, N}, Acc) ->
  Acc - N;
process_transaction({deposit, N}, Acc) ->
  Acc + N;
process_transaction({interest, N}, Acc) ->
  Acc*(1+(N/100)).

process_transactions([], Acc) -> Acc;
process_transactions([H | T], Acc) ->
  process_transactions(T, process_transaction(H, Acc)).

process_transactions_cum([], _Acc) -> [];
process_transactions_cum([H | T], Acc) ->
  Acc = process_transaction(H, Acc),
  [Acc | process_transactions_cum(T, Acc)].


% bank_statement(WTree, SrcKey, DstKey, InitialBalance)
%   Compute a bank statement for an account starting with a balance of InitialBalance.
%   WTree is a worker-tree.  The list of transactions to be performed for the statement
%     are stored in a list distributed across the workers.
%   SrcKey is the name of the list of transactions.
%   DstKey is the name under which to save the list of after-transaction balances.
bank_statement(W, SrcKey, DstKey, InitialBalance) ->
  Leaf1 = fun(ProcState) ->
    T = compact_transactions(wtree:get(ProcState, SrcKey)),
    {T, process_transactions(T, 0)}
  end,
  Leaf2 = fun(ProcState, AccIn) ->
    {_T, N} = AccIn,
    Src = wtree:get(ProcState, SrcKey),  % get the original list
    Result = process_transactions_cum(Src, N),    % compute the cummulative sum
    wtree:put(ProcState, DstKey, Result) % save the result -- must be the last expression
  end,                                   %   in the Leaf2 function
  Combine = fun({T1, N1}, {T2, _N2}) -> {compact_transactions(T1 ++ T2),
                                        process_transactions(T2, N1)} end,
  wtree:scan(W, Leaf1, Leaf2, Combine, {[], InitialBalance}).

% sample_transactions provides the transaction list for the example from
%   the problem statement.
sample_transactions() ->  [ % a list of 21 transactions for testing bank_statement
  % the commented columns correspond to the balance after each transaction with an
  %  initial balance of    0.00      1000.00
  {deposit,  100.00},  % 100.00      1100.00
  {deposit,  500.00},  % 600.00      1600.00
  {withdraw,  50.00},  % 550.00      1550.00
  {withdraw,  25.00},  % 525.00      1525.00
  {withdraw,  14.00},  % 511.00      1511.00
  {withdraw,  11.00},  % 500.00      1500.00
  {interest,   5.00},  % 525.00      1575.00
  {withdraw,  17.00},  % 508.00      1558.00
  {deposit,   42.00},  % 550.00      1600.00
  {interest,  -3.00},  % 533.50      1552.00
  {withdraw, 123.45},  % 410.05      1428.55
  {deposit,   19.95},  % 430.00      1448.50
  {interest,   2.33},  % 440.02      1482.25
  {deposit,   -0.33},  % 439.69      1481.92
  {withdraw, 192.68},  % 247.01      1289.24
  {withdraw, 300.00},  % -52.99       989.24
  {interest,  20.00},  % -63.59      1187.09
  {deposit,   10.00},  % -53.59      1197.09
  {interest,  -5.00},  % -50.91      1137.23
  {withdraw,  14.00},  % -64.91      1123.23
  {deposit,  100.00}   %  35.09      1223.23
].



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                          %
% Q5: sliding average                                                      %
%                                                                          %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

last_two(Data) ->
  DLen = length(Data),
  if
    DLen >= 2 -> lists:sublist(Data,DLen-1,2);
    true -> Data
  end.

sliding_average(WTree, SrcKey, DstKey, Kernel, InitialPrefix) ->
  Leaf1 = fun(ProcState) ->
    last_two(wtree:get(ProcState, SrcKey))
  end,
  Leaf2 = fun(ProcState, AccIn) ->
    Src = wtree:get(ProcState, SrcKey),
    %Result = sliding_average(AccIn++Src, Kernel),
    Result = sliding_average_seq(Src, Kernel, AccIn),
    wtree:put(ProcState, DstKey, Result)
  end,
  Combine = fun(Left, Right) -> last_two(Left++Right) end,
  wtree:scan(WTree, Leaf1, Leaf2, Combine, InitialPrefix).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The next three functions: dp, sliding_average_help, and        %
%   sliding_average_seq provide a sequential implementation      %
%   of sliding_average.  You can use this to get a clearer       %
%   idea of what sliding_average should do.  In particular:      %
%     sliding_average_equiv(SrcList, Kernel, InitialPrefix) ->   %
%       W = wtree:create(),                                      %
%       workers:update(W, src, misc:cut(SrcList, W)),            %
%       sliding_average(W, src, dst, Kernel, InitialPrefix),     %
%       Result = lists:append(workers:retrieve(W, dst)),         %
%       wtree:reap(W), % clean-up                                %
%       Result.                                                  %
%   should return the same thing as sliding_average_help.        %
%   Of course, you may use any of this code, "as is" or with     %
%   modifications, in your solution.                             %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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

% avg_demo() performs the computation described in the Figure for the
%   sliding window question and returns the result.
avg_demo() ->
  Kernel =  [0.25, 0.25, 0.5],
  SrcList = [1, 2, 10, 12, 14, 2, 2, 2, 0, -1, -8, 14],
  InitialPrefix = [0, 0],
  DstList = sliding_average_seq(SrcList, Kernel, InitialPrefix),
  DstList.
