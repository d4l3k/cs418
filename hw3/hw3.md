# CPSC 418: Homework 3

Tristan Rice, q7w9a, 25886145

# Question 2

## (b) (4 points) Report the elapsed time for

Note: Compiled with `gcc -O3` on thetis.

### i. Sorting an array of 1,000,000 random elements.

```
$ ./hw3 random array 1000000 100
random array n=1000000 n_trial=100 t_avg=7.868e-02
```

### ii. Sorting an array of 1,000,000 ascending elements.

```
./hw3 ascending array 1000000 100
ascending array n=1000000 n_trial=100 t_avg=2.112e-02
```

### iii. Sorting a list of 1,000,000 random elements.

```
./hw3 random list 1000000 100
random list n=1000000 n_trial=100 t_avg=2.174e-01
```

### iv. Sorting a list of 1,000,000 ascending elements.

```
./hw3 ascending list 1000000 100
ascending list n=1000000 n_trial=100 t_avg=8.000e-04
```

## (c) (8 points) Which takes longer: sorting an array of ascending elements or sorting an array of random elements? By what factor?

Sorting an array of ascending elements is significantly faster, by almost a
factor of 4.

Branch prediction plays a large role in this, since at every compare operation
the left element will be smaller than the right one. This allows for it to
predict the correct branch in every iteration of the merge operation, this saves
wasted cycles due to prediction errors.

We also get a lot of cache performance since the merge operation only needs to
access the first element of the right side since all the left elements will be
smaller than all those on the right. This means that sorting ascending numbers
needs to access half the number of elements as sorting random numbers in each
merge step leading to vast performance improvements since there will be way
fewer cache misses.

## (d) (8 points) Which takes longer: sorting a list of ascending elements or sorting a list of random elements? By what factor? Is the ratio bigger or smaller than for arrays?

Sorting a list of ascending elements is much faster than sorting a list of
random elements by a factor of ~270. The ratio is much larger than for arrays.

Sorting lists is much harder on the cache since you don't get the benefit of
sequences of elements being sequential in memory. Thus, sorting the random list
is much slower than a random array.

However, with lists we get a large performance increase with ascending numbers.
The same factors that increase performance for ascending arrays also play in
here. As well as the fact that with sorted lists there are no need to do
extraneous memory copying with a temp array as with sorting arrays. Thus, there
are no changes to the list in memory which allow for good caching and no slow
writes.


# Question 3

## (a) Derive a formula for p(k).

$p(k) = 6^k$.

## (b) (5 points) What is the bisection width of a machine with $10^k$ nodes?

For k=1 nodes, the bandwidth is 5.

For k=2 nodes, there are 6 top level switches, each with 10 connections. We have
to cut half of those to bisect. Thus the width is 30.

For k=3 nodes, there are still 6 top level switches, each with 10 connections.
Thus there is still bandwidth of 30.

## (c) How long does it take to send these messages?

Each node on the left is sending 1KB to the opposite side. The total amount of
data that needs to be transfered is $1KB(\frac{10^{k}}{2})$. Since there is a
fixed bandwidth across the bisection, the amount of time will be

$$\frac{1KB(\frac{10^{k}}{2})}{30 gb/s}$$

$$=133.33ns(10^{k})$$

When $k=4$, it will take 1.3333 milliseconds.

## (d) What is the bisection width of a toroidal machine with $10^k$ nodes?

Since it's a grid, we can bisect by cutting anywhere aligned with the grid. This
would give us a bisection width of $2(10^{k/2})$, since the grid has height
$10^{k/2}$, and the edges wrap around.

## (e)  If we only consider the time to transfer the data across the network bisection, how long does it take to send these messages?

Once again we have to send $1KB(\frac{10^k}{2})$.
The total bandwidth is $2(10^{k/2})gbps$.

$$\frac{1KB(\frac{10^k}{2})}{2(10^{k/2})gbps}$$
$$=2\mu s (10^{k/2})$$

$$ 2\mu s (10^{4/2}) = 200 \mu s$$

