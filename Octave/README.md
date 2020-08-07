## Octave tutorial:
### Basic:
* % comment
* ~= not equal
* 0 false
* 1 true
* PS1(‘>> ‘); change the prompt
* Put an semi column at the end can suppress the output
* disp(xxxx) display sth on screen
* format long/short  change the format of output
* A = [1 2; 3 4; 5 6]   A is a 3*2 matrix. And the semi column says go to the next row of matrix.
* V = 1:0.1:2. Set V to a brunch of elements start form 1, increment 0.1 to 2. Result is a 11*1 matrix.
* ones(2,3) 2*3 matrix with all positions 1.
* zeros(1,3) ...
* rand(3,3) 3*3 of all random numbers between 0 and 1
* randn(1,3) three values drawn form a Gaussian distribution with mean 0 and variance or standard deviation equal to 1.
* hist(w) plot a histogram of w
* hist(w,50) plot a histogram with 50 bins.
* eye(4) create a 4*4 identity matrix
* help eye/rand/help help document

### Move data around:
* size(A) return 1*2 matrix of the size of matrix A
* size(A,1) return the first-dimension of of A(3 in 3*2)
* length(A) the longest dimension of a matrix (3 in 3*2)
---
* pwd/ls/cd
* load priceY.dat or load(‘priceY.dat’)   To load files(single quote to represent strings)
* who    Shows the variables in the current scope(priceY is also in the list and can be used(A 47*1 matrix))
* whos  Same as who but also list size,buyers....
* clear priceY   Variable priceY will disappear
* v = priceY(1:10)  now v stores the first 10 elements of vector Y
* save hello.mat v;  save v to a file named hello.mat
* save hello.txt v -ascii  so that the file can be read by human. The pervious one is in binary form
* clear  clear all variables in the workspace
* A(3,2) indexing to 3,2 element of the matrix A
* A(2,:)  second row of A
* A(:,2)  second column of A
* A([1 3],:)  All elements in row 1 and 3
* Can do assignment in the above notation too
* A = [A,[100;101;102]]  append another column to right 
* A(:)     Put all elements of A into a single vector
* C = [A B]  concat A B together, A on the left, B on right 
* C = [A;B]  A on top, B on bottom
* a = 1, b = 2, c = 3  comma chaining of function calls. Carrey out 3 comments at the same time

### Computation:
* A .* B  compute A with corresponding elements in B
* A .^ 2  element wise squaring of A
* log(a)  exp(a)...abs(a)
* v + 1  add 1 to each element of v
* A’  A transpose
* max(v) the biggest value in vector v, while max(A) returns the largest value of each column in A
* [val, ind] = max(a)  val = largest value in a, ind = the index of that number
* v < 3       element wise comparison 
* find(a < 3)  returns the index of elements less than 3
* magic(3)  generate 3*3 matrix that each row, each column, each diagonals all add up to the same number
* sum(a)    Add up all elements in a
* prod(a)  the product of all elements in a
* floor(a) rounds down, while ceil(a) rounds up
* max(rand(3),rand(3))  a 3*3 matrix with max number in each position
* max(A,[],1) max in each column in A
* .........2  max in each row in A
* max(A)  the default is each row, and if want to find the max in whole matrix, use max(max(A))
* sum(A,1)  column sum
* sum(A,2)  row sum
* flipud(A)  matrix flips up down
* pinv(A)  inverse of A

### Plots:
* t = [0:0.01;0.98];
* y1 = sin(2*pi*4*t);
* plot(t,y1);     t as x-axis, y1 as y-axis
* hold on;        Plot new figure on the top of the previous one
* plot(t,y2,’r’);   R is the color red
* xlabel(‘time’)
* title(‘my plot’)
* print -dpng ‘myPlot.png’    Save as a Image
* close  close figure
* figure(1); plot(t,y1);
* figure(2); plot(t,y2);     Create two figures
* subplot(1,2,1);   Divided figure to 1*2 grid, access the first one 
* axis([0.5 1 -1 1])  set x from 0.5 to 1, y to -1 to 1
* clf;    Clear figure
* image(magic(15)), colarbar, colarmap  create a figure with a element in magic(15) a block of color that corresponds to colorbar.

### Loops:
```
for i = 1 : 10,
 	v(i) = 2 ^ i;
  end;
```

```i = 1;
while i <= 5.
	v(i) = 100;
	i = i + 1;
end;```

```if i == 6,
	disp(‘the value is 6’);
elseif i == 7,
	...;
else
	disp(‘...’);
end;```

```If ....,
	....;
end;```

### Function:
Create a file named by function name(end in .m), declare function there.

E.g., in squareThisNumber.m file
```function y = squareThisNumber(x)
y = x^2;```

In file, y is the value to return; x is the argument. y = x^2 is the function body. Just move to the proper root and enter squareThisNumber(5) to call function. Or use  addpath(‘C:\Users\...’) to add path in to the Octave search path so that Octave search function there.

Octave can also create function that returns two values.
```function [y1,y2] = squareAndCubeThisNumber(x)
y1 = x^2;
y2 = x^3;```

E.g. 
`[a,b] = squareAndCubeThisNumber(5);`
And then a = 25, b = 125

